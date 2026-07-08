#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end (E2E) profiling of the interactive TurboDiffusion pipeline.

Models the online streaming use-case where every request pays:
    text prompt  ─►  UMT5 encode  ─►  DiT sampling  ─►  VAE decode  ─►  frames
The model + VAE + text encoder are all resident in GPU memory (no offload).
Frames are NOT saved (streaming out over the wire, not to disk).

We report per-stage median time and the aggregate E2E FPS, so you can tell
whether text or VAE decode moves the needle relative to sampling — and how
much CP speeds up the whole request, not just the DiT.

Usage on the 5090 box:
    cd /root/autodl-tmp/TurboDiffusion
    git pull

    # 1-GPU (baseline for the streaming request budget)
    bash run_e2e_profile.sh

    # 6-GPU CP sweep for 1.3B
    MODEL=Wan2.1-1.3B bash run_e2e_profile.sh

    # 14B sweep
    MODEL=Wan2.1-14B bash run_e2e_profile.sh

Or invoke this python entry directly (single rank):
    python turbodiffusion/inference/wan2.1_t2v_e2e_profile.py \
        --model Wan2.1-1.3B \
        --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
        --num_steps 4 --warmup 3 --repeats 10
"""
import argparse
import math
import os
import sys
import time

# Make imaginaire / rcm importable regardless of cwd
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(os.path.dirname(_here)) if os.path.basename(os.path.dirname(_here)) == "turbodiffusion" else os.path.dirname(_here)
_pkg = os.path.join(_repo, "turbodiffusion")
_inference = os.path.join(_pkg, "inference")
for _p in (_pkg, _inference):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.distributed as dist
from einops import repeat

from imaginaire.utils import log
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import UMT5EncoderModel  # keep-resident encoder
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model


# --- Prompt bank: rotate to prevent per-prompt caching artifacts ---
_PROMPTS = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon.",
    "A drone flies through a canyon at golden hour, dust motes catching the light.",
    "A robot barista slides a latte across a chrome counter in slow motion.",
    "Waves crash against a rocky coastline as a lighthouse beam sweeps the darkness.",
    "A samurai draws a katana in a snow-covered bamboo forest.",
    "Fireflies drift over a still pond mirroring a full moon.",
    "A vintage train pulls into a foggy alpine station at dawn.",
    "A child's paper boat sails across a rainy city puddle.",
    "A kite made of stained glass tumbles above desert dunes.",
    "A cat leaps from rooftop to rooftop over a rain-slick market.",
    "A jazz trio plays on a Manhattan rooftop as the skyline glows.",
    "A cyclist races down a mountain switchback at sunset.",
    "An astronaut floats through a garden of zero-gravity flowers.",
    "A candle flickers on a wooden desk beside an open leather journal.",
    "A firefighter runs through a corridor of embers to rescue a puppy.",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TurboDiffusion end-to-end (text + sample + vae) profiling")
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"],
                        default="Wan2.1-1.3B")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--vae_path", type=str,
                        default="checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--aspect_ratio", default="16:9")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla")
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--quant_linear", action="store_true")
    parser.add_argument("--default_norm", action="store_true")
    parser.add_argument("--enable_sac", action="store_true")
    parser.add_argument("--sampler_dtype", choices=["fp32", "fp64"], default="fp64",
                        help="See wan2.1_t2v_infer.py — default fp64 preserves quality.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed warmup requests")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Timed requests")
    parser.add_argument("--profile_csv", type=str, default=None)
    parser.add_argument("--enable_parallelism", action="store_true",
                        help="Ulysses CP; launch with torchrun --nproc_per_node=N")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def setup_device(enable_parallelism: bool):
    if not enable_parallelism or "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def init_cp_group(world_size: int, local_rank: int):
    if world_size < 2:
        return None
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return dist.new_group(ranks=list(range(world_size)))


def _stats(name, times):
    """Return (name, mean, median, min, max, stdev, fps)."""
    import statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return name, mean, median, min(times), max(times), stdev


def main():
    args = parse_arguments()
    rank, world_size, local_rank = setup_device(args.enable_parallelism)
    is_main = rank == 0

    if is_main:
        log.info(f"E2E profile config:")
        for k in ("model", "dit_path", "resolution", "num_frames", "num_steps",
                  "attention_type", "sla_topk", "quant_linear",
                  "sampler_dtype", "warmup", "repeats"):
            log.info(f"  {k:<18}: {getattr(args, k)}")
        if world_size > 1:
            log.info(f"  world_size        : {world_size}")

    # -------- Model init (all resident, streaming style) --------
    # Text encoder: keep resident (interactive prompts arrive one at a time).
    # We instantiate directly instead of using get_umt5_embedding()'s lazy singleton
    # so we can time encode() alone without lazy-load noise polluting the first call.
    text_encoder = UMT5EncoderModel(
        text_len=512, device="cuda",
        checkpoint_path=args.text_encoder_path,
    )
    if is_main:
        log.info("text encoder resident")

    # DiT
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # CP group (late — after text/VAE loaders)
    cp_group = init_cp_group(world_size, local_rank)
    if cp_group is not None:
        net.enable_context_parallel(cp_group)
        if is_main:
            log.info(f"context parallel enabled: world_size={world_size}")

    net.cuda()

    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    sampler_dtype = torch.float64 if args.sampler_dtype == "fp64" else torch.float32
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=sampler_dtype, device="cuda",
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    ones = torch.ones(1, 1, device="cuda", dtype=sampler_dtype)
    total_steps = t_steps.shape[0] - 1

    # -------- Stage functions --------
    def stage_text(prompt: str):
        """UMT5 encode. Returns condition dict fed to net."""
        with torch.no_grad():
            emb = text_encoder(prompt, device="cuda").to(**tensor_kwargs)
        return {"crossattn_emb": repeat(emb, "b l d -> (k b) l d", k=1)}

    def stage_sample(condition, seed: int):
        """Full 4-step sampling loop."""
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        noise = torch.randn(1, *state_shape, dtype=torch.float32,
                            device="cuda", generator=g)
        x = noise.to(sampler_dtype) * t_steps[0]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            with torch.no_grad():
                v = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(sampler_dtype)
                x = (1 - t_next) * (x - t_cur * v) + t_next * torch.randn(
                    *x.shape, dtype=sampler_dtype, device="cuda", generator=g)
        return x

    def stage_decode(latent):
        """VAE decode to pixel frames. Runs on rank 0 only; other ranks return None."""
        if not is_main:
            return None
        with torch.no_grad():
            video = tokenizer.decode(latent.float())
        return video

    def run_one(prompt: str, seed: int, times: dict):
        """One end-to-end request. Records CUDA-synced per-stage timing on rank 0."""
        if world_size > 1:
            dist.barrier()

        # Text
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond = stage_text(prompt)
        torch.cuda.synchronize()
        t_text = time.perf_counter() - t0

        # Sample
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        latent = stage_sample(cond, seed)
        torch.cuda.synchronize()
        t_sample = time.perf_counter() - t0

        # Decode (rank 0 only)
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = stage_decode(latent)
        torch.cuda.synchronize()
        t_decode = time.perf_counter() - t0

        if is_main:
            t_total = t_text + t_sample + t_decode
            times["text"].append(t_text)
            times["sample"].append(t_sample)
            times["decode"].append(t_decode)
            times["total"].append(t_total)

    # -------- Warmup + repeats --------
    times = {"text": [], "sample": [], "decode": [], "total": []}

    for i in range(args.warmup):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond = stage_text(prompt)
        lat = stage_sample(cond, args.seed + 10000 + i)
        _ = stage_decode(lat)
        torch.cuda.synchronize()
        if is_main:
            log.info(f"[e2e-warmup {i+1}/{args.warmup}] "
                     f"total={time.perf_counter()-t0:.3f}s")

    for r in range(args.repeats):
        prompt = _PROMPTS[r % len(_PROMPTS)]
        run_one(prompt, args.seed + r, times)
        if is_main:
            log.info(f"[e2e-repeat {r+1}/{args.repeats}] "
                     f"text={times['text'][-1]:.3f}s  "
                     f"sample={times['sample'][-1]:.3f}s  "
                     f"decode={times['decode'][-1]:.3f}s  "
                     f"total={times['total'][-1]:.3f}s")

    # -------- Summary --------
    if is_main:
        print()
        print(f"==== E2E summary (world_size={world_size}, {args.model}, "
              f"{args.resolution}, {args.num_frames} frames, {args.num_steps} steps) ====")
        header = f"{'stage':>8} | {'median(s)':>10} | {'mean(s)':>9} | {'min(s)':>8} | {'max(s)':>8} | {'stdev(s)':>9}"
        print(header)
        print("-" * len(header))
        for stage in ("text", "sample", "decode", "total"):
            _, mean, median, tmin, tmax, stdev = _stats(stage, times[stage])
            print(f"{stage:>8} | {median:>10.3f} | {mean:>9.3f} | "
                  f"{tmin:>8.3f} | {tmax:>8.3f} | {stdev:>9.3f}")

        # FPS relative to E2E total (streaming budget)
        import statistics
        med_total = statistics.median(times["total"])
        med_sample = statistics.median(times["sample"])
        fps_e2e = args.num_frames / med_total
        fps_sample_only = args.num_frames / med_sample
        realtime_ratio = fps_e2e / 16.0  # 16 fps is the target playback rate
        print(f"\nFPS (median):")
        print(f"  E2E (text+sample+decode) : {fps_e2e:>7.2f}  "
              f"[{med_total:.3f}s per request; {realtime_ratio:.2f}x realtime at 16fps]")
        print(f"  sample-only              : {fps_sample_only:>7.2f}  "
              f"[{med_sample:.3f}s per request]")

        # Composition
        med_text = statistics.median(times["text"])
        med_decode = statistics.median(times["decode"])
        print(f"\nStage share of E2E median:")
        for name, v in (("text", med_text), ("sample", med_sample), ("decode", med_decode)):
            print(f"  {name:<8}: {v:.3f}s  ({v/med_total*100:5.1f}%)")

        if args.profile_csv:
            import csv, pathlib
            pathlib.Path(args.profile_csv).parent.mkdir(parents=True, exist_ok=True)
            new = not pathlib.Path(args.profile_csv).exists()
            with open(args.profile_csv, "a", newline="") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow([
                        "world_size", "model", "resolution", "num_frames",
                        "num_steps", "warmup", "repeats",
                        "text_median_s", "sample_median_s", "decode_median_s",
                        "total_median_s",
                        "text_mean_s", "sample_mean_s", "decode_mean_s",
                        "total_mean_s",
                        "fps_e2e_median", "fps_sample_median",
                    ])
                stats = {s: _stats(s, times[s]) for s in ("text", "sample", "decode", "total")}
                wr.writerow([
                    world_size, args.model, args.resolution, args.num_frames,
                    total_steps, args.warmup, args.repeats,
                    f"{stats['text'][2]:.4f}", f"{stats['sample'][2]:.4f}",
                    f"{stats['decode'][2]:.4f}", f"{stats['total'][2]:.4f}",
                    f"{stats['text'][1]:.4f}", f"{stats['sample'][1]:.4f}",
                    f"{stats['decode'][1]:.4f}", f"{stats['total'][1]:.4f}",
                    f"{fps_e2e:.4f}", f"{fps_sample_only:.4f}",
                ])
            log.info(f"[e2e] appended to {args.profile_csv}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
