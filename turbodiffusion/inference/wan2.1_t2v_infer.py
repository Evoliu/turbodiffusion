# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.1 T2V")
    parser.add_argument("--dit_path", type=str, required=True, help="Custom path to the DiT model checkpoint for distilled models")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=80, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation (required unless --serve)")
    parser.add_argument("--resolution", default="480p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    parser.add_argument("--enable_sac", action="store_true", help="[Debug] Re-enable Selective Activation Checkpoint. Inference disables it by default because SAC wraps every block in checkpoint_wrapper, adding host-side overhead with no memory benefit when torch.no_grad() is active.")
    parser.add_argument("--sampler_dtype", choices=["fp32", "fp64"], default="fp64",
                        help="[Debug] Precision of the outer sampling loop state (x, t_steps, elementwise ops "
                             "around net(...)). The DiT itself always runs in bf16. Default fp64 matches the "
                             "original training/inference recipe; 4-step distilled sampling appears to be very "
                             "sensitive to this (fp32 diverges — cosine sim ~0.67 to fp64 output). Use --sampler_dtype fp32 "
                             "only to A/B the fp64 kernel cost on consumer GPUs (RTX 5090 fp64 = fp32/64).")
    parser.add_argument("--serve", action="store_true", help="Launch interactive TUI server mode (keeps model loaded)")
    parser.add_argument("--enable_parallelism", action="store_true", help="Enable multi-GPU context (sequence) parallelism to speed up a single inference. Launch with torchrun --nproc_per_node=N")
    parser.add_argument("--profile", action="store_true", help="Profiling mode: run the sampling loop repeatedly and print timing stats (mean/median/min/max/FPS). Skips VAE decode + save.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup iterations under --profile (default 3).")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations under --profile (default 10).")
    parser.add_argument("--profile_csv", type=str, default=None, help="If set with --profile, append the summary row to this CSV path.")
    parser.add_argument("--compile", action="store_true",
                        help="[Experimental] Wrap the DiT with torch.compile(). Fuses elementwise "
                             "ops (adaLN scale/shift/gate, add/mul chains) that show up in the "
                             "profiler as ~1s of aten::copy_/add/mul; leaves custom kernels "
                             "(SageSLA, Int8Linear, FastRMSNorm) unchanged. Extra ~30-90s at first "
                             "warmup for graph compilation; measured repeats amortize it. Set "
                             "--compile_mode to pick backend/mode.")
    parser.add_argument("--compile_mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode. reduce-overhead uses CUDA Graphs internally when "
                             "safe, which is what host-bound sampling wants. max-autotune spends "
                             "much longer at first warmup to find optimal tiles.")
    parser.add_argument("--compile_dynamic", action="store_true",
                        help="Pass dynamic=True to torch.compile. Slower for a fixed shape (which "
                             "is our case since latent shape is constant), but faster to compile "
                             "and doesn't recompile if input shape changes. Default False.")
    parser.add_argument("--compile_fullgraph", action="store_true",
                        help="Pass fullgraph=True — makes torch.compile HARD-FAIL on any graph "
                             "break instead of silently falling back. Use this to diagnose "
                             "which custom kernel is breaking dynamo. Default False (allow breaks).")
    return parser.parse_args()


def setup_device(enable_parallelism: bool):
    """Pin this process to its own GPU as early as possible (before any CUDA
    work), so per-rank models land on distinct devices. Does NOT init the
    process group yet. Returns (rank, world_size, local_rank)."""
    if not enable_parallelism or "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def init_cp_group(world_size: int, local_rank: int):
    """Initialize torch.distributed and build a single CP group spanning all
    ranks (Ulysses all-to-all sequence parallel). Call this AFTER loading the
    text encoder / VAE, since their loaders only fetch weights on rank 0 and
    rely on sync_model_states, which no-ops while the group is uninitialized
    (letting every rank load independently instead)."""
    if world_size < 2:
        return None
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return dist.new_group(ranks=list(range(world_size)))


if __name__ == "__main__":
    args = parse_arguments()

    # Handle serve mode
    if args.serve:
        # Set mode to t2v for the TUI server
        args.mode = "t2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)

    # Validate prompt is provided for one-shot mode
    if args.prompt is None:
        log.error("--prompt is required (unless using --serve mode)")
        exit(1)

    # Pin each process to its own GPU FIRST (before any CUDA allocation), but
    # defer process-group init until after the text encoder / VAE are loaded.
    rank, world_size, local_rank = setup_device(args.enable_parallelism)
    is_main_process = rank == 0
    if world_size > 1:
        log.info(f"Context parallelism: rank {rank}/{world_size} on cuda:{local_rank}")

    log.info(f"Computing embedding for prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
    clear_umt5_memory()

    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT model.")

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # Now that per-rank models are loaded independently (text encoder / VAE
    # loaders only fetch weights on rank 0 + rely on sync_model_states, which
    # no-ops while the group is uninitialized), bring up the CP group.
    cp_group = init_cp_group(world_size, local_rank)
    if cp_group is not None:
        net.enable_context_parallel(cp_group)
        log.info(f"Model context parallel enabled across {world_size} GPUs.")

    # Optional torch.compile wrap. Placed AFTER CP setup so compile sees the
    # already-hooked module (a2a hooks live inside enable_context_parallel).
    if args.compile:
        # torch.compile with reduce-overhead mode uses CUDA Graphs, which is
        # the whole point on host-bound 5090 (see profiler: 30% Command Buffer
        # Full). Graph breaks on SageSLA / Int8Linear are expected — dynamo
        # falls back to eager for those and re-enters compile after, without
        # error unless fullgraph=True.
        log.info(f"Compiling DiT with torch.compile(mode='{args.compile_mode}', "
                 f"dynamic={args.compile_dynamic}, fullgraph={args.compile_fullgraph})...")
        # Suppress dynamo errors so a bad compile falls back to eager rather
        # than crashing — but we log via TORCH_LOGS=recompiles for diagnosis.
        torch._dynamo.config.suppress_errors = not args.compile_fullgraph
        _compile_kwargs = dict(mode=args.compile_mode)
        if args.compile_dynamic:
            _compile_kwargs["dynamic"] = True
        if args.compile_fullgraph:
            _compile_kwargs["fullgraph"] = True
        net = torch.compile(net, **_compile_kwargs)
        log.info("torch.compile applied. First warmup will be slower (graph compilation).")


    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    to_show = []

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    # mid_t = [1.3, 1.0, 0.6][: args.num_steps - 1]
    # For better visual quality
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    # Sampler-loop precision. DiT internals still run in bf16 (tensor_kwargs);
    # this only controls x / t_steps / elementwise ops around the net call.
    sampler_dtype = torch.float64 if args.sampler_dtype == "fp64" else torch.float32

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=sampler_dtype,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling steps
    ones = torch.ones(init_noise.size(0), 1, device=init_noise.device, dtype=sampler_dtype)
    total_steps = t_steps.shape[0] - 1
    net.cuda()

    def _run_sampling(seed_offset: int = 0) -> torch.Tensor:
        """Run one full sampling loop. Returns the final latent x.
        seed_offset lets profile repeats use distinct RNG streams while keeping
        the shape / kernel path identical across iterations."""
        gen = torch.Generator(device=tensor_kwargs["device"])
        gen.manual_seed(args.seed + seed_offset)
        noise = torch.randn(
            args.num_samples, *state_shape,
            dtype=torch.float32, device=tensor_kwargs["device"], generator=gen,
        )
        x = noise.to(sampler_dtype) * t_steps[0]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            with torch.no_grad():
                v_pred = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(sampler_dtype)
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape, dtype=sampler_dtype,
                    device=tensor_kwargs["device"], generator=gen,
                )
        return x

    if args.profile:
        # Profiling mode: warmup + repeats, print stats. Skip VAE decode / save.
        import statistics
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()

        for w in range(args.warmup):
            _ = _run_sampling(seed_offset=w)
            torch.cuda.synchronize()
            if is_main_process:
                log.info(f"[profile] warmup {w+1}/{args.warmup} done")

        times = []
        for r in range(args.repeats):
            if world_size > 1:
                dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _run_sampling(seed_offset=args.warmup + r)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            if is_main_process:
                log.info(f"[profile] repeat {r+1}/{args.repeats}: {dt:.3f}s")

        if is_main_process:
            mean = statistics.mean(times)
            median = statistics.median(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0.0
            tmin, tmax = min(times), max(times)
            fps = args.num_frames / median
            log.success(
                f"[profile] world_size={world_size} steps={total_steps} "
                f"num_frames={args.num_frames} | "
                f"mean={mean:.3f}s median={median:.3f}s "
                f"min={tmin:.3f}s max={tmax:.3f}s stdev={stdev:.3f}s | "
                f"FPS(median)={fps:.2f}"
            )
            if args.profile_csv:
                import csv, pathlib
                pathlib.Path(args.profile_csv).parent.mkdir(parents=True, exist_ok=True)
                new = not pathlib.Path(args.profile_csv).exists()
                with open(args.profile_csv, "a", newline="") as f:
                    wr = csv.writer(f)
                    if new:
                        wr.writerow(["world_size", "model", "resolution", "num_frames",
                                     "num_steps", "warmup", "repeats",
                                     "mean_s", "median_s", "min_s", "max_s", "stdev_s",
                                     "fps_median"])
                    wr.writerow([world_size, args.model, args.resolution, args.num_frames,
                                 total_steps, args.warmup, args.repeats,
                                 f"{mean:.4f}", f"{median:.4f}", f"{tmin:.4f}",
                                 f"{tmax:.4f}", f"{stdev:.4f}", f"{fps:.4f}"])
                log.info(f"[profile] appended to {args.profile_csv}")

        net.cpu()
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        exit(0)

    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    x = init_noise.to(sampler_dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps, disable=not is_main_process)):
        with torch.no_grad():
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                sampler_dtype
            )
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=sampler_dtype,
                device=tensor_kwargs["device"],
                generator=generator,
            )
    torch.cuda.synchronize()
    if is_main_process:
        log.info(f"[benchmark] sampling loop: {time.perf_counter() - _t0:.2f}s "
                 f"({total_steps} steps, world_size={world_size})")
    samples = x.float()
    net.cpu()
    torch.cuda.empty_cache()

    if is_main_process:
        with torch.no_grad():
            video = tokenizer.decode(samples)

        to_show.append(video.float().cpu())

        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

        save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
        log.success(f"Video saved to: {args.save_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
