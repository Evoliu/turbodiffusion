#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level probe for the Wan2.1-1.3B sampling loop.

Runs three diagnostics in one process:
  1. GPU state at start + end (SM clock, power, temp) via nvidia-smi.
  2. Torch runtime flags that affect kernel dispatch (dynamo, cudnn, SDP).
  3. torch.profiler over ONE steady-state sampling loop (after 3 warmups),
     printing the top-25 CUDA kernels by self time, plus a chrome trace at
     /tmp/turbo.trace.json.

Usage on the 5090 box:
    cd /root/autodl-tmp/TurboDiffusion
    export PYTHONPATH=turbodiffusion HF_ENDPOINT=https://hf-mirror.com \
           TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
    python scripts/profile_probe.py

Optional overrides via env:
    MODEL, DIT_PATH, VAE_PATH, TEXT_ENC_PATH, RESOLUTION, NUM_FRAMES,
    NUM_STEPS, ATTENTION_TYPE, SLA_TOPK, QUANT (1/0), WARMUP, TRACE_PATH.
"""
import math
import os
import subprocess
import sys
import time

# Make imaginaire / rcm importable regardless of cwd
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
_pkg = os.path.join(_repo, "turbodiffusion")
_inference = os.path.join(_pkg, "inference")
for _p in (_pkg, _inference):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from einops import repeat


def _nvsmi(tag: str):
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,clocks.sm,clocks.max.sm,"
             "power.draw,power.limit,temperature.gpu,utilization.gpu,"
             "memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        print(f"[nvsmi:{tag}] " + out.strip().replace("\n", "  |  "))
    except Exception as e:
        print(f"[nvsmi:{tag}] FAILED: {e}")


def _flags():
    import torch._dynamo
    print("---- torch runtime flags ----")
    print(f"  torch                     : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"  device                    : {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)}")
    print(f"  dynamo.suppress_errors    : {torch._dynamo.config.suppress_errors}")
    print(f"  cudnn.benchmark           : {torch.backends.cudnn.benchmark}")
    print(f"  cudnn.allow_tf32          : {torch.backends.cudnn.allow_tf32}")
    print(f"  matmul.allow_tf32         : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cuda.flash_sdp_enabled    : {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  cuda.mem_efficient_sdp    : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  cuda.math_sdp             : {torch.backends.cuda.math_sdp_enabled()}")
    for m in ("flash_attn", "spas_sage_attn", "turbo_diffusion_ops",
              "triton", "transformers"):
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", "ok")
            print(f"  {m:<25}: {v}")
        except Exception as e:
            print(f"  {m:<25}: MISSING ({type(e).__name__})")


class _Args:  # duck-typed like argparse.Namespace for modify_model.create_model
    pass


def main():
    _flags()
    _nvsmi("before")

    from imaginaire.utils import log  # noqa: F401  (registers logger)
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
    from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    from modify_model import tensor_kwargs, create_model

    args = _Args()
    args.model = os.environ.get("MODEL", "Wan2.1-1.3B")
    args.dit_path = os.environ.get(
        "DIT_PATH", "checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth")
    args.vae_path = os.environ.get("VAE_PATH", "checkpoints/Wan2.1_VAE.pth")
    args.text_encoder_path = os.environ.get(
        "TEXT_ENC_PATH", "checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    args.num_samples = 1
    args.num_steps = int(os.environ.get("NUM_STEPS", 4))
    args.sigma_max = 80.0
    args.num_frames = int(os.environ.get("NUM_FRAMES", 81))
    args.prompt = "profile test"
    args.resolution = os.environ.get("RESOLUTION", "480p")
    args.aspect_ratio = "16:9"
    args.seed = 0
    args.attention_type = os.environ.get("ATTENTION_TYPE", "sagesla")
    args.sla_topk = float(os.environ.get("SLA_TOPK", 0.1))
    args.quant_linear = os.environ.get("QUANT", "1") == "1"
    args.default_norm = False
    warmup = int(os.environ.get("WARMUP", 3))
    trace_path = os.environ.get("TRACE_PATH", "/tmp/turbo.trace.json")

    print(f"---- config ----")
    for k in ("model", "dit_path", "resolution", "num_frames", "num_steps",
              "attention_type", "sla_topk", "quant_linear"):
        print(f"  {k:<20}: {getattr(args, k)}")

    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=args.prompt,
        ).to(**tensor_kwargs)
    clear_umt5_memory()

    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    condition = {
        "crossattn_emb": repeat(text_emb.to(**tensor_kwargs),
                                "b l d -> (k b) l d", k=1)
    }
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64, device=tensor_kwargs["device"],
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
    ones = torch.ones(1, 1, device=tensor_kwargs["device"], dtype=torch.float64)
    net.cuda()

    def _run(seed: int):
        g = torch.Generator(device=tensor_kwargs["device"])
        g.manual_seed(seed)
        n = torch.randn(1, *state_shape, dtype=torch.float32,
                        device=tensor_kwargs["device"], generator=g)
        x = n.to(torch.float64) * t_steps[0]
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            with torch.no_grad():
                v = net(
                    x_B_C_T_H_W=x.to(**tensor_kwargs),
                    timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                    **condition,
                ).to(torch.float64)
                x = (1 - t_next) * (x - t_cur * v) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32,
                    device=tensor_kwargs["device"], generator=g)
        return x

    # --- Warmup ---
    for i in range(warmup):
        t0 = time.perf_counter()
        _run(i)
        torch.cuda.synchronize()
        print(f"[warmup {i+1}/{warmup}] {time.perf_counter()-t0:.3f}s")

    # --- One measured run WITHOUT profiler (baseline for this process) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _run(1000)
    torch.cuda.synchronize()
    baseline = time.perf_counter() - t0
    print(f"[baseline no-profiler] {baseline:.3f}s")

    # --- Profiled run ---
    from torch.profiler import profile, ProfilerActivity
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run(1001)
        torch.cuda.synchronize()
        profiled = time.perf_counter() - t0
    print(f"[with-profiler] {profiled:.3f}s "
          f"(overhead vs baseline: {(profiled - baseline)*1000:.0f}ms)")

    print("\n---- Top-25 CUDA kernels by self CUDA time ----")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25))

    try:
        prof.export_chrome_trace(trace_path)
        print(f"\n[trace] chrome trace: {trace_path} "
              f"(open in chrome://tracing or perfetto.dev)")
    except Exception as e:
        print(f"[trace] export failed: {e}")

    _nvsmi("after")


if __name__ == "__main__":
    main()
