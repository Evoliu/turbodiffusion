#!/usr/bin/env bash
# TurboDiffusion multi-GPU CP profiling on 5090 (6 cards).
# Sweeps NPROC in 1 2 3 4 6 (skips 5: Wan2.1-1.3B num_heads=12 not divisible by 5).
# Each NPROC: warmup=3 (default) + repeats=10 (default), reports median FPS.
set -euo pipefail

cd "$(dirname "$0")"

# ===== Tunables =====
NPROC_LIST=${NPROC_LIST:-"1 2 3 4 6"}   # 6-card 5090; 5 skipped (12 % 5 != 0)
MODEL=${MODEL:-Wan2.1-1.3B}
DIT_PATH=${DIT_PATH:-checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth}
RESOLUTION=${RESOLUTION:-480p}
NUM_FRAMES=${NUM_FRAMES:-81}
NUM_STEPS=${NUM_STEPS:-4}
SLA_TOPK=${SLA_TOPK:-0.1}
WARMUP=${WARMUP:-3}
REPEATS=${REPEATS:-10}
PROMPT=${PROMPT:-"A stylish woman walks down a Tokyo street filled with warm glowing neon."}
CSV=${CSV:-output/profile.csv}
MASTER_PORT=${MASTER_PORT:-29511}

# ===== Env =====
export PYTHONPATH=turbodiffusion
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$CSV")"
: > "${CSV}.new"   # start fresh; move on success at end

# num_heads for the head-divisibility check
if [ "$MODEL" = "Wan2.1-14B" ]; then
    NUM_HEADS=40
else
    NUM_HEADS=12
fi

COMMON_ARGS=(
    --model "$MODEL"
    --dit_path "$DIT_PATH"
    --resolution "$RESOLUTION"
    --prompt "$PROMPT"
    --num_samples 1
    --num_frames "$NUM_FRAMES"
    --num_steps "$NUM_STEPS"
    --quant_linear
    --attention_type sagesla
    --sla_topk "$SLA_TOPK"
    --profile
    --warmup "$WARMUP"
    --repeats "$REPEATS"
    --profile_csv "$CSV"
)

echo "=== Profile config ==="
echo "  MODEL=$MODEL  RES=$RESOLUTION  FRAMES=$NUM_FRAMES  STEPS=$NUM_STEPS"
echo "  WARMUP=$WARMUP  REPEATS=$REPEATS  NPROC_LIST=[$NPROC_LIST]  NUM_HEADS=$NUM_HEADS"
echo "  CSV=$CSV"
echo

# Wipe old CSV so the summary at the end reflects THIS run only
rm -f "$CSV"

for NPROC in $NPROC_LIST; do
    if [ "$NPROC" -gt 1 ] && [ $((NUM_HEADS % NPROC)) -ne 0 ]; then
        echo ">>> SKIP NPROC=$NPROC (num_heads=$NUM_HEADS not divisible)"
        continue
    fi
    echo ">>> Running NPROC=$NPROC ..."
    if [ "$NPROC" -eq 1 ]; then
        unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
        # Pin to GPU 0 for the single-card run
        export CUDA_VISIBLE_DEVICES=0
        python -u turbodiffusion/inference/wan2.1_t2v_infer.py "${COMMON_ARGS[@]}"
    else
        # Use first NPROC of the 6 available GPUs
        GPUS=$(seq -s, 0 $((NPROC - 1)))
        export CUDA_VISIBLE_DEVICES="$GPUS"
        echo "    CUDA_VISIBLE_DEVICES=$GPUS"
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
            turbodiffusion/inference/wan2.1_t2v_infer.py \
            "${COMMON_ARGS[@]}" \
            --enable_parallelism
    fi
    echo
done

echo "=== Summary (from $CSV) ==="
python -u - "$CSV" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("(no rows)"); sys.exit(0)
cols = ["world_size", "median_s", "mean_s", "min_s", "max_s", "stdev_s", "fps_median"]
print(f"{'NPROC':>5} | {'median(s)':>10} | {'mean(s)':>9} | {'min(s)':>8} | {'max(s)':>8} | {'stdev(s)':>9} | {'FPS':>8}")
print("-" * 74)
base = None
for r in rows:
    ws = int(r["world_size"])
    med = float(r["median_s"]); fps = float(r["fps_median"])
    if ws == 1: base = med
    speedup = f" ({base/med:.2f}x)" if base else ""
    print(f"{ws:>5} | {med:>10.3f} | {float(r['mean_s']):>9.3f} | "
          f"{float(r['min_s']):>8.3f} | {float(r['max_s']):>8.3f} | "
          f"{float(r['stdev_s']):>9.3f} | {fps:>8.2f}{speedup}")
PY
