#!/usr/bin/env bash
# End-to-end (text + sample + decode) profiling sweep for TurboDiffusion.
# Same NPROC divisor logic as run_profile.sh but drives wan2.1_t2v_e2e_profile.py.
set -euo pipefail

cd "$(dirname "$0")"

# ===== Tunables =====
MODEL=${MODEL:-Wan2.1-1.3B}
MAX_GPUS=${MAX_GPUS:-6}
if [ "$MODEL" = "Wan2.1-14B" ]; then
    DIT_PATH=${DIT_PATH:-checkpoints/TurboWan2.1-T2V-14B-480P-quant.pth}
    NUM_HEADS=40
elif [ "$MODEL" = "Wan2.1-1.3B" ]; then
    DIT_PATH=${DIT_PATH:-checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth}
    NUM_HEADS=12
else
    echo "Unsupported MODEL=$MODEL"; exit 1
fi
RESOLUTION=${RESOLUTION:-480p}
NUM_FRAMES=${NUM_FRAMES:-81}
NUM_STEPS=${NUM_STEPS:-4}
SLA_TOPK=${SLA_TOPK:-0.1}
WARMUP=${WARMUP:-3}
REPEATS=${REPEATS:-10}
CSV=${CSV:-output/e2e_profile_${MODEL//./pt}.csv}
MASTER_PORT=${MASTER_PORT:-29511}

if [ -z "${NPROC_LIST:-}" ]; then
    NPROC_LIST=""
    for n in $(seq 1 "$MAX_GPUS"); do
        [ $((NUM_HEADS % n)) -eq 0 ] && NPROC_LIST+="$n "
    done
fi

export PYTHONPATH=turbodiffusion
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$CSV")"

COMMON_ARGS=(
    --model "$MODEL"
    --dit_path "$DIT_PATH"
    --resolution "$RESOLUTION"
    --num_frames "$NUM_FRAMES"
    --num_steps "$NUM_STEPS"
    --quant_linear
    --attention_type sagesla
    --sla_topk "$SLA_TOPK"
    --warmup "$WARMUP"
    --repeats "$REPEATS"
    --profile_csv "$CSV"
)

echo "=== E2E profile config ==="
echo "  MODEL=$MODEL  DIT=$DIT_PATH  NUM_HEADS=$NUM_HEADS"
echo "  RES=$RESOLUTION  FRAMES=$NUM_FRAMES  STEPS=$NUM_STEPS"
echo "  WARMUP=$WARMUP  REPEATS=$REPEATS  NPROC_LIST=[$NPROC_LIST]"
echo "  CSV=$CSV"
echo

if [ ! -f "$DIT_PATH" ]; then
    echo "ERROR: DIT weights not found at $DIT_PATH"; exit 1
fi
rm -f "$CSV"

for NPROC in $NPROC_LIST; do
    if [ "$NPROC" -gt 1 ] && [ $((NUM_HEADS % NPROC)) -ne 0 ]; then
        echo ">>> SKIP NPROC=$NPROC"; continue
    fi
    echo ">>> Running NPROC=$NPROC ..."
    if [ "$NPROC" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
        python -u turbodiffusion/inference/wan2.1_t2v_e2e_profile.py "${COMMON_ARGS[@]}"
    else
        GPUS=$(seq -s, 0 $((NPROC - 1)))
        export CUDA_VISIBLE_DEVICES="$GPUS"
        echo "    CUDA_VISIBLE_DEVICES=$GPUS"
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
            turbodiffusion/inference/wan2.1_t2v_e2e_profile.py \
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
print(f"{'NPROC':>5} | {'text(s)':>8} | {'sample(s)':>10} | {'decode(s)':>10} | {'total(s)':>9} | {'E2E FPS':>8} | {'sample FPS':>10}")
print("-" * 82)
base_total = base_sample = None
for r in rows:
    ws = int(r["world_size"])
    text = float(r["text_median_s"]); samp = float(r["sample_median_s"])
    dec = float(r["decode_median_s"]); tot = float(r["total_median_s"])
    fps_e = float(r["fps_e2e_median"]); fps_s = float(r["fps_sample_median"])
    if ws == 1:
        base_total, base_sample = tot, samp
    su_tot = f" ({base_total/tot:.2f}x)" if base_total else ""
    su_s = f" ({base_sample/samp:.2f}x)" if base_sample else ""
    print(f"{ws:>5} | {text:>8.3f} | {samp:>10.3f} | {dec:>10.3f} | "
          f"{tot:>9.3f}{su_tot} | {fps_e:>8.2f} | {fps_s:>10.2f}{su_s}")
PY
