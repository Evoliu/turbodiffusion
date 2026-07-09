#!/usr/bin/env bash
# Sample nvidia-smi at 200ms cadence while the inference runs.
# Usage on the 5090 box:
#
#   cd /root/autodl-tmp/TurboDiffusion
#   git pull
#   bash monitor_clocks.sh
#
# This starts monitoring first, then launches the profile run itself. Monitoring
# auto-stops when the profile finishes. Logs land in output/clocks_<timestamp>.csv
# and are also summarized to stdout so you can just paste the tail.

set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-Wan2.1-1.3B}
DIT_PATH=${DIT_PATH:-checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth}
GPU=${GPU:-0}                    # single GPU to watch
WARMUP=${WARMUP:-3}
REPEATS=${REPEATS:-10}
INTERVAL_MS=${INTERVAL_MS:-200}  # sampling cadence (min 100)

ts=$(date +%Y%m%d_%H%M%S)
LOG_DIR=output
LOG_RAW="$LOG_DIR/clocks_${ts}.csv"
LOG_SUM="$LOG_DIR/clocks_${ts}.summary.txt"
mkdir -p "$LOG_DIR"

export PYTHONPATH=turbodiffusion
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== monitor_clocks.sh ==="
echo "  MODEL=$MODEL  GPU=$GPU  interval=${INTERVAL_MS}ms"
echo "  raw log: $LOG_RAW"
echo

# 1) Start the sampler in the background. We poll a specific GPU by index
#    (query-gpu with -i so all rows are that GPU).
echo "timestamp,gpu,sm_clock_mhz,max_sm_clock_mhz,mem_clock_mhz,power_w,power_limit_w,temp_c,util_pct,mem_mb" > "$LOG_RAW"
(
    while true; do
        # -i requires a UUID or index; use 0 (post CUDA_VISIBLE_DEVICES the physical
        # index in nvidia-smi is still the original one, so query all and grep GPU)
        ts_iso=$(date +%s.%3N)
        nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used \
                   --format=csv,noheader,nounits 2>/dev/null | \
            awk -v ts="$ts_iso" -F', *' '{print ts","$1","$2","$3","$4","$5","$6","$7","$8","$9}' >> "$LOG_RAW"
        # Sleep in ms; if bash lacks fractional sleep, fall back to 1s
        sleep_s=$(awk -v ms="$INTERVAL_MS" 'BEGIN{printf "%.3f", ms/1000}')
        sleep "$sleep_s" 2>/dev/null || sleep 1
    done
) &
MON_PID=$!

# Guarantee we kill the monitor on any exit
trap "kill $MON_PID 2>/dev/null || true" EXIT

# 2) Run the profile (blocks). Warmup + repeats identical to run_profile.sh 1-GPU.
echo ">>> Running inference under monitor..."
python -u turbodiffusion/inference/wan2.1_t2v_infer.py \
    --model "$MODEL" \
    --dit_path "$DIT_PATH" \
    --resolution 480p --prompt "clock monitor test" \
    --num_samples 1 --num_steps 4 \
    --quant_linear --attention_type sagesla --sla_topk 0.1 \
    --profile --warmup "$WARMUP" --repeats "$REPEATS" 2>&1 | tee "$LOG_DIR/inference_${ts}.log"

# 3) Stop monitor
kill $MON_PID 2>/dev/null || true
sleep 0.3
echo
echo ">>> Monitor stopped. Summarizing..."

# 4) Summarize with python (only rows where GPU util > 30% — i.e. actual work)
python - "$LOG_RAW" "$GPU" > "$LOG_SUM" <<'PY'
import csv, sys, statistics as st
raw, gpu = sys.argv[1], int(sys.argv[2])
rows = []
with open(raw) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            if int(row["gpu"]) != gpu: continue
            rows.append({k: float(v) for k, v in row.items() if k != "timestamp"})
        except (ValueError, KeyError):
            continue

if not rows:
    print("NO SAMPLES CAPTURED"); sys.exit(0)

def stats(name, vals):
    return (name, st.mean(vals), st.median(vals), min(vals), max(vals),
            st.stdev(vals) if len(vals) > 1 else 0.0)

# Separate idle (util <= 20%) from active samples so the median doesn't get
# dragged down by pre/post idle gaps.
active = [r for r in rows if r["util_pct"] > 20]
idle = [r for r in rows if r["util_pct"] <= 20]

print(f"total samples : {len(rows)}")
print(f"active samples: {len(active)}  (util > 20%)")
print(f"idle samples  : {len(idle)}")
if not active:
    print("!! no active samples — sampling cadence may have missed the busy window"); sys.exit(0)

max_clock = rows[0]["max_sm_clock_mhz"]
print(f"max SM clock  : {max_clock:.0f} MHz")
print(f"power limit   : {rows[0]['power_limit_w']:.1f} W")
print()

hdr = f"{'metric':<18} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'stdev':>10}"
print(hdr); print("-"*len(hdr))
for name, key in (("sm_clock (MHz)",     "sm_clock_mhz"),
                  ("mem_clock (MHz)",    "mem_clock_mhz"),
                  ("power (W)",          "power_w"),
                  ("temp (C)",           "temp_c"),
                  ("util (%)",           "util_pct"),
                  ("mem used (MB)",      "mem_mb")):
    vals = [r[key] for r in active]
    _, mn, md, lo, hi, sd = stats(name, vals)
    print(f"{name:<18} {mn:>10.1f} {md:>10.1f} {lo:>10.1f} {hi:>10.1f} {sd:>10.2f}")

med_clock = st.median(r["sm_clock_mhz"] for r in active)
med_pwr = st.median(r["power_w"] for r in active)
pwr_lim = rows[0]["power_limit_w"]
print()
print(f"SM clock utilization  : {med_clock/max_clock*100:.1f}% of max ({med_clock:.0f}/{max_clock:.0f} MHz)")
print(f"power utilization     : {med_pwr/pwr_lim*100:.1f}% of limit ({med_pwr:.1f}/{pwr_lim:.1f} W)")

# Verdict
print()
if med_clock < 0.85 * max_clock:
    print(f"VERDICT: GPU is clock-limited (running at {med_clock/max_clock*100:.0f}% of max SM clock).")
    if med_pwr < 0.7 * pwr_lim:
        print("        Power is well below limit -> likely thermal cap, driver throttle, or")
        print("        BIOS/vBIOS clock floor set by AutoDL. Not fixable from user space.")
    else:
        print("        Power is near limit -> hitting TDP throttle. Also not user-fixable.")
elif st.median(r["util_pct"] for r in active) < 85:
    print("VERDICT: GPU is host-bound (util below 85% while clock is high).")
    print("        Root cause is CPU-side kernel dispatch, not hardware.")
else:
    print("VERDICT: GPU is running at spec. 40% gap vs README is unlikely to be hardware.")
PY

echo
echo "=== Summary ($LOG_SUM) ==="
cat "$LOG_SUM"
echo
echo "Full raw samples in: $LOG_RAW"
