#!/usr/bin/env bash
# TurboDiffusion Wan2.1 T2V 推理脚本 (A800 / sm_80 已验证)
# 复现 README 的单卡 5090 推理路径: 量化 + quant_linear + sagesla
set -euo pipefail

cd "$(dirname "$0")"

# ===== 可调参数 =====
GPU=${GPU:-7}                                   # 单卡模式使用的 GPU 编号 (NPROC=1)
NPROC=${NPROC:-1}                               # 并行卡数; >1 时启用多卡上下文并行 (torchrun)
GPUS=${GPUS:-0,1,2,3}                            # 多卡模式使用的 GPU 列表 (NPROC>1 时生效)
MODEL=${MODEL:-Wan2.1-14B}                       # Wan2.1-14B 或 Wan2.1-1.3B
PROMPT=${PROMPT:-"A yound women is selling some product in a studio in front of camera"}
NUM_STEPS=${NUM_STEPS:-4}                        # 1~4, 越大质量越好越慢
SLA_TOPK=${SLA_TOPK:-0.1}
SAVE_PATH=${SAVE_PATH:-output/test.mp4}

# DIT 权重路径: 未显式指定时按 MODEL 自动选择
if [ "$MODEL" = "Wan2.1-14B" ]; then
    DIT_PATH=${DIT_PATH:-ckpts/TurboWan2.1-T2V-14B-480P-quant.pth}
else
    DIT_PATH=${DIT_PATH:-checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth}
fi

# ===== 环境 =====
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH

# 代理 + HF 镜像 (huggingface.co 被墙, tokenizer 走镜像)
export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

export PYTHONPATH=turbodiffusion

PYTHON=/root/paddlejob/workspace/miniconda3/envs/turbo/bin/python
TORCHRUN=/root/paddlejob/workspace/miniconda3/envs/turbo/bin/torchrun

mkdir -p "$(dirname "$SAVE_PATH")"

# 注意: 不要加 --default_norm (量化路径必须用 FastRMSNorm)
COMMON_ARGS=(
    --model "$MODEL"
    --dit_path "$DIT_PATH"
    --resolution 480p
    --prompt "$PROMPT"
    --num_samples 1
    --num_steps "$NUM_STEPS"
    --quant_linear
    --attention_type sagesla
    --sla_topk "$SLA_TOPK"
    --save_path "$SAVE_PATH"
)

if [ "$NPROC" -gt 1 ]; then
    # 多卡: Ulysses 上下文并行, 加速单次推理 (需 num_heads 可被 NPROC 整除)
    export CUDA_VISIBLE_DEVICES=$GPUS
    echo "Running with context parallelism: NPROC=$NPROC on GPUs [$GPUS]"
    $TORCHRUN --nproc_per_node="$NPROC" --master_port=29511 \
        turbodiffusion/inference/wan2.1_t2v_infer.py \
        "${COMMON_ARGS[@]}" \
        --enable_parallelism
else
    # 单卡
    export CUDA_VISIBLE_DEVICES=$GPU
    $PYTHON -u turbodiffusion/inference/wan2.1_t2v_infer.py "${COMMON_ARGS[@]}"
fi

echo "Done. Video saved to: $SAVE_PATH"
