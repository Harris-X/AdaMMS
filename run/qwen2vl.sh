#!/bin/bash
# GPUS=0,1,2,3 USE_TORCHRUN=never bash /root/autodl-tmp/AdaMMS/run/mplug_owl2.sh
# ==============================================================================
#                  mplug-owl2-llama2-7b 模型评测脚本 (最终修正版 v2)
# ==============================================================================

#!/usr/bin/env bash

# --- 1. 基本设置 ---
# 支持逗号分隔的多 GPU 列表，例如 "0,1,2,3"；也兼容旧变量 GPU（单卡）
GPUS=${GPUS:-}
GPU=${GPU:-}
if [ -z "$GPUS" ] && [ -n "$GPU" ]; then
    GPUS="$GPU"
fi
# 默认使用 0 号卡（如未指定）
GPUS=${GPUS:-1}

# 模型路径现在已在 config.py 中设置，此处仅作记录
MODEL_PATH="/root/autodl-tmp/AdaMMS/merged_models_stage3/Qwen2-VL-7B-Instruct/dgsm_merged"
EVAL_DIR="${EVAL_DIR:-./Qwen2_VL_7B_dgsm_1006_2258_a0_5_eval_results}"  # 评测结果保存目录


# 自动定位 VLMEvalKit 目录（以当前脚本所在目录为基准）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLMKIT_DIR="${VLMKIT_DIR:-$(cd "$SCRIPT_DIR/../VLMEvalKit" && pwd)}"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 '$MODEL_PATH' 不存在. 请检查路径是否正确."
    exit 1
fi

# --- 2. 评测任务列表 --- MMMU_DEV_VAL MME SEEDBench_IMG 
TASK_LIST="MMMU_DEV_VAL MME SEEDBench_IMG OCRBench TextVQA_VAL GQA_TestDev_Balanced VizWiz"


# --- 3. 环境准备 ---
echo "--- 准备环境和目录 ---"
mkdir -p $EVAL_DIR
MODEL_NAME=$(basename $MODEL_PATH)

echo "===================================================="
echo "  评测模型: $MODEL_NAME"
echo "  模型路径: $MODEL_PATH (通过 config.py 加载)"
echo "  使用 GPU: $GPUS"
echo "  评测任务: $TASK_LIST"
echo "  结果将保存至: $EVAL_DIR"
echo "===================================================="

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# --- 4. 进入VLMEvalKit目录并运行评测 ---
cd "$VLMKIT_DIR"



# --- 3.1 启动方式：根据 GPU 数量自动选择 python 或 torchrun ---
export CUDA_VISIBLE_DEVICES="$GPUS"
# 避免某些机型/拓扑下 NCCL 的 P2P/IB 报错，默认关闭，可通过环境变量覆盖
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# 允许手动控制是否使用 torchrun：auto | always | never
USE_TORCHRUN=${USE_TORCHRUN:-auto}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
case "$USE_TORCHRUN" in
    always)
        LAUNCHER=(torchrun --nproc-per-node=$NPROC)
        ;;
    never)
        LAUNCHER=(python)
        ;;
    *) # auto
        if [ "$NPROC" -gt 1 ]; then
            LAUNCHER=(torchrun --nproc-per-node=$NPROC)
        else
            LAUNCHER=(python)
        fi
        ;;
esac

for task in $TASK_LIST; do
    echo "--- 正在评测任务: $task ---"
    
    # run.py 会自动从 config.py 中读取 'mPLUG-Owl2' 的路径。
    "${LAUNCHER[@]}" run.py \
        --data $task \
        --model Qwen2-VL-7B-Instruct \
        --work-dir $EVAL_DIR \
        --verbose \
        --mode all

    # 检查上一个命令是否成功；若 torchrun 失败则自动回退到 python 模式再尝试一次
    status=$?
    if [ $status -ne 0 ]; then
        echo "警告: 任务 '$task' 使用 ${LAUNCHER[0]} 失败(退出码 $status)。"
        if [ "${LAUNCHER[0]}" = "torchrun" ]; then
            echo "尝试回退到单进程 python 模式 (device_map=auto, 多卡分片，非多进程并行)。"
            LAUNCHER_FALLBACK=(python)
            "${LAUNCHER_FALLBACK[@]}" run.py \
                --data $task \
                --model Qwen2-VL-7B-Instruct \
                --work-dir $EVAL_DIR \
                --verbose \
                --mode all
            status=$?
        fi

        if [ $status -ne 0 ]; then
            echo "错误: 任务 '$task' 评测失败."
        fi
    fi
done

# --- 5. 结束并显示评测摘要 ---
minute=$((SECONDS / 60))
second=$((SECONDS % 60))
echo "===================================================="
echo "      模型 '$MODEL_NAME' 评测完成"
echo "      总耗时: $minute 分 $second 秒"
echo "      所有结果已保存至: $EVAL_DIR"
echo "===================================================="

# 显示评测结果摘要
echo "--- 评测结果摘要 ---"
find $EVAL_DIR -name "*.csv" -exec echo {} \; -exec cat {} \; -exec echo "" \;