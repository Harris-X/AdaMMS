#!/bin/bash

# ==============================================================================
#                  Qwen2-VL-7B-Instruct 模型评测脚本 (修复 flash-attn 兼容性)
# ==============================================================================

# --- 1. 基本设置 ---
GPU=6                     
MODEL_PATH="/home/user/xieqiuhao/AdaMMS/merged_models/idream-sams-dream-0.1-0.8"
EVAL_DIR="/home/user/xieqiuhao/AdaMMS/eval_results_single_run/merged_models/idream-sams-dream-0.1-0.8"
VLMKIT_DIR="/home/user/xieqiuhao/AdaMMS/VLMEvalKit"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 '$MODEL_PATH' 不存在. 请检查路径是否正确."
    exit 1
fi

# --- 2. 评测任务列表 ---
TASK_LIST="MME SEEDBench_IMG" #OCRBench TextVQA_VAL 

# --- 3. 环境准备 ---
echo "--- 准备环境和目录 ---"
mkdir -p $EVAL_DIR
MODEL_NAME=$(basename $MODEL_PATH)

echo "===================================================="
echo "  评测模型: $MODEL_NAME"
echo "  模型路径: $MODEL_PATH"
echo "  评测任务: $TASK_LIST"
echo "  结果将保存至: $EVAL_DIR"
echo "===================================================="

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# --- 4. 进入VLMEvalKit目录并修复环境 ---
cd $VLMKIT_DIR

# 如果仍然失败，我们将在运行时禁用 FlashAttention
echo "--- 开始评测任务 ---"

for task in $TASK_LIST; do
    echo "--- 正在评测任务: $task ---"
    
    # 尝试正常运行
    CUDA_VISIBLE_DEVICES=$GPU python run.py \
        --data $task \
        --model Qwen2-VL-7B-Instruct \
        --work-dir $EVAL_DIR \
        --verbose \
        --mode all
    
    # 如果失败，尝试禁用 FlashAttention
    if [ $? -ne 0 ]; then
        echo "正常运行失败，尝试禁用 FlashAttention..."
        CUDA_VISIBLE_DEVICES=$GPU DISABLE_FLASH_ATTN=1 python -c "
import torch
torch.backends.cuda.enable_flash_sdp(False)
import subprocess
import sys
subprocess.run([sys.executable, 'run.py', '--data', '$task', '--model', 'Qwen2-VL-7B-Instruct', '--work-dir', '$EVAL_DIR', '--verbose', '--mode', 'all'])
"
    fi
    
    if [ $? -ne 0 ]; then
        echo "错误: 任务 '$task' 评测失败."
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

echo "--- 评测结果摘要 ---"
find $EVAL_DIR -name "*.csv" -exec echo {} \; -exec cat {} \; -exec echo "" \;