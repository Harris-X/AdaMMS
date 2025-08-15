#!/bin/bash

# ==============================================================================
#                  mplug-owl2-llama2-7b 模型评测脚本 (最终修正版 v2)
# ==============================================================================

# --- 1. 基本设置 ---
GPU=7                     # 设置要使用的 GPU ID
# 模型路径现在已在 config.py 中设置，此处仅作记录
MODEL_PATH="/home/user/xieqiuhao/AdaMMS/downloaded_models/mplug-owl2-llama2-7b"
EVAL_DIR="./mplug_owl2_eval_results"  # 评测结果保存目录
VLMKIT_DIR="/home/user/xieqiuhao/AdaMMS/VLMEvalKit"  # VLMEvalKit 目录

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 '$MODEL_PATH' 不存在. 请检查路径是否正确."
    exit 1
fi

# --- 2. 评测任务列表 ---
TASK_LIST="MMMU_DEV_VAL MME SEEDBench_IMG OCRBench TextVQA_VAL GQA_TestDev_Balanced VizWiz"


# --- 3. 环境准备 ---
echo "--- 准备环境和目录 ---"
mkdir -p $EVAL_DIR
MODEL_NAME=$(basename $MODEL_PATH)

echo "===================================================="
echo "  评测模型: $MODEL_NAME"
echo "  模型路径: $MODEL_PATH (通过 config.py 加载)"
echo "  评测任务: $TASK_LIST"
echo "  结果将保存至: $EVAL_DIR"
echo "===================================================="

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# --- 4. 进入VLMEvalKit目录并运行评测 ---
cd $VLMKIT_DIR

# 修正：确保安装 mplug-owl2 所需的、且唯一的 transformers 版本
echo "--- 正在检查并设置正确的 transformers 版本 (4.33.0) ---"
pip install transformers==4.33.0

for task in $TASK_LIST; do
    echo "--- 正在评测任务: $task ---"
    
    # run.py 会自动从 config.py 中读取 'mPLUG-Owl2' 的路径。
    python run.py \
        --data $task \
        --model mPLUG-Owl2 \
        --work-dir $EVAL_DIR \
        --verbose \
        --mode all
    
    # 检查上一个命令是否成功
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

# 显示评测结果摘要
echo "--- 评测结果摘要 ---"
find $EVAL_DIR -name "*.csv" -exec echo {} \; -exec cat {} \; -exec echo "" \;