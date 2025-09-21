#!/bin/bash

# ==============================================================================
#                  快捷评测脚本: 评测指定的融合模型
# ==============================================================================

# --- 1. 基本设置 (请根据你的环境修改) ---
GPU=2 # 请设置为你希望使用的 GPU ID 6 
PORT=29517 # 建议为每个独立脚本使用不同端口，避免冲突
MODEL_PATH="/home/user/xieqiuhao/AdaMMS/downloaded_models/llava-v1.5-7b" # <--- 在这里设置你要评测的模型路径 merged_models/grafted-s1.0-c0.0
EVAL_BASE="./eval_results_single_run" # 评测结果的根目录 

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 '$MODEL_PATH' 不存在. 请检查路径是否正确."
    exit 1
fi

# --- 2. 评测任务列表 ---
# 根据需要取消注释或添加任务
# 完整任务列表参考:ok_vqa  textvqa_val vizwiz_vqa_val gqa mme seedbench ocrbench mmmu_val
TASK_LIST="mmmu_val mme ocrbench textvqa_val vizwiz_vqa_val ok_vqa gqa seedbench  "

# --- 3. 环境准备 ---
echo "--- Preparing environment and directories ---"
# 从模型路径中提取模型名称用于输出目录
MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${EVAL_BASE}/${MODEL_NAME}
mkdir -p $OUTPUT_DIR
# conda activate lmms-cogvlm # 如果需要，请取消注释以激活你的 conda 环境

echo "===================================================="
echo "  评测模型: $MODEL_NAME"
echo "  模型路径: $MODEL_PATH"
echo "  评测任务: $TASK_LIST"
echo "  结果将保存到: $OUTPUT_DIR"
echo "===================================================="

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# --- 4. 运行评测 ---
for task in $TASK_LIST; do
    echo "--- 正在运行评测任务: $task ---"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
        --model llava_v1.5_7b \
        --model_args pretrained=$MODEL_PATH \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity INFO \
        --output_path $OUTPUT_DIR
    
    # 检查上一个命令是否成功
    if [ $? -ne 0 ]; then
        echo "错误: 任务 '$task' 评测失败."
        # 如果希望在失败时停止脚本，可以取消此行注释: exit 1
    fi
done

# --- 5. 结束 ---
minute=$((SECONDS / 60))
second=$((SECONDS % 60))
echo "===================================================="
echo "      模型 '$MODEL_NAME' 评测完成."
echo "      总耗时: $minute 分 $second 秒"
echo "      所有结果已保存到: $OUTPUT_DIR"
echo "===================================================="