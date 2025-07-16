#!/bin/bash

# ==============================================================================
# AdaMMS: Test SVD-Merging (LLaVA-OneVision-7B -> Qwen2-VL-7B)
# ==============================================================================

# --- 1. 基本设置 ---
GPU=4 # 请设置为你希望使用的 GPU ID
PORT=29518 # 使用与其它脚本不同的端口
EVAL_BASE=./eval_results_svd_merging # 为 SVD 评测创建独立的目录

# 模型路径
QWEN2_VL_PATH="/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-VL-7B-Instruct"

# 评测任务列表
TASK_LIST="mmmu_val gqa textvqa_val vizwiz_vqa_val" 

# --- 2. 环境准备 ---
echo "--- Preparing environment and directories ---"
mkdir -p $EVAL_BASE
conda activate lmms-cogvlm # 确保已激活正确的 conda 环境

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# # --- 3. 评测原始模型 (Baseline) ---
# echo "===================================================="
# echo "           PART 1: Evaluating Base Model"
# echo "===================================================="
# echo "--- Evaluating Base Model: Qwen2-VL-7B ---"
# output_path_qwen=${EVAL_BASE}/baseline_qwen2_vl
# for task in $TASK_LIST; do
#     CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
#         --model qwen2_vl --model_args pretrained=$QWEN2_VL_PATH \
#         --tasks $task --batch_size 1 --log_samples --output_path $output_path_qwen
# done

# --- 4. 评测 SVD-Merging ---
echo "===================================================="
echo "        PART 2: Evaluating SVD-Merging"
echo "===================================================="
MERGE_SCRIPT=merge/llava-qwen2qwenvl_svd_merging.py

# 测试不同的 SVD 秩
for rank in 32 64 128 256; do
    echo "--- Merging & Evaluating with SVD-Merging, rank=$rank ---"
    ckpt_path="checkpoints/qwens-svd-rank-${rank}"
    
    python3 $MERGE_SCRIPT --output $ckpt_path --rank $rank
    
    output_path=${EVAL_BASE}/svd_rank_${rank}
    for task in $TASK_LIST; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
            --model qwen2_vl --model_args pretrained=$ckpt_path \
            --tasks $task --batch_size 1 --log_samples --output_path $output_path
    done
    # rm -rf $ckpt_path # 清理检查点以节省空间
done

# --- 5. 结束 ---
echo "===================================================="
echo "          All SVD-Merging evaluations finished."
echo "===================================================="

minute=$((SECONDS / 60))
echo "Total elapsed time: $minute mins"
echo "All results have been saved to the directory: $EVAL_BASE"