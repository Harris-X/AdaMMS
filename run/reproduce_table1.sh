#!/bin/bash

# ==============================================================================
# AdaMMS: 复现 Table 1 (LLaVA-OneVision-7B -> Qwen2-VL-7B) 的评测脚本
# ==============================================================================

# --- 1. 基本设置 ---
GPU=2 # 请设置为你希望使用的 GPU ID
PORT=29516 # 建议为每个独立脚本使用不同端口，避免冲突
EVAL_BASE=./eval_results_table1 # 为本次复现创建一个独立的评测结果目录

# 模型路径 (从融合脚本中获取)
QWEN2_VL_PATH="/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-VL-7B-Instruct"
LLAVA_ONEVISION_PATH="/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/lmms-lab_llava-onevision-qwen2-7b-si"

# 评测任务列表 (请根据 Table 1 的实际任务进行调整)
TASK_LIST="mmmu_val gqa textvqa_val vizwiz_vqa_val " # 缺少 mme seedbench ok_vqa  ocrbench

# --- 2. 环境准备 ---
echo "--- Preparing environment and directories ---"
mkdir -p $EVAL_BASE
conda activate lmms-cogvlm # 确保已激活包含 lmms-eval 和模型依赖的环境

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# # --- 3. 评测原始模型 (Baselines) ---
# echo "===================================================="
# echo "           PART 1: Evaluating Base Models"
# echo "===================================================="

# 评测 Qwen2-VL-7B (原始模型)
echo "--- Evaluating Base Model: Qwen2-VL-7B ---"
output_path_qwen=${EVAL_BASE}/baseline_qwen2_vl
for task in $TASK_LIST; do
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
        --model qwen2_vl --model_args pretrained=$QWEN2_VL_PATH \
        --tasks $task --batch_size 1 --log_samples --output_path $output_path_qwen 
done

# # 评测 LLaVA-OneVision-7B (原始模型) 分开测
# echo "--- Evaluating Base Model: LLaVA-OneVision-7B ---"
# output_path_onevision=${EVAL_BASE}/baseline_llava_onevision
# for task in $TASK_LIST; do
#     CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
#         --model llava_onevision --model_args pretrained=$LLAVA_ONEVISION_PATH \
#         --tasks $task --batch_size 1 --log_samples --output_path $output_path_onevision
# done


# --- 4. 评测线性插值 (Linear Interpolation) ---
echo "===================================================="
echo "        PART 2: Evaluating Linear Interpolation"
echo "===================================================="
MERGE_SCRIPT_INTERP=merge/llava-qwen2qwenvl.py

for alpha in 1.0 0.9 0.8 0.7 0.6 0.5 0.4  ; do #1.0 0.9 0.8 0.7 0.6 0.5 0.4
    echo "--- Merging & Evaluating with Interpolation, alpha=$alpha ---"
    ckpt_path="checkpoints/qwens-interp-alpha-${alpha}"
    
    python3 $MERGE_SCRIPT_INTERP --output $ckpt_path --alpha $alpha --interpolation
    
    output_path=${EVAL_BASE}/interpolation_alpha_${alpha}
    for task in $TASK_LIST; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
            --model qwen2_vl --model_args pretrained=$ckpt_path \
            --tasks $task --batch_size 1 --log_samples --output_path $output_path 
    done
    rm -rf $ckpt_path # 清理检查点
done


# # --- 5. 评测非线性融合 (TIES-Merging & others) ---
# echo "===================================================="
# echo "        PART 3: Evaluating Non-Linear Merging"
# echo "===================================================="
MERGE_SCRIPT_TIES=merge/llava-qwen2qwenvl_ties_merging.py

for strategy in "task_arithmetic" "dare_ties" "metagpt"; do
    echo "--- Merging & Evaluating with TIES strategy: $strategy ---"
    # TIES 融合脚本通常自己管理输出路径，我们这里指定一个基础路径
    # 注意：TIES 脚本可能需要不同的参数，如 -K。这里使用默认值。
    ckpt_path="checkpoints/qwens-ties-${strategy}"

    python3 $MERGE_SCRIPT_TIES --output $ckpt_path --strategy $strategy
    
    output_path=${EVAL_BASE}/ties_${strategy}
    for task in $TASK_LIST; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
            --model qwen2_vl --model_args pretrained=$ckpt_path \
            --tasks $task --batch_size 1 --log_samples --output_path $output_path
    done
    rm -rf $ckpt_path # 清理检查点
done

# 运行搜索脚本，找到最佳 alpha
# 假设 view_log_delta_perdata_search_limit.py 可以接受一个参数来指定日志目录
echo "==> Searching for the best alpha in logs..."
python search/view_log_delta_perdata_search_limit.py --log_path $EVAL_BASE

# --- 6. 结束 ---
echo "===================================================="
echo "          All evaluations for Table 1 finished."
echo "===================================================="

minute=$((SECONDS / 60))
echo "Total elapsed time: $minute mins"
echo "All results have been saved to the directory: $EVAL_BASE"
echo "Please collect the 'results.json' from each sub-directory to compile Table 1."
