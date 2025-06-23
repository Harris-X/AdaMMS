#!/bin/bash

# ==============================================================================
# AdaMMS: 测试 TIES-Merging 策略 (LLaVA-OneVision-7B -> Qwen2-VL-7B)
# ==============================================================================

# --- 1. 基本设置 ---
GPU=2 # 请设置为你希望使用的 GPU ID
PORT=29517 # 使用与原脚本不同的端口避免冲突
EVAL_BASE=./eval_results_ties_merging # 为 TIES-Merging 创建独立的评测结果目录

# 模型路径 (从融合脚本中获取)
QWEN2_VL_PATH="/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/Qwen_Qwen2-VL-7B-Instruct"
LLAVA_ONEVISION_PATH="/home/data2t1/xieqiuhao/AdaMMS/downloaded_models/lmms-lab_llava-onevision-qwen2-7b-si"

# 评测任务列表 (与 Table 1 相同)
TASK_LIST="mmmu_val gqa textvqa_val vizwiz_vqa_val " 

# --- 2. 环境准备 ---
echo "--- Preparing environment and directories ---"
mkdir -p $EVAL_BASE
conda activate lmms-cogvlm # 确保已激活包含 lmms-eval 和模型依赖的环境

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# --- 3. 评测原始模型 (Baseline) ---
echo "===================================================="
echo "           PART 1: Evaluating Base Model"
echo "===================================================="

# 评测 Qwen2-VL-7B (原始模型)
echo "--- Evaluating Base Model: Qwen2-VL-7B ---"
output_path_qwen=${EVAL_BASE}/baseline_qwen2_vl
for task in $TASK_LIST; do
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
        --model qwen2_vl --model_args pretrained=$QWEN2_VL_PATH \
        --tasks $task --batch_size 1 --log_samples --output_path $output_path_qwen 
done

# --- 4. 评测 TIES-Merging 策略 ---
echo "===================================================="
echo "        PART 2: Evaluating TIES-Merging Strategies"
echo "===================================================="
MERGE_SCRIPT=merge/llava-qwen2qwenvl_ties_merging.py

# 测试 ties 策略，不同的 K 值
for K in 0.3 0.5 0.7; do
    for tiesalpha in 0.3 0.5 0.7; do
        echo "--- Merging & Evaluating with TIES strategy, K=$K, tiesalpha=$tiesalpha ---"
        ckpt_path="checkpoints/qwens-ties-K-${K}-alpha-${tiesalpha}"
        
        python3 $MERGE_SCRIPT --output $ckpt_path --strategy ties --K $K --tiesalpha $tiesalpha
        
        output_path=${EVAL_BASE}/ties_K_${K}_alpha_${tiesalpha}
        for task in $TASK_LIST; do
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
                --model qwen2_vl --model_args pretrained=$ckpt_path \
                --tasks $task --batch_size 1 --log_samples --output_path $output_path 
        done
        rm -rf $ckpt_path # 清理检查点
    done
done

# 测试 task_arithmetic 策略
for K in 0.3 0.5 0.7; do
    for tiesalpha in 0.3 0.5 0.7; do
        echo "--- Merging & Evaluating with task_arithmetic strategy, K=$K, tiesalpha=$tiesalpha ---"
        ckpt_path="checkpoints/qwens-task_arithmetic-K-${K}-alpha-${tiesalpha}"
        
        python3 $MERGE_SCRIPT --output $ckpt_path --strategy task_arithmetic --K $K --tiesalpha $tiesalpha
        
        output_path=${EVAL_BASE}/task_arithmetic_K_${K}_alpha_${tiesalpha}
        for task in $TASK_LIST; do
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
                --model qwen2_vl --model_args pretrained=$ckpt_path \
                --tasks $task --batch_size 1 --log_samples --output_path $output_path 
        done
        rm -rf $ckpt_path # 清理检查点
    done
done

# --- 5. 测试线性插值（作为对照组）---
echo "===================================================="
echo "        PART 3: Evaluating Linear Interpolation (as control group)"
echo "===================================================="

for alpha in 0.5 0.7; do
    echo "--- Merging & Evaluating with Interpolation, alpha=$alpha ---"
    ckpt_path="checkpoints/qwens-interp-alpha-${alpha}"
    
    python3 $MERGE_SCRIPT --output $ckpt_path --alpha $alpha --interpolation
    
    output_path=${EVAL_BASE}/interpolation_alpha_${alpha}
    for task in $TASK_LIST; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
            --model qwen2_vl --model_args pretrained=$ckpt_path \
            --tasks $task --batch_size 1 --log_samples --output_path $output_path 
    done
    rm -rf $ckpt_path # 清理检查点
done

# 运行搜索脚本，找到最佳配置
echo "==> Searching for the best configuration in logs..."
python search/view_log_delta_perdata_search_limit.py --log_path $EVAL_BASE

# --- 6. 结束 ---
echo "===================================================="
echo "          All TIES-Merging evaluations finished."
echo "===================================================="

minute=$((SECONDS / 60))
echo "Total elapsed time: $minute mins"
echo "All results have been saved to the directory: $EVAL_BASE"
echo "Please collect the 'results.json' from each sub-directory to compare different merging strategies."