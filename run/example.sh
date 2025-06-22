#!/bin/bash
# GPU 和端口设置
GPU=1 # 请设置为你希望使用的 GPU ID
PORT=29515

# 融合脚本设置
MERGE_SCRIPT=merge/llava-qwen2qwenvl.py
# 定义一个基础名称，用于生成检查点和日志目录
MERGE_NAME_BASE=qwens

# 评测结果和日志的根目录
EVAL_BASE=./eval_results

# 确保评测日志目录存在
mkdir -p $EVAL_BASE

# 计时器
date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# 激活 Conda 环境
# 请确保已激活包含 lmms-eval 和模型依赖的环境
conda activate lmms-cogvlm # 根据你的环境名称修改

# 循环测试不同的 alpha 值
for alpha in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 ; do
    echo "===================================================="
    echo "           MERGING & EVALUATING alpha=$alpha"
    echo "===================================================="

    # 根据 alpha 构建本次运行的检查点路径
    ckpt_path="checkpoints/${MERGE_NAME_BASE}-alpha-${alpha}-interpolation"
    
    echo "==> Merging models into: $ckpt_path"
    # 执行融合脚本，--interpolation 参数会附加到路径名中
    python3 $MERGE_SCRIPT --output $ckpt_path --alpha $alpha --interpolation
    
    # 评测日志输出路径
    output_path=${EVAL_BASE}/interpolation_${MERGE_NAME_BASE}_${alpha}

    # 循环评测多个任务 # "mme" "mmmu_val" "nocaps_val" "vizwiz_vqa_val" "seedbench"  "gqa" "ok_vqa" "refcoco_bbox_testA" "refcocog_bbox_test" "refcoco+_bbox_testA" "mmbench" "ocrbench"
    for task in "mme" "mmmu_val" "nocaps_val" "vizwiz_vqa_val" "seedbench"  "gqa" "ok_vqa" "refcoco_bbox_testA" "refcocog_bbox_test" "refcoco+_bbox_testA" "mmbench" "ocrbench" ;
    do
        echo "----------------------------------------------------"
        echo "==> Evaluating task: ${task} for alpha: ${alpha}"
        echo "----------------------------------------------------"
        
        # 使用 accelerate 启动评测
        # --model 设置为 qwen2_vl
        # --model_args 只需提供 pretrained 路径，lmms-eval 会自动加载
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
             --num_processes=1 \
             --gpu_ids $GPU \
             --main_process_port $PORT \
             -m lmms_eval \
             --model qwen2_vl \
             --model_args pretrained=$ckpt_path \
             --tasks $task \
             --batch_size 1 \
             --log_samples \
             --log_samples_suffix interpolation_${MERGE_NAME_BASE}_${alpha}_${task} \
             --output_path $output_path

        # 检查上一个命令的退出状态
        if [ $? -ne 0 ]; then
            echo "Evaluation failed for task: ${task} with alpha: ${alpha}"
            # 你可以在这里决定是退出脚本还是继续下一个任务
            # exit 1 
        fi
    done

    echo "==> Cleaning up checkpoint: $ckpt_path"
    # 评测完成后删除临时的模型检查点以节省空间
    rm -rf $ckpt_path
done

echo "===================================================="
echo "          All evaluations finished."
echo "===================================================="

# 运行搜索脚本，找到最佳 alpha
# 假设 view_log_delta_perdata_search_limit.py 可以接受一个参数来指定日志目录
echo "==> Searching for the best alpha in logs..."
python search/view_log_delta_perdata_search_limit.py --log_path $EVAL_BASE

# 输出总耗时
minute=$((SECONDS / 60))
echo "Total elapsed time: $minute mins"