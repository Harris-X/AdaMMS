#!/bin/bash
GPU=5
PORT=29515

MERGE_BASE=YOUR_PATH
MERGE_NAME=YOUR_FILE_NAME
MERGE_SCRIPT=${MERGE_BASE}${MERGE_NAME}.py

TASK=ok_vqa

prompt_type=vqa
ckpt_name=cogvlm-chat-hf
date +"%Y-%m-%d %H:%M:%S"
SECONDS=0
conda activate lmms-cogvlm

for alpha in  1.0 0.9 0.8 0.7 0.6 0.5 0.4 ; do
    echo "Generating alpha=$alpha..."

    ckpt_path="checkpoints/${alpha}-${MERGE_NAME}-/$ckpt_name"
    cd YOUR_PATH
    echo "merging------"
    python3 $MERGE_SCRIPT --output $ckpt_path --alpha $alpha --interpolation \
        --base COGVLM_PATH  --llava_base LLAVA_PATH

    
    cd eval    
    echo $ckpt_path
    output_path=${EVAL_BASE}/interpolation_${MERGE_NAME}_${alpha}

EVAL_BASE=YOUR_PATH
for task in   "mme" "mmmu_val" "nocaps_val" "vizwiz_vqa_val" "seedbench"  "gqa" "ok_vqa" "refcoco_bbox_testA" "refcocog_bbox_test" "refcoco+_bbox_testA" "mmbench" "ocrbench" ;
do
    echo "eval---------"
    echo ${task} 
    echo $PORT
    echo $GPU
    CUDA_AVAILABLE_DEVICES=$GPU accelerate launch \
         --num_processes=1 \
         --gpu_ids $GPU \
         --main_process_port $PORT \
         -m lmms_eval \
         --model cogvlm \
         --model_args pretrained=$ckpt_path,tokenizer="lmsys/vicuna-7b-v1.5",prompt_type=$prompt_type \
         --tasks $task \
         --batch_size 1 \
        --log_samples \
         --log_samples_suffix interpolation_${MERGE_NAME}_${alpha}_${task} \
         --output_path $output_path
date +"%Y-%m-%d %H:%M:%S"
minute=$((SECONDS / 60))   
echo "Elapsed time: $minute mins"

minute=$((SECONDS / 60))   
echo "Elapsed time: $minute mins"
echol "--+++++++++++++++++"
echo "eval---------"
echo ${task}
#     echo "------------"
echo $PORT
done

 rm -r $ckpt_path
done

python search/view_log_delta_perdata_search_limit.py