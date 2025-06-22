#!/bin/bash

for alpha in 1.0 0.9 0.8 0.7 0.6 0.5 0.4; do
    echo "===> Alpha: $alpha"
    
    # Merge
    python3 $MERGE_SCRIPT --output $ckpt_path --alpha $alpha --interpolation \
        --base COGVLM_PATH --llava_base LLAVA_PATH

    # Evaluate
    for task in "mme" "mmmu_val" "nocaps_val" "vizwiz_vqa_val" "seedbench"  "gqa" "ok_vqa" "refcoco_bbox_testA" "refcocog_bbox_test" "refcoco+_bbox_testA" "mmbench" "ocrbench" ; do
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
            --num_processes=1 \
            -m lmms_eval \
            --model cogvlm \
            --model_args pretrained=$ckpt_path,... \
            --tasks $task \
            --log_samples \
            --output_path $output_path
    done

    rm -rf $ckpt_path
done