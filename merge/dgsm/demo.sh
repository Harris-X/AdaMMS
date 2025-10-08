# Stage-1
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/mplug-owl2-llama2-7b --rank 128 --save activations/dgsm_stage1_base.pt
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/llava-v1.5-7b --rank 128 --save activations/dgsm_stage1_donor.pt

# Stage-2 (启用动态映射)
python merge/dgsm/dgsm_stage2_dynamic_gwd_ori.py \
  --subs-a activations/dgsm_stage1_base.pt \
  --subs-b activations/dgsm_stage1_donor.pt \
  --save activations/dgsm_stage2_base_TO_donor.pt \
  --dist-mode us --use-pot \
  --gamma 4 --cost-scale 1 \
  --dynamic-steps 15 --dynamic-lr 5e-3 --dynamic-reg 1e-3 --dynamic-report --verbose


python merge/dgsm/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/dgsm_stage1_base.pt \
  --subs-b activations/dgsm_stage1_donor.pt \
  --save activations/dgsm_stage2_base_TO_donor_entropic.pt \
  --dist-mode us --gamma 4 --cost-scale 1 \
  --dynamic-steps 8 --dynamic-lr 1e-3 --dynamic-reg 5e-4 \
  --dynamic-gw-mode entropic --dynamic-gw-eps 0.06 \
  --dynamic-gw-outer 3 --dynamic-gw-sinkhorn 15 \
  --grad-clip 0.5 \
  --dynamic-log-file activations/dgsm_stage2_dynamic_log_entropic.csv \
  --dynamic-report --verbose

python merge/dgsm/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/dgsm_stage1_base.pt \
  --subs-b activations/dgsm_stage1_donor.pt \
  --save activations/dgsm_stage2_base_TO_donor_entropic.pt \
  --dist-mode us --use-pot --pot-entropic-eps 0.05 \
  --gamma 4 --cost-scale 1 \
  --dynamic-steps 8 --dynamic-lr 1e-3 --dynamic-reg 5e-4 \
  --dyn-loss entropic --entropic-eps 0.06 --entropic-iters 10 --entropic-sinkhorn 10 \
  --verbose --dynamic-report


# Stage-3 (使用动态映射 M)
python merge/dgsm/dgsm_stage3_merge.py \
  --base-model downloaded_models/mplug-owl2-llama2-7b \
  --donor-model downloaded_models/llava-v1.5-7b \
  --stage2 activations/dgsm_stage2_base_TO_donor.pt \
  --output-dir merged_models_stage3 \
  --cost-scale 1.0 --gamma 4.0 --ortho-scale 0.5 \
  --fallback-alpha 0.5 --bias-alpha 0.5 \
  --use-dynamic-m



# B. 推荐平衡版
python merge/dgsm/dgsm_stage1_subspace.py \
  --model-dir downloaded_models/mplug-owl2-llama2-7b \
  --rank 128 \
  --save activations/dgsm_stage1_base_r128.pt


python merge/dgsm/dgsm_stage1_subspace.py \
  --model-dir downloaded_models/llava-v1.5-7b \
  --rank 128 \
  --save activations/dgsm_stage1_donor_r128.pt

python merge/dgsm/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/dgsm_stage1_base_r128.pt \
  --subs-b activations/dgsm_stage1_donor_r128.pt \
  --save activations/dgsm_stage2_base_TO_donor_balanced.pt \
  --dist-mode us \
  --dynamic-steps 15 \
  --dyn-loss entropic \
  --entropic-eps 0.07 \
  --entropic-iters 4 \
  --entropic-sinkhorn 5 \
  --dynamic-lr 4e-3 \
  --dynamic-reg 7e-4 \
  --gamma 4.5 --cost-scale 48 \
  --max-rank 128 \
  --use-pot --pot-entropic-eps 0.04 \
  --dynamic-report --verbose

python merge/dgsm/dgsm_stage3_merge.py \
  --base-model downloaded_models/mplug-owl2-llama2-7b \
  --donor-model downloaded_models/llava-v1.5-7b \
  --stage2 activations/dgsm_stage2_base_TO_donor_balanced.pt \
  --output-dir merged_models_stage3 \
  --cost-scale 48 --gamma 4.5 \
  --ortho-scale 0.5 --fallback-alpha 0.5 --bias-alpha 0.5 \
  --use-dynamic-m \
  --base-subs activations/dgsm_stage1_base_r128.pt



python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/Qwen2-VL-7B-Instruct --rank 128 --save activations/dgsm_stage1_base_qwen.pt
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/llava-onevision-qwen2-7b-si-hf --rank 128 --save activations/dgsm_stage1_donor_qwen.pt

python merge/dgsm/dgsm_stage2_dynamic_gwd_ori.py \
  --subs-a activations/dgsm_stage1_base_qwen.pt \
  --subs-b activations/dgsm_stage1_donor_qwen.pt \
  --save activations/dgsm_stage2_base_TO_donor_qwen.pt \
  --dist-mode us --use-pot \
  --gamma 4 --cost-scale 1 \
  --dynamic-steps 15 --dynamic-lr 5e-3 --dynamic-reg 1e-3 --dynamic-report --verbose

python merge/dgsm/dgsm_stage3_merge.py \
  --base-model downloaded_models/Qwen2-VL-7B-Instruct \
  --donor-model downloaded_models/llava-onevision-qwen2-7b-si-hf \
  --stage2 activations/dgsm_stage2_base_TO_donor_qwen_fast_v3.pt \
  --output-dir merged_models_stage3 \
  --cost-scale 1.0 --gamma 4.0 --ortho-scale 0.5 \
  --fallback-alpha 0.5 --bias-alpha 0.5 \
  --use-dynamic-m