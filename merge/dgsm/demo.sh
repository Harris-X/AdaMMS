# Stage-1
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/mplug-owl2-llama2-7b --rank 64 --save activations/dgsm_stage1_base.pt
python merge/dgsm/dgsm_stage1_subspace.py --model-dir downloaded_models/llava-v1.5-7b --rank 64 --save activations/dgsm_stage1_donor.pt

# Stage-2 (启用动态映射)
python merge/dgsm/dgsm_stage2_dynamic_gwd.py \
  --subs-a activations/dgsm_stage1_base.pt \
  --subs-b activations/dgsm_stage1_donor.pt \
  --save activations/dgsm_stage2_base_TO_donor.pt \
  --dist-mode us --use-pot \
  --gamma 4 --cost-scale 1 \
  --dynamic-steps 15 --dynamic-lr 5e-3 --dynamic-reg 1e-3 --dynamic-report --verbose

# Stage-3 (使用动态映射 M)
python merge/dgsm/dgsm_stage3_merge.py \
  --base-model downloaded_models/mplug-owl2-llama2-7b \
  --donor-model downloaded_models/llava-v1.5-7b \
  --stage2 activations/dgsm_stage2_base_TO_donor_entropic_heads_norm.pt \
  --output-dir merged_models_stage3 \
  --cost-scale 1.0 --gamma 4.0 --ortho-scale 0.5 \
  --fallback-alpha 0.5 --bias-alpha 0.5 \
  --use-dynamic-m