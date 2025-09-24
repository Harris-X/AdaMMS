python merge/FAM/fam_mapping.py \
  --acts_a activations/mPLUG-Owl2_meta.pt \
  --acts_b activations/llava_v1.5_7b_meta.pt \
  --sigma 0.1 \
  --fos-threshold 0.3 \
  --top-ratio 0.2 \
  --module-regex "mlp\.|self_attn\.|down_proj|up_proj|gate_proj|q_proj|k_proj|v_proj|o_proj|dense|fc|ffn" \
  --exclude-regex "lm_head|embed|embedding" \
  --feature-weights 1.0 1.0 1.0