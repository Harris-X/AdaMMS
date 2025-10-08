#!/usr/bin/env bash
set -euo pipefail
# DGSM-TEFM++ end-to-end runner
# Models
PRE="/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-7B-Instruct"
A="/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct"
B="/root/autodl-tmp/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"
WK="/root/autodl-tmp/AdaMMS/work/dgsmp"
OUT="/root/autodl-tmp/AdaMMS/merged_models_stage3"
mkdir -p "$WK"

echo "[Run] Stage-1"
python -u -m merge.dgsmp.dgsmp_stage1_svd_ortho \
  --model-a "$A" --model-b "$B" --model-pre "$PRE" \
  --save "$WK/stage1.pt" --rank 128 --verbose

echo "[Run] Stage-2"
python -u -m merge.dgsmp.dgsmp_stage2_ties_meta \
  --stage1 "$WK/stage1.pt" --model-a "$A" --model-b "$B" --model-pre "$PRE" \
  --save "$WK/stage2.pt"

echo "[Run] Stage-3"
python -u -m merge.dgsmp.dgsmp_stage3_fuse \
  --stage1 "$WK/stage1.pt" --stage2 "$WK/stage2.pt" \
  --model-a "$A" --model-b "$B" --model-pre "$PRE" \
  --output-dir "$OUT" --rank 128 --ortho-blend 0.5 --ties-lambda 1.0 --bias-alpha 0.5

echo "[Out] $OUT/$(basename "$A")/dgsmp_merged"
