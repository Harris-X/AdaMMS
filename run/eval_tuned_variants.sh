#!/bin/bash
set -euo pipefail

# 遍历 OUT_ROOT 下的多个合并结果目录并批量评测
# 用法示例：
#   OUT_ROOT=/root/autodl-tmp/AdaMMS/merged_models_stage3 \
#   GPU=0 TASK_LIST="mme ocrbench gqa" bash run/eval_tuned_variants.sh

OUT_ROOT=${OUT_ROOT:-"/root/autodl-tmp/AdaMMS/merged_models_stage3"}
GPU=${GPU:-0}
PORT_BASE=${PORT_BASE:-29700}
TASK_LIST=${TASK_LIST:-"mme ocrbench gqa"}

find "$OUT_ROOT" -maxdepth 3 -type d -name dgsm_merged | while read -r model_path; do
	tag=$(echo "$model_path" | sed 's#.*/merged_models_stage3/##; s#/dgsm_merged##')
	echo "[BatchEval] $tag -> $model_path"
	MODEL_PATH="$model_path" GPU="$GPU" PORT=$((PORT_BASE + RANDOM % 50)) TASK_LIST="$TASK_LIST" bash /root/autodl-tmp/AdaMMS/run/eval_single_model.sh || true
done

