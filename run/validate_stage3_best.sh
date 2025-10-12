#!/bin/bash
set -euo pipefail

# 快速验证 DGSM-TEFM Stage-3（v0.5）并生成推荐权重
# 默认使用已验证表现最优的一组超参（基于 OCRBench 807 分）
# 可通过环境变量覆盖关键参数以进一步调试。

ROOT=${ROOT:-/root/autodl-tmp/AdaMMS}
MODEL_A=${MODEL_A:-"$ROOT/downloaded_models/Qwen2-VL-7B-Instruct"}
MODEL_B=${MODEL_B:-"$ROOT/downloaded_models/llava-onevision-qwen2-7b-si-hf"}
WORK=${WORK:-"$ROOT/work/dgsm"}
OUT_ROOT=${OUT_ROOT:-"$ROOT/merged_models_stage3"}
EVAL_BASE=${EVAL_BASE:-"$ROOT/eval_runs"}
RANK=${RANK:-128}

# Stage-1 / Stage-2 产物路径（若不存在则自动生成）
SUBS_A="$WORK/stage1_A_r${RANK}.pt"
SUBS_B="$WORK/stage1_B_r${RANK}.pt"
STAGE2="$WORK/stage2_r${RANK}_us_on_gw_e0.5_it30_reg0.05_ds8.pt"

# Stage-2 超参（可按需调整）
DIST_MODE=${DIST_MODE:-us}
POT_MODE=${POT_MODE:-on}
POT_METHOD=${POT_METHOD:-gw}
POT_EPS=${POT_EPS:-0.5}
ITERS=${ITERS:-30}
SINK_REG=${SINK_REG:-0.05}
DYN_STEPS=${DYN_STEPS:-8}
DYN_LR=${DYN_LR:-0.01}
DYN_REG=${DYN_REG:-0.001}
DYN_LOSS=${DYN_LOSS:-hybrid}
DYN_MIX_ALPHA=${DYN_MIX_ALPHA:-0.2}

# Stage-3 推荐超参（基于当前最佳评分）
GAMMA=${GAMMA:-3.5}
COST_SCALE=${COST_SCALE:-0.8}
ORTHO_SCALE=${ORTHO_SCALE:-0.3}
FALLBACK_ALPHA=${FALLBACK_ALPHA:-0.6}
BIAS_ALPHA=${BIAS_ALPHA:-0.6}
TFI_THRESHOLD=${TFI_THRESHOLD:-0.01}
TFI_TOPK=${TFI_TOPK:-64}
MAPPING_BETA=${MAPPING_BETA:-0.12}
USE_DYNAMIC_M=${USE_DYNAMIC_M:-1}
USE_LAMBDA_EST=${USE_LAMBDA_EST:-1}
ORTHO_ADAPT=${ORTHO_ADAPT:-1}

mkdir -p "$WORK"

if [ ! -f "$SUBS_A" ]; then
  echo "[Stage-1] Extracting base model subspaces -> $SUBS_A"
  python -m merge.dgsm_v0_5.dgsm_stage1_subspace --model-dir "$MODEL_A" --save "$SUBS_A" --rank "$RANK" --cuda
else
  echo "[Stage-1] Using cached base subspaces: $SUBS_A"
fi

if [ ! -f "$SUBS_B" ]; then
  echo "[Stage-1] Extracting donor model subspaces -> $SUBS_B"
  python -m merge.dgsm_v0_5.dgsm_stage1_subspace --model-dir "$MODEL_B" --save "$SUBS_B" --rank "$RANK" --cuda
else
  echo "[Stage-1] Using cached donor subspaces: $SUBS_B"
fi

if [ ! -f "$STAGE2" ] || [ "${FORCE_STAGE2:-0}" = "1" ]; then
  echo "[Stage-2] Running DGSM alignment -> $STAGE2"
  python -m merge.dgsm_v0_5.dgsm_stage2_dynamic_gwd \
    --subs-a "$SUBS_A" \
    --subs-b "$SUBS_B" \
    --save "$STAGE2" \
    --dist-mode "$DIST_MODE" \
    --pot "$POT_MODE" --pot-method "$POT_METHOD" --pot-eps "$POT_EPS" --pot-max-iter 50 \
    --iters "$ITERS" --sink-reg "$SINK_REG" --tol 5e-4 --patience 3 \
    --dynamic-steps "$DYN_STEPS" --dynamic-lr "$DYN_LR" --dynamic-reg "$DYN_REG" \
    --dyn-loss "$DYN_LOSS" --dyn-mix-alpha "$DYN_MIX_ALPHA" \
    --gamma "$GAMMA" --cost-scale "$COST_SCALE" \
    --cpu
else
  echo "[Stage-2] Using cached alignment: $STAGE2"
fi

MERGED_OUT="$OUT_ROOT/$(basename "$MODEL_A")/dgsm_merged"

echo "[Stage-3] Launching TEFM integration merge -> $MERGED_OUT"
python -m merge.dgsm_v0_5.dgsm_stage3_merge \
  --base-model "$MODEL_A" --donor-model "$MODEL_B" \
  --stage2 "$STAGE2" --output-dir "$OUT_ROOT" \
  --gamma "$GAMMA" --cost-scale "$COST_SCALE" \
  --ortho-scale "$ORTHO_SCALE" --fallback-alpha "$FALLBACK_ALPHA" --bias-alpha "$BIAS_ALPHA" \
  $( [ "$USE_DYNAMIC_M" = "1" ] && echo --use-dynamic-m ) \
  $( [ "$USE_LAMBDA_EST" = "1" ] && echo --use-lambda-est ) \
  $( [ "$ORTHO_ADAPT" = "1" ] && echo --ortho-adapt ) \
  --base-subs "$SUBS_A" \
  --tfi-threshold "$TFI_THRESHOLD" --tfi-topk "$TFI_TOPK" --mapping-beta "$MAPPING_BETA"

echo "[Stage-3] 完成。融合权重位于: $MERGED_OUT"

if [ "${RUN_EVAL:-0}" = "1" ]; then
  echo "[Eval] RUN_EVAL=1 -> 调用 eval_single_model.sh (需要已登录 Hugging Face)"
  MODEL_PATH="$MERGED_OUT" \
  EVAL_BASE="$EVAL_BASE/dgsm_best_stage3" \
  GPU=${GPU:-0} \
  TASK_LIST="${TASK_LIST:-ocrbench}" \
  bash "$ROOT/run/eval_single_model.sh"
fi
