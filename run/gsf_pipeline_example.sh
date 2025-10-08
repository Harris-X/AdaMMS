#!/usr/bin/env bash
# ============================================================================
# GSF-TEFM 全流程示例脚本 (Stage-1 -> Stage-2 -> Stage-2b -> Stage-3)
# ----------------------------------------------------------------------------
# 用途: 生成子空间, 进行 Gromov-Wasserstein 对齐, 神经元群定位 (行级 λ), 并完成最终融合权重合并。
# 运行前请确认: 两个待融合模型已解压在 downloaded_models/ 目录下。
# 默认示例: base = mplug-owl2-llama2-7b, donor = llava-v1.5-7b
# ----------------------------------------------------------------------------
# 可用环境变量覆盖:
#   BASE_MODEL_DIR, DONOR_MODEL_DIR, RANK, STAGE2_DIST_MODE, STAGE2_ITERS,
#   STAGE2_GAMMA, STAGE2_COST_SCALE, STAGE2_USE_POT (1/0),
#   STAGE2B_K, STAGE2B_ALPHA, STAGE2B_BETA, STAGE2B_GAMMA, STAGE2B_COST_SCALE,
#   OUTPUT_DIR
# 示例:
#   BASE_MODEL_DIR=downloaded_models/mplug-owl2-llama2-7b \
#   DONOR_MODEL_DIR=downloaded_models/llava-v1.5-7b \
#   RANK=64 STAGE2_USE_POT=1 bash run/gsf_pipeline_example.sh
# ============================================================================
set -euo pipefail

echo "[GSF] === 参数解析 ==="
BASE_MODEL_DIR=${BASE_MODEL_DIR:-downloaded_models/mplug-owl2-llama2-7b}
DONOR_MODEL_DIR=${DONOR_MODEL_DIR:-downloaded_models/llava-v1.5-7b}
RANK=${RANK:-64}
OUTPUT_DIR=${OUTPUT_DIR:-merged_models_stage3}

# Stage-2 主要超参
STAGE2_DIST_MODE=${STAGE2_DIST_MODE:-us}          # u / us / usn
STAGE2_ITERS=${STAGE2_ITERS:-30}
STAGE2_SINK_REG=${STAGE2_SINK_REG:-0.05}
STAGE2_TOL=${STAGE2_TOL:-5e-4}
STAGE2_PATIENCE=${STAGE2_PATIENCE:-3}
STAGE2_GAMMA=${STAGE2_GAMMA:-4.0}                 # 用于估算 λ (可不用于 Stage-3 直接再设)
STAGE2_COST_SCALE=${STAGE2_COST_SCALE:-1.0}
STAGE2_USE_POT=${STAGE2_USE_POT:-1}               # 1 使用 POT (若已安装), 0 仅近似
STAGE2_VERBOSE=${STAGE2_VERBOSE:-0}

# Stage-2b 行级 / 组级 λ 参数
STAGE2B_K=${STAGE2B_K:-8}
STAGE2B_ALPHA=${STAGE2B_ALPHA:-0.5}
STAGE2B_BETA=${STAGE2B_BETA:-0.5}
STAGE2B_GAMMA=${STAGE2B_GAMMA:-4.0}
STAGE2B_COST_SCALE=${STAGE2B_COST_SCALE:-1.0}

# Stage-3 合并控制
MERGE_GAMMA=${MERGE_GAMMA:-4.0}
MERGE_COST_SCALE=${MERGE_COST_SCALE:-1.0}
MERGE_ORTHO_SCALE=${MERGE_ORTHO_SCALE:-0.5}
FALLBACK_ALPHA=${FALLBACK_ALPHA:-0.5}
BIAS_ALPHA=${BIAS_ALPHA:-0.5}

echo "[GSF] base=${BASE_MODEL_DIR} donor=${DONOR_MODEL_DIR} rank=${RANK}";
echo "[GSF] Stage2: dist=${STAGE2_DIST_MODE} iters=${STAGE2_ITERS} gamma=${STAGE2_GAMMA} cost_scale=${STAGE2_COST_SCALE} use_pot=${STAGE2_USE_POT}";
echo "[GSF] Stage2b: K=${STAGE2B_K} alpha=${STAGE2B_ALPHA} beta=${STAGE2B_BETA} gamma=${STAGE2B_GAMMA} cost_scale=${STAGE2B_COST_SCALE}";
echo "[GSF] Stage3: gamma=${MERGE_GAMMA} cost_scale=${MERGE_COST_SCALE} ortho_scale=${MERGE_ORTHO_SCALE}";

DATE_TAG=$(date +%Y%m%d_%H%M%S)
ACT_DIR=activations
mkdir -p "$ACT_DIR"

BASE_TAG=$(basename "$BASE_MODEL_DIR")
DONOR_TAG=$(basename "$DONOR_MODEL_DIR")

STAGE1_A_FILE=${STAGE1_A_FILE:-$ACT_DIR/gsf_stage1_${BASE_TAG}.pt}
STAGE1_B_FILE=${STAGE1_B_FILE:-$ACT_DIR/gsf_stage1_${DONOR_TAG}.pt}
STAGE2_FILE=${STAGE2_FILE:-$ACT_DIR/gsf_stage2_${BASE_TAG}_TO_${DONOR_TAG}.pt}
STAGE2B_FILE=${STAGE2B_FILE:-$ACT_DIR/gsf_stage2b_${BASE_TAG}_TO_${DONOR_TAG}.pt}

PY_STAGE1=merge/gsf/gsf_stage1_subspace.py
PY_STAGE2=merge/gsf/gsf_stage2_gwd.py
PY_STAGE2B=merge/gsf/gsf_stage2b_group_loc.py
PY_STAGE3=merge/gsf/gsf_stage3_merge.py

# ----------------------------------------------------------------------------
# Stage-1: 提取两个模型的子空间
# ----------------------------------------------------------------------------
if [ ! -f "$STAGE1_A_FILE" ]; then
  echo "[Stage-1] 提取 base 子空间 -> $STAGE1_A_FILE"
  python $PY_STAGE1 --model-dir "$BASE_MODEL_DIR" --rank $RANK --save "$STAGE1_A_FILE" --verbose
else
  echo "[Stage-1] 跳过 base (已存在 $STAGE1_A_FILE)"
fi

if [ ! -f "$STAGE1_B_FILE" ]; then
  echo "[Stage-1] 提取 donor 子空间 -> $STAGE1_B_FILE"
  python $PY_STAGE1 --model-dir "$DONOR_MODEL_DIR" --rank $RANK --save "$STAGE1_B_FILE" --verbose
else
  echo "[Stage-1] 跳过 donor (已存在 $STAGE1_B_FILE)"
fi

# ----------------------------------------------------------------------------
# Stage-2: Gromov-Wasserstein 对齐 + ψ 编码
# ----------------------------------------------------------------------------
if [ ! -f "$STAGE2_FILE" ]; then
  echo "[Stage-2] 运行对齐 -> $STAGE2_FILE"
  EXTRA=""
  if [ "$STAGE2_USE_POT" = "1" ]; then EXTRA="--use-pot"; fi
  if [ "$STAGE2_VERBOSE" = "1" ]; then EXTRA="$EXTRA --verbose"; fi
  python $PY_STAGE2 \
    --subs-a "$STAGE1_A_FILE" \
    --subs-b "$STAGE1_B_FILE" \
    --save "$STAGE2_FILE" \
    --dist-mode "$STAGE2_DIST_MODE" \
    --iters $STAGE2_ITERS --sink-reg $STAGE2_SINK_REG --tol $STAGE2_TOL --patience $STAGE2_PATIENCE \
    --gamma $STAGE2_GAMMA --cost-scale $STAGE2_COST_SCALE $EXTRA
else
  echo "[Stage-2] 跳过 (已存在 $STAGE2_FILE)"
fi

# ----------------------------------------------------------------------------
# Stage-2b: 神经元群定位 + 行级 λ
# ----------------------------------------------------------------------------
if [ ! -f "$STAGE2B_FILE" ]; then
  echo "[Stage-2b] 计算行级 λ -> $STAGE2B_FILE"
  python $PY_STAGE2B \
    --subs-a "$STAGE1_A_FILE" \
    --subs-b "$STAGE1_B_FILE" \
    --stage2 "$STAGE2_FILE" \
    --save "$STAGE2B_FILE" \
    --k $STAGE2B_K --gamma $STAGE2B_GAMMA --cost-scale $STAGE2B_COST_SCALE \
    --alpha $STAGE2B_ALPHA --beta $STAGE2B_BETA --verbose
else
  echo "[Stage-2b] 跳过 (已存在 $STAGE2B_FILE)"
fi

# ----------------------------------------------------------------------------
# Stage-3: 权重融合 (使用 Stage-2 / 可选 Stage-2b 行级 λ)
# ----------------------------------------------------------------------------
echo "[Stage-3] 合并权重 -> $OUTPUT_DIR"
python $PY_STAGE3 \
  --base-model "$BASE_MODEL_DIR" \
  --donor-model "$DONOR_MODEL_DIR" \
  --stage2 "$STAGE2_FILE" \
  --stage2b "$STAGE2B_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --cost-scale $MERGE_COST_SCALE --gamma $MERGE_GAMMA \
  --ortho-scale $MERGE_ORTHO_SCALE \
  --fallback-alpha $FALLBACK_ALPHA --bias-alpha $BIAS_ALPHA

MERGED_DIR="$OUTPUT_DIR/$(basename $BASE_MODEL_DIR)/gsf_merged"
echo "[GSF] 完成: 合并模型目录 => $MERGED_DIR"
echo "[GSF] 可用于后续评测 (需在 VLMEvalKit config 中指向该路径, 或覆盖原 base 模型路径)"

cat <<EOF
---------------------------------- 使用示例 ----------------------------------
1) 仅快速运行 (默认参数):
   bash run/gsf_pipeline_example.sh

2) 指定 rank=96, 关闭 POT, K=12:
   RANK=96 STAGE2_USE_POT=0 STAGE2B_K=12 bash run/gsf_pipeline_example.sh

3) 使用已有 Stage-1 结果 (不重新计算): 先手动放置/保留 activations/gsf_stage1_*.pt 然后再运行即可自动跳过。

4) 评测合并模型 (示例，多任务脚本可参考 run/mplug_owl2.sh):
   修改 run/mplug_owl2.sh 中 MODEL_PATH 指向: $MERGED_DIR
   或临时运行: (确保 VLMEvalKit 能找到路径)
     MODEL_PATH=$MERGED_DIR GPUS=0 USE_TORCHRUN=never bash run/mplug_owl2.sh

5) 只重跑 Stage-2 / 2b (调整超参): 删除对应文件后再运行脚本；Stage-1 与其它阶段会自动复用。

输出文件:
  - $STAGE1_A_FILE / $STAGE1_B_FILE : Stage-1 子空间
  - $STAGE2_FILE : Stage-2 GWD 对齐结果
  - $STAGE2B_FILE : Stage-2b 行级 / 组级 λ
  - $MERGED_DIR : Stage-3 融合权重
-------------------------------------------------------------------------------
EOF
