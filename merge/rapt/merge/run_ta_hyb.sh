#!/usr/bin/env bash
# ==============================================================================
# TA_hyb / MRC —— RAPT V0.3 当前最优配置的独立单文件实现入口
# (迁移自 rapt_V0_3 的 ta_hyb_standalone.py + run_crossarch_batch.sh / run_mrc_gen.sh)
#
# 配方 (TA_hyb):
#   W* = (1-ia)·W_host + ia·W_donor + alpha·λ·U_τB diag([h]_+) V_τBᵀ
#
# 超参全部来自 merge/rapt/config.yaml:
#   merge.rank / alpha / interp_alpha / multiway / device
#   merge.variants.ortho_scale / conflict_filter
#   merge.mrc_interp_alpha.<group>   (开启 ortho_scale 时使用)
#   groups.<group>.anchor / host / donor
#
# 用法:
#   bash merge/rapt/merge/run_ta_hyb.sh --group g1
#   bash merge/rapt/merge/run_ta_hyb.sh --group g3_fwd --ortho-scale
#   RANK=256 ALPHA=0.4 bash merge/rapt/merge/run_ta_hyb.sh --group g1   # 临时覆盖
#
# 产物: <output>/<host_basename>/ta_hyb_standalone_r{R}_a{A}_ia{IA}[_suffix]
# ==============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT=$(cd "${HERE}/../../.." &>/dev/null && pwd)
CFG_FILE=${RAPT_CONFIG:-"${ROOT}/merge/rapt/config.yaml"}
CFG_PY="${HERE}/../tools/cfg.py"

BOOT_PY=${PY:-python3}
PY=${PY:-$("$BOOT_PY" "$CFG_PY" --get paths.python 2>/dev/null || echo python3)}
[ -x "$PY" ] || PY=python3
eval "$("$PY" "$CFG_PY" --shell "$CFG_FILE")" || { echo "[ERROR] 读取配置失败: $CFG_FILE" >&2; exit 1; }

GROUP=${GROUP:-g1}
RANK=${RANK:-$CFG_merge_rank}
ALPHA=${ALPHA:-$CFG_merge_alpha}
TAG_SUFFIX=${TAG_SUFFIX:-}
MULTIWAY=${MULTIWAY:-$CFG_merge_multiway}
DEVICE=${DEVICE:-$CFG_merge_device}
NO_PROGRESS=${NO_PROGRESS:-1}
DM=${DM:-"${ROOT}/${CFG_paths_models}"}
OUT_ROOT=${OUT_ROOT:-"${ROOT}/${CFG_paths_merged_models}"}

if [ -z "${ORTHO_SCALE:-}" ]; then
  if [ "$CFG_merge_variants_ortho_scale" = "true" ]; then ORTHO_SCALE=1; else ORTHO_SCALE=0; fi
fi
if [ -z "${CONFLICT_FILTER:-}" ]; then
  if [ "$CFG_merge_variants_conflict_filter" = "true" ]; then CONFLICT_FILTER=1; else CONFLICT_FILTER=0; fi
fi

usage() {
  cat <<EOF
用法: bash merge/rapt/merge/run_ta_hyb.sh [选项]
  --group STR         g1 | g3_fwd | g3_rev (默认: g1)
  --anchor PATH       预训练锚点 W_0 (默认取 config groups.<g>.anchor)
  --host PATH         host 模型目录    (默认取 config groups.<g>.host)
  --donor PATH        donor 模型目录   (默认取 config groups.<g>.donor)
  --output DIR        产物根目录 (默认: merge/rapt/merged_models/<group>)
  --rank INT          秩 (默认 config merge.rank = 128)
  --alpha FLOAT       band 注入强度 (默认 config merge.alpha = 0.5)
  --interp-alpha F    插值系数 (默认 config merge.interp_alpha;
                      开启 --ortho-scale 时用 merge.mrc_interp_alpha.<g>)
  --tag-suffix STR    产物名后缀
  --multiway STR      output_sum|mean (默认 config merge.multiway)
  --ortho-scale       方案 B: 互补度自适应 (MRC)
  --conflict-filter   方案 C: 逐方向冲突过滤
  --device STR        auto|cpu|cuda (默认 config merge.device)
  --progress          显示进度条
  -h, --help          显示帮助
EOF
}

ANCHOR=""
HOST=""
DONOR=""
OUT=""
IA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUP=$2; shift 2 ;;
    --anchor) ANCHOR=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --donor) DONOR=$2; shift 2 ;;
    --output) OUT=$2; shift 2 ;;
    --rank) RANK=$2; shift 2 ;;
    --alpha) ALPHA=$2; shift 2 ;;
    --interp-alpha) IA=$2; shift 2 ;;
    --tag-suffix) TAG_SUFFIX=$2; shift 2 ;;
    --multiway) MULTIWAY=$2; shift 2 ;;
    --ortho-scale) ORTHO_SCALE=1; shift ;;
    --conflict-filter) CONFLICT_FILTER=1; shift ;;
    --device) DEVICE=$2; shift 2 ;;
    --progress) NO_PROGRESS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

# 组预设: 唯一来源 = config.yaml
ANCHOR_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.anchor" 2>/dev/null || true)
HOST_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.host" 2>/dev/null || true)
DONOR_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.donor" 2>/dev/null || true)
[ -n "$ANCHOR_NAME" ] && [ -n "$HOST_NAME" ] && [ -n "$DONOR_NAME" ] || {
  echo "[ERROR] config 中不存在 groups.${GROUP} (仅支持 g1 | g3_fwd | g3_rev)" >&2; exit 1;
}
ANCHOR=${ANCHOR:-"$DM/$ANCHOR_NAME"}
HOST=${HOST:-"$DM/$HOST_NAME"}
DONOR=${DONOR:-"$DM/$DONOR_NAME"}
OUT=${OUT:-"$OUT_ROOT/$GROUP"}

# ia: TA_hyb 用 merge.interp_alpha; MRC(ortho) 用 merge.mrc_interp_alpha.<group>
if [ -z "$IA" ]; then
  if [ "$ORTHO_SCALE" = "1" ]; then
    IA=$("$PY" "$CFG_PY" --get "merge.mrc_interp_alpha.${GROUP}" 2>/dev/null || true)
  fi
  [ -n "$IA" ] || IA=$CFG_merge_interp_alpha
fi

for d in "$ANCHOR" "$HOST" "$DONOR"; do
  [ -d "$d" ] || { echo "[ERROR] 模型目录不存在: $d" >&2; exit 1; }
done
mkdir -p "$OUT"

ARGS=(
  --anchor-model "$ANCHOR"
  --host-model "$HOST"
  --donor-model "$DONOR"
  --output-dir "$OUT"
  --rank "$RANK" --alpha "$ALPHA" --interp-alpha "$IA"
  --multiway-combine "$MULTIWAY"
  --device "$DEVICE"
)
[ "$ORTHO_SCALE" = "1" ] && ARGS+=(--ortho-scale)
[ "$CONFLICT_FILTER" = "1" ] && ARGS+=(--conflict-filter)
[ -n "$TAG_SUFFIX" ] && ARGS+=(--tag-suffix "$TAG_SUFFIX")
[ "$NO_PROGRESS" = "1" ] && ARGS+=(--no-progress)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] TA_hyb  group=${GROUP}  config=${CFG_FILE}"
  echo "  anchor=$ANCHOR"
  echo "  host  =$HOST"
  echo "  donor =$DONOR"
  echo "  rank=$RANK alpha=$ALPHA interp_alpha=$IA"
  echo "  multiway=$MULTIWAY ortho_scale=$ORTHO_SCALE conflict_filter=$CONFLICT_FILTER"
  echo "  output=$OUT"
  echo "  cmd: $PY -m merge.rapt.ta_hyb_standalone ${ARGS[*]}"
  exit 0
fi

echo "=============================================="
echo "  RAPT TA_hyb  group=${GROUP}  config=${CFG_FILE}"
echo "  anchor=$(basename "$ANCHOR")"
echo "  host  =$(basename "$HOST")"
echo "  donor =$(basename "$DONOR")"
echo "  rank=${RANK} alpha=${ALPHA} interp_alpha=${IA}"
echo "  multiway=${MULTIWAY} ortho_scale=${ORTHO_SCALE} conflict_filter=${CONFLICT_FILTER}"
echo "  output=${OUT}"
echo "=============================================="

"$PY" -m merge.rapt.ta_hyb_standalone "${ARGS[@]}"

echo "[OK] TA_hyb 生成完成, 产物在: $OUT/$(basename "$HOST")"
