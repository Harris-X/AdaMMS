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
source "${HERE}/../tools/cfg_bootstrap.sh" "$@"

GROUP=${GROUP:-${CFG_GROUP_ARG:-g1}}
GROUP_NAME=${CFG_GROUP:-$GROUP}
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
  --group STR         组名 (g1/g3_fwd/...) 或组 YAML 路径 (默认: g1)
  --anchor PATH       预训练锚点 W_0 (默认取组配置 anchor)
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

# 组预设: 唯一来源 = configs/groups/<group>.yaml (经 cfg.py 合并后注入 groups.<name>)
CFG_GET_GROUP() {  # <field>
  "$PY" "$CFG_PY" --file "$CFG_FILE" "${GROUP_ARGS[@]}" --get "groups.${GROUP_NAME}.$1" 2>/dev/null || true
}
ANCHOR_NAME=$(CFG_GET_GROUP anchor)
HOST_NAME=$(CFG_GET_GROUP host)
DONOR_NAME=$(CFG_GET_GROUP donor)
[ -n "$ANCHOR_NAME" ] && [ -n "$HOST_NAME" ] && [ -n "$DONOR_NAME" ] || {
  echo "[ERROR] 未找到模型组 '$GROUP' 的 anchor/host/donor" >&2
  echo "        期望配置文件: $(cfg 提示 2>/dev/null || true) merge/rapt/configs/groups/<name>.yaml" >&2
  echo "        可用组: $("$PY" "$CFG_PY" --file "$CFG_FILE" --list-groups | tail -n +2 | awk '{print $1}' | tr '\n' ' ')" >&2
  exit 1;
}
ANCHOR=${ANCHOR:-"$DM/$ANCHOR_NAME"}
HOST=${HOST:-"$DM/$HOST_NAME"}
DONOR=${DONOR:-"$DM/$DONOR_NAME"}
OUT=${OUT:-"$OUT_ROOT/$GROUP_NAME"}

GROUP_BACKEND=$(CFG_GET_GROUP backend)
if [ "$GROUP_BACKEND" = "pipeline" ]; then
  echo "[ERROR] 组 '$GROUP_NAME' 配置为 backend=pipeline, 不能走 TA_hyb 单文件入口。" >&2
  echo "        请用: bash merge/rapt/merge/run_merge_all.sh --group ${CFG_GROUP_ARG:-$GROUP_NAME}" >&2
  echo "        或:   bash merge/rapt/merge/run_stage1.sh --group ... && run_stage2.sh && run_stage3.sh" >&2
  exit 1
fi
GROUP_STATUS=$(CFG_GET_GROUP status)
if [ -n "$GROUP_STATUS" ] && [ "$GROUP_STATUS" != "ready" ] && [ "${ALLOW_UNREADY:-0}" != "1" ]; then
  echo "[WARN] 组 '$GROUP_NAME' status=$GROUP_STATUS (非 ready): 结果可能只覆盖命名/形状匹配的模块。" >&2
  echo "       如需强制运行: ALLOW_UNREADY=1 ..." >&2
fi

# ia: 优先取组配置; TA_hyb 用 interp_alpha, MRC(ortho) 用 mrc_interp_alpha
if [ -z "$IA" ]; then
  if [ "$ORTHO_SCALE" = "1" ]; then
    IA=$(CFG_GET_GROUP mrc_interp_alpha)
  fi
  [ -n "$IA" ] || IA=$(CFG_GET_GROUP interp_alpha)
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
  echo "[DRY_RUN] TA_hyb  group=${GROUP_NAME}  config=${CFG_FILE}"
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
