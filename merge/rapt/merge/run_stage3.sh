#!/usr/bin/env bash
# ==============================================================================
# RAPT V0.3 Stage-3: target × gate 解耦的保护性注入 (迁移自 rapt_V0_3)
#
# 默认即已确认的 TA_hyb 最优点 (方法说明 §21.6):
#   --mode hybrid --interp-alpha 0.1 --target task_delta --gate apt
#   --alpha 0.5 --identity-mapping
# 单文件等价实现见 merge/rapt/ta_hyb_standalone.py; 本脚本保留完整三阶段管线,
# 用于需要换 target / gate / budget 的消融。
#
# 默认值来自 merge/rapt/config.yaml:
#   merge.mode / target / gate / alpha / interp_alpha / identity_mapping
#   merge.rank / paths.work / paths.merged_models
#   groups.<group>.{host,donor}
#
# 用法:
#   bash merge/rapt/merge/run_stage3.sh --group g1
#   bash merge/rapt/merge/run_stage3.sh --group g1 --target operator_response --gate none
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
DM=${DM:-"${ROOT}/${CFG_paths_models}"}
RANK=${RANK:-$CFG_merge_rank}
MODE=${MODE:-$CFG_merge_mode}
INJECTION=${INJECTION:-}
TARGET=${TARGET:-$CFG_merge_target}
GATE=${GATE:-$CFG_merge_gate}
ALPHA=${ALPHA:-$CFG_merge_alpha}
IA=${IA:-$CFG_merge_interp_alpha}
BUDGET=${BUDGET:-}
INJECT_BUDGET=${INJECT_BUDGET:-}
TAG_SUFFIX=${TAG_SUFFIX:-}
NO_PROGRESS=${NO_PROGRESS:-1}
SUBS_A_SIDE=${SUBS_A_SIDE:-donor}
HOST=${HOST:-}
DONOR=${DONOR:-}
WORK=${WORK:-"${ROOT}/${CFG_paths_work}/$GROUP"}
OUT=${OUT:-"${ROOT}/${CFG_paths_merged_models}/$GROUP"}
IDENTITY_MAPPING=${IDENTITY_MAPPING:-}

if [ -z "$IDENTITY_MAPPING" ]; then
  if [ "$CFG_merge_identity_mapping" = "true" ]; then IDENTITY_MAPPING=1; else IDENTITY_MAPPING=0; fi
fi

usage() {
  cat <<EOF
用法: bash merge/rapt/merge/run_stage3.sh [选项]
  --group STR         g1 | g3_fwd | g3_rev (默认: g1)
  --host PATH         host 模型目录   (默认 config groups.<g>.host)
  --donor PATH        donor 模型目录  (默认 config groups.<g>.donor)
  --work DIR          Stage-1/2 缓存目录 (默认 merge/rapt/work/<group>)
  --output-dir DIR    合并产物目录    (默认 merge/rapt/merged_models/<group>)
  --rank INT          秩 (默认 config merge.rank)
  --mode STR          inject|hybrid|direct (默认 config merge.mode)
  --injection STR     diag|subspace|bidir
  --target STR        task_delta|transport_spectrum|operator_response|
                      full_donor_response|calibrated_transport|hard_transport
  --gate STR          apt|component|mean|energy|none|norm_match|
                      mismatch_component_donor|host_overlap
  --alpha FLOAT       注入强度 (默认 config merge.alpha)
  --interp-alpha F    插值系数 (默认 config merge.interp_alpha)
  --budget STR        raw|rel (默认不传, 保持 stage3 内部默认)
  --inject-budget STR raw|rel
  --tag-suffix STR    产物名后缀
  --subs-a-side STR   donor|host (默认 donor, 即 TA 分支口径)
  --no-identity-mapping  关闭 identity-mapping
  --progress          显示进度条
  -h, --help          显示帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUP=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --donor) DONOR=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --output-dir) OUT=$2; shift 2 ;;
    --rank) RANK=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --injection) INJECTION=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    --gate) GATE=$2; shift 2 ;;
    --alpha) ALPHA=$2; shift 2 ;;
    --interp-alpha) IA=$2; shift 2 ;;
    --budget) BUDGET=$2; shift 2 ;;
    --inject-budget) INJECT_BUDGET=$2; shift 2 ;;
    --tag-suffix) TAG_SUFFIX=$2; shift 2 ;;
    --subs-a-side) SUBS_A_SIDE=$2; shift 2 ;;
    --no-identity-mapping) IDENTITY_MAPPING=0; shift ;;
    --progress) NO_PROGRESS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$HOST" ] || [ -z "$DONOR" ]; then
  HOST_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.host" 2>/dev/null || true)
  DONOR_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.donor" 2>/dev/null || true)
  [ -n "$HOST_NAME" ] && [ -n "$DONOR_NAME" ] || {
    echo "[ERROR] config 中不存在 groups.${GROUP}" >&2; exit 1; }
  HOST=${HOST:-"$DM/$HOST_NAME"}
  DONOR=${DONOR:-"$DM/$DONOR_NAME"}
fi
for d in "$HOST" "$DONOR"; do
  [ -d "$d" ] || { echo "[ERROR] 模型目录不存在: $d" >&2; exit 1; }
done
[ -d "$WORK" ] || { echo "[ERROR] 缓存目录不存在: $WORK (先跑 run_stage1.sh / run_stage2.sh)" >&2; exit 1; }
mkdir -p "$OUT"

# TA 分支: 注入基 = donor 的 τ 分解, host τ 仅用于门控/诊断
if [ "$SUBS_A_SIDE" = "donor" ]; then
  BASE_SUBS="$WORK/stage1_B_r${RANK}_tau.pt"
  DONOR_SUBS="$WORK/stage1_A_r${RANK}_tau.pt"
else
  BASE_SUBS="$WORK/stage1_A_r${RANK}_tau.pt"
  DONOR_SUBS="$WORK/stage1_B_r${RANK}_tau.pt"
fi
STAGE2="$WORK/stage2_dual_r${RANK}_tau${CFG_merge_stage2_tau}_ta.pt"
for f in "$BASE_SUBS" "$DONOR_SUBS" "$STAGE2"; do
  [ -f "$f" ] || { echo "[ERROR] 缺少缓存: $f" >&2; exit 1; }
done

ARGS=(
  --base-model "$HOST"
  --base-subs "$BASE_SUBS"
  --donor-subs "$DONOR_SUBS"
  --stage2 "$STAGE2"
  --donor-model "$DONOR"
  --output-dir "$OUT" --rank "$RANK"
  --mode "$MODE" --target "$TARGET" --gate "$GATE" --alpha "$ALPHA"
)
[ -n "$BUDGET" ] && ARGS+=(--budget "$BUDGET")
[ -n "$INJECT_BUDGET" ] && ARGS+=(--inject-budget "$INJECT_BUDGET")
[ "$MODE" = "hybrid" ] && ARGS+=(--interp-alpha "$IA")
[ -n "$INJECTION" ] && ARGS+=(--injection "$INJECTION")
[ "$IDENTITY_MAPPING" = "1" ] && ARGS+=(--identity-mapping)
[ -n "$TAG_SUFFIX" ] && ARGS+=(--tag-suffix "$TAG_SUFFIX")
[ "$NO_PROGRESS" = "1" ] && ARGS+=(--no-progress)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Stage-3 group=${GROUP} config=${CFG_FILE}"
  echo "  host=$HOST  donor=$DONOR  work=$WORK"
  echo "  mode=$MODE target=$TARGET gate=$GATE alpha=$ALPHA interp_alpha=$IA"
  echo "  identity_mapping=$IDENTITY_MAPPING  subs_a_side=$SUBS_A_SIDE"
  echo "  output=$OUT"
  echo "  cmd: $PY -m merge.rapt.rapt_v3_stage3_injection ${ARGS[*]}"
  exit 0
fi

echo "=============================================="
echo "  RAPT Stage-3  group=${GROUP}  config=${CFG_FILE}"
echo "  mode=${MODE} target=${TARGET} gate=${GATE}"
echo "  alpha=${ALPHA} interp_alpha=${IA} rank=${RANK}"
echo "  host=$(basename "$HOST")  donor=$(basename "$DONOR")"
echo "  work=${WORK}"
echo "  output=${OUT}"
echo "=============================================="

"$PY" -m merge.rapt.rapt_v3_stage3_injection "${ARGS[@]}"

echo "[OK] Stage-3 完成, 产物目录: $OUT"
