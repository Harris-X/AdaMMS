#!/usr/bin/env bash
# ==============================================================================
# RAPT V0.3 Stage-1: 方向精确的固定秩 SVD 子空间提取 (迁移自 rapt_V0_3)
#
# 默认值全部来自 merge/rapt/config.yaml:
#   merge.rank / paths.work / paths.models / groups.<group>.{anchor,host,donor}
#
# 用法:
#   bash merge/rapt/merge/run_stage1.sh --group g1
#   bash merge/rapt/merge/run_stage1.sh \
#       --host  downloaded_models/Qwen2-VL-7B-Instruct \
#       --donor downloaded_models/llava-onevision-qwen2-7b-si-hf \
#       --anchor downloaded_models/Qwen2-7B \
#       --rank 128 --work merge/rapt/work/g1
#
# 产物 (work 目录):
#   stage1_A_r{R}_tau.pt   host  任务向量 τ_A = W_A − W_0 的 SVD 缓存
#   stage1_B_r{R}_tau.pt   donor 任务向量 τ_B = W_B − W_0 的 SVD 缓存
# 传 --no-anchor 时分解绝对权重 (V0.1/V0.2 语义), 产物去 _tau 后缀。
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
CUDA=${CUDA:-0}
NO_PROGRESS=${NO_PROGRESS:-1}
ANCHOR=${ANCHOR:-}
HOST=${HOST:-}
DONOR=${DONOR:-}
WORK=${WORK:-"${ROOT}/${CFG_paths_work}/$GROUP"}

usage() {
  cat <<EOF
用法: bash merge/rapt/merge/run_stage1.sh [选项]
  --group STR     g1 | g3_fwd | g3_rev  (默认: g1; 决定 host/donor/anchor)
  --host   PATH   host 模型目录        (默认取 config groups.<g>.host)
  --donor  PATH   donor 模型目录       (默认取 config groups.<g>.donor)
  --anchor PATH   预训练锚点 W_0       (默认取 config groups.<g>.anchor; none=绝对权重)
  --rank   INT    秩                   (默认 config merge.rank)
  --work   DIR    缓存输出目录         (默认 merge/rapt/work/<group>)
  --cuda          使用 GPU 做截断 SVD  (默认 CPU)
  --progress      显示进度条           (默认关闭)
  -h, --help      显示帮助
EOF
}

ANCHOR_GIVEN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --group)     GROUP=$2; shift 2 ;;
    --host)      HOST=$2; shift 2 ;;
    --donor)     DONOR=$2; shift 2 ;;
    --anchor)    ANCHOR=$2; [ "$2" = "none" ] && ANCHOR_GIVEN=0; shift 2 ;;
    --rank)      RANK=$2; shift 2 ;;
    --work)      WORK=$2; shift 2 ;;
    --cuda)      CUDA=1; shift ;;
    --progress)  NO_PROGRESS=0; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$HOST" ] || [ -z "$DONOR" ] || [ -z "$ANCHOR" ]; then
  HOST_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.host" 2>/dev/null || true)
  DONOR_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.donor" 2>/dev/null || true)
  ANCHOR_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.anchor" 2>/dev/null || true)
  [ -n "$HOST_NAME" ] && [ -n "$DONOR_NAME" ] || {
    echo "[ERROR] config 中不存在 groups.${GROUP}" >&2; exit 1; }
  HOST=${HOST:-"$DM/$HOST_NAME"}
  DONOR=${DONOR:-"$DM/$DONOR_NAME"}
  ANCHOR=${ANCHOR:-"$DM/$ANCHOR_NAME"}
fi

for d in "$HOST" "$DONOR"; do
  [ -d "$d" ] || { echo "[ERROR] 模型目录不存在: $d" >&2; exit 1; }
done
if [ "$ANCHOR_GIVEN" = "1" ]; then
  [ -d "$ANCHOR" ] || { echo "[ERROR] 锚点目录不存在: $ANCHOR" >&2; exit 1; }
  SUFFIX="_tau"
  ANCHOR_ARGS=(--anchor-model "$ANCHOR")
else
  SUFFIX=""
  ANCHOR_ARGS=()
fi

mkdir -p "$WORK"
EXTRA=()
[ "$CUDA" = "1" ] && EXTRA+=(--cuda)
[ "$NO_PROGRESS" = "1" ] && EXTRA+=(--no-progress)

echo "=============================================="
echo "  RAPT Stage-1  group=${GROUP}"
echo "  host=$(basename "$HOST")  donor=$(basename "$DONOR")"
echo "  anchor=${ANCHOR:-<absolute-weight>}  rank=${RANK}"
echo "  work=${WORK}  config=${CFG_FILE}"
echo "=============================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Stage-1 group=${GROUP} config=${CFG_FILE}"
  echo "  host=$HOST"
  echo "  donor=$DONOR"
  echo "  anchor=${ANCHOR:-<absolute-weight>}  rank=$RANK  work=$WORK"
  echo "  cmd: $PY -m merge.rapt.rapt_v3_stage1_subspace --model-dir <host|donor> --rank $RANK ${ANCHOR_ARGS[*]} ${EXTRA[*]}"
  exit 0
fi

echo "=== [1/2] Stage-1 host: $(basename "$HOST") ==="
"$PY" -m merge.rapt.rapt_v3_stage1_subspace \
  --model-dir "$HOST" --rank "$RANK" \
  "${ANCHOR_ARGS[@]}" "${EXTRA[@]}" \
  --save "$WORK/stage1_A_r${RANK}${SUFFIX}.pt"

echo "=== [2/2] Stage-1 donor: $(basename "$DONOR") ==="
"$PY" -m merge.rapt.rapt_v3_stage1_subspace \
  --model-dir "$DONOR" --rank "$RANK" \
  "${ANCHOR_ARGS[@]}" "${EXTRA[@]}" \
  --save "$WORK/stage1_B_r${RANK}${SUFFIX}.pt"

echo "[OK] Stage-1 完成: $WORK/stage1_A_r${RANK}${SUFFIX}.pt, $WORK/stage1_B_r${RANK}${SUFFIX}.pt"
