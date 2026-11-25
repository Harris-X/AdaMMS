#!/usr/bin/env bash
# ==============================================================================
# RAPT V0.3 Stage-2: 免训练算子对应 + 双 target 谱/算子系数 (迁移自 rapt_V0_3)
#
# 默认值来自 merge/rapt/config.yaml + configs/groups/<group>.yaml:
#   merge.stage2.tau / sinkhorn_iters / merge.rank / paths.work (组可覆盖)
#
# 用法:
#   bash merge/rapt/merge/run_stage2.sh --group g1
#   bash merge/rapt/merge/run_stage2.sh \
#       --subs-a merge/rapt/work/g1/stage1_A_r128_tau.pt \
#       --subs-b merge/rapt/work/g1/stage1_B_r128_tau.pt \
#       --save   merge/rapt/work/g1/stage2_dual_r128_tau0.15_ta.pt
# ==============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${HERE}/../tools/cfg_bootstrap.sh" "$@"

GROUP=${GROUP:-${CFG_GROUP_ARG:-g1}}
GROUP_NAME=${CFG_GROUP:-$GROUP}
CFG_GET_GROUP() {  # <field>
  "$PY" "$CFG_PY" --file "$CFG_FILE" "${GROUP_ARGS[@]}" --get "groups.${GROUP_NAME}.$1" 2>/dev/null || true
}
RANK=${RANK:-$(CFG_GET_GROUP rank)}
[ -n "$RANK" ] || RANK=$CFG_merge_rank
TAU=${TAU:-$CFG_merge_stage2_tau}
ITERS=${ITERS:-$CFG_merge_stage2_sinkhorn_iters}
WORK=${WORK:-"${ROOT}/${CFG_paths_work}/$GROUP_NAME"}
SUBS_A=${SUBS_A:-}
SUBS_B=${SUBS_B:-}
SAVE=${SAVE:-}
SCORE_NORM=${SCORE_NORM:-none}
VERBOSE=${VERBOSE:-0}

usage() {
  cat <<EOF
用法: bash merge/rapt/merge/run_stage2.sh [选项]
  --group STR     g1 | g3_fwd | g3_rev  (默认: g1; 只影响默认缓存路径)
  --work DIR      Stage-1 缓存目录      (默认 merge/rapt/work/<group>)
  --subs-a PATH   host 侧 Stage-1 缓存  (默认 <work>/stage1_A_r<R>_tau.pt)
  --subs-b PATH   donor 侧 Stage-1 缓存 (默认 <work>/stage1_B_r<R>_tau.pt)
  --save   PATH   Stage-2 缓存输出      (默认 <work>/stage2_dual_r<R>_tau0.15_ta.pt)
  --tau    FLOAT  Sinkhorn 温度         (默认 config merge.stage2.tau = 0.15)
  --iters  INT    Sinkhorn 迭代数       (默认 config = 50)
  --rank   INT    最大秩                (默认 config merge.rank = 128)
  --score-norm STR  none|center_scale    (默认 none)
  --verbose       打印逐模块诊断
  -h, --help      显示帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUP=$2; shift 2 ;;
    --work)  WORK=$2; shift 2 ;;
    --subs-a) SUBS_A=$2; shift 2 ;;
    --subs-b) SUBS_B=$2; shift 2 ;;
    --save)   SAVE=$2; shift 2 ;;
    --tau)    TAU=$2; shift 2 ;;
    --iters)  ITERS=$2; shift 2 ;;
    --rank)   RANK=$2; shift 2 ;;
    --score-norm) SCORE_NORM=$2; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

SUBS_A=${SUBS_A:-"$WORK/stage1_A_r${RANK}_tau.pt"}
SUBS_B=${SUBS_B:-"$WORK/stage1_B_r${RANK}_tau.pt"}
SAVE=${SAVE:-"$WORK/stage2_dual_r${RANK}_tau${TAU}_ta.pt"}

EXTRA=()
[ "$VERBOSE" = "1" ] && EXTRA+=(--verbose)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] Stage-2 group=${GROUP_NAME:-$GROUP} config=${CFG_FILE}"
  echo "  work=$WORK"
  echo "  subs-a=$SUBS_A"
  echo "  subs-b=$SUBS_B"
  echo "  save=$SAVE"
  echo "  tau=$TAU  iters=$ITERS  max_rank=$RANK  score_norm=$SCORE_NORM"
  exit 0
fi

for f in "$SUBS_A" "$SUBS_B"; do
  [ -f "$f" ] || { echo "[ERROR] 缓存不存在: $f (先跑 run_stage1.sh)" >&2; exit 1; }
done
mkdir -p "$(dirname "$SAVE")"

echo "=============================================="
echo "  RAPT Stage-2  group=${GROUP}  config=${CFG_FILE}"
echo "  tau=${TAU}  iters=${ITERS}  max_rank=${RANK}"
echo "  subs-a=$(basename "$SUBS_A")"
echo "  subs-b=$(basename "$SUBS_B")"
echo "  save=$(basename "$SAVE")"
echo "=============================================="

"$PY" -m merge.rapt.rapt_v3_stage2_mapping \
  --subs-a "$SUBS_A" --subs-b "$SUBS_B" --save "$SAVE" \
  --tau "$TAU" --sinkhorn-iters "$ITERS" --max-rank "$RANK" \
  --score-norm "$SCORE_NORM" "${EXTRA[@]}"

echo "[OK] Stage-2 完成: $SAVE"
