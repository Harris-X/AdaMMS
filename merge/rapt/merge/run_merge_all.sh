#!/usr/bin/env bash
# ==============================================================================
# RAPT V0.3 多组模型一键合并
#
# 组与超参全部来自 merge/rapt/config.yaml:
#   merge.groups / backend / rank / alpha / interp_alpha / variants / multiway / device
#   groups.<g>.anchor / host / donor
#
# 两种后端:
#   BACKEND=ta_hyb   (config 默认) 单文件实现, 无 Stage-1/2 缓存
#   BACKEND=pipeline       完整 Stage-1 -> Stage-2 -> Stage-3 三阶段管线
#
# 用法:
#   bash merge/rapt/merge/run_merge_all.sh
#   bash merge/rapt/merge/run_merge_all.sh --group g1
#   BACKEND=pipeline bash merge/rapt/merge/run_merge_all.sh
#   ORTHO_SCALE=1 bash merge/rapt/merge/run_merge_all.sh      # MRC (方案 B)
#   RANK=256 bash merge/rapt/merge/run_merge_all.sh           # 临时覆盖
# ==============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT=$(cd "${HERE}/../../.." &>/dev/null && pwd)
CFG_FILE=${RAPT_CONFIG:-"${ROOT}/merge/rapt/config.yaml"}
CFG_PY="${HERE}/../tools/cfg.py"

BOOT_PY=${PY:-python3}
PY=${PY:-$("$BOOT_PY" "$CFG_PY" --get paths.python 2>/dev/null || echo python3)}
[ -x "$PY" ] || PY=python3
eval "$("$PY" "$CFG_PY" --shell "$CFG_FILE")" || { echo "[ERROR] 读取配置失败: $CFG_FILE" >&2; exit 1; }

GROUPS=${GROUP:-${GROUPS:-$CFG_merge_groups}}
BACKEND=${BACKEND:-$CFG_merge_backend}
RANK=${RANK:-$CFG_merge_rank}
ALPHA=${ALPHA:-$CFG_merge_alpha}
ORTHO_SCALE=${ORTHO_SCALE:-}
CONFLICT_FILTER=${CONFLICT_FILTER:-}
MULTIWAY=${MULTIWAY:-$CFG_merge_multiway}
DEVICE=${DEVICE:-$CFG_merge_device}
DM=${DM:-"${ROOT}/${CFG_paths_models}"}
OUT_ROOT=${OUT_ROOT:-"${ROOT}/${CFG_paths_merged_models}"}
WORK_ROOT=${WORK_ROOT:-"${ROOT}/${CFG_paths_work}"}
NO_PROGRESS=${NO_PROGRESS:-1}

while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUPS=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; exit 1 ;;
  esac
done

echo "================================================================"
echo "  RAPT 一键合并  groups='$GROUPS'  backend=$BACKEND"
echo "  config=$CFG_FILE  sha=$("$PY" "$CFG_PY" --sha256 "$CFG_FILE" | cut -c1-12)"
echo "  rank=$RANK alpha=$ALPHA variants.ortho=$ORTHO_SCALE conflict=$CONFLICT_FILTER"
echo "  output_root=$OUT_ROOT"
echo "================================================================"

RUN_ONE() {
  local g=$1
  if [ "$BACKEND" = "ta_hyb" ]; then
    local args=(--group "$g" --rank "$RANK" --alpha "$ALPHA"
                --multiway "$MULTIWAY" --device "$DEVICE")
    [ -n "$ORTHO_SCALE" ] && args+=(--ortho-scale)
    [ -n "$CONFLICT_FILTER" ] && args+=(--conflict-filter)
    [ "$NO_PROGRESS" = "1" ] && args+=(--no-progress)
    bash "$HERE/run_ta_hyb.sh" "${args[@]}"
  else
    local anchor_name host_name donor_name anchor host donor
    anchor_name=$("$PY" "$CFG_PY" --get "groups.${g}.anchor")
    host_name=$("$PY" "$CFG_PY" --get "groups.${g}.host")
    donor_name=$("$PY" "$CFG_PY" --get "groups.${g}.donor")
    anchor="$DM/$anchor_name"; host="$DM/$host_name"; donor="$DM/$donor_name"
    local work="$WORK_ROOT/$g" out="$OUT_ROOT/$g"
    for d in "$anchor" "$host" "$donor"; do
      [ -d "$d" ] || { echo "[ERROR][$g] 模型目录不存在: $d" >&2; return 1; }
    done
    mkdir -p "$work" "$out"
    bash "$HERE/run_stage1.sh" --host "$host" --donor "$donor" --anchor "$anchor" \
        --rank "$RANK" --work "$work"
    bash "$HERE/run_stage2.sh" \
        --subs-a "$work/stage1_A_r${RANK}_tau.pt" \
        --subs-b "$work/stage1_B_r${RANK}_tau.pt" \
        --save "$work/stage2_dual_r${RANK}_tau0.15_ta.pt" \
        --tau "$CFG_merge_stage2_tau" --iters "$CFG_merge_stage2_sinkhorn_iters" --rank "$RANK"
    local ia
    ia=$("$PY" "$CFG_PY" --get "groups.${g}.interp_alpha")
    bash "$HERE/run_stage3.sh" --host "$host" --donor "$donor" --work "$work" \
        --output-dir "$out" --rank "$RANK" \
        --mode "$CFG_merge_mode" --target "$CFG_merge_target" --gate "$CFG_merge_gate" \
        --alpha "$ALPHA" --interp-alpha "$ia"
  fi
}

for g in $GROUPS; do
  echo ""
  echo "################################################################"
  echo "# [$g] backend=$BACKEND"
  echo "################################################################"
  RUN_ONE "$g" || echo "[ERROR] 组 $g 合并失败, 继续下一组" >&2
done

echo ""
echo "[ALL DONE] 产物根目录: $OUT_ROOT"
ls -d "$OUT_ROOT"/*/* 2>/dev/null | head -20 || true
