#!/usr/bin/env bash
# ==============================================================================
# base 与 merged 同参数配对评测 (lmms-eval 与 VLMEvalKit 各跑一遍)
#
# 所有超参来自 merge/rapt/config.yaml:
#   eval.lmms_eval.* / eval.vlmeval.*     (两套框架的固定口径)
#   eval.pairs.roles / both_frameworks    (配对角色与是否两框架都跑)
#   groups.<group>.host                   (base 模型)
#
# 因此 base 与 merged 不可能拿到不同参数: 两者都调用同一脚本、同一份 config。
#
# 用法:
#   bash merge/rapt/eval/run_base_and_merged.sh --group g1 --merged <merged_dir>
#   bash merge/rapt/eval/run_base_and_merged.sh --group g3_fwd \
#       --merged merge/rapt/merged_models/g3_fwd/llava-v1.5-7b/<tag>
#   EVALS="lmms" bash ...        # 只跑 lmms-eval
#   EVALS="vlmeval" bash ...     # 只跑 VLMEvalKit
#
# 输出:
#   merge/rapt/eval_results/lmms/<tag>_base 与 <tag>_merged
#   merge/rapt/eval_results/vlmeval/<tag>_base 与 <tag>_merged
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

GROUP=${GROUP:-g1}
MERGED=${MERGED:-}
BASE=${BASE:-}
GPU=${GPU:-$CFG_eval_lmms_eval_gpu}
PAIR_TAG=${PAIR_TAG:-$GROUP}
DM=${DM:-"${ROOT}/${CFG_paths_models}"}

if [ -z "${EVALS:-}" ]; then
  if [ "$CFG_eval_pairs_both_frameworks" = "true" ]; then EVALS="lmms vlmeval"; else EVALS="lmms"; fi
fi

usage() {
  cat <<EOF
用法: bash merge/rapt/eval/run_base_and_merged.sh [选项]
  --group STR    g1 | g3_fwd | g3_rev           (默认: g1)
  --merged PATH  merged 模型目录 (必填)
  --base PATH    base 模型目录 (默认 = config 的 groups.<group>.host)
  --evals STR    "lmms" | "vlmeval" | "lmms vlmeval"
  --tag STR      输出目录前缀 (默认 = group)
  --gpu ID       显卡编号 (默认取 config)
  -h, --help     显示帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUP=$2; shift 2 ;;
    --merged) MERGED=$2; shift 2 ;;
    --base) BASE=$2; shift 2 ;;
    --evals) EVALS=$2; shift 2 ;;
    --tag) PAIR_TAG=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

[ -n "$MERGED" ] || { echo "[ERROR] --merged 为必填" >&2; usage; exit 1; }
[ -d "$MERGED" ] || { echo "[ERROR] merged 目录不存在: $MERGED" >&2; exit 1; }

# base = 该组的 host (config 唯一来源)
HOST_NAME=$("$PY" "$CFG_PY" --get "groups.${GROUP}.host" 2>/dev/null || true)
[ -n "$HOST_NAME" ] || { echo "[ERROR] config 中不存在 groups.${GROUP}.host" >&2; exit 1; }
BASE=${BASE:-"$DM/$HOST_NAME"}
[ -d "$BASE" ] || { echo "[ERROR] base 目录不存在: $BASE" >&2; exit 1; }

echo "================================================================"
echo "  配对评测 group=$GROUP  tag=$PAIR_TAG  evals='$EVALS'  gpu=$GPU"
echo "  config=$CFG_FILE  sha=$("$PY" "$CFG_PY" --sha256 "$CFG_FILE" | cut -c1-12)"
echo "  base  =$BASE"
echo "  merged=$MERGED"
echo "  (base/merged 均从同一 config 读取超参, 无任何角色相关分支)"
echo "================================================================"

run_one() {
  local role=$1 path=$2
  local name="${PAIR_TAG}_${role}"
  for ev in $EVALS; do
    case "$ev" in
      lmms)
        echo ""
        echo "---------- [$name] lmms-eval ----------"
        GPU="$GPU" bash "$HERE/eval_lmms.sh" "$path" "$name"
        ;;
      vlmeval)
        echo ""
        echo "---------- [$name] VLMEvalKit ----------"
        GPU="$GPU" bash "$HERE/eval_vlmeval.sh" "$path" "$name"
        ;;
      *) echo "[ERROR] 未知评测器: $ev (仅支持 lmms|vlmeval)" >&2; return 1 ;;
    esac
  done
}

# 角色顺序来自 config eval.pairs.roles; 两角色都调用同一脚本与同一 config
for role in $CFG_eval_pairs_roles; do
  case "$role" in
    base)   run_one base   "$BASE"   || echo "[WARN] base 评测有失败任务" ;;
    merged) run_one merged "$MERGED" || echo "[WARN] merged 评测有失败任务" ;;
    *) echo "[WARN] config eval.pairs.roles 含未知角色: $role (跳过)" ;;
  esac
done

echo ""
echo "================================================================"
echo "  配对评测完成"
echo "  lmms-eval  : ${ROOT}/${CFG_paths_eval_results}/lmms/${PAIR_TAG}_base"
echo "               ${ROOT}/${CFG_paths_eval_results}/lmms/${PAIR_TAG}_merged"
echo "  VLMEvalKit : ${ROOT}/${CFG_paths_eval_results}/vlmeval/${PAIR_TAG}_base"
echo "               ${ROOT}/${CFG_paths_eval_results}/vlmeval/${PAIR_TAG}_merged"
echo "  汇总       : ${PY} ${HERE}/summarize_eval8.py \\"
echo "               ${ROOT}/${CFG_paths_eval_results}/lmms/${PAIR_TAG}_base \\"
echo "               ${ROOT}/${CFG_paths_eval_results}/lmms/${PAIR_TAG}_merged"
echo "================================================================"
