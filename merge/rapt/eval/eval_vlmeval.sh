#!/usr/bin/env bash
# ==============================================================================
# RAPT 统一 VLMEvalKit 评测 (5 数据集, base / merged 通用)
#
# 所有超参来自 merge/rapt/config.yaml (唯一事实来源):
#   eval.vlmeval.datasets / min_pixels / max_pixels / max_new_tokens
# 当前固定口径:
#   min_pixels=1003520  max_pixels=12845056
#   max_new_tokens=null -> 保持框架默认 2048 (Qwen2VLChat.__init__)
#
# 适用模型族:
#   qwen2vl : Qwen2VLChat  (像素参数生效)
#   llava   : LLaVA 类     (VLMEvalKit 不支持像素参数, 用框架默认)
#   mplug   : mPLUG_Owl2   (同上; 需要 mplug_owl2 包)
#   auto    : 由 config.json architectures 推断 (默认)
#
# 用法:
#   bash merge/rapt/eval/eval_vlmeval.sh <model_path> [run_name]
#   DATASETS="TextVQA_VAL ChartQA_TEST" bash ...   # 临时覆盖数据集
#   FAMILY=qwen2vl ...                            # 跳过自动推断
#
# 输出: merge/rapt/eval_results/vlmeval/<run_name>/<timestamp>/
# ==============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${HERE}/../tools/cfg_bootstrap.sh" "$@"

MODEL_PATH=${1:-}
RUN_NAME=${2:-}
if [ -z "$MODEL_PATH" ]; then
  echo "用法: bash merge/rapt/eval/eval_vlmeval.sh <model_path> [run_name]" >&2
  exit 1
fi
[ -d "$MODEL_PATH" ] || { echo "[ERROR] 模型目录不存在: $MODEL_PATH" >&2; exit 1; }
VLM_EVALKIT_DIR=${VLM_EVALKIT_DIR:-"${ROOT}/${CFG_paths_vlmevalkit_repo}"}
[ -f "$VLM_EVALKIT_DIR/eval.sh" ] || { echo "[ERROR] 未找到 VLMEvalKit/eval.sh: $VLM_EVALKIT_DIR" >&2; exit 1; }

GPU=${GPU:-$CFG_eval_vlmeval_gpu}
FAMILY=${FAMILY:-auto}
DATASETS=${DATASETS:-$CFG_eval_vlmeval_datasets}
MODE=${MODE:-$CFG_eval_vlmeval_mode}
NPROC_PER_NODE=${NPROC_PER_NODE:-$CFG_eval_vlmeval_nproc_per_node}
MASTER_PORT=${MASTER_PORT:-0}          # 0 = 自动挑空闲端口
CONDA_ENV=${CONDA_ENV:-vita_qwen2vl}
SKIP_CONDA=${SKIP_CONDA:-1}            # 默认要求已在正确环境; 置 0 则 conda activate

# REUSE: config 的布尔 -> 0/1
if [ -z "${REUSE:-}" ]; then
  if [ "$CFG_eval_vlmeval_reuse" = "true" ]; then REUSE=1; else REUSE=0; fi
fi

# ---- 统一超参 (Qwen2VL 口径; 非 Qwen2VL 由 VLMEvalKit 忽略像素参数) ----
MIN_PIXELS=${MIN_PIXELS:-$CFG_eval_vlmeval_min_pixels}
MAX_PIXELS=${MAX_PIXELS:-$CFG_eval_vlmeval_max_pixels}
# config 中为 null -> 空串 -> VLMEvalKit 不显式设置, 保持类默认 2048
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-$CFG_eval_vlmeval_max_new_tokens}

EVAL_BASE=${EVAL_BASE:-"${ROOT}/${CFG_paths_eval_results}/vlmeval"}
RUN_NAME=${RUN_NAME:-$(basename "$MODEL_PATH")}
RUN_TS=${RUN_TS:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-"${ROOT}/${CFG_paths_logs}"}
mkdir -p "$EVAL_BASE" "$LOG_DIR"
LOG="$LOG_DIR/eval_vlmeval_${RUN_NAME}.log"

if [ "$FAMILY" = "auto" ]; then
  FAMILY=$("$PY" - "$MODEL_PATH" <<'PY'
import json, os, sys
try:
    arch = (json.load(open(os.path.join(sys.argv[1], "config.json"))).get("architectures") or [""])[0].lower()
except Exception:
    arch = ""
name = sys.argv[1].lower()
if "qwen2vl" in arch or "qwen2vl" in name:
    print("qwen2vl")
elif "mplug" in arch or "mplug" in name:
    print("mplug")
elif "llava" in arch or "llava" in name:
    print("llava")
else:
    print("unknown")
PY
)
fi

case "$FAMILY" in
  qwen2vl) MODEL_CLASS=Qwen2VLChat; DEFAULT_MODELS=Qwen2-VL-7B-Instruct ;;
  llava)   MODEL_CLASS=LLaVA;       DEFAULT_MODELS=llava-v1.5-7b ;;
  mplug)   MODEL_CLASS=mPLUG_Owl2;  DEFAULT_MODELS=mplug-owl2-llama2-7b ;;
  *) echo "[ERROR] 无法推断模型族; 请显式设置 FAMILY=qwen2vl|llava|mplug" >&2; exit 1 ;;
esac
MODELS=${MODELS:-$DEFAULT_MODELS}

# ---- DRY_RUN: 只打印解析结果, 不加载模型 ----
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] VLMEvalKit  run_name=$RUN_NAME"
  echo "  config=$CFG_FILE  sha=${CFG_SHA:0:12}"
  echo "  model_path=$MODEL_PATH  family=$FAMILY  class=$MODEL_CLASS"
  echo "  datasets=$DATASETS"
  echo "  min_pixels=$MIN_PIXELS  max_pixels=$MAX_PIXELS"
  echo "  max_new_tokens=${MAX_NEW_TOKENS:-<framework default 2048>}"
  echo "  mode=$MODE  reuse=$REUSE  nproc=$NPROC_PER_NODE"
  echo "  output=${EVAL_BASE}/${RUN_NAME}/<timestamp>"
  exit 0
fi

if [ "$MASTER_PORT" = "0" ]; then
  MASTER_PORT=$("$PY" - <<'PY'
import socket
for p in range(29600, 29800):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", p)); print(p); break
    except OSError:
        continue
    finally:
        s.close()
PY
)
fi

if [ "$REUSE" = "0" ]; then
  WORK_DIR="$EVAL_BASE/$RUN_NAME/$RUN_TS"
else
  WORK_DIR="$EVAL_BASE/$RUN_NAME"
fi
mkdir -p "$WORK_DIR"

# 记录实际生效的参数 (与 lmms-eval manifest 同一套审计口径)
"$PY" - "$WORK_DIR/rapt_eval_manifest.json" "$RUN_NAME" "$MODEL_PATH" "$FAMILY" \
  "$MODEL_CLASS" "$DATASETS" "$MIN_PIXELS" "$MAX_PIXELS" "$MAX_NEW_TOKENS" "$CFG_SHA" <<'PY'
import json, sys, datetime
path, name, model, family, cls, datasets, minp, maxp, tokens, sha = sys.argv[1:]
json.dump({
    "name": name,
    "model_path": model,
    "family": family,
    "model_class": cls,
    "datasets": datasets.split(),
    "eval_script": "merge/rapt/eval/eval_vlmeval.sh",
    "config_sha256": sha,
    "min_pixels": int(minp) if family == "qwen2vl" else None,
    "max_pixels": int(maxp) if family == "qwen2vl" else None,
    "max_new_tokens": int(tokens) if tokens else "<framework-default:2048>",
    "started_at": datetime.datetime.now().strftime("%F %T"),
}, open(path, "w"), ensure_ascii=False, indent=2)
PY

echo "=============================================="
echo "  VLMEvalKit: ${RUN_NAME}  (family=${FAMILY}, class=${MODEL_CLASS})"
echo "  config=${CFG_FILE}  sha=${CFG_SHA:0:12}"
echo "  model_path=${MODEL_PATH}"
echo "  datasets=${DATASETS}"
echo "  min_pixels=${MIN_PIXELS} max_pixels=${MAX_PIXELS}"
echo "  max_new_tokens=${MAX_NEW_TOKENS:-<framework default 2048>}"
echo "  work_dir=${WORK_DIR}"
echo "  gpu=${GPU} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "=============================================="

cd "$VLM_EVALKIT_DIR"

if [ "$SKIP_CONDA" != "1" ]; then
  if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || { echo "[ERROR] conda activate $CONDA_ENV 失败" >&2; exit 1; }
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export LAUNCH_BACKEND=${LAUNCH_BACKEND:-auto}

DATASETS="$DATASETS" \
MODELS="$MODELS" \
MODEL_PATH="$MODEL_PATH" \
MODEL_CLASS="$MODEL_CLASS" \
MODEL_ALIAS="$RUN_NAME" \
MIN_PIXELS="$MIN_PIXELS" \
MAX_PIXELS="$MAX_PIXELS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
MASTER_PORT="$MASTER_PORT" \
MODE="$MODE" \
REUSE="$REUSE" \
WORK_DIR="$WORK_DIR" \
bash "$VLM_EVALKIT_DIR/eval.sh" >> "$LOG" 2>&1 && RC=0 || RC=$?

sed -i -E 's/hf_[A-Za-z0-9]{30,}/<HF_TOKEN_REDACTED>/g' "$LOG" 2>/dev/null || true
echo "[eval_vlmeval] ${RUN_NAME} 完成 rc=$RC  -> $WORK_DIR  (日志: $LOG)"
exit $RC
