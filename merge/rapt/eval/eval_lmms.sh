#!/usr/bin/env bash
# ==============================================================================
# RAPT 统一 lmms-eval 评测 (8 任务, 逐任务落盘, 可断点续跑)
#
# 所有超参来自 merge/rapt/config.yaml (唯一事实来源):
#   eval.lmms_eval.tasks / min_pixels / max_pixels / max_num_frames /
#   max_new_tokens / temperature / batch_size
# 当前固定口径: min_pixels=3136, max_pixels=12845056,
#               max_num_frames=16, max_new_tokens=256, temperature=0
#   (= eval_qwen2vl_lmms.sh 默认; base 与 merged 完全相同)
#
# 适用模型族 (FAMILY):
#   qwen2vl : Qwen2-VL 系 (上述 4 个参数全部生效)
#   llava   : LLaVA-1.5 系  (框架不支持像素参数, 仅统一 max_new_tokens)
#   mplug   : mPLUG-Owl2 系  (同上)
#   auto    : 由 config.json 的 architectures 自动推断 (默认)
#
# 用法:
#   bash merge/rapt/eval/eval_lmms.sh <model_path> [run_name]
#   RAPT_CONFIG=/path/to/config.yaml bash ...      # 换配置
#   TASKS="mmmu_val mme ocrbench" bash ...          # 临时覆盖任务子集
#   NO_STAGE=1 bash ...                             # 不拷贝到本地盘
#
# 输出: merge/rapt/eval_results/lmms/<run_name>/<task>/<model>/<ts>_results.json
#       merge/rapt/eval_results/lmms/<run_name>/run_manifest.json
# ==============================================================================
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT=$(cd "${HERE}/../../.." &>/dev/null && pwd)
CFG_FILE=${RAPT_CONFIG:-"${ROOT}/merge/rapt/config.yaml"}
CFG_PY="${HERE}/../tools/cfg.py"

BOOT_PY=${PY:-python3}
PY=${PY:-$("$BOOT_PY" "$CFG_PY" --get paths.python 2>/dev/null || echo python3)}
[ -x "$PY" ] || PY=python3
eval "$("$PY" "$CFG_PY" --shell "$CFG_FILE")" || { echo "[ERROR] 读取配置失败: $CFG_FILE" >&2; exit 1; }
CFG_SHA=$("$PY" "$CFG_PY" --sha256 "$CFG_FILE")

MODEL_PATH=${1:-}
RUN_NAME=${2:-}
if [ -z "$MODEL_PATH" ]; then
  echo "用法: bash merge/rapt/eval/eval_lmms.sh <model_path> [run_name]" >&2
  exit 1
fi
[ -d "$MODEL_PATH" ] || { echo "[ERROR] 模型目录不存在: $MODEL_PATH" >&2; exit 1; }

GPU=${GPU:-$CFG_eval_lmms_eval_gpu}
FAMILY=${FAMILY:-auto}
EVAL_BASE=${EVAL_BASE:-"${ROOT}/${CFG_paths_eval_results}/lmms"}
RUN_NAME=${RUN_NAME:-$(basename "$MODEL_PATH")}
OUT="$EVAL_BASE/$RUN_NAME"
LOG_DIR=${LOG_DIR:-"${ROOT}/${CFG_paths_logs}"}
mkdir -p "$OUT" "$LOG_DIR"
LOG="$LOG_DIR/eval_lmms_${RUN_NAME}.log"

# ---- 统一超参: 默认取 config; 环境变量仅作单次临时覆盖 (会写入 manifest) ----
MIN_PIXELS=${MIN_PIXELS:-$CFG_eval_lmms_eval_min_pixels}
MAX_PIXELS=${MAX_PIXELS:-$CFG_eval_lmms_eval_max_pixels}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-$CFG_eval_lmms_eval_max_num_frames}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-$CFG_eval_lmms_eval_max_new_tokens}
TEMPERATURE=${TEMPERATURE:-$CFG_eval_lmms_eval_temperature}
BATCH_SIZE=${BATCH_SIZE:-$CFG_eval_lmms_eval_batch_size}
TASK_ORDER=${TASKS:-$CFG_eval_lmms_eval_tasks}

# ---- 模型族推断 ----
if [ "$FAMILY" = "auto" ]; then
  FAMILY=$("$PY" - "$MODEL_PATH" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], "config.json")
try:
    arch = (json.load(open(p)).get("architectures") or [""])[0].lower()
except Exception:
    arch = ""
name = sys.argv[1].lower()
if "qwen2vl" in arch or "qwen2vl" in name:
    print("qwen2vl")
elif "mplugowl" in arch or "mplug" in name:
    print("mplug")
elif "llava" in arch or "llava" in name:
    print("llava")
else:
    print("unknown")
PY
)
fi
case "$FAMILY" in
  qwen2vl|llava|mplug) : ;;
  *) echo "[ERROR] 无法推断模型族; 请显式设置 FAMILY=qwen2vl|llava|mplug" >&2; exit 1 ;;
esac

# ---- DRY_RUN: 只打印解析结果, 不加载模型 ----
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] lmms-eval  run_name=$RUN_NAME"
  echo "  config=$CFG_FILE  sha=${CFG_SHA:0:12}"
  echo "  model_path=$MODEL_PATH  family=$FAMILY"
  echo "  tasks=$TASK_ORDER"
  if [ "$FAMILY" = "qwen2vl" ]; then
    echo "  min_pixels=$MIN_PIXELS  max_pixels=$MAX_PIXELS"
    echo "  max_num_frames=$MAX_NUM_FRAMES"
  else
    echo "  min_pixels/max_pixels/max_num_frames=N/A (${FAMILY} adapter 不支持)"
  fi
  echo "  max_new_tokens=$MAX_NEW_TOKENS  temperature=$TEMPERATURE  batch_size=$BATCH_SIZE"
  echo "  output=$OUT"
  exit 0
fi

# ---- 本地盘 stage (NFS 读 7B 权重很慢; 已完整则复用) ----
EVAL_PATH=$MODEL_PATH
STAGE_ROOT=${STAGE_ROOT:-/root/rapt_stage}
if [ -z "${NO_STAGE:-}" ]; then
  if [ "$CFG_eval_lmms_eval_stage_local" = "true" ]; then NO_STAGE=0; else NO_STAGE=1; fi
fi
if [ -d "$MODEL_PATH" ] && [ "$NO_STAGE" != "1" ]; then
  mkdir -p "$STAGE_ROOT"
  STAGED="$STAGE_ROOT/$RUN_NAME"
  if [ -f "$STAGED/config.json" ] && ls "$STAGED"/*.safetensors >/dev/null 2>&1; then
    echo "[eval_lmms] 复用已 stage 的 $STAGED"
    EVAL_PATH="$STAGED"
  else
    rm -rf "$STAGED"
    echo "[eval_lmms] staging $RUN_NAME -> $STAGED ($(date '+%F %T'))"
    if cp -rL "$MODEL_PATH" "$STAGED" 2>>"$LOG_DIR/stage_err.log"; then
      EVAL_PATH="$STAGED"
    else
      echo "[eval_lmms] stage 失败, 回退直接读原路径"
      EVAL_PATH="$MODEL_PATH"
    fi
  fi
fi

# ---- 防覆盖/防沿用 + 记录实际生效超参 ----
MANIFEST=$OUT/run_manifest.json
if [ -f "$MANIFEST" ]; then
  PREV=$("$PY" -c "import json;print(json.load(open('$MANIFEST')).get('model_path',''))" 2>/dev/null || echo "")
  if [ "$PREV" != "$MODEL_PATH" ]; then
    echo "[ABORT] 输出名 '$RUN_NAME' 已绑定到另一个模型: $PREV != $MODEL_PATH" >&2
    exit 1
  fi
fi
"$PY" - "$MANIFEST" "$RUN_NAME" "$MODEL_PATH" "$FAMILY" "$TASK_ORDER" \
  "$MIN_PIXELS" "$MAX_PIXELS" "$MAX_NUM_FRAMES" "$MAX_NEW_TOKENS" "$TEMPERATURE" \
  "$BATCH_SIZE" "$CFG_SHA" <<'PY'
import json, os, sys, datetime
(path, name, model, family, tasks, minp, maxp, frames, tokens, temp,
 batch, sha) = sys.argv[1:]
json.dump({
    "name": name,
    "model_path": model,
    "family": family,
    "tasks": tasks.split(),
    "eval_script": "merge/rapt/eval/eval_lmms.sh",
    "config_sha256": sha,
    "min_pixels": int(minp) if family == "qwen2vl" else None,
    "max_pixels": int(maxp) if family == "qwen2vl" else None,
    "max_num_frames": int(frames) if family == "qwen2vl" else None,
    "max_new_tokens": int(tokens),
    "temperature": float(temp),
    "batch_size": int(batch),
    "started_at": datetime.datetime.now().strftime("%F %T"),
}, open(path, "w"), ensure_ascii=False, indent=2)
PY

# ---- 按模型族构造 model_args / gen_kwargs ----
GEN_KWARGS="max_new_tokens=${MAX_NEW_TOKENS},temperature=${TEMPERATURE}"
MODEL_ARGS="pretrained=${EVAL_PATH},device_map=auto"
if [ "$FAMILY" = "qwen2vl" ]; then
  MODEL_ARGS="${MODEL_ARGS},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},max_num_frames=${MAX_NUM_FRAMES}"
elif [ "$FAMILY" = "llava" ]; then
  # lmms-eval 的 LLaVA builder 依据 model_name 是否含 "llava" 选分支;
  # 合并产物名通常不含 llava, 必须显式提示 (与原 eval_single_model_llava_v15.sh 一致)。
  LLAVA_MODEL_NAME=${LLAVA_MODEL_NAME:-$CFG_eval_lmms_eval_llava_model_name}
  MODEL_ARGS="${MODEL_ARGS},model_name=${LLAVA_MODEL_NAME}"
fi

echo "=============================================="
echo "  lmms-eval: ${RUN_NAME}  (family=${FAMILY})"
echo "  config=${CFG_FILE}  sha=${CFG_SHA:0:12}"
echo "  model_path=${MODEL_PATH}"
echo "  eval_path =${EVAL_PATH}"
echo "  tasks=${TASK_ORDER}"
echo "  min_pixels=${MIN_PIXELS} max_pixels=${MAX_PIXELS}"
echo "  max_num_frames=${MAX_NUM_FRAMES} max_new_tokens=${MAX_NEW_TOKENS}"
echo "  temperature=${TEMPERATURE} batch_size=${BATCH_SIZE}"
echo "  output=${OUT}"
echo "=============================================="

# ---- 逐任务评测 (已完成自动跳过) ----
for T in $TASK_ORDER; do
  TDIR=$OUT/$T
  mkdir -p "$TDIR"
  if ls "$TDIR"/*/*_results.json >/dev/null 2>&1; then
    echo "[eval_lmms] ${RUN_NAME} / ${T} 已完成, 跳过"
    continue
  fi
  echo "[eval_lmms] === ${RUN_NAME} / ${T} 开始 $(date '+%F %T') ==="
  case "$FAMILY" in
    qwen2vl) LM_MODEL=qwen2_vl ;;
    llava)   LM_MODEL=llava ;;
    mplug)   LM_MODEL=mplug_owl2 ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m lmms_eval \
    --model "$LM_MODEL" --model_args "$MODEL_ARGS" \
    --gen_kwargs "$GEN_KWARGS" \
    --tasks "$T" --batch_size "$BATCH_SIZE" --log_samples \
    --output_path "$TDIR" >> "$LOG" 2>&1
  RC=$?
  echo "[eval_lmms] === ${RUN_NAME} / ${T} 结束 rc=$RC $(date '+%F %T') ==="
  "$PY" "$HERE/summarize_eval8.py" "$OUT" 2>/dev/null || true
done

# ---- 清理本地 stage, 打码日志里的 HF token ----
[ "$EVAL_PATH" != "$MODEL_PATH" ] && rm -rf "$STAGE_ROOT/$RUN_NAME"
sed -i -E 's/hf_[A-Za-z0-9]{30,}/<HF_TOKEN_REDACTED>/g' "$LOG" 2>/dev/null || true
echo "[eval_lmms] ${RUN_NAME} ALL DONE $(date '+%F %T')  -> $OUT"
