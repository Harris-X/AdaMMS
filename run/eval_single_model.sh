#!/bin/bash

# ==============================================================================
#                  快捷评测脚本: 评测指定的融合模型
# ==============================================================================

# --- 1. 基本设置 (可通过环境变量覆盖) ---
GPU=${GPU:-0}                                 # 可设置希望使用的 GPU ID 
PORT=${PORT:-29517}                           # 建议为每个独立脚本使用不同端口，避免冲突
MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/AdaMMS/merged_models_stage3/Qwen2-VL-7B-Instruct/dgsm_merged"}
EVAL_BASE=${EVAL_BASE:-"./eval_results_single_run"}  # 评测结果根目录 
HF_TOKEN=${HF_TOKEN:-}                        # 可 export HF_TOKEN=xxxx 自动登录 Hugging Face
# 使用镜像站（如国内环境）：可通过环境变量覆盖默认镜像地址
HF_MIRROR_URL=${HF_MIRROR_URL:-https://hf-mirror.com}

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径 '$MODEL_PATH' 不存在. 请检查路径是否正确."
    exit 1
fi

# --- 2. 评测任务列表 ---
# 可通过环境变量 TASK_LIST 覆盖；默认一组代表性任务（可较快跑完）
# 完整任务列表参考: ok_vqa textvqa_val vizwiz_vqa_val gqa mme seedbench ocrbench mmmu_val
TASK_LIST=${TASK_LIST:-"mmmu_val mme ocrbench gqa"}

# --- 3. 环境准备 ---
echo "--- Preparing environment and directories ---"
# 从模型路径中提取模型名称用于输出目录
MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${EVAL_BASE}/${MODEL_NAME}
mkdir -p $OUTPUT_DIR
# conda activate lmms-cogvlm # 如果需要，请取消注释以激活你的 conda 环境

echo "===================================================="
echo "  评测模型: $MODEL_NAME"
echo "  模型路径: $MODEL_PATH"
echo "  评测任务: $TASK_LIST"
echo "  结果将保存到: $OUTPUT_DIR"
echo "===================================================="

date +"%Y-%m-%d %H:%M:%S"
SECONDS=0

# 设置数值表达式线程，避免 numexpr 报错
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-64}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-32}

# 启用更快的传输（可选）
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# 配置 Hugging Face 镜像端点（若需要）
echo "--- 使用 Hugging Face 镜像: $HF_MIRROR_URL ---"
export HF_ENDPOINT="$HF_MIRROR_URL"
# 建议设置缓存路径（可选）
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}

# 检查 transformers 是否支持 Qwen2-VL，若不支持则升级
echo "--- 检查 transformers 是否支持 Qwen2-VL ---"
TF_CHECK=$(python - << 'PY'
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor  # noqa: F401
    print('OK')
except Exception:
    print('MISS')
PY
)
if [ "$TF_CHECK" != "OK" ]; then
    echo "未检测到 Qwen2-VL 类，正在安装合适版本的 transformers/tokenizers..."
    pip install "transformers>=4.45,<5" "tokenizers>=0.15" --upgrade || {
        echo "错误: 升级 transformers 失败。"; exit 1; }
else
    echo "transformers 已支持 Qwen2-VL (跳过安装)"
fi

# Qwen-VL 评测依赖（qwen-vl-utils）
echo "--- 检查 qwen-vl-utils ---"
QWEN_UTILS=$(python - << 'PY'
try:
    import qwen_vl_utils  # noqa: F401
    print('OK')
except Exception:
    print('MISS')
PY
)
if [ "$QWEN_UTILS" != "OK" ]; then
    pip install qwen-vl-utils || true
fi

# 检查 lmms_eval 是否可用，否则本地可编辑安装
echo "--- 检查 lmms_eval 包是否可用 ---"
LMMS_CHECK=$(python - << 'PY'
try:
    import lmms_eval  # noqa: F401
    print('OK')
except Exception:
    print('MISS')
PY
)
if [ "$LMMS_CHECK" != "OK" ]; then
    echo "未检测到 lmms_eval，正在本地安装 (editable)..."
    pip install -e ./lmms-eval || { echo "错误: 安装 lmms-eval 失败"; exit 1; }
else
    echo "lmms_eval 已安装 (跳过)"
fi

# Hugging Face 登录：优先使用已登录状态；若未登录且提供了 HF_TOKEN，则自动登录；否则提示并退出
echo "--- 检查 Hugging Face 登录状态 ---"
HF_LOGIN=$(python - << 'PY'
from huggingface_hub.utils import get_token
print('LOGGED_IN' if get_token() else 'NO_TOKEN')
PY
)
if [ "$HF_LOGIN" = "NO_TOKEN" ]; then
    if [ -n "$HF_TOKEN" ]; then
        echo "未登录，正在用环境变量 HF_TOKEN 进行登录..."
        python - << PY
from huggingface_hub import login
import os
token=os.environ.get('HF_TOKEN')
assert token, 'HF_TOKEN is empty'
login(token=token, add_to_git_credential=True)
print('Hugging Face 登录成功')
PY
        export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    else
        echo "错误: 需要 Hugging Face Token 才能下载某些评测数据。请先执行:"
        echo "  export HF_TOKEN=\"<你的token>\""
        echo "或提前执行 'huggingface-cli login' 完成登录后再运行本脚本。"
        exit 1
    fi
else
    echo "已检测到 Hugging Face 登录状态。"
fi

# --- 4. 运行评测 ---
for task in $TASK_LIST; do
    echo "--- 正在运行评测任务: $task ---"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --gpu_ids $GPU --main_process_port $PORT -m lmms_eval \
        --model qwen2_vl \
        --model_args pretrained=$MODEL_PATH \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity INFO \
        --output_path $OUTPUT_DIR
    
    # 检查上一个命令是否成功
    if [ $? -ne 0 ]; then
        echo "错误: 任务 '$task' 评测失败."
        # 如果希望在失败时停止脚本，可以取消此行注释: exit 1
    fi
done

# --- 5. 结束 ---
minute=$((SECONDS / 60))
second=$((SECONDS % 60))
echo "===================================================="
echo "      模型 '$MODEL_NAME' 评测完成."
echo "      总耗时: $minute 分 $second 秒"
echo "      所有结果已保存到: $OUTPUT_DIR"
echo "===================================================="