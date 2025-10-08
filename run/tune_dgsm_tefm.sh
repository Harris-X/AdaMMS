#!/bin/bash
set -euo pipefail

# 快速调参与评测入口（可根据需要扩展网格）

# 默认较小网格以便快速启动；如需扩大搜索范围，可在环境变量中覆盖
export RANKS="128"
export DIST_MODES="us"
export USE_POT="on"
export POT_METHODS="gw"
export POT_EPS_LIST="0.5"
export ITERS_LIST="30"
export SINK_REG_LIST="0.05"

export DYN_STEPS_LIST="8 0"
export DYN_LR_LIST="0.01"
export DYN_REG_LIST="1e-3"
export DYN_LOSS_LIST="hybrid"
export DYN_MIX_ALPHA_LIST="0.2"

export GAMMAS="3.5 4.0 5.0"
export COST_SCALES="0.8 1.0 1.2"
export ORTHO_SCALES="0.3 0.5 0.7"
export FALLBACK_ALPHAS="0.4 0.5 0.6"
export BIAS_ALPHAS="0.4 0.5 0.6"
export USE_DYNAMIC_M="1"
export STAGE2_CPU="1"            # 避免 Stage-2 在 GPU 上 OOM
export RANK_THRESHOLD=""         # 可设置 0.95 进行能量阈值自适应 r
export USE_LAMBDA_EST="1"        # 使用 per-layer lambda
export ORTHO_ADAPT="1"           # 启用基于熵的正交项自适应

# 评测任务快速集
export TASK_LIST=${TASK_LIST:-"mmmu_val mme ocrbench gqa"}

bash /root/autodl-tmp/AdaMMS/run/auto_search_dgsm.sh

echo "[Tune] 完成初步搜索。汇总：python run/summarize_eval.py <eval_base>"

