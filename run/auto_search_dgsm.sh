#!/bin/bash
set -euo pipefail

# 自动化 DGSM-TEFM 超参数搜索与评测
# - 运行 Stage-1/2/3（可复用已有产物）
# - 对关键超参做网格搜索
# - 调用 eval_single_model.sh 评测并汇总

# 1) 路径配置（可通过环境变量覆盖）
PRE=${PRE:-"/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-7B-Instruct"}
MODEL_A=${MODEL_A:-"/root/autodl-tmp/AdaMMS/downloaded_models/Qwen2-VL-7B-Instruct"}
MODEL_B=${MODEL_B:-"/root/autodl-tmp/AdaMMS/downloaded_models/llava-onevision-qwen2-7b-si-hf"}
WORK=${WORK:-"/root/autodl-tmp/AdaMMS/work/dgsm"}
OUT_ROOT=${OUT_ROOT:-"/root/autodl-tmp/AdaMMS/merged_models_stage3"}
EVAL_BASE=${EVAL_BASE:-"/root/autodl-tmp/AdaMMS/eval_runs"}

mkdir -p "$WORK"

# 2) SVD rank 与 Stage-2/3 超参网格
RANKS=${RANKS:-"96 128"}
DIST_MODES=${DIST_MODES:-"us usn"}
USE_POT=${USE_POT:-"on"}             # auto|on|off
POT_METHODS=${POT_METHODS:-"gw entropic"}
POT_EPS_LIST=${POT_EPS_LIST:-"0.3 0.5"}
ITERS_LIST=${ITERS_LIST:-"20 30"}
SINK_REG_LIST=${SINK_REG_LIST:-"0.03 0.05"}
RANK_THRESHOLD=${RANK_THRESHOLD:-}     # e.g. 0.95 to auto-select r by energy
STAGE2_CPU=${STAGE2_CPU:-1}            # 1=force Stage-2 on CPU to avoid GPU OOM

# 动态映射参数
DYN_STEPS_LIST=${DYN_STEPS_LIST:-"0 8"}
DYN_LR_LIST=${DYN_LR_LIST:-"0.01"}
DYN_REG_LIST=${DYN_REG_LIST:-"1e-3"}
DYN_LOSS_LIST=${DYN_LOSS_LIST:-"hybrid"}
DYN_MIX_ALPHA_LIST=${DYN_MIX_ALPHA_LIST:-"0.2"}

# Stage-3 融合参数
GAMMAS=${GAMMAS:-"3.5 4.0 5.0"}
COST_SCALES=${COST_SCALES:-"0.8 1.0 1.2"}
ORTHO_SCALES=${ORTHO_SCALES:-"0.3 0.5 0.7"}
FALLBACK_ALPHAS=${FALLBACK_ALPHAS:-"0.4 0.5 0.6"}
BIAS_ALPHAS=${BIAS_ALPHAS:-"0.4 0.5 0.6"}
USE_DYNAMIC_M=${USE_DYNAMIC_M:-"1"}  # 1 启用 / 0 关闭

# 评测配置（可复用 eval_single_model.sh）
GPU=${GPU:-0}
PORT_BASE=${PORT_BASE:-29600}
TASK_LIST=${TASK_LIST:-"mme ocrbench gqa"}

echo "[AutoSearch] A=$MODEL_A B=$MODEL_B Pre=$PRE"

run_stage1() {
	local model_dir=$1
	local save_path=$2
	local rank=$3
	if [ -f "$save_path" ]; then
		echo "[Stage-1] Skip existing: $save_path"
	else
		echo "[Stage-1] Extract $model_dir -> $save_path (rank=$rank)"
			python -m merge.dgsm.dgsm_stage1_subspace --model-dir "$model_dir" --save "$save_path" --rank "$rank" --cuda
	fi
}

run_stage2() {
	local subs_a=$1
	local subs_b=$2
	local save_path=$3
	local dist_mode=$4
	local pot=$5
	local pot_method=$6
	local pot_eps=$7
	local iters=$8
	local sink_reg=$9
	local dyn_steps=${10}
	local dyn_lr=${11}
	local dyn_reg=${12}
	local dyn_loss=${13}
	local dyn_mix_alpha=${14}
	if [ -f "$save_path" ]; then
		echo "[Stage-2] Skip existing: $save_path"
	else
		echo "[Stage-2] $save_path (dist=$dist_mode pot=$pot/$pot_method eps=$pot_eps it=$iters reg=$sink_reg dyn_steps=$dyn_steps)"
		python -m merge.dgsm.dgsm_stage2_dynamic_gwd \
			--subs-a "$subs_a" \
			--subs-b "$subs_b" \
			--save "$save_path" \
			--dist-mode "$dist_mode" \
			--pot "$pot" --pot-method "$pot_method" --pot-eps "$pot_eps" --pot-max-iter 50 \
			--iters "$iters" --sink-reg "$sink_reg" --tol 5e-4 --patience 3 \
			--dynamic-steps "$dyn_steps" --dynamic-lr "$dyn_lr" --dynamic-reg "$dyn_reg" \
				--dyn-loss "$dyn_loss" --dyn-mix-alpha "$dyn_mix_alpha" \
				$( [ -n "$RANK_THRESHOLD" ] && echo --rank-threshold "$RANK_THRESHOLD" ) \
				$( [ "$STAGE2_CPU" = "1" ] && echo --cpu )
	fi
}

run_stage3_and_eval() {
	local stage2=$1
	local base_subs=$2
	local gamma=$3
	local cost_scale=$4
	local ortho_scale=$5
	local fallback_alpha=$6
	local bias_alpha=$7
	local use_dyn_m=$8
	local tag=$9

	local out_dir="$OUT_ROOT"
	python -m merge.dgsm.dgsm_stage3_merge \
		--base-model "$MODEL_A" --donor-model "$MODEL_B" \
		--stage2 "$stage2" --output-dir "$out_dir" \
		--gamma "$gamma" --cost-scale "$cost_scale" \
		--ortho-scale "$ortho_scale" --fallback-alpha "$fallback_alpha" --bias-alpha "$bias_alpha" \
			$( [ "$use_dyn_m" = "1" ] && echo "--use-dynamic-m" ) \
			$( [ "${USE_LAMBDA_EST:-1}" = "1" ] && echo "--use-lambda-est" ) \
			$( [ -n "${LAM_SCALE:-}" ] && echo --lam-scale "$LAM_SCALE" ) \
			$( [ "${ORTHO_ADAPT:-1}" = "1" ] && echo "--ortho-adapt" ) \
		--base-subs "$base_subs"

	local merged_dir="$OUT_ROOT/$(basename "$MODEL_A")/dgsm_merged"
	local run_tag="dgsm_${tag}_g${gamma}_cs${cost_scale}_os${ortho_scale}_fa${fallback_alpha}_ba${bias_alpha}_dm${use_dyn_m}"
	local eval_dir="$EVAL_BASE/$run_tag"
	echo "[Eval] $merged_dir -> $eval_dir"
	MODEL_PATH="$merged_dir" EVAL_BASE="$eval_dir" GPU="$GPU" PORT=$((PORT_BASE + RANDOM % 50)) TASK_LIST="$TASK_LIST" bash /root/autodl-tmp/AdaMMS/run/eval_single_model.sh || true
}

# 主循环
for rank in $RANKS; do
	SUBS_A="$WORK/stage1_A_r${rank}.pt"
	SUBS_B="$WORK/stage1_B_r${rank}.pt"
	run_stage1 "$MODEL_A" "$SUBS_A" "$rank"
	run_stage1 "$MODEL_B" "$SUBS_B" "$rank"

	for dist in $DIST_MODES; do
		for potm in $POT_METHODS; do
			for peps in $POT_EPS_LIST; do
				for iters in $ITERS_LIST; do
					for reg in $SINK_REG_LIST; do
						for dstep in $DYN_STEPS_LIST; do
							for dlr in $DYN_LR_LIST; do
								for dreg in $DYN_REG_LIST; do
									for dloss in $DYN_LOSS_LIST; do
										for dalpha in $DYN_MIX_ALPHA_LIST; do
											STAGE2="$WORK/stage2_r${rank}_${dist}_${USE_POT}_${potm}_e${peps}_it${iters}_reg${reg}_ds${dstep}.pt"
											run_stage2 "$SUBS_A" "$SUBS_B" "$STAGE2" "$dist" "$USE_POT" "$potm" "$peps" "$iters" "$reg" "$dstep" "$dlr" "$dreg" "$dloss" "$dalpha"
											for g in $GAMMAS; do
												for cs in $COST_SCALES; do
													for os in $ORTHO_SCALES; do
														for fa in $FALLBACK_ALPHAS; do
															for ba in $BIAS_ALPHAS; do
																TAG="r${rank}_${dist}_${USE_POT}_${potm}_e${peps}_it${iters}_reg${reg}_ds${dstep}"
																run_stage3_and_eval "$STAGE2" "$SUBS_A" "$g" "$cs" "$os" "$fa" "$ba" "$USE_DYNAMIC_M" "$TAG"
															done
														done
													done
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

echo "[AutoSearch] All jobs launched. Results in $EVAL_BASE"
