#!/usr/bin/env bash
# ==============================================================================
# RAPT 多模型组一键合并 (每组一个配置文件)
#
# 组定义: merge/rapt/configs/groups/<name>.yaml
#   每个组文件自带 anchor / host / donor / family / interp_alpha /
#   mrc_interp_alpha / backend, 并可覆盖主 config 的 merge 段。
#
# 用法:
#   bash merge/rapt/merge/run_merge_all.sh                          # 默认组列表
#   bash merge/rapt/merge/run_merge_all.sh --group g1               # 单组
#   bash merge/rapt/merge/run_merge_all.sh --groups "g1 g3_rev"     # 多组
#   bash merge/rapt/merge/run_merge_all.sh --group ./my_group.yaml  # 任意实验配置
#   bash merge/rapt/merge/run_merge_all.sh --group cogvlm_llava_fwd --allow-unready
#
# 后端: 每组配置的 backend 优先 (ta_hyb | pipeline); 未写则用主 config merge.backend。
#   可用 --backend 全局强制 (一般不推荐, 会覆盖组自身选择)。
#
# 超参: 不在本脚本传; 各组从自己的 YAML 读取。临时覆盖用环境变量, 例如
#   RANK=256 ALPHA=0.4 bash merge/rapt/merge/run_merge_all.sh --group g1
# ==============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${HERE}/../tools/cfg_bootstrap.sh" "$@"

# 注意: bash 有内置只读数组 GROUPS (用户组 ID), 不能用 GROUPS 作环境变量
GROUPS_LIST=${RAPT_GROUPS:-$CFG_merge_groups}
BACKEND_OVERRIDE=""
ALLOW_UNREADY=${ALLOW_UNREADY:-0}

while [ $# -gt 0 ]; do
  case "$1" in
    --group) GROUPS_LIST=$2; shift 2 ;;
    --groups) GROUPS_LIST=$2; shift 2 ;;
    --group=*) GROUPS_LIST=${1#--group=}; shift ;;
    --groups=*) GROUPS_LIST=${1#--groups=}; shift ;;
    --backend) BACKEND_OVERRIDE=$2; shift 2 ;;
    --allow-unready) ALLOW_UNREADY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; exit 1 ;;
  esac
done

CFG_GROUP_FIELD() {  # <group> <field>
  "$PY" "$CFG_PY" --file "$CFG_FILE" --group "$1" --get "groups.$("$PY" "$CFG_PY" --file "$CFG_FILE" --group "$1" --get active_group).$2" 2>/dev/null || true
}

echo "================================================================"
echo "  RAPT 一键合并  groups='$GROUPS_LIST'"
echo "  config=$CFG_FILE  sha=${CFG_SHA:0:12}"
echo "  override backend=${BACKEND_OVERRIDE:-<per-group>}  allow_unready=$ALLOW_UNREADY"
echo "================================================================"

RUN_ONE() {
  local g=$1 active status backend
  active=$("$PY" "$CFG_PY" --file "$CFG_FILE" --group "$g" --get active_group 2>/dev/null || true)
  if [ -z "$active" ]; then
    echo "[ERROR] 组 '$g' 无法解析 (缺 configs/groups/<name>.yaml 或文件格式错误)" >&2
    return 1
  fi
  status=$("$PY" "$CFG_PY" --file "$CFG_FILE" --group "$g" --get "groups.${active}.status" 2>/dev/null || true)
  backend=$("$PY" "$CFG_PY" --file "$CFG_FILE" --group "$g" --get "groups.${active}.backend" 2>/dev/null || true)
  [ -n "$BACKEND_OVERRIDE" ] && backend=$BACKEND_OVERRIDE
  [ -n "$backend" ] || backend=$CFG_merge_backend

  echo ""
  echo "################################################################"
  echo "# group=$active  backend=$backend  status=${status:-unknown}"
  echo "################################################################"

  if [ -n "$status" ] && [ "$status" != "ready" ] && [ "$ALLOW_UNREADY" != "1" ]; then
    echo "[SKIP] 组 '$active' status=$status, 默认不运行。" >&2
    echo "       该组需要额外适配; 强制运行请加 --allow-unready (结果不应作为完整合并且报)。" >&2
    return 2
  fi
  if [ -n "$status" ] && [ "$status" != "ready" ]; then
    echo "[WARN] 组 '$active' status=$status, 已按 ALLOW_UNREADY=1 强制运行" >&2
  fi

  case "$backend" in
    ta_hyb)
      bash "$HERE/run_ta_hyb.sh" --group "$g"
      ;;
    pipeline)
      bash "$HERE/run_stage1.sh" --group "$g" || return 1
      bash "$HERE/run_stage2.sh" --group "$g" || return 1
      bash "$HERE/run_stage3.sh" --group "$g" || return 1
      ;;
    *)
      echo "[ERROR] 组 '$active' 的 backend 非法: $backend (仅支持 ta_hyb|pipeline)" >&2
      return 1
      ;;
  esac
}

RC=0
for g in $GROUPS_LIST; do
  RUN_ONE "$g"
  rc=$?
  [ $rc -eq 0 ] || RC=1
done

echo ""
echo "[DONE] rc=$RC  产物根目录: ${ROOT}/${CFG_paths_merged_models}"
ls -d "${ROOT}/${CFG_paths_merged_models}"/*/* 2>/dev/null | head -20 || true
exit $RC
