#!/usr/bin/env bash
# ==============================================================================
# 公共配置引导 (source 使用)
#
# 调用脚本在 source 前定义 HERE (脚本自身目录), 然后:
#     source "$HERE/../tools/cfg_bootstrap.sh" "$@"
#
# 完成后可用:
#   ROOT          项目根
#   PY            配置中 paths.python (或环境变量 PY / python3 回落)
#   CFG_FILE      主配置
#   CFG_SHA       有效配置 (含组覆盖) 的指纹
#   CFG_GROUP_ARG 命令行/environment 指定的模型组 (NAME 或 YAML 路径)
#   GROUP_ARGS    传给 cfg.py 的参数数组 (可为空)
#   CFG_GROUP     组配置中的 active_group (未指定组时为空)
#   GROUP_NAME    CFG_GROUP 或 CFG_GROUP_ARG 的合并结果, 供输出目录使用
#
# 超参规则: config.yaml 与 configs/groups/<group>.yaml 合并后才是有效配置。
# 环境变量仍可在脚本内单次覆盖, 且会写入产物 manifest。
# ==============================================================================
RAPT_TOOLS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${RAPT_TOOLS_DIR}/../../.." &>/dev/null && pwd)
CFG_FILE=${RAPT_CONFIG:-"${ROOT}/merge/rapt/config.yaml"}
CFG_PY="${RAPT_TOOLS_DIR}/cfg.py"

_pre_group_arg() {
  local g="" args=("$@") i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
      --group)   g="${args[i + 1]:-}" ;;
      --group=*) g="${args[i]#--group=}" ;;
    esac
  done
  printf '%s' "$g"
}

# 优先级: RAPT_GROUP > 命令行 --group
CFG_GROUP_ARG=${RAPT_GROUP:-}
if [ -z "$CFG_GROUP_ARG" ]; then
  CFG_GROUP_ARG=$(_pre_group_arg "$@")
fi
GROUP_ARGS=()
if [ -n "$CFG_GROUP_ARG" ]; then
  GROUP_ARGS=(--group "$CFG_GROUP_ARG")
fi

BOOT_PY=${PY:-python3}
PY=${PY:-$("$BOOT_PY" "$CFG_PY" --get paths.python --file "$CFG_FILE" 2>/dev/null || echo python3)}
[ -x "$PY" ] || PY=python3

CFG_SHELL=$("$PY" "$CFG_PY" --shell --file "$CFG_FILE" "${GROUP_ARGS[@]}") || {
  echo "[ERROR] 读取配置失败: $CFG_FILE ${CFG_GROUP_ARG:+ (group=$CFG_GROUP_ARG)}" >&2
  exit 1
}
eval "$CFG_SHELL"
CFG_SHA=$("$PY" "$CFG_PY" --sha256 --file "$CFG_FILE" "${GROUP_ARGS[@]}")
GROUP_NAME=${CFG_GROUP:-$CFG_GROUP_ARG}
