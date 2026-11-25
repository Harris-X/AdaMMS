#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 merge/rapt/config.yaml (+ 可选模型组配置), 供 shell / python 脚本统一取参。

配置分层:
  1) config.yaml                 全局默认 (paths / merge / eval)
  2) configs/groups/<group>.yaml 单个模型组 (anchor/host/donor/family + 可选覆盖)

用法:
  cfg.py --get eval.lmms_eval.min_pixels
  cfg.py --shell                          # 输出 CFG_xxx='yyy', 供 bash eval
  cfg.py --json                           # 输出整份(合并后)配置
  cfg.py --sha256                         # 有效配置的指纹
  cfg.py --list-groups                    # 列出可用模型组
  cfg.py --resolve-group g1               # 输出组配置文件绝对路径

  --group NAME|PATH   加载模型组:
      NAME  -> configs/groups/NAME.yaml
      PATH  -> 直接读该 YAML (可放任意实验配置, 不必改主 config)
  可作用于所有子命令。

--shell 变量名规则: 点路径 -> 下划线, 前缀 CFG_。
例如 eval.lmms_eval.min_pixels -> CFG_eval_lmms_eval_min_pixels。
列表以空格连接; null 输出空串; 布尔输出 true/false。
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from copy import deepcopy

try:
    import yaml
except Exception:  # noqa: BLE001
    sys.exit("[FATAL] 需要 PyYAML: pip install pyyaml")


DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
GROUP_DIR_NAME = os.path.join("configs", "groups")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def group_dir(cfg_path):
    return os.path.join(os.path.dirname(os.path.abspath(cfg_path)), GROUP_DIR_NAME)


def resolve_group(name, cfg_path):
    """NAME 或 PATH -> 组配置绝对路径; 找不到返回 None (回退主配置里的 groups 段)。"""
    if not name:
        return None
    if os.sep in name or name.endswith((".yaml", ".yml")):
        p = os.path.abspath(os.path.expanduser(name))
        return p if os.path.isfile(p) else None
    p = os.path.join(group_dir(cfg_path), name + ".yaml")
    return p if os.path.isfile(p) else None


def apply_group(cfg, group_path):
    group = load(group_path)
    name = str(group.get("name") or os.path.splitext(os.path.basename(group_path))[0])
    overlay = {k: group[k] for k in ("paths", "merge", "eval") if isinstance(group.get(k), dict)}
    eff = deep_merge(cfg, overlay)
    meta = {
        k: v for k, v in group.items()
        if k not in ("paths", "merge", "eval")
    }
    meta.setdefault("description", "")
    eff.setdefault("groups", {})
    eff["groups"][name] = meta
    eff["active_group"] = name
    eff["group_file"] = os.path.abspath(group_path)
    return eff, name


def flatten(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten(v, key)
    elif isinstance(node, list):
        yield prefix, " ".join("" if x is None else str(x) for x in node)
    else:
        if node is None:
            yield prefix, ""
        elif isinstance(node, bool):
            yield prefix, "true" if node else "false"
        else:
            yield prefix, str(node)


def list_groups(cfg_path):
    d = group_dir(cfg_path)
    if not os.path.isdir(d):
        return []
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith((".yaml", ".yml")):
            continue
        p = os.path.join(d, fn)
        try:
            g = load(p)
        except Exception:  # noqa: BLE001
            rows.append((os.path.splitext(fn)[0], "<parse error>", "", p))
            continue
        rows.append((
            str(g.get("name") or os.path.splitext(fn)[0]),
            str(g.get("description", "")),
            str(g.get("status", "ready")),
            p,
        ))
    return rows


def take_opts(args):
    """抽取 --file / --group, 返回 (剩余参数, cfg_path, group_name)。"""
    cfg_path = DEFAULT
    group = ""
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--file":
            cfg_path = args[i + 1]; i += 2
        elif a.startswith("--file="):
            cfg_path = a.split("=", 1)[1]; i += 1
        elif a == "--group":
            group = args[i + 1]; i += 2
        elif a.startswith("--group="):
            group = a.split("=", 1)[1]; i += 1
        else:
            rest.append(a); i += 1
    return rest, cfg_path, group


def main():
    args = sys.argv[1:]
    rest, cfg_path, group = take_opts(args)
    if not rest or rest[0] in ("-h", "--help"):
        print(__doc__)
        return
    if not os.path.exists(cfg_path):
        sys.exit(f"[FATAL] 配置文件不存在: {cfg_path}")

    if rest[0] == "--list-groups":
        print(f"{'group':28s} {'status':18s} description")
        for name, desc, status, path in list_groups(cfg_path):
            print(f"{name:28s} {status:18s} {desc}    ({path})")
        return
    if rest[0] == "--resolve-group":
        p = resolve_group(rest[1] if len(rest) > 1 else group, cfg_path)
        if not p:
            sys.exit(f"[FATAL] 未找到模型组配置: {rest[1] if len(rest) > 1 else group}")
        print(p)
        return

    cfg = load(cfg_path)
    active = ""
    if group:
        gp = resolve_group(group, cfg_path)
        if gp is None:
            # 回退: 主配置 groups 段里已有该组 (旧格式兼容)
            if isinstance(cfg.get("groups"), dict) and group in cfg["groups"]:
                active = group
            elif os.sep in group or group.endswith((".yaml", ".yml")):
                sys.exit(f"[FATAL] 未找到模型组配置文件: {group}")
            else:
                sys.exit(f"[FATAL] 未找到模型组 '{group}': 期望 {os.path.join(group_dir(cfg_path), group + '.yaml')}")
        else:
            cfg, active = apply_group(cfg, gp)

    if rest[0] == "--json":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if rest[0] == "--sha256":
        payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str).encode()
        print(hashlib.sha256(payload).hexdigest())
        return
    if rest[0] == "--shell":
        for key, val in flatten(cfg):
            name = "CFG_" + key.replace(".", "_").replace("-", "_")
            print(f"{name}={shlex.quote(val)}")
        print(f"CFG_GROUP={shlex.quote(active)}")
        return
    if rest[0] == "--get":
        key = rest[1]
        cur = cfg
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                sys.exit(f"[FATAL] 配置项不存在: {key}")
            cur = cur[part]
        if isinstance(cur, list):
            print(" ".join("" if x is None else str(x) for x in cur))
        elif isinstance(cur, dict):
            print(json.dumps(cur, ensure_ascii=False))
        elif cur is None:
            print("")
        elif isinstance(cur, bool):
            print("true" if cur else "false")
        else:
            print(cur)
        return

    sys.exit(f"[FATAL] 未知参数: {rest[0]}")


if __name__ == "__main__":
    main()
