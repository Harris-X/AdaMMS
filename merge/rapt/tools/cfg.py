#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 merge/rapt/config.yaml, 供 shell / python 脚本统一取参。

用法:
  python3 merge/rapt/tools/cfg.py --get eval.lmms_eval.min_pixels
  python3 merge/rapt/tools/cfg.py --shell          # 输出 CFG_xxx='yyy', 供 bash eval
  python3 merge/rapt/tools/cfg.py --json           # 输出整份配置
  python3 merge/rapt/tools/cfg.py --sha256         # 输出配置内容哈希

--shell 的变量名规则: 点路径 -> 下划线, 前缀 CFG_。
例如 eval.lmms_eval.min_pixels -> CFG_eval_lmms_eval_min_pixels。
列表以空格连接; null 输出空串; 布尔输出 true/false。
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys

try:
    import yaml
except Exception:  # noqa: BLE001
    sys.exit("[FATAL] 需要 PyYAML: pip install pyyaml")


DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def main():
    path = DEFAULT
    args = sys.argv[1:]
    if "--file" in args:
        i = args.index("--file")
        path = args[i + 1]
        del args[i:i + 2]
    if not os.path.exists(path):
        sys.exit(f"[FATAL] 配置文件不存在: {path}")

    cfg = load(path)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    if args[0] == "--json":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if args[0] == "--sha256":
        print(hashlib.sha256(open(path, "rb").read()).hexdigest())
        return
    if args[0] == "--shell":
        for key, val in flatten(cfg):
            name = "CFG_" + key.replace(".", "_").replace("-", "_")
            print(f"{name}={shlex.quote(val)}")
        return
    if args[0] == "--get":
        key = args[1]
        cur = cfg
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                sys.exit(f"[FATAL] 配置项不存在: {key}")
            cur = cur[part]
        if isinstance(cur, list):
            print(" ".join("" if x is None else str(x) for x in cur))
        elif cur is None:
            print("")
        elif isinstance(cur, bool):
            print("true" if cur else "false")
        else:
            print(cur)
        return

    sys.exit(f"[FATAL] 未知参数: {args[0]}")


if __name__ == "__main__":
    main()
