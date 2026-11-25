#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对 bootstrap: 在相同样本上比较我们的候选与 AdaMMS alpha=0.1。

点估计的 SUM3 标准误约 2.2 分, 无法区分 214.5 与 214.9;
配对比较对同一样本重采样, 消除共享方差, 区间显著更窄。

用法:
  python merge/rapt/eval/paired_bootstrap.py <我们的目录> [<我们的目录> ...]
其中 <我们的目录> 形如 reproduce_table1/eval_screen3/L0_interp_idx
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = "/root/autodl-tmp/AdaMMS"
# 参照目录: 默认 AdaMMS α=0.1; 可用 RAPT_REF_DIR 换成其它候选 (如 eval_v3 下的 full8 目录),
# 用于「同 harness、同样本」的候选间配对对照。
ADA_DIR = os.environ.get("RAPT_REF_DIR",
                         os.path.join(ROOT, "eval_results_single_run/adamms-0.1-qwenvl"))
TASKS5 = ["mmmu_val", "mme", "ocrbench", "gqa", "vizwiz_vqa_val"]
TASKS3 = ["mmmu_val", "mme", "ocrbench"]
# 8 任务口径与 compile_table.py 一致: seedbench 用 seed_image, okvqa 用 ok_vqa_val2014
TASKS8 = ["mmmu_val", "ocrbench", "mme", "vizwiz_vqa_val",
          "textvqa_val", "ok_vqa_val2014", "gqa", "seedbench"]
_which = os.environ.get("RAPT_TASKS", "5")
TASKS = {"8": TASKS8, "5": TASKS5, "3": TASKS3}[_which]
N_BOOT = 10000
SEED = 12345


def _latest_samples(dirpath: str, task: str) -> str | None:
    files = glob.glob(os.path.join(dirpath, "**", f"*samples_{task}.jsonl"), recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def _correct(j: dict, task: str) -> float:
    if task == "mmmu_val":
        m = j.get("mmmu_acc") or {}
        ans, pred = m.get("answer"), (m.get("parsed_pred") or [None])[0]
        return 1.0 if ans is not None and ans == pred else 0.0
    if task == "ocrbench":
        m = j.get("ocrbench_accuracy") or {}
        return float(m.get("score", 0) or 0)
    if task == "mme":
        m = j.get("mme_perception_score") or j.get("mme_cognition_score") or {}
        return float(m.get("score", 0) or 0)
    if task in ("gqa", "vizwiz_vqa_val", "textvqa_val", "ok_vqa_val2014"):
        return float(j.get("exact_match", 0) or 0)
    if task == "seedbench":
        # 与 compile_table.py 同口径: 用 seed_image (不是 seed_all); 字段是 {pred, answer}
        m = j.get("seed_image") or {}
        if isinstance(m, dict):
            return 1.0 if m.get("pred") == m.get("answer") else 0.0
        return float(m or 0)
    raise ValueError(task)


def load(path: str, task: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            key = j.get("doc_id")
            out[key] = _correct(j, task)
    return out


def task_score(vals: dict, task: str) -> float:
    """逐样本准确率 (0-100)。注意 MME 官方聚合按类别加权, 与逐样本均值尺度不同;
    这里统一用逐样本准确率, 使配对差值可比。"""
    n = len(vals)
    if n == 0:
        return float("nan")
    return sum(vals.values()) / n * 100.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    ours = sys.argv[1:]

    ada = {}
    for t in TASKS:
        p = _latest_samples(ADA_DIR, t)
        ada[t] = load(p, t) if p else {}
    ada_sum = sum(task_score(ada[t], t) for t in TASKS)
    print(f"AdaMMS a=0.1 参照: " + " ".join(f"{t}={task_score(ada[t], t):.4f}" for t in TASKS)
          + f"  SUM3={ada_sum:.4f}  (样本数 " + ",".join(str(len(ada[t])) for t in TASKS) + ")")
    print()

    rng = np.random.default_rng(SEED)
    for d in ours:
        cur = {}
        for t in TASKS:
            p = _latest_samples(d, t)
            cur[t] = load(p, t) if p else {}
        cur_sum = sum(task_score(cur[t], t) for t in TASKS)

        # 配对: 取交集样本, 构造逐样本贡献向量
        diffs, weights = [], []
        detail = []
        for t in TASKS:
            keys = [k for k in cur[t] if k in ada[t]]
            if not keys:
                continue
            o = np.array([cur[t][k] for k in keys], dtype=float)
            a = np.array([ada[t][k] for k in keys], dtype=float)
            w = 100.0 / len(keys)
            diffs.append((o - a) * w)
            weights.append(len(keys))
            detail.append(f"{t}:n={len(keys)},Δ={float(((o-a)*w).sum()):+.3f}")
        d_all = np.concatenate(diffs)
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, d_all.size, d_all.size)
            boots[i] = d_all[idx].sum()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "显著" if lo > 0 or hi < 0 else "不显著"
        print(f"{os.path.basename(d):26s} SUM3={cur_sum:.4f} Δ={cur_sum-ada_sum:+.4f} "
              f"[95% CI {lo:+.3f}, {hi:+.3f}] {sig}")
        print(f"{'':26s} 逐任务: " + " | ".join(detail))


if __name__ == "__main__":
    main()
