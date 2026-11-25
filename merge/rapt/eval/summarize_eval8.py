#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按任务汇总 eval8 的逐任务结果 (支持部分完成), 口径与 collect_results.py 一致。

用法:
  python merge/rapt/eval/summarize_eval8.py <eval8输出目录> [<目录2> ...]

每个任务的结果来自 <目录>/<task>/<model>/<timestamp>_results.json
(即 eval8.sh 的逐任务落盘结构)。任务未完成时显示 "进行中 n/8"。
"""
import glob
import json
import os
import sys

# 按样本数升序 —— 也是 eval8.sh 的执行次序
TASKS = ["mmmu_val", "ocrbench", "mme", "vizwiz_vqa_val",
         "textvqa_val", "ok_vqa_val2014", "gqa", "seedbench"]
HEAD = ["MMMU", "OCR", "MME", "VizWiz", "TextVQA", "OKVQA", "GQA", "Seed"]
N_SAMPLES = {"mmmu_val": 1800, "ocrbench": 2000, "mme": 4748, "vizwiz_vqa_val": 8638,
             "textvqa_val": 10000, "ok_vqa_val2014": 10092, "gqa": 25156,
             "seedbench": 35980}

METRICS = {
    "mmmu_val": lambda v: v["mmmu_acc,none"] * 100,
    "mme": lambda v: (v["mme_perception_score,none"] + v["mme_cognition_score,none"]) / 28.0,
    "seedbench": lambda v: v["seed_image,none"] * 100,
    "ocrbench": lambda v: v["ocrbench_accuracy,none"] * 100,
    "textvqa_val": lambda v: v["exact_match,none"] * 100,
    "ok_vqa_val2014": lambda v: v["exact_match,none"] * 100,
    "gqa": lambda v: v["exact_match,none"] * 100,
    "vizwiz_vqa_val": lambda v: v["exact_match,none"] * 100,
}

REFS = {
    "base":    [50.44, 79.60, 82.15, 68.86, 81.37, 52.07, 62.27, 76.06],   # 552.83
    "adamms":  [50.89, 80.70, 83.28, 68.21, 80.99, 52.89, 62.37, 76.13],   # 555.46
    "metagpt": [50.78, 80.90, 83.25, 68.94, 81.19, 52.29, 62.47, 76.25],   # 556.07
}


def collect(dirpath):
    """收集已完成任务的指标。

    同时兼容两种落盘结构:
      A) 逐任务 (eval8.sh 新版): <dir>/<task>/<model>/<ts>_results.json
      B) 单次调用 (旧版):        <dir>/<model>/<ts>_results.json  内含多个任务
    同一任务取 mtime 最新的文件。
    """
    out = {}
    for task, fn in METRICS.items():
        files = glob.glob(os.path.join(dirpath, task, "**", "*_results.json"),
                          recursive=True)
        if not files:                      # 回退: 旧结构
            files = glob.glob(os.path.join(dirpath, "**", "*_results.json"),
                              recursive=True)
        files.sort(key=os.path.getmtime)
        for f in files:
            try:
                res = json.load(open(f)).get("results", {})
            except Exception:
                continue
            if task in res:
                try:
                    out[task] = fn(res[task])
                except Exception:
                    pass
    return out


def show(name, vals):
    ad = sum(REFS["adamms"])
    cells, tot = [], 0.0
    for i, t in enumerate(TASKS):
        if t in vals:
            cells.append(f"{vals[t]:6.2f}({vals[t]-REFS['adamms'][i]:+.2f})")
            tot += vals[t]
        else:
            cells.append("      --    ")
    n = len(vals)
    done_samples = sum(N_SAMPLES[t] for t in vals)
    total_samples = sum(N_SAMPLES.values())
    if n == len(TASKS):
        mark = " <<< 超过 AdaMMS" if tot > ad else ""
        print(f"{name:26s} " + " ".join(cells)
              + f" | SUM8={tot:7.2f} Δ={tot-ad:+.2f}{mark}")
    else:
        print(f"{name:26s} " + " ".join(cells)
              + f" | 进行中 {n}/8 任务 (样本 {done_samples}/{total_samples}"
              + f" = {100*done_samples/total_samples:.0f}%)  部分和={tot:.2f}")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return
    print(f"{'方法':26s} " + " ".join(f"{h:>12s}" for h in HEAD)
          + " | 括号内为相对 AdaMMS α=0.1 的差")
    print("-" * 140)
    show("[参照] base", dict(zip(TASKS, REFS["base"])))
    show("[参照] adamms a=0.1", dict(zip(TASKS, REFS["adamms"])))
    show("[参照] metagpt", dict(zip(TASKS, REFS["metagpt"])))
    print("-" * 140)
    for d in dirs:
        if not os.path.isdir(d):
            print(f"{d}: 目录不存在")
            continue
        show(os.path.basename(d.rstrip("/")), collect(d))


if __name__ == "__main__":
    main()
