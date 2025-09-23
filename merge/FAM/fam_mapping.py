#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlatAlignMerge (FAM) - Step 2: 确定神经元之间的映射（平坦导向相似性对齐）

本脚本基于已缓存的激活与 FAI 统计（由 merge/FAM/cache_activation_new.py 生成），
对模型 A 与模型 B 的对应模块执行神经元级的一对一匹配。

核心：
  - 使用每个神经元的特征向量 f_i = [A_i, MI_i, FAI_i] 进行余弦相似度度量
  - 加入平坦度惩罚：exp(-|H_A_i - H_B_j| / sigma)
  - FOS(i,j) = cos(f_A_i, f_B_j) * exp(-|H_A_i - H_B_j| / sigma)
  - 通过匈牙利算法（最大权匹配，等价于最小化 -FOS）或贪心回退进行映射

输入：
  - 两个激活缓存文件路径（torch.save 的 .pt），其中每个模块包含：
        - output/input（平均激活）
        - fai（可选）
        - fai_A（可选）
        - fai_MI（可选）
        - fai_H（可选）
        - fai_n_samples（可选）

输出：
  - 一个 .pt 文件，字典形式：
        { module_name: { 'pairs': List[(iA, jB, fos)], 'size_A': H_A, 'size_B': H_B, 'sigma': sigma, 'threshold': thr } }

注意：若部分 FAI 向量缺失，则回退使用 output 的绝对值作为 A，MI=0，H=0，FAI=|A|。
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Dict, List, Tuple

import torch
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _canon_module_name(name: str) -> str:
    k = name.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    return k


def _load_acts(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    d = torch.load(path, map_location="cpu")
    # 规范化 key
    return { _canon_module_name(k): v for k, v in d.items() }


def _select_modules(intersection_keys: List[str], include_regex: str | None, exclude_regex: str | None) -> List[str]:
    import re
    sel = []
    pat_inc = re.compile(include_regex) if include_regex else None
    pat_exc = re.compile(exclude_regex) if exclude_regex else None
    for k in intersection_keys:
        if pat_inc and (not pat_inc.search(k)):
            continue
        if pat_exc and pat_exc.search(k):
            continue
        sel.append(k)
    return sel


def _normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x)
    return x / n


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < eps:
        return 0.0
    return float(np.dot(a, b) / den)


def _build_feat(blk: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (FAI, A, MI, H) 都是 [H] 的 numpy 向量，必要时回退。
    回退策略：
      - 若没有 fai_A/MI/H，则用 |output| 作为 A，MI=0，H=0；fai=|A|
    """
    H = None
    # 优先使用缓存的 FAI 家族
    fai = blk.get("fai")
    A = blk.get("fai_A")
    MI = blk.get("fai_MI")
    Hdiag = blk.get("fai_H")
    if (A is None or MI is None or Hdiag is None):
        # 回退
        out = blk.get("output")
        if out is None:
            raise ValueError("Module block missing both FAI and output vector.")
        out_np = torch.abs(out).float().cpu().numpy()
        H = out_np.shape[0]
        A_np = out_np
        MI_np = np.zeros_like(A_np)
        H_np = np.zeros_like(A_np)
        fai_np = A_np.copy()
        return fai_np, A_np, MI_np, H_np

    # 正常路径
    A_np = A.float().cpu().numpy()
    MI_np = MI.float().cpu().numpy()
    H_np = Hdiag.float().cpu().numpy()
    if fai is None:
        fai_np = (np.abs(A_np) * np.abs(MI_np)) / (np.abs(H_np) + 1e-6)
    else:
        fai_np = fai.float().cpu().numpy()
    return fai_np, A_np, MI_np, H_np


def _preselect_top_idx(score: np.ndarray, top_ratio: float | None, top_k: int | None) -> np.ndarray:
    n = score.shape[0]
    if top_k is not None and top_k > 0:
        k = min(n, top_k)
    elif top_ratio is not None and top_ratio > 0:
        k = max(1, int(round(n * top_ratio)))
    else:
        return np.arange(n)
    idx = np.argpartition(-score, kth=k-1)[:k]
    return np.sort(idx)


def _hungarian_max(sim: np.ndarray) -> List[Tuple[int, int, float]]:
    """最大化 sim；scipy 的 linear_sum_assignment 是最小化，所以取 cost=-sim。
    若无 scipy，回退到贪婪最大匹配。
    """
    m, n = sim.shape
    if _HAVE_SCIPY:
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(int(r), int(c), float(sim[r, c])) for r, c in zip(row_ind, col_ind)]
        return pairs
    # greedy fallback
    used_r = set(); used_c = set(); pairs: List[Tuple[int,int,float]] = []
    # 展平排序
    flat_idx = np.dstack(np.unravel_index(np.argsort(sim, axis=None)[::-1], sim.shape))[0]
    for r, c in flat_idx:
        r = int(r); c = int(c)
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c, float(sim[r, c])))
        used_r.add(r); used_c.add(c)
        if len(used_r) >= m or len(used_c) >= n:
            break
    return pairs


def compute_module_mapping(blockA: Dict[str, torch.Tensor], blockB: Dict[str, torch.Tensor],
                           sigma: float, fos_threshold: float,
                           top_ratio: float | None, top_k: int | None,
                           feature_weights: Tuple[float,float,float] = (1.0,1.0,1.0)) -> Dict[str, torch.Tensor]:
    """对单个模块计算映射，返回：
        {'pairs': LongTensor[K,2], 'scores': FloatTensor[K], 'size_A': H_A, 'size_B': H_B}
    """
    # 取特征
    faiA, AA, MIA, HA = _build_feat(blockA)
    faiB, AB, MIB, HB = _build_feat(blockB)

    H_A = AA.shape[0]; H_B = AB.shape[0]

    # 预选（按 FAI）
    idxA = _preselect_top_idx(faiA, top_ratio, top_k)
    idxB = _preselect_top_idx(faiB, top_ratio, top_k)

    # 构造特征向量 [w0*A, w1*MI, w2*FAI]
    w0, w1, w2 = feature_weights
    FeatA = np.stack([w0*AA[idxA], w1*MIA[idxA], w2*faiA[idxA]], axis=1)
    FeatB = np.stack([w0*AB[idxB], w1*MIB[idxB], w2*faiB[idxB]], axis=1)

    # 归一化后余弦相似
    FeatA = np.apply_along_axis(_normalize_vec, 1, FeatA)
    FeatB = np.apply_along_axis(_normalize_vec, 1, FeatB)

    # 计算相似度矩阵（余弦）
    # sim[i,j] = cos(FeatA[i], FeatB[j])，使用矩阵乘法实现
    sim = FeatA @ FeatB.T  # 形状 [Na, Nb]
    sim = np.clip(sim, -1.0, 1.0)

    # 平坦度惩罚项
    HA_sel = HA[idxA][:, None]
    HB_sel = HB[idxB][None, :]
    flat_penalty = np.exp(-np.abs(HA_sel - HB_sel) / max(1e-8, float(sigma)))

    fos = sim * flat_penalty  # FOS 矩阵

    # 匹配
    pairs = _hungarian_max(fos)
    # 过滤阈值
    pairs = [(idxA[i], idxB[j], s) for i, j, s in pairs if s >= fos_threshold]

    if not pairs:
        return {
            'pairs': torch.empty(0, 2, dtype=torch.long),
            'scores': torch.empty(0, dtype=torch.float32),
            'size_A': torch.tensor(H_A, dtype=torch.long),
            'size_B': torch.tensor(H_B, dtype=torch.long),
        }

    pairs_arr = np.array([[pa[0], pa[1]] for pa in pairs], dtype=np.int64)
    scores_arr = np.array([pa[2] for pa in pairs], dtype=np.float32)
    return {
        'pairs': torch.from_numpy(pairs_arr),
        'scores': torch.from_numpy(scores_arr),
        'size_A': torch.tensor(H_A, dtype=torch.long),
        'size_B': torch.tensor(H_B, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description="FAM Step2: 平坦导向相似性对齐（神经元映射）")
    parser.add_argument('--acts_a', type=str, required=True, help='模型A的激活/FAI缓存文件（.pt）')
    parser.add_argument('--acts_b', type=str, required=True, help='模型B的激活/FAI缓存文件（.pt）')
    parser.add_argument('--save', type=str, default=None, help='输出映射文件路径（.pt）；默认保存在 activations/ 下')

    parser.add_argument('--sigma', type=float, default=0.1, help='平坦度差异惩罚的温度项 sigma')
    parser.add_argument('--fos-threshold', type=float, default=0.3, help='FOS 阈值，低于该值的配对将被丢弃')
    parser.add_argument('--topk', type=int, default=None, help='每个模块按 FAI 选取的 Top-K 神经元（A/B 各自）')
    parser.add_argument('--top-ratio', type=float, default=0.2, help='或按比例选择 Top-ratio（0~1）')

    parser.add_argument('--module-regex', type=str, default=None, help='仅匹配模块名满足该正则者')
    parser.add_argument('--exclude-regex', type=str, default=r"lm_head|embed|embedding", help='排除的模块名正则')
    parser.add_argument('--feature-weights', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        help='特征权重 (w_A, w_MI, w_FAI)，用于构造特征向量')

    args = parser.parse_args()

    actsA = _load_acts(args.acts_a)
    actsB = _load_acts(args.acts_b)

    inter_keys = sorted(set(actsA.keys()) & set(actsB.keys()))
    sel_modules = _select_modules(inter_keys, args.module_regex, args.exclude_regex)

    mapping: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in sel_modules:
        try:
            res = compute_module_mapping(
                actsA[name], actsB[name],
                sigma=float(args.sigma),
                fos_threshold=float(args.fos_threshold),
                top_ratio=float(args.top_ratio) if args.top_ratio is not None else None,
                top_k=int(args.topk) if args.topk is not None else None,
                feature_weights=tuple(args.feature_weights),
            )
            mapping[name] = res
        except Exception as e:
            print(f"[Warn] 模块 {name} 映射失败: {type(e).__name__}: {e}")
            continue

    save_path = args.save
    if not save_path:
        os.makedirs('activations', exist_ok=True)
        a_label = osp.basename(args.acts_a).replace('.pt', '')
        b_label = osp.basename(args.acts_b).replace('.pt', '')
        save_path = osp.join('activations', f"mapping_{a_label}_TO_{b_label}_sigma{args.sigma}_thr{args.fos_threshold}.pt")

    torch.save({
        'mapping': mapping,
        'sigma': float(args.sigma),
        'fos_threshold': float(args.fos_threshold),
        'top_ratio': float(args.top_ratio) if args.top_ratio is not None else None,
        'topk': int(args.topk) if args.topk is not None else None,
        'feature_weights': tuple(args.feature_weights),
    }, save_path)
    print(f"[Done] FAM 神经元映射已保存: {save_path}")


if __name__ == '__main__':
    main()
