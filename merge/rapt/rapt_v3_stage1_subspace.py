#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPT V0.3 Stage 1: 方向精确的固定秩 SVD 子空间提取。

与 V0.1 的差别 (均为工程前置, 不改变分解对象):
  * 每个模块记录权重指纹, 使 Stage-2/Stage-3 能校验分量身份;
  * 记录截断边界谱隙 gap = sigma_r - sigma_{r+1} 与尾谱能量 (精确模式);
  * 缓存带 SCHEMA_VERSION, 不匹配时由下游显式拒绝。

**TA 分支 (V0.3.13 新增, 可选)**: `--anchor-model` 指定共同基座锚点 W_0 时,
分解对象由绝对权重 W 改为 **task vector τ = W − W_0** (Ilharco et al. 2023 的 task arithmetic 对象)。
动机: ‖τ‖ ≪ ‖W_0‖ 时 U_W 的 top-r 由**预训练**方向主导, 而 W_B − W_A = τ_B − τ_A
是两个**微调**方向之差, 与预训练主方向近似正交 —— 这解释了 §8.8.7 的 98.86% 不可达。
改用 U_τ 作保护域后, Stage-2/Stage-3 全部机制原样复用 (接口不变, 仅换分解对象)。
未指定锚点时行为与旧版逐位一致 (`decomp_object="absolute_weight"`)。

来源契约: 复用 Eckart-Young(1936) 的给定矩阵低秩最小二乘近似; 随机化路径复用
Halko et al.(2011) 的 range finding 与小矩阵 SVD。task vector 的线性结构沿用
Ilharco et al.(2023, arXiv 2212.04089); 本文件只做分解, 不新增方法学保证。
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import torch
from tqdm import tqdm

try:
    from .utils import load_weights, need_merge
    from .rapt_v3_common import canon_svd_signs, canon_module_from_param_key, weight_fingerprint
except Exception:  # 直接脚本运行
    from utils import load_weights, need_merge
    from rapt_v3_common import canon_svd_signs, canon_module_from_param_key, weight_fingerprint

SCHEMA_VERSION = "rapt-v2.0"


def _anchor_index(anchor_model: str):
    """canonical 模块名 -> 锚点模型的 2D 参数键 (跨命名规范对齐)。"""
    weights_0 = load_weights(anchor_model)
    index = {}
    for k, w in weights_0.items():
        if k.endswith(".weight") and w.ndim == 2 and need_merge(k):
            index[canon_module_from_param_key(k)] = k
    return weights_0, index


def extract(args: argparse.Namespace) -> None:
    print("\n--- [RAPT v3 Stage-1: Direction-Accurate SVD + Cache Identity] ---")
    weights = load_weights(args.model_dir)
    rank = int(args.rank)

    anchor_weights = anchor_index = None
    decomp_object = "absolute_weight"
    if getattr(args, "anchor_model", None):
        decomp_object = "task_vector"
        print(f"[TA] 分解对象改为 task vector τ = W − W_anchor; anchor={args.anchor_model}")
        anchor_weights, anchor_index = _anchor_index(args.anchor_model)
        print(f"[TA] 锚点语言侧 2D 模块 {len(anchor_index)} 个")
    minus_weights = minus_index = None
    if getattr(args, "minus_model", None):
        decomp_object = "delta"
        print(f"[Δ] 分解对象改为 Δ = W_model − W_minus; minus={args.minus_model}")
        minus_weights, minus_index = _anchor_index(args.minus_model)
        print(f"[Δ] 被减模型语言侧 2D 模块 {len(minus_index)} 个")
    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    out: dict = {}
    total = kept = anchor_missing = 0
    iterator = tqdm(list(weights.items()), desc="RAPT v3 Stage-1 SVD", disable=bool(args.no_progress))
    for key, weight in iterator:
        if not (key.endswith(".weight") and weight.ndim == 2 and need_merge(key)):
            continue
        total += 1
        if decomp_object == "task_vector":
            akey = anchor_index.get(canon_module_from_param_key(key))
            W_0 = anchor_weights.get(akey) if akey is not None else None
            if W_0 is None or W_0.shape != weight.shape:
                anchor_missing += 1
                continue
            W = (weight.float() - W_0.float()).cuda() if use_cuda else (weight.float() - W_0.float())
            del W_0
        elif decomp_object == "delta":
            mkey = minus_index.get(canon_module_from_param_key(key))
            W_m = minus_weights.get(mkey) if mkey is not None else None
            if W_m is None or W_m.shape != weight.shape:
                anchor_missing += 1
                continue
            W = (weight.float() - W_m.float()).cuda() if use_cuda else (weight.float() - W_m.float())
            del W_m
        else:
            W = weight.float().cuda() if use_cuda else weight.float()
        d_out, d_in = W.shape
        r = min(rank, d_out, d_in)
        if r <= 0:
            continue
        sigma_next = None
        tail_energy = None
        gap = None
        try:
            if args.low_rank_svd and r < min(d_out, d_in):
                q = min(min(d_out, d_in), r + 16)
                U, S, V = torch.svd_lowrank(W, q=q, niter=2)
                U, S, Vh = U[:, :r], S[:r], V[:, :r].T
                # 近似模式没有精确尾谱, 只记录重构残差作为近似证书
                tail_energy = float((W - (U * S.unsqueeze(0)) @ Vh).norm().item())
            else:
                U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
                U, S, Vh = U_full[:, :r], S_full[:r], Vh_full[:r, :]
                if S_full.numel() > r:
                    sigma_next = float(S_full[r].item())
                    gap = float((S_full[r - 1] - S_full[r]).item())
                    tail_energy = float((S_full[r:] ** 2).sum().item())
                else:
                    sigma_next = 0.0
                    gap = float(S_full[r - 1].item())
                    tail_energy = 0.0
                del U_full, Vh_full
        except RuntimeError as exc:
            print(f"[Warn] SVD failed for {key}: {exc}")
            continue
        U, Vh = canon_svd_signs(U.contiguous(), Vh.contiguous())
        mod = canon_module_from_param_key(key)
        out[mod] = {
            "U": U.cpu(),
            "S": S.contiguous().cpu(),
            "Vh": Vh.cpu(),
            "rank_used": torch.tensor(r),
            "shape": torch.tensor([d_out, d_in]),
            "fingerprint": weight_fingerprint(W),
            "sigma_next": torch.tensor(float(sigma_next) if sigma_next is not None else -1.0),
            "boundary_gap": torch.tensor(float(gap) if gap is not None else -1.0),
            "tail_energy": torch.tensor(float(tail_energy) if tail_energy is not None else -1.0),
        }
        kept += 1
        if args.verbose:
            print(
                f"[RAPTv2-SVD] {mod}: shape=({d_out},{d_in}) r={r} "
                f"gap={gap if gap is None else round(gap, 4)} mode={'lowrank' if args.low_rank_svd else 'exact'}"
            )
    save_dir = osp.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "subspaces": out,
            "rank": rank,
            "model_dir": args.model_dir,
            "svd_mode": "randomized_lowrank" if args.low_rank_svd else "exact",
            "schema_version": SCHEMA_VERSION,
            "code_version": "rapt_V0_3",
            "decomp_object": decomp_object,
            "anchor_model": getattr(args, "anchor_model", None),
            "minus_model": getattr(args, "minus_model", None),
            "anchor_missing_modules": anchor_missing,
        },
        args.save,
    )
    print(f"[Done] {kept}/{total} saved -> {args.save} "
          f"(decomp_object={decomp_object}"
          + (f", anchor_missing={anchor_missing}"
             if decomp_object in ("task_vector", "delta") else "") + ")")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAPT v3 Stage-1 direction-accurate SVD")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--low-rank-svd", action="store_true",
                    help="固定 oversampling(+16) 与 2 次幂迭代的加速分支")
    ap.add_argument("--anchor-model", dest="anchor_model", default=None,
                    help="TA 分支: 共同基座锚点目录; 指定后分解对象变为 task vector W − W_anchor")
    ap.add_argument("--minus-model", dest="minus_model", default=None,
                    help="§18 分支: 再减去该模型, 即分解对象变为 W_model − W_minus。"
                         "用于生成 Δ = W_B − W_A 的主方向基 (--model-dir 传 donor, "
                         "此处传 host), 供 --injection delta_lr 使用")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    extract(parse_args())
