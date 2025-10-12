#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DGSM-TEFM Stage-3: Dynamic-Adjusted Subspace Fusion Merge

与原 GSF Stage-3 基本相同, 区别:
  * 若 Stage-2 (DGSM) 中存在动态映射矩阵 M (r x r), 在重建 donor 差分投影时可选择使用对齐奇异方向加权。
  * 增加 --use-dynamic-m 选项: 启用时若记录 M 则先用 W_A 的 U_A 基底重投影，donor 差分投影方向按 M 的列混合。

保守实现: 为避免过度复杂化, 这里仍只对投影差分 τ_proj 进行缩放, 未改变正交残差处理。
"""
from __future__ import annotations
import argparse, json, math, os, os.path as osp
from collections import defaultdict
from typing import Dict, Tuple
import torch, safetensors.torch
from tqdm import tqdm
try:
    from .utils import load_weights, need_merge  # type: ignore
except Exception:
    from utils import load_weights, need_merge  # type: ignore

def _load_subspaces(path: str):
    try:
        obj = torch.load(path, map_location='cpu')
        return obj.get('subspaces', obj)
    except Exception:
        return None

EPS=1e-8

def _canon_param_key(param_key: str) -> str:
    k = param_key.replace('language_model.model.','model.').replace('language_model.','model.')
    if 'layers' in k:
        pos=k.find('layers'); k='model.'+k[pos:]
    return k

def _module_from_param_key(param_key:str)->str:
    k=_canon_param_key(param_key); parts=k.split('.')
    if len(parts)>=2: parts=parts[:-1]
    return '.'.join(parts)

def _alt_donor_key(donor:dict, base_key:str)->str | None:
    """在 donor 权重字典里为 base_key 寻找兼容命名的键。
    规则：
      1) 原样匹配 base_key
      2) 尝试在前面加 "language_model." 前缀（OneVision/QwenVL 常见命名）
    命中则返回匹配到的键，否则返回 None
    """
    if base_key in donor:
        return base_key
    alt = f"language_model.{base_key}"
    if alt in donor:
        return alt
    return None

def _sigmoid(x:float)->float:
    try: return 1.0/(1.0+math.exp(-x))
    except OverflowError: return 0.0 if x<0 else 1.0

def _svd_trunc(W:torch.Tensor, r:int):
    Wf=W.float(); r_use=min(r,Wf.shape[0],Wf.shape[1])
    if r_use<=0: return None
    try:
        U,S,Vh=torch.linalg.svd(Wf, full_matrices=False); return U[:,:r_use].contiguous(), S[:r_use].contiguous()
    except Exception: return None


def _softmax_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max() if x.numel() > 0 else x
    return torch.exp(x) / torch.exp(x).sum().clamp_min(EPS)


def _compute_tfi(S: torch.Tensor, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Localization step (DGSM-TEFM_v0_5.md §4) to obtain TFI scores.

    Returns (TFI, p, bar_d, bar_h, bar_s).
    """
    p = _softmax_tensor(S)
    bar_s = float(psi[0]) if psi.numel() >= 1 else float((p * S).sum().item())
    bar_h = float(psi[1]) if psi.numel() >= 2 else float(-(p * (p.clamp_min(EPS).log())).sum().item())
    bar_d = float(psi[2]) if psi.numel() >= 3 else 0.0
    denom = bar_h + EPS
    tfi_base = max(0.0, bar_s * (1.0 - bar_d))
    tfi = p * (tfi_base / denom)
    return tfi, p, bar_d, bar_h, bar_s


def _select_tfi_mask(scores: torch.Tensor, threshold: float, topk: int) -> torch.Tensor:
    mask = scores >= threshold if threshold > 0 else torch.ones_like(scores, dtype=torch.bool)
    if topk > 0:
        k = min(topk, scores.numel())
        top_idx = torch.topk(scores, k).indices
        top_mask = torch.zeros_like(scores, dtype=torch.bool)
        top_mask[top_idx] = True
        mask = mask | top_mask
    if mask.sum() == 0:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[scores.argmax()] = True
    return mask


def _normalize_transport(pi: torch.Tensor, iters: int = 4) -> torch.Tensor:
    if pi.ndim != 2:
        return pi
    r, c = pi.shape
    if r == 0 or c == 0:
        return pi
    target_row = 1.0 / c
    target_col = 1.0 / r
    pi = pi / pi.sum().clamp_min(EPS)
    for _ in range(iters):
        row_sum = pi.sum(dim=1, keepdim=True).clamp_min(EPS)
        pi = pi * (target_row / row_sum)
        col_sum = pi.sum(dim=0, keepdim=True).clamp_min(EPS)
        pi = pi * (target_col / col_sum)
    return pi / pi.sum().clamp_min(EPS)


def _save_model(args, merged:Dict[str,torch.Tensor]):
    base_dir=osp.basename(args.base_model.rstrip(os.sep))
    out_root=osp.join(args.output_dir, base_dir, 'dgsm_merged'); os.makedirs(out_root, exist_ok=True)
    sft_index=osp.join(args.base_model,'model.safetensors.index.json')
    bin_index=osp.join(args.base_model,'pytorch_model.bin.index.json')
    def copy_side():
        for fn in os.listdir(args.base_model):
            if fn.endswith(('.json','.model','.py','.md')):
                try:
                    src=osp.join(args.base_model,fn); dst=osp.join(out_root,fn)
                    if not osp.exists(dst):
                        import shutil; shutil.copy(src,dst)
                except Exception: pass
    if osp.exists(sft_index):
        with open(sft_index,'r') as f: index=json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard,sd in shards.items(): safetensors.torch.save_file(sd, osp.join(out_root,shard))
        copy_side(); print(f"[Save] Sharded safetensors -> {out_root}"); return
    if osp.exists(bin_index):
        with open(bin_index,'r') as f: index=json.load(f)['weight_map']
        shards=defaultdict(dict)
        for k,v in merged.items():
            if k in index: shards[index[k]][k]=v
        for shard,sd in shards.items(): torch.save(sd, osp.join(out_root,shard))
        copy_side(); print(f"[Save] Sharded .bin -> {out_root}"); return
    sft_single=osp.join(args.base_model,'model.safetensors')
    bin_single=osp.join(args.base_model,'pytorch_model.bin')
    if osp.exists(sft_single): safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Single safetensors -> {out_root}"); return
    if osp.exists(bin_single): torch.save(merged, osp.join(out_root,'pytorch_model.bin')); copy_side(); print(f"[Save] Single .bin -> {out_root}"); return
    safetensors.torch.save_file(merged, osp.join(out_root,'model.safetensors')); copy_side(); print(f"[Save] Default safetensors -> {out_root}")


def dgsm_merge(args: argparse.Namespace):
    print("\n--- [DGSM Stage-3: Dynamic Subspace Fusion Merge] ---")
    weights_A=load_weights(args.base_model); weights_B=load_weights(args.donor_model)
    stage2=torch.load(args.stage2, map_location='cpu'); modules_info=stage2.get('modules', stage2)
    base_subs = _load_subspaces(args.base_subs) if args.base_subs else None
    if base_subs is not None:
        print(f"[Info] Reusing Stage-1 base subspaces: {len(base_subs)} entries")
    merged=weights_A.copy(); stat_layers=stat_used=0
    missing_in_B=0; missing_examples=[]
    reuse_ok=reuse_fail=svd_fallback=0
    for k in tqdm(list(weights_A.keys()), desc='DGSM Merge'):
        if not need_merge(k): continue
        if k.endswith('.weight') and weights_A[k].ndim==2:
            mod=_module_from_param_key(k); blk=modules_info.get(mod)
            W_A=weights_A[k].float(); W_B=weights_B.get(k, None)
            if W_B is None:
                alt_k = _alt_donor_key(weights_B, k)
                if alt_k is None:
                    missing_in_B += 1
                    if len(missing_examples) < 10:
                        missing_examples.append(k)
                    continue
                W_B = weights_B[alt_k]
            W_B=W_B.float(); stat_layers+=1
            if blk is not None:
                gwd_cost=float(blk['gwd_cost']); cost_norm=min(1.0, gwd_cost/max(EPS,float(args.cost_scale)))
                # 优先使用 Stage-2 层级 lambda 估计（若启用并可用），否则用全局 gamma/cost_scale 计算
                if bool(getattr(args, 'use_lambda_est', False)) and ('lambda_est' in blk):
                    lam = float(blk['lambda_est'])
                    if getattr(args, 'lam_scale', None) is not None:
                        lam = max(0.0, min(1.0, lam * float(args.lam_scale)))
                else:
                    lam=_sigmoid(float(args.gamma)*(1.0-cost_norm))
                # 基于 Stage-2 的 psi_A 熵自适应调整正交项权重（越确定性，越小的正交注入）
                base_ortho=float(args.ortho_scale)
                if bool(getattr(args, 'ortho_adapt', False)) and ('psi_A' in blk):
                    psiA = blk['psi_A']  # [bar_s, bar_h, bar_d]
                    bar_h = float(psiA[1]) if torch.is_tensor(psiA) else float(psiA[1])
                    # 归一化熵: 经验范围 ~ [0, ln(r)]，用 r 的对数进行归一
                    r_eff = int(blk.get('rank_A', torch.tensor(1)))
                    import math as _m
                    h_max = max(1e-6, _m.log(max(2, r_eff)))
                    h_norm = min(1.0, max(0.0, bar_h / h_max))
                    # 低熵(更确定) -> 减小正交注入；高熵 -> 增大到 base_ortho
                    ortho_scale = base_ortho * h_norm
                else:
                    ortho_scale=base_ortho
                r=int(blk['rank_A'])
                U_A=None
                if base_subs is not None and mod in base_subs:
                    try:
                        U_full = base_subs[mod]['U']  # d_out x r_full
                        if U_full.shape[0] == W_A.shape[0]:
                            U_A = U_full[:, :r].contiguous().float(); reuse_ok += 1
                        else:
                            reuse_fail += 1
                    except Exception:
                        reuse_fail += 1
                if U_A is None:
                    svdA=_svd_trunc(W_A, r)
                    if svdA is None: continue
                    U_A,_=svdA; svd_fallback += 1
                tau = W_B - W_A
                coef = U_A.T @ tau  # r x d_in

                tefm_weights = None
                lam_doc = None
                has_tefm = all(key in blk for key in ('pi', 'S_A', 'S_B', 'psi_A', 'psi_B'))
                M_tensor = blk['M'].float() if (args.use_dynamic_m and ('M' in blk)) else None

                if has_tefm:
                    # Localization: TFI_{l,i} = p_{A,i} * (\bar{s}_{A,l} (1-\bar{d}_l)) / (\bar{h}_{A,l}+\epsilon)
                    S_A = blk['S_A'].float()
                    S_B = blk['S_B'].float()
                    psi_A = blk['psi_A'].float()
                    psi_B = blk['psi_B'].float()
                    tfi, p_A, bar_d, _, _ = _compute_tfi(S_A, psi_A)
                    mask = _select_tfi_mask(tfi, float(getattr(args, 'tfi_threshold', 0.0)), int(getattr(args, 'tfi_topk', 0)))

                    # Mapping: 调整 π*_{k,p} ← π*_{k,p} · TOS_{k,p} / Z，其中 TOS 基于余弦相似和 e^{-β \bar{d}_l}
                    p_B = _softmax_tensor(S_B)
                    bar_d_B = float(psi_B[2]) if psi_B.numel() >= 3 else bar_d
                    psiA_comp = torch.stack([S_A, p_A, S_A.new_full(S_A.shape, bar_d)], dim=1)
                    psiB_comp = torch.stack([S_B, p_B, S_B.new_full(S_B.shape, bar_d_B)], dim=1)
                    normA = psiA_comp.norm(dim=1, keepdim=True).clamp_min(EPS)
                    normB = psiB_comp.norm(dim=1, keepdim=True).clamp_min(EPS)
                    cos_sim = (psiA_comp @ psiB_comp.T) / (normA @ normB.T)
                    beta = float(getattr(args, 'mapping_beta', 0.1))
                    tos = torch.relu(cos_sim) * math.exp(-beta * bar_d)
                    pi = blk['pi'].float()
                    pi_adj = _normalize_transport((pi * tos).clamp_min(EPS))
                    if M_tensor is not None:
                        pi_effective = M_tensor @ pi_adj
                    else:
                        pi_effective = pi_adj
                    alignment = pi_effective @ S_B
                    alignment = alignment * (tfi / (tfi.max().clamp_min(EPS)))
                    alignment = alignment * mask.float()
                    if alignment.abs().sum() <= EPS:
                        alignment = mask.float()
                    alignment = alignment.clamp_min(0.0)
                    if alignment.max() > 0:
                        alignment = alignment / alignment.max()
                    tefm_weights = alignment.to(coef.dtype)

                    # 融合系数: λ_l = sigmoid(-\bar{d}_l) * mean(TFI_selected) 与原 λ 取平均
                    tfi_sel = tfi[mask]
                    if tfi_sel.numel() > 0:
                        tfi_norm = (tfi_sel / tfi_sel.max().clamp_min(EPS)).mean().item()
                        lam_doc = (1.0 / (1.0 + math.exp(bar_d))) * tfi_norm
                    else:
                        lam_doc = 0.0

                # 动态 M 处理 (Fusion 前的子空间映射): tau_proj = U_A * (M_l (·))
                if M_tensor is not None:
                    coef = M_tensor @ coef

                if tefm_weights is not None:
                    # Fusion: τ_proj' = U_A (M_l π^* S_B) with TFI-selected components (DGSM-TEFM_v0_5.md §4)
                    coef = coef * tefm_weights.unsqueeze(1)

                tau_proj = U_A @ coef
                tau_ortho = tau - tau_proj
                if lam_doc is not None:
                    lam = max(0.0, min(1.0, 0.5 * (lam + lam_doc)))
                W_new = W_A + lam * tau_proj + lam * ortho_scale * tau_ortho
                merged[k]=W_new.to(weights_A[k].dtype); stat_used+=1
            else:
                alpha=float(args.fallback_alpha); merged[k]=(1-alpha)*W_A + alpha*W_B
        elif k.endswith('.bias') and weights_A[k].ndim==1:
            W_A=weights_A[k].float(); W_B=weights_B.get(k, None)
            if W_B is None:
                alt_k = _alt_donor_key(weights_B, k)
                if alt_k is None:
                    continue
                W_B = weights_B[alt_k]
            W_B=W_B.float(); alpha=float(args.bias_alpha)
            merged[k]=(1-alpha)*W_A + alpha*W_B
    _save_model(args, merged)
    meta=dict(base_model=args.base_model, donor_model=args.donor_model, stage2=args.stage2,
              cost_scale=float(args.cost_scale), gamma=float(args.gamma), ortho_scale=float(args.ortho_scale),
              fallback_alpha=float(args.fallback_alpha), bias_alpha=float(args.bias_alpha),
              use_dynamic_m=bool(args.use_dynamic_m), stat_layers=stat_layers, stat_used=stat_used,
              base_subs=args.base_subs, reuse_ok=reuse_ok, reuse_fail=reuse_fail, svd_fallback=svd_fallback,
              tfi_threshold=float(getattr(args, 'tfi_threshold', 0.0)),
              tfi_topk=int(getattr(args, 'tfi_topk', 0)),
              mapping_beta=float(getattr(args, 'mapping_beta', 0.1)))
    base_dir=osp.basename(args.base_model.rstrip(os.sep))
    out_root=osp.join(args.output_dir, base_dir, 'dgsm_merged'); os.makedirs(out_root, exist_ok=True)
    with open(osp.join(out_root,'merge_meta_dgsm.json'),'w') as f: json.dump(meta,f,indent=2)
    print(f"[Done] DGSM merge complete: layers={stat_layers}, used={stat_used}, reuse_U={reuse_ok}, svd_fallback={svd_fallback}, reuse_fail={reuse_fail}")
    if missing_in_B>0:
        print(f"  [Warn] {missing_in_B} 处权重在 donor 中未找到（已跳过）。示例: {missing_examples}")


def parse_args():
    ap=argparse.ArgumentParser(description='DGSM Stage-3 Merge')
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--donor-model', required=True)
    ap.add_argument('--stage2', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--cost-scale', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=4.0)
    ap.add_argument('--ortho-scale', type=float, default=0.5)
    ap.add_argument('--fallback-alpha', type=float, default=0.5)
    ap.add_argument('--bias-alpha', type=float, default=0.5)
    ap.add_argument('--use-dynamic-m', action='store_true')
    ap.add_argument('--base-subs', type=str, default=None, help='Stage-1 base 子空间文件 (含 U)，提供可跳过重复 SVD')
    ap.add_argument('--use-lambda-est', action='store_true', help='使用 Stage-2 中每层的 lambda_est 作为融合强度')
    ap.add_argument('--lam-scale', type=float, default=None, help='对 per-layer lambda_est 进行缩放的系数, 可微调融合强度')
    ap.add_argument('--ortho-adapt', action='store_true', help='根据 Stage-2 的 psi 熵自适应调整正交项注入强度')
    ap.add_argument('--tfi-threshold', type=float, default=0.0, help='Localization 阶段的 TFI 阈值 (DGSM-TEFM_v0_5.md §4)')
    ap.add_argument('--tfi-topk', type=int, default=0, help='Localization 阶段仅保留前 k 个 TFI 分量 (0 表示保留全部)')
    ap.add_argument('--mapping-beta', type=float, default=0.1, help='Mapping 阶段 TOS 权重中的 β 系数 (e^{-β \bar{d}_l})')
    return ap.parse_args()

if __name__=='__main__':
    dgsm_merge(parse_args())
