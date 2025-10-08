#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DGSM-TEFM Stage-1: Subspace Extraction (same as GSF, standalone copy)
"""
from __future__ import annotations
import argparse
import os
import os.path as osp
from typing import Dict
import torch
from tqdm import tqdm
from .utils import load_weights, need_merge  # type: ignore

def _canon_module_from_param(key: str) -> str:
    k = key.replace("language_model.model.", "model.").replace("language_model.", "model.")
    if "layers" in k:
        pos = k.find("layers")
        k = "model." + k[pos:]
    parts = k.split('.')
    if parts[-1] in ('weight','bias'):
        parts = parts[:-1]
    return '.'.join(parts)

def extract(args: argparse.Namespace):
    print("\n--- [DGSM Stage-1: Subspace Extraction] ---")
    weights = load_weights(args.model_dir)
    rank = int(args.rank)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    total=kept=0
    use_cuda = bool(getattr(args, 'cuda', False)) and torch.cuda.is_available()
    for k,W in tqdm(list(weights.items()), desc='Stage-1 SVD'):
        if not (k.endswith('.weight') and W.ndim==2 and need_merge(k)):
            continue
        total+=1
        Wf = W.float().cuda() if use_cuda else W.float()
        d_out,d_in = Wf.shape
        r_use = min(rank,d_out,d_in)
        if r_use<=0: continue
        try_lowrank = (r_use < min(d_out,d_in)//2 and hasattr(torch,'svd_lowrank'))
        try:
            if try_lowrank:
                try:
                    U,S,V = torch.svd_lowrank(Wf, q=r_use)
                except Exception:
                    U,S,Vh = torch.linalg.svd(Wf, full_matrices=False); V=Vh.T
            else:
                U,S,Vh = torch.linalg.svd(Wf, full_matrices=False); V=Vh.T
        except RuntimeError as e:
            print(f"[Warn] SVD fail {k}: {e}"); continue
        U,S = U[:,:r_use].contiguous(), S[:r_use].contiguous()
        if use_cuda:
            U = U.cpu(); S = S.cpu()
        mod = _canon_module_from_param(k)
        out[mod] = {'U':U.cpu(),'S':S.cpu(),'rank_used':torch.tensor(int(r_use)),'shape':torch.tensor([d_out,d_in])}
        kept+=1
        if args.verbose:
            print(f"[SVD] {mod}: shape=({d_out},{d_in}) r={r_use}")
    torch.save({'subspaces':out,'rank':rank,'model_dir':args.model_dir}, args.save)
    print(f"[Done] {kept}/{total} saved -> {args.save}")

def parse_args():
    ap=argparse.ArgumentParser(description='DGSM Stage-1')
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--save',required=True)
    ap.add_argument('--rank',type=int,default=64)
    ap.add_argument('--verbose',action='store_true')
    ap.add_argument('--cuda',action='store_true', help='在可用时使用 CUDA 加速 SVD')
    return ap.parse_args()

if __name__=='__main__':
    extract(parse_args())
