#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPT V0.3 公共数学与配对工具 (自包含, 不依赖 V0.1)。

包含:
  * 模块名规范化与 host<->donor 配对 (含 CogVLM 分支别名);
  * SVD 联合符号规范化;
  * log-domain Sinkhorn (Cuturi 2013 原义的熵正则 OT 求解);
  * 列正交化 orth (正对角 QR) 与 APT 事后结构差异 c_m;
  * 权重指纹 (缓存绑定分量身份)。

来源契约: 熵正则 OT 求解沿用 Cuturi(2013); 非负锥投影/投影最优性沿用 Boyd(2014);
c_m 为 APT 参考稿给出的 discrepancy, 本文件只做忠实实现, 不新增保证。
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch

EPS = 1e-8


# --------------------------------------------------------------------------
# 模块名规范化与配对
# --------------------------------------------------------------------------
def canon_module_name(name: str) -> str:
    """language_model.model.* / language_model.* -> model.*; transformer.layers.* -> model.layers.*"""
    k = name
    if k.startswith("language_model.model."):
        k = "model." + k[len("language_model.model."):]
    elif k.startswith("language_model."):
        k = "model." + k[len("language_model."):]
    k = k.replace("transformer.layers.", "model.layers.")
    return k


def canon_module_from_param_key(key: str) -> str:
    """把参数键 (含 .weight/.bias) 规范化为模块键。"""
    k = key.replace("language_model.model.", "model.").replace("language_model.", "model.")
    k = k.replace("transformer.layers.", "model.layers.")
    if "layers" in k:
        k = "model." + k[k.find("layers"):]
    parts = k.split(".")
    if parts[-1] in ("weight", "bias"):
        parts = parts[:-1]
    return ".".join(parts)


def strip_multiway_segment(name: str) -> str:
    """去掉 multiway.* 段, 使多路模块与普通模块可对齐。"""
    if ".multiway." not in name:
        return name
    parts, out, skip = name.split("."), [], False
    for part in parts:
        if skip:
            skip = False
            continue
        if part == "multiway":
            skip = True
            continue
        out.append(part)
    return ".".join(out)


def group_by_canonical(subspaces: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for name in subspaces.keys():
        canon = strip_multiway_segment(canon_module_name(name))
        grouped.setdefault(canon, []).append(name)
    return grouped


def _cogvlm_alias_for_base(base_canon: str, donor_canons: set) -> Optional[str]:
    """base 侧 CogVLM 命名 -> donor 标准命名。"""
    if ".self_attn." in base_canon:
        head = base_canon.split(".self_attn.")[0]
        for cand in (
            head + ".self_attn.language_expert_query_key_value",
            head + ".self_attn.vision_expert_query_key_value",
        ):
            if cand in donor_canons:
                return cand
        if base_canon.endswith(".self_attn.o_proj"):
            for cand in (
                head + ".self_attn.language_expert_dense",
                head + ".self_attn.vision_expert_dense",
            ):
                if cand in donor_canons:
                    return cand
    if ".mlp." in base_canon:
        head, tail = base_canon.split(".mlp.", 1)
        for cand in (head + ".mlp.language_mlp." + tail, head + ".mlp.vision_mlp." + tail):
            if cand in donor_canons:
                return cand
    # 反向: base 为 CogVLM 分支命名 -> donor 标准命名
    if ".self_attn.language_expert_query_key_value" in base_canon or ".self_attn.vision_expert_query_key_value" in base_canon:
        head = base_canon.split(".self_attn.")[0]
        for tgt in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"):
            if head + tgt in donor_canons:
                return head + tgt
    if ".self_attn.language_expert_dense" in base_canon or ".self_attn.vision_expert_dense" in base_canon:
        head = base_canon.split(".self_attn.")[0]
        if head + ".self_attn.o_proj" in donor_canons:
            return head + ".self_attn.o_proj"
    for seg, repl in ((".mlp.language_mlp.", ".mlp."), (".mlp.vision_mlp.", ".mlp.")):
        if seg in base_canon:
            head, tail = base_canon.split(seg, 1)
            cand = head + repl + tail
            if cand in donor_canons:
                return cand
    return None


def _cogvlm_alias_for_donor(donor_canon: str, base_canons: set) -> Optional[str]:
    """donor 标准命名 -> base 侧 CogVLM 命名。"""
    if ".self_attn." in donor_canon:
        head = donor_canon.split(".self_attn.")[0]
        if any(donor_canon.endswith(f".self_attn.{x}_proj") for x in ("q", "k", "v")):
            for cand in (
                head + ".self_attn.language_expert_query_key_value",
                head + ".self_attn.vision_expert_query_key_value",
            ):
                if cand in base_canons:
                    return cand
        if donor_canon.endswith(".self_attn.o_proj"):
            for cand in (
                head + ".self_attn.language_expert_dense",
                head + ".self_attn.vision_expert_dense",
            ):
                if cand in base_canons:
                    return cand
    if ".mlp." in donor_canon:
        head, tail = donor_canon.split(".mlp.", 1)
        for cand in (head + ".mlp.language_mlp." + tail, head + ".mlp.vision_mlp." + tail):
            if cand in base_canons:
                return cand
    return None


def pair_modules(
    subsA: Dict[str, Dict[str, torch.Tensor]],
    subsB: Dict[str, Dict[str, torch.Tensor]],
    verbose: bool = False,
) -> List[Tuple[str, str, str]]:
    """返回 [(host_key, donor_key, canonical), ...]; 多对一由排序稳定决定, 不静默覆盖。"""
    groupA, groupB = group_by_canonical(subsA), group_by_canonical(subsB)
    keysA, keysB = set(groupA.keys()), set(groupB.keys())
    inter = sorted(keysA & keysB)
    onlyA, onlyB = sorted(keysA - keysB), sorted(keysB - keysA)
    if onlyA or onlyB:
        print(
            f"[Stage-2][Warn] A/B 子空间键不完全一致: A={len(keysA)} B={len(keysB)} "
            f"交集={len(inter)} A-Only={len(onlyA)} B-Only={len(onlyB)}"
        )
        if onlyA:
            print("  [Only in A] 示例:", [(k, groupA[k][:2]) for k in onlyA[:5]])
        if onlyB:
            print("  [Only in B] 示例:", [(k, groupB[k][:2]) for k in onlyB[:5]])

    pairs: List[Tuple[str, str, str]] = []
    for canon in inter:
        base_list, donor_list = sorted(groupA[canon]), sorted(groupB[canon])
        if len(donor_list) == len(base_list):
            pairs.extend((b, d, canon) for b, d in zip(base_list, donor_list))
        else:
            pairs.extend((b, donor_list[0], canon) for b in base_list)
    mapped = 0
    for canon in onlyA:
        mapped_canon = _cogvlm_alias_for_base(canon, keysB)
        if mapped_canon is None:
            continue
        base_list, donor_list = sorted(groupA[canon]), sorted(groupB[mapped_canon])
        if len(donor_list) == len(base_list):
            pairs.extend((b, d, canon) for b, d in zip(base_list, donor_list))
        else:
            pairs.extend((b, donor_list[0], canon) for b in base_list)
        mapped += 1
    if mapped > 0 and verbose:
        print(f"[Stage-2] CogVLM alias mapping pairs added: {mapped}")
    mapped_b = 0
    for canon in onlyB:
        mapped_canon = _cogvlm_alias_for_donor(canon, keysA)
        if mapped_canon is None:
            continue
        donor_list, base_list = sorted(groupB[canon]), sorted(groupA[mapped_canon])
        if len(donor_list) == len(base_list):
            pairs.extend((b, d, mapped_canon) for b, d in zip(base_list, donor_list))
        else:
            pairs.extend((base_list[0], d, mapped_canon) for d in donor_list)
        mapped_b += 1
    if mapped_b > 0 and verbose:
        print(f"[Stage-2] CogVLM reverse alias mapping pairs added: {mapped_b}")
    return pairs


def load_subspaces(path: Optional[str]):
    if not path:
        return None
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj.get("subspaces", obj)


def load_subspaces_full(path: Optional[str]):
    """返回 (subspaces, meta_obj); meta 含 decomp_object / anchor_model / rank 等 Stage-1 记录。

    用于 Stage-3 判断保护域是「绝对权重基」还是「task vector 基」——
    两者注入语义不同 (前者保护预训练方向, 后者保护微调方向), 必须在产物 meta 中留痕。
    """
    if not path:
        return None, {}
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "subspaces" not in obj:
        return obj, {}
    return obj["subspaces"], obj


# --------------------------------------------------------------------------
# SVD 符号规范 / 正交化 / OT / discrepancy
# --------------------------------------------------------------------------
def canon_svd_signs(U: torch.Tensor, Vh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """最大绝对坐标为正的联合符号规范; 只固定 (u,v) 与 (-u,-v) 的等价歧义。"""
    idx = U.abs().argmax(dim=0)
    col = torch.arange(U.shape[1], device=U.device)
    signs = U[idx, col].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return U * signs.unsqueeze(0), Vh * signs.unsqueeze(1)


def orth(X: torch.Tensor) -> torch.Tensor:
    """正对角 QR 列正交化 (与 APT-local 的 Orth 约定一致)。"""
    Q, R = torch.linalg.qr(X)
    d = torch.diagonal(R)
    signs = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    return Q * signs.unsqueeze(0)


def sinkhorn(A: torch.Tensor, tau: float = 0.15, iters: int = 50,
             anneal: int = 1) -> torch.Tensor:
    """log-domain Sinkhorn: 把 relevance 分数 A 投影为双随机软对应矩阵 (作用于 A/tau)。

    anneal > 1 时启用 **ε-scaling (退火)**: 温度从 tau*2^(anneal-1) 逐步降到 tau,
    每一步用上一步的对偶势 (u, v) 做 warm start。

    为什么需要: 固定 tau 时行边际残差以约 1/sqrt(iters) 衰减, 对小 tau (锐化 plan) 极慢。
    实测 (tau_bar=1.0, 规范化打分, 128x128):
        iters=50 -> row_res 2.2e-2 ; iters=1000 -> 1.1e-3   (仍不达标)
    退火后可在相同预算内把残差压到 1e-6 量级, 同时保留锐化 plan 的熵。

    对偶势形式与逐次行列归一化在收敛点等价; anneal=1 时退化为原实现。
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    r = A.shape[0]
    anneal = max(1, int(anneal))
    # 双随机口径: 行/列和都为 1 (不是联合分布的 1/r), 与原实现一致。
    log_a = torch.zeros(r, device=A.device, dtype=A.dtype)
    u = torch.zeros(r, device=A.device, dtype=A.dtype)
    v = torch.zeros(r, device=A.device, dtype=A.dtype)
    logK = A / max(tau, EPS)
    for s in range(anneal):
        tau_s = tau * (2.0 ** (anneal - 1 - s))     # 由大到小
        logK = A / max(tau_s, EPS)
        for _ in range(int(iters)):
            u = log_a - torch.logsumexp(logK + v.unsqueeze(0), dim=1)
            v = log_a - torch.logsumexp(logK + u.unsqueeze(1), dim=0)
    return torch.exp(u.unsqueeze(1) + logK + v.unsqueeze(0))


def structural_divergence(U_A: torch.Tensor, U_B: torch.Tensor, M: torch.Tensor, r: int) -> float:
    """c_m = ||U_A^T Orth(U_B M^T) - I||_F^2 / r (带基取向的 discrepancy, 非纯 principal angle)。"""
    with torch.no_grad():
        U_B_bar = orth(U_B @ M.T)
        X = U_A.T @ U_B_bar
        I = torch.eye(r, device=X.device, dtype=X.dtype)
        return float(((X - I) ** 2).sum().item() / max(r, 1))


# --------------------------------------------------------------------------
# 权重指纹 (缓存绑定分量身份)
# --------------------------------------------------------------------------
def weight_fingerprint(W: torch.Tensor, n_sample: int = 2048) -> str:
    """低成本指纹: shape/dtype/范数/首尾采样统计的 sha1 摘要。"""
    Wf = W.detach().float().reshape(-1)
    n = Wf.numel()
    if n == 0:
        return "empty"
    k = min(int(n_sample), n)
    head = Wf[:k]
    tail = Wf[-k:]
    stats = torch.stack([
        torch.tensor(float(Wf.norm().item())),
        torch.tensor(float(Wf.sum().item())),
        torch.tensor(float(Wf.abs().sum().item())),
        torch.tensor(float(head.sum().item())),
        torch.tensor(float(tail.sum().item())),
        torch.tensor(float(Wf.max().item())),
        torch.tensor(float(Wf.min().item())),
    ])
    payload = f"{tuple(W.shape)}|{W.dtype}|" + ",".join(f"{v:.6e}" for v in stats.tolist())
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
