#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPT V0.3 Stage 2: 免训练算子对应 + 双 target 谱/算子系数。

核心 (与 APT 的 learned translator 不同的实现方式):
    S_ij = <B_A,i, B_B,j>_F = (u_A,i^T u_B,j)(v_A,i^T v_B,j)   # 有符号, 符号不变
    C_ij = ||B_A,i - B_B,j||_F^2 / 4 = (1 - S_ij) / 2          # 归一化 cost ∈ [0,1]
    M    = Sinkhorn(S / tau)                                    # 熵正则 OT (Cuturi 2013 原义)
    P    = M / rowsum(M)                                        # 行归一化条件分布

同时产出两套"可迁移盈余"的 target 原料 (Stage-3 用 --target 选择):
    E0 运输谱:   sigma_hat_B = P @ sigma_B           (先平均 donor 谱强度)
    E1 算子响应: t_operator  = S @ sigma_B           (直接读 donor 截断算子在 host atom 上的有符号投影)

以及全部门控量:
    rho_i        = [sum_j P_ij S_ij]_+               (component 逐方向)
    lambda_apt   = 1 / (1 + c_m),  c_m = ||U_A^T Orth(U_B M^T) - I||_F^2 / r
    lambda_apt_identity (P=I 对照, 用于 --identity-mapping)
    lambda_energy = 运输能量中落在同向 operator pair 的比例

来源契约: S/C 是本地对 atom 表示的建模; Sinkhorn 复用 Cuturi(2013) 的给定 cost 求解;
c_m 忠实复现 APT 参考稿的 discrepancy; 本文件不新增任务收益保证。
"""
from __future__ import annotations

import argparse
import math
import os

import torch
from tqdm import tqdm

try:
    from .rapt_v3_common import (
        EPS,
        load_subspaces,
        pair_modules,
        sinkhorn,
        structural_divergence,
    )
except Exception:  # 直接脚本运行
    from rapt_v3_common import (
        EPS,
        load_subspaces,
        pair_modules,
        sinkhorn,
        structural_divergence,
    )

SCHEMA_VERSION = "rapt-v2.1"   # v2.1 起新增 q_raw / c_op / g6 / g7 (T17 失配正则收缩)


def operator_similarity_and_cost(UA, VhA, UB, VhB):
    """返回 (S, C); S 保留符号, C=(1-S)/2 ∈ [0,1]。"""
    if UA.shape[0] != UB.shape[0] or VhA.shape[1] != VhB.shape[1]:
        raise ValueError(
            "RAPT v3 要求 canonicalize 后模块同形: "
            f"U rows {UA.shape[0]} vs {UB.shape[0]}, V width {VhA.shape[1]} vs {VhB.shape[1]}"
        )
    left = UA.T @ UB
    right = VhA @ VhB.T
    similarity = (left * right).clamp(-1.0, 1.0)
    cost = ((1.0 - similarity) * 0.5).clamp(0.0, 1.0)
    return similarity, cost


def component_reliability(P: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
    """rho_i = [<B_A,i, sum_j P_ij B_B,j>_F]_+ = [sum_j P_ij S_ij]_+, ∈ [0,1]。"""
    if P.shape != similarity.shape:
        raise ValueError(f"P/similarity shape mismatch: {P.shape} vs {similarity.shape}")
    return (P * similarity).sum(dim=1).clamp(0.0, 1.0)


def operator_compatibility(P: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
    """q_i = sum_j P_ij S_ij, **保留符号** ∈ [−1,1] (T17)。

    与 rho 的区别: rho 先截正部再当注入比例; q 不截断, 用于计算期望 operator 平方失配
        c_i^op = sum_j P_ij ||B_Ai - B_Bj||_F^2 = 2(1 - q_i)
    再由 APT 倒数衰减得到 gate g_i = 1/(1+c_i^op) = 1/(3-2q_i) ∈ [1/5, 1]。
    """
    if P.shape != similarity.shape:
        raise ValueError(f"P/similarity shape mismatch: {P.shape} vs {similarity.shape}")
    return (P * similarity).sum(dim=1).clamp(-1.0, 1.0)


def mismatch_gate_from_q(q: torch.Tensor) -> torch.Tensor:
    """g = 1/(1 + 2(1-q)) = 1/(3-2q); q∈[-1,1] => g∈[1/5,1] (T17.2)。"""
    return 1.0 / (3.0 - 2.0 * q)


def _as_float(x: torch.Tensor) -> torch.Tensor:
    """保留输入浮点精度 (FP64 测试) ; 整型/半精度统一升到 float32。"""
    if not x.is_floating_point():
        return x.float()
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def normalize_score(similarity: torch.Tensor, mode: str = "none"):
    """打分尺度规范化 (§11.4), 使 plan 的温度不依赖 S 的量纲。

    center_scale:  S_c = H S H  (H = I - 11ᵀ/r, 去掉行列常数)
                   s   = ‖S_c‖_F / r
                   送入 Sinkhorn 的是 S_c / s —— 无量纲, 且对
                   ① 行/列常数平移 ② 正比例重标 ③ 独立行列置换 均不变。

    返回 (score_for_plan, scale, degenerate)。
    degenerate=True 表示 s→0 (没有可分辨的对应信号), 此时不做规范化以免除极小值。
    注意: gate/失配量仍必须用**原始** S, 不能用规范化后的分数冒充 [-1,1] 几何内积。
    """
    if mode != "center_scale":
        return similarity, None, False
    r = similarity.shape[0]
    if r < 2:
        return similarity, None, True
    H = torch.eye(r, device=similarity.device, dtype=similarity.dtype) \
        - torch.ones(r, r, device=similarity.device, dtype=similarity.dtype) / r
    S_c = H @ similarity @ H
    s = float(S_c.norm() / r)
    if s < 1e-9:
        return similarity, 0.0, True
    return S_c / s, s, False


def map_module(blkA, blkB, tau=0.15, sinkhorn_iters=50, max_rank=128,
               score_norm="none"):
    """对单个同形模块对求 S/P/双 target/门控量; 形状不兼容返回 None。"""
    UA, SA, VhA = _as_float(blkA["U"]), _as_float(blkA["S"]), _as_float(blkA["Vh"])
    UB, SB, VhB = _as_float(blkB["U"]), _as_float(blkB["S"]), _as_float(blkB["Vh"])
    r = min(SA.numel(), SB.numel(), int(max_rank))
    if r <= 0:
        return None
    UA, SA, VhA = UA[:, :r], SA[:r], VhA[:r, :]
    UB, SB, VhB = UB[:, :r], SB[:r], VhB[:r, :]
    if UA.shape[0] != UB.shape[0] or VhA.shape[1] != VhB.shape[1]:
        return None

    similarity, cost = operator_similarity_and_cost(UA, VhA, UB, VhB)
    score_plan, score_scale, score_degen = normalize_score(similarity, mode=score_norm)
    M = sinkhorn(score_plan, tau=float(tau), iters=int(sinkhorn_iters))
    P = M / M.sum(dim=1, keepdim=True).clamp_min(EPS)

    # ---- 两套 target 原料 ----
    sigma_hat_B = P @ SB            # E0: 运输谱 (先平均 donor 强度)
    t_operator = similarity @ SB    # E1: 算子响应系数 (有符号投影, 不做概率平均)
    h_ot = sigma_hat_B - SA
    h_op = t_operator - SA

    rho = component_reliability(P, similarity)
    # ---- T17: 期望 operator 平方失配与倒数收缩 gate (G6 模块级 / G7 逐组件) ----
    q_raw = operator_compatibility(P, similarity)          # ∈ [−1,1], 保留符号
    c_op = 2.0 * (1.0 - q_raw)                             # = Σ_j P_ij ||B_Ai-B_Bj||_F^2 ∈ [0,4]
    g7 = mismatch_gate_from_q(q_raw)                       # 逐组件, ∈ [1/5,1]
    g6_scalar = float(mismatch_gate_from_q(q_raw.mean()).item())   # 先平均失配再取倒数
    q_mean = float(q_raw.mean().item())

    c_apt = structural_divergence(UA, UB, M, r)
    lambda_apt = 1.0 / (1.0 + c_apt)
    eye = torch.eye(r, device=similarity.device, dtype=similarity.dtype)
    c_apt_identity = structural_divergence(UA, UB, eye, r)
    lambda_apt_identity = 1.0 / (1.0 + c_apt_identity)

    # RAPT-Energy: 被搬运的 donor 谱能量中, 落在兼容 operator pair 上的比例
    e_send = M * (SB.view(1, -1) ** 2)
    denom = e_send.sum().clamp_min(EPS)
    lambda_energy = float(((e_send * similarity.clamp_min(0.0) ** 2).sum() / denom).item())

    avg_cost = float((P * cost).sum(dim=1).mean().item())
    entropy = -(P * P.clamp_min(EPS).log()).sum(dim=1).mean()
    entropy = float((entropy / math.log(max(2, r))).item())
    concentration = float(P.max(dim=1).values.mean().item())
    row_residual = float((M.sum(dim=1) - 1.0).abs().max().item())
    col_residual = float((M.sum(dim=0) - 1.0).abs().max().item())

    return {
        "M": M.cpu(),
        "P": P.cpu(),
        "operator_similarity": similarity.cpu(),
        "operator_cost": cost.cpu(),
        "rho_component": rho.cpu(),
        "q_raw": q_raw.cpu(),
        "q_mean": torch.tensor(q_mean),
        "c_op": c_op.cpu(),
        "g7_mismatch": g7.cpu(),
        "g6_mismatch": torch.tensor(g6_scalar),
        "rho_mean": torch.tensor(float(rho.mean().item())),
        "rho_var": torch.tensor(float(rho.var(unbiased=False).item())),
        "rho_zero_ratio": torch.tensor(float((rho <= EPS).float().mean().item())),
        "sigma_hat_B": sigma_hat_B.cpu(),
        "t_operator": t_operator.cpu(),
        "h_ot": h_ot.cpu(),
        "h_op": h_op.cpu(),
        "S_A": SA.cpu(),
        "S_B": SB.cpu(),
        "rank_A": torch.tensor(r),
        "rank_B": torch.tensor(r),
        "shape_match": torch.tensor(True),
        "c_apt": torch.tensor(c_apt),
        "c_apt_identity": torch.tensor(c_apt_identity),
        "lambda_apt": torch.tensor(lambda_apt),
        "lambda_apt_identity": torch.tensor(lambda_apt_identity),
        "lambda_energy": torch.tensor(lambda_energy),
        "average_operator_cost": torch.tensor(avg_cost),
        "mapping_entropy": torch.tensor(entropy),
        "mapping_concentration": torch.tensor(concentration),
        "sinkhorn_row_residual": torch.tensor(row_residual),
        "sinkhorn_col_residual": torch.tensor(col_residual),
        "score_norm": score_norm,
        "score_scale": (torch.tensor(score_scale) if score_scale is not None
                        else torch.tensor(float("nan"))),
        "score_degenerate": torch.tensor(bool(score_degen)),
    }


def stage2(args: argparse.Namespace) -> None:
    print("\n--- [RAPT v3 Stage-2: Operator Mapping + Dual Target] ---")
    subsA = load_subspaces(args.subs_a)
    subsB = load_subspaces(args.subs_b)
    pairs = pair_modules(subsA, subsB, verbose=bool(args.verbose))
    if not pairs:
        raise RuntimeError("No module pairs found after canonicalization")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_mod: dict = {}
    skipped = []
    agg = {k: [] for k in (
        "rho_mean", "rho_var", "lambda_energy", "cost", "entropy", "row_res", "col_res",
        "h_ot_pos", "h_op_pos", "lambda_apt", "lambda_apt_identity",
        "q_mean", "g6", "g7",
    )}

    iterator = tqdm(pairs, desc="Stage-2 RAPT v3 Mapping", disable=bool(args.no_progress))
    for name, donor_name, canon in iterator:
        blkA = {k: v.float().to(device) for k, v in subsA[name].items() if isinstance(v, torch.Tensor)}
        blkB = {k: v.float().to(device) for k, v in subsB[donor_name].items() if isinstance(v, torch.Tensor)}
        rec = map_module(
            blkA, blkB,
            tau=float(args.tau),
            sinkhorn_iters=int(args.sinkhorn_iters),
            max_rank=int(args.max_rank),
            score_norm=str(getattr(args, 'score_norm', 'none')),
        )
        if rec is None:
            skipped.append((name, donor_name))
            if args.verbose:
                print(f"[RAPTv2][Skip] shape mismatch: {name} <- {donor_name}")
            continue
        rec["donor_module"] = donor_name
        rec["canonical_key"] = canon
        rec["host_fingerprint"] = str(subsA[name].get("fingerprint", ""))
        rec["donor_fingerprint"] = str(subsB[donor_name].get("fingerprint", ""))
        out_mod[name] = rec

        agg["rho_mean"].append(float(rec["rho_mean"]))
        agg["rho_var"].append(float(rec["rho_var"]))
        agg["lambda_energy"].append(float(rec["lambda_energy"]))
        agg["cost"].append(float(rec["average_operator_cost"]))
        agg["entropy"].append(float(rec["mapping_entropy"]))
        agg["row_res"].append(float(rec["sinkhorn_row_residual"]))
        agg["col_res"].append(float(rec["sinkhorn_col_residual"]))
        agg["h_ot_pos"].append(float((rec["h_ot"] > 0).float().mean().item()))
        agg["h_op_pos"].append(float((rec["h_op"] > 0).float().mean().item()))
        agg["lambda_apt"].append(float(rec["lambda_apt"]))
        agg["lambda_apt_identity"].append(float(rec["lambda_apt_identity"]))
        agg["q_mean"].append(float(rec["q_mean"]))
        agg["g6"].append(float(rec["g6_mismatch"]))
        agg["g7"].append(float(rec["g7_mismatch"].mean()))
        if args.verbose:
            print(
                f"[RAPTv2] {name}: rho={float(rec['rho_mean']):.4f} "
                f"h_ot+={float((rec['h_ot'] > 0).float().mean()):.3f} "
                f"h_op+={float((rec['h_op'] > 0).float().mean()):.3f} "
                f"lam_apt={float(rec['lambda_apt']):.4f} r={int(rec['rank_A'])}"
            )

    def _mean(v):
        return sum(v) / len(v) if v else None

    meta = {
        "method": "RAPT-V0.3-OperatorDualTarget",
        "code_version": "rapt_V0_3",
        "schema_version": SCHEMA_VERSION,
        "subs_a": args.subs_a,
        "subs_b": args.subs_b,
        "tau": float(args.tau),
        "sinkhorn_iters": int(args.sinkhorn_iters),
        "score_norm": str(getattr(args, "score_norm", "none")),
        "max_rank": int(args.max_rank),
        "n_pairs": len(pairs),
        "n_modules": len(out_mod),
        "n_shape_skipped": len(skipped),
        "shape_skipped_examples": skipped[:10],
        "rho_mean": _mean(agg["rho_mean"]),
        "q_mean": _mean(agg["q_mean"]),
        "g6_mismatch_mean": _mean(agg["g6"]),
        "g7_mismatch_mean": _mean(agg["g7"]),
        "rho_var_mean": _mean(agg["rho_var"]),
        "lambda_apt_mean": _mean(agg["lambda_apt"]),
        "lambda_apt_identity_mean": _mean(agg["lambda_apt_identity"]),
        "lambda_energy_mean": _mean(agg["lambda_energy"]),
        "average_operator_cost_mean": _mean(agg["cost"]),
        "mapping_entropy_mean": _mean(agg["entropy"]),
        "h_ot_positive_ratio_mean": _mean(agg["h_ot_pos"]),
        "h_op_positive_ratio_mean": _mean(agg["h_op_pos"]),
        "sinkhorn_row_residual_max": max(agg["row_res"]) if agg["row_res"] else None,
        "sinkhorn_col_residual_max": max(agg["col_res"]) if agg["col_res"] else None,
    }
    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save({"modules": out_mod, "meta": meta}, args.save)
    print(f"[Done] RAPT v3 Stage-2 -> {args.save}; mapped={len(out_mod)} skipped={len(skipped)}")
    if out_mod:
        print(
            f"  rho_mean={meta['rho_mean']:.4f}, q_mean={meta['q_mean']:.4f}, "
            f"g6_mean={meta['g6_mismatch_mean']:.4f}, g7_mean={meta['g7_mismatch_mean']:.4f}, "
            f"lam_apt_mean={meta['lambda_apt_mean']:.4f}, "
            f"h_ot+={meta['h_ot_positive_ratio_mean']:.4f}, h_op+={meta['h_op_positive_ratio_mean']:.4f}, "
            f"entropy={meta['mapping_entropy_mean']:.4f}, "
            f"residual(row/col)={meta['sinkhorn_row_residual_max']:.2e}/"
            f"{meta['sinkhorn_col_residual_max']:.2e}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAPT v3 Stage-2 operator correspondence (dual target)")
    ap.add_argument("--subs-a", required=True)
    ap.add_argument("--subs-b", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--sinkhorn-iters", type=int, default=50)
    ap.add_argument("--score-norm", dest="score_norm", default="none",
                    choices=["none", "center_scale"],
                    help="§11.4 打分尺度规范化: center_scale 使 plan 温度不依赖 S 量纲")
    ap.add_argument("--max-rank", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    stage2(parse_args())
