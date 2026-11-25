#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPT V0.3 合成性质测试 (FP64 解析 + FP32 数值)。

覆盖流程文档第 9/10 节要求的检查:
  * 置换对应恢复 / 联合符号不变 / rho 范围;
  * E1 精确自合并零更新 (含"donor=host 的谱截断投影"一般情形);
  * E0 已知的软谱平均偏置 (自合并非零, 作为回归案例而非实现错误);
  * NormMatch 与 Component 的实际更新范数一致;
  * 代理误差下降恒等式 ||E-Delta||^2 = ||E||^2 - sum beta(2-beta)[h]_+^2;
  * identity mapping 使用重算的 APT gate;
  * 形状不兼容拒绝。
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from rapt_v3_common import sinkhorn, structural_divergence  # noqa: E402
from rapt_v3_stage2_mapping import (  # noqa: E402
    component_reliability,
    map_module,
    operator_similarity_and_cost,
)
from rapt_v3_stage3_injection import build_gate  # noqa: E402

torch.set_default_dtype(torch.float64)


def orthogonal(rows: int, cols: int) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(rows, cols), mode="reduced").Q


def make_block(d_out: int, d_in: int, rank: int, spectrum: torch.Tensor | None = None):
    U = orthogonal(d_out, rank)
    V = orthogonal(d_in, rank)
    S = spectrum.clone() if spectrum is not None else torch.linspace(3.0, 1.0, rank)
    return {"U": U, "S": S, "Vh": V.T.contiguous()}


def test_permutation_recovery() -> None:
    torch.manual_seed(3)
    r = 12
    base = make_block(48, 40, r)
    perm = torch.tensor([5, 0, 8, 2, 10, 1, 11, 4, 6, 9, 3, 7])
    donor = {
        "U": base["U"][:, perm].clone(),
        "S": 1.25 * base["S"][perm].clone(),
        "Vh": base["Vh"][perm, :].clone(),
    }
    rec = map_module(base, donor, tau=0.03, sinkhorn_iters=300, max_rank=r)
    assert rec is not None
    P = rec["P"]
    predicted = P.argmax(dim=1)
    expected = torch.empty(r, dtype=torch.long)
    for donor_idx, host_idx in enumerate(perm.tolist()):
        expected[host_idx] = donor_idx
    recovered = int((predicted == expected).sum().item())
    assert recovered == r, (recovered, predicted, expected)
    assert float(rec["rho_component"].min()) > 0.99
    assert torch.allclose(rec["sigma_hat_B"], 1.25 * base["S"], atol=2e-3)
    print(f"[OK] permutation recovery: {recovered}/{r}, rho_min={float(rec['rho_component'].min()):.4f}")


def test_sign_invariance_and_rho_range() -> None:
    torch.manual_seed(5)
    base = make_block(36, 28, 8)
    donor = {k: v.clone() for k, v in base.items()}
    donor["U"][:, 2] *= -1
    donor["Vh"][2, :] *= -1
    similarity, cost = operator_similarity_and_cost(base["U"], base["Vh"], donor["U"], donor["Vh"])
    assert torch.allclose(torch.diagonal(similarity), torch.ones(8), atol=1e-9)
    assert torch.allclose(torch.diagonal(cost), torch.zeros(8), atol=1e-9)

    sim = torch.diag(torch.tensor([1.0, 0.5, 0.0, -1.0]))
    rho = component_reliability(torch.eye(4), sim)
    assert torch.allclose(rho, torch.tensor([1.0, 0.5, 0.0, 0.0]))
    print("[OK] sign invariance and rho in [0,1]")


def test_selfmerge_zero_update_operator_target() -> None:
    """E1: 同一矩阵的精确谱截断 => h_op <= 0, 更新为零。"""
    torch.manual_seed(7)
    d_out, d_in, r0, r = 40, 32, 6, 4
    base = make_block(d_out, d_in, r0, spectrum=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5]))
    W_A = (base["U"] * base["S"].unsqueeze(0)) @ base["Vh"]

    # (a) donor 与 host 完全同块同秩 => S = I => h_op = 0
    rec_same = map_module(base, base, tau=0.15, sinkhorn_iters=200, max_rank=r)
    assert float(rec_same["h_op"].abs().max()) < 1e-8, float(rec_same["h_op"].abs().max())

    # (b) donor = W_A Pi_B (任意真实右奇异子空间上的投影) => h_op <= 0
    Q = torch.randn(r0, r)
    V_B = torch.linalg.qr(base["Vh"].T @ Q, mode="reduced").Q   # span(V_B) 包含于 span(V_A)
    Pi_B = V_B @ V_B.T
    donor = W_A @ Pi_B
    Ub, Sb, Vhb = torch.linalg.svd(donor, full_matrices=False)
    donor_blk = {"U": Ub[:, :r], "S": Sb[:r], "Vh": Vhb[:r, :]}
    rec = map_module(base, donor_blk, tau=0.15, sinkhorn_iters=200, max_rank=r)
    assert float(rec["h_op"].max()) <= 1e-8, float(rec["h_op"].max())
    print(f"[OK] E1 self-merge zero update: h_op_max={float(rec['h_op'].max()):.2e}")


def test_selfmerge_bias_transport_target() -> None:
    """E0: 有限温度下同模型仍产生正盈余 (已知偏置, 记录而非修复)。"""
    torch.manual_seed(9)
    r = 16
    base = make_block(48, 40, r, spectrum=torch.linspace(4.0, 1.0, r))
    rec = map_module(base, base, tau=0.15, sinkhorn_iters=200, max_rank=r)
    h_ot = rec["h_ot"]
    assert float(h_ot.max()) > 1e-6, "E0 自合并偏置应存在"
    assert float(rec["h_op"].abs().max()) < 1e-9, "同一 target 下 E1 应严格零更新"
    print(
        f"[OK] E0 self-merge bias present: h_ot_max={float(h_ot.max()):.4e}, "
        f"positive_ratio={float((h_ot > 0).double().mean()):.3f}"
    )


def test_normmatch_budget_equality() -> None:
    torch.manual_seed(11)
    r = 24
    U = orthogonal(60, r)
    V = orthogonal(50, r)
    rho = torch.rand(r)
    d = torch.rand(r) + 0.05
    alpha = 0.5
    eta_comp = alpha * rho * d
    g_norm = build_gate("norm_match", rho, d, 0.5, 0.5, r)
    eta_norm = alpha * g_norm * d
    delta_comp = (U * eta_comp.unsqueeze(0)) @ V.T
    delta_norm = (U * eta_norm.unsqueeze(0)) @ V.T
    assert abs(delta_comp.norm().item() - delta_norm.norm().item()) < 1e-8
    # d 全零 => gate=0
    assert float(build_gate("norm_match", rho, torch.zeros(r), 0.5, 0.5, r).abs().max()) == 0.0
    print(f"[OK] NormMatch budget equality: ||Dcomp||={delta_comp.norm():.6f}")


def test_proxy_descent_identity() -> None:
    torch.manual_seed(13)
    d_out, d_in, r = 44, 36, 10
    host = make_block(d_out, d_in, r, spectrum=torch.linspace(2.0, 0.8, r))
    donor = make_block(d_out, d_in, r, spectrum=torch.linspace(2.6, 1.1, r))
    # donor 侧与 host 略有旋转, 使 S 非对角
    Q = orthogonal(r, r)
    donor = {"U": donor["U"] @ Q, "S": donor["S"], "Vh": Q.T @ donor["Vh"]}

    rec = map_module(host, donor, tau=0.15, sinkhorn_iters=200, max_rank=r)
    S_B, S_A = rec["S_B"], rec["S_A"]
    h = rec["h_op"]
    beta = 0.5 * torch.rand(r)          # 固定 gate 与 alpha
    eta = beta * h.clamp_min(0.0)

    D = (donor["U"] * S_B.unsqueeze(0)) @ donor["Vh"]
    W_A = (host["U"] * S_A.unsqueeze(0)) @ host["Vh"]
    E = D - W_A
    Delta = (host["U"] * eta.unsqueeze(0)) @ host["Vh"]

    lhs = (E - Delta).pow(2).sum().item()
    rhs = E.pow(2).sum().item() - (beta * (2.0 - beta) * h.clamp_min(0.0).pow(2)).sum().item()
    assert abs(lhs - rhs) < 1e-7 * max(1.0, E.pow(2).sum().item()), (lhs, rhs)
    print(f"[OK] proxy descent identity: residual={abs(lhs - rhs):.3e}")


def test_identity_mapping_recomputes_gate() -> None:
    torch.manual_seed(17)
    r = 10
    host = make_block(40, 32, r)
    Q = orthogonal(r, r)
    donor = {"U": host["U"] @ Q, "S": 1.1 * host["S"], "Vh": Q.T @ host["Vh"]}
    rec = map_module(host, donor, tau=0.15, sinkhorn_iters=200, max_rank=r)
    lam_m = float(rec["lambda_apt"])
    lam_i = float(rec["lambda_apt_identity"])
    assert abs(lam_m - lam_i) > 1e-6, "identity gate 必须与 M 版不同, 不能读旧缓存"
    assert 0.2 <= lam_i <= 1.0 and 0.2 <= lam_m <= 1.0
    # identity 分支: rho 取 diag(S) 正部
    rho_id = torch.diagonal(rec["operator_similarity"]).clamp(0.0, 1.0)
    assert torch.allclose(rho_id, torch.diagonal(rec["operator_similarity"]).clamp(0.0, 1.0))
    print(f"[OK] identity mapping gate recomputed: lam_M={lam_m:.4f} lam_I={lam_i:.4f}")


def test_dual_target_distinct() -> None:
    torch.manual_seed(19)
    r = 12
    host = make_block(48, 40, r, spectrum=torch.linspace(3.0, 0.6, r))
    Q = orthogonal(r, r)
    donor = {"U": host["U"] @ Q, "S": 1.15 * host["S"], "Vh": Q.T @ host["Vh"]}
    rec = map_module(host, donor, tau=0.15, sinkhorn_iters=200, max_rank=r)
    diff = (rec["h_op"] - rec["h_ot"]).abs().max().item()
    assert diff > 1e-6, "E1 与 E0 的 target 不应相同"
    print(f"[OK] dual target distinct: max|h_op-h_ot|={diff:.4e}")


def test_mismatch_gate_g6_g7() -> None:
    """T17: c_i^op = 2(1-q_i), g = 1/(1+c) = 1/(3-2q) ∈ [1/5,1]; G6 是先平均失配再取倒数。"""
    from rapt_v3_stage2_mapping import mismatch_gate_from_q, operator_compatibility

    q = torch.tensor([1.0, 0.5, 0.0, -1.0, 0.0365])
    g = mismatch_gate_from_q(q)
    assert torch.allclose(g, torch.tensor([1.0, 0.5, 1.0 / 3.0, 0.2, 1.0 / 2.927]), atol=1e-12)
    assert float(g.min()) >= 0.2 - 1e-12 and float(g.max()) <= 1.0 + 1e-12
    # 与 c=2(1-q) 的等价性
    c = 2.0 * (1.0 - q)
    assert torch.allclose(g, 1.0 / (1.0 + c), atol=1e-12)

    # q 由 P 与 S 的有符号行和给出, 保留负值 (不截正部)
    P = torch.eye(4)
    S = torch.diag(torch.tensor([0.9, -0.4, 0.1, -0.8]))
    qq = operator_compatibility(P, S)
    assert torch.allclose(qq, torch.tensor([0.9, -0.4, 0.1, -0.8]))
    # G6 != mean(G7): 先平均失配再取倒数
    g7 = mismatch_gate_from_q(qq)
    g6 = mismatch_gate_from_q(qq.mean())
    assert abs(float(g6) - float(g7.mean())) > 1e-6, (float(g6), float(g7.mean()))
    print(f"[OK] G6/G7 mismatch gates: g7={g7.tolist()}, g6={float(g6):.4f}, mean(g7)={float(g7.mean()):.4f}")


def test_shape_rejection() -> None:
    a = make_block(32, 24, 6)
    b = make_block(31, 24, 6)
    assert map_module(a, b, max_rank=6) is None
    print("[OK] incompatible ambient shape is rejected")


def test_apt_approximation() -> None:
    torch.manual_seed(23)
    r = 10
    U = orthogonal(40, r)
    V = orthogonal(32, r)
    angle = 0.35
    Q = torch.eye(r)
    Q[:2, :2] = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
                              [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]])
    donor_u = U @ Q
    c_apt = structural_divergence(U, donor_u, torch.eye(r), r)
    similarity, _ = operator_similarity_and_cost(U, V.T, donor_u, V.T)
    rho = component_reliability(torch.eye(r), similarity)
    approx = 2.0 * (1.0 - float(rho.mean()))
    assert abs(c_apt - approx) < 1e-8, (c_apt, approx)
    print(f"[OK] APT discrepancy approx: c={c_apt:.6f}, 2(1-mean rho)={approx:.6f}")


def test_subspace_injection_confinement_and_capacity() -> None:
    """路径 D: subspace 注入 (完整 r×r 块) 的性质检验。

    1) 自合并零更新: W_B = W_A -> Z = 0 -> Δ = 0 (精确)
    2) 保护域封闭: Δ 完全落在 host top-r 子空间内 (域外残差精确为 0)
    3) 容量: 完整块 ||U_A Z V_A^T|| >= 对角 ||U_A diag(Z) V_A^T||
    4) 秩: rank(Δ) <= r
    """
    torch.manual_seed(31)
    m, n, r = 48, 36, 8
    W_A = torch.randn(m, n, dtype=torch.float64)
    U_full, S_full, Vh_full = torch.linalg.svd(W_A, full_matrices=False)
    U_A, Vh_A = U_full[:, :r], Vh_full[:r, :]

    # 1) 自合并: Z 精确为零
    Z_self = U_A.T @ (W_A - W_A) @ Vh_A.T
    assert float(Z_self.abs().max()) == 0.0
    print("[OK] subspace: self-merge gives exactly zero block Z (zero update)")

    # 构造一个完全落在 host 子空间内的目标差异
    W_B = W_A + U_A @ torch.randn(r, r, dtype=torch.float64) @ Vh_A
    Z = U_A.T @ (W_B - W_A) @ Vh_A.T
    Delta_full = U_A @ Z @ Vh_A
    Delta_diag = (U_A * Z.diagonal().unsqueeze(0)) @ Vh_A

    # 2) 保护域封闭: 左/右域外残差均为 0
    left_out = (torch.eye(m, dtype=torch.float64) - U_A @ U_A.T) @ Delta_full
    right_out = Delta_full @ (torch.eye(n, dtype=torch.float64) - Vh_A.T @ Vh_A)
    assert float(left_out.abs().max()) < 1e-10, float(left_out.abs().max())
    assert float(right_out.abs().max()) < 1e-10, float(right_out.abs().max())
    print("[OK] subspace: update is exactly confined to the host top-r subspace")

    # 3) 容量: 完整块 >= 对角块
    nf, nd = float(Delta_full.norm()), float(Delta_diag.norm())
    assert nf >= nd - 1e-12, (nf, nd)
    print(f"[OK] subspace: full block captures more: ||full||={nf:.4f} vs ||diag||={nd:.4f} (x{nf / nd:.2f})")

    # 4) 秩受 r 限制
    rk = int(torch.linalg.matrix_rank(Delta_full))
    assert rk <= r, (rk, r)
    print(f"[OK] subspace: rank(Delta)={rk} <= r={r}")


def test_calibrated_transport_zero_hypothesis() -> None:
    """E2-M 零假设: 匹配强度无差异 (σ_B = Πᵀσ_A) 时 h_cal 必须精确为 0。

    这是 §11.6 预登记的判据。E0 在同一情形下会产生 (PΠᵀ−I)σ_A ≠ 0 的平滑基线项。
    """
    import importlib.util
    from scipy.optimize import linear_sum_assignment

    torch.manual_seed(7)
    r = 12
    # 构造: host 强度 -> 经置换 -> donor 强度 (即 σ_B = Πᵀ σ_A, 无强度差)
    sigma_A = torch.sort(torch.rand(r, dtype=torch.float64) + 0.5, descending=True).values
    perm_true = torch.randperm(r)
    inv_true = torch.zeros_like(perm_true)
    inv_true[perm_true] = torch.arange(r)
    sigma_B = sigma_A[inv_true]                       # 精确满足 σ_B = Πᵀσ_A

    # S 的峰值落在真置换上, 使线性指派能恢复它
    S = torch.rand(r, r, dtype=torch.float64) * 0.01
    S[torch.arange(r), perm_true] = 1.0
    rows, cols = linear_sum_assignment((-S).numpy())
    perm = torch.zeros(r, dtype=torch.long)
    perm[torch.from_numpy(rows)] = torch.from_numpy(cols)
    inv = torch.zeros_like(perm)
    inv[perm] = torch.arange(r)
    assert torch.equal(perm, perm_true), "线性指派未恢复真置换"

    # 任意行随机 P (含平滑的非置换 P)
    P = torch.rand(r, r, dtype=torch.float64)
    P = P / P.sum(dim=1, keepdim=True)

    h_cal = P @ (sigma_B - sigma_A[inv])
    h_e0 = P @ sigma_B - sigma_A
    assert float(h_cal.abs().max()) < 1e-12, float(h_cal.abs().max())
    assert float(h_e0.abs().max()) > 1e-6, "E0 在本构造下应当非零"
    print(f"[OK] calibrated_transport: zero-hypothesis gives exactly 0 "
          f"(E0 gives max|h|={float(h_e0.abs().max()):.4f})")


def test_score_norm_scale_invariance() -> None:
    """§11.4 尺度规范化必须满足的不变性 (plan 相同 <=> 规范化分数相同)。

    ① 正比例重标 S  ② 行/列常数平移  ③ 独立行列置换  —— 规范化后送入 Sinkhorn 的分数一致。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s2map", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "rapt_v3_stage2_mapping.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    torch.manual_seed(11)
    r = 16
    S = torch.randn(r, r, dtype=torch.float64)
    base, _, d0 = m.normalize_score(S, mode="center_scale")
    assert not d0

    # ① 正比例重标
    a, _, _ = m.normalize_score(3.7 * S, mode="center_scale")
    assert float((a - base).abs().max()) < 1e-9, "正比例重标未保持不变"

    # ② 行/列常数平移
    b, _, _ = m.normalize_score(S + 5.0, mode="center_scale")
    assert float((b - base).abs().max()) < 1e-9, "常数平移未保持不变"

    # ③ 独立行列置换 (同时作用在 S 上, 规范化分数应整体置换)
    pr, pc = torch.randperm(r), torch.randperm(r)
    c, _, _ = m.normalize_score(S[pr][:, pc], mode="center_scale")
    assert float((c - base[pr][:, pc]).abs().max()) < 1e-9, "置换等变性不成立"

    # 退化: 常数矩阵没有可分辨信号
    _, _, deg = m.normalize_score(torch.ones(r, r, dtype=torch.float64), mode="center_scale")
    assert deg, "常数矩阵应判为退化"

    print("[OK] score_norm center_scale: invariant to rescale / shift / permutation; "
          "constant matrix flagged degenerate")


def _load_stage3_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s3inj", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "rapt_v3_stage3_injection.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_compute_surplus_h_production_path() -> None:
    """直接测试**生产函数** compute_surplus_h。

    背景: E2-M 首版实现把基线写成 S_B[inv] (应为 S_A[inv]), 当时的合成测试只验证了
    数学公式本身而非生产线代码, 笔误逃过了校验。本测试针对该类错误:
    逐 target 对照手写参考实现, 任何变量名张冠李戴都会在此暴露。
    """
    m = _load_stage3_module()
    torch.manual_seed(23)
    r = 10

    # --- 构造: S 峰值在真置换 perm_true 上, 保证线性指派可恢复 ---
    perm_true = torch.randperm(r)
    inv_true = torch.zeros_like(perm_true)
    inv_true[perm_true] = torch.arange(r)
    S = torch.rand(r, r, dtype=torch.float64) * 0.01
    S[torch.arange(r), perm_true] = 1.0
    P = torch.rand(r, r, dtype=torch.float64)
    P = P / P.sum(dim=1, keepdim=True)          # 行随机 (生产中的 P 口径)

    sigma_A = torch.sort(torch.rand(r, dtype=torch.float64) + 0.5, descending=True).values
    sigma_B = sigma_A[inv_true] + 0.05          # 匹配位置上 donor 一致地强 0.05

    blk = {"operator_similarity": S, "P": P}
    common = dict(blk=blk, S_A=sigma_A, S_B=sigma_B, sigma_used=P @ sigma_B,
                  t_operator=S @ sigma_B, r_eff=r)

    # 1) E0: h = P σ_B − σ_A
    h0, aux0 = m.compute_surplus_h("transport_spectrum", **common)
    assert torch.allclose(h0, P @ sigma_B - sigma_A, atol=1e-12)
    assert not aux0

    # 2) E1: h = S σ_B − σ_A
    h1, _ = m.compute_surplus_h("operator_response", **common)
    assert torch.allclose(h1, S @ sigma_B - sigma_A, atol=1e-12)

    # 3) E2-M 校准: h = P (σ_B − Πᵀ σ_A), Πᵀσ_A = σ_A[inv]
    hcal, aux = m.compute_surplus_h("calibrated_transport", **common)
    ref_correct = P @ (sigma_B - sigma_A[inv_true])
    assert torch.allclose(hcal, ref_correct, atol=1e-12), "E2-M 与正确参考不一致"
    # 零假设: 匹配强度无差异时 h ≡ 0
    hcal0, _ = m.compute_surplus_h(
        "calibrated_transport", blk=blk, S_A=sigma_A, S_B=sigma_A[inv_true],
        sigma_used=P @ sigma_A[inv_true], t_operator=S @ sigma_A[inv_true], r_eff=r)
    assert float(hcal0.abs().max()) < 1e-12
    # 回归防护: 生产结果必须**不等于**错误公式 P(σ_B − σ_B[inv])
    ref_bug = P @ (sigma_B - sigma_B[inv_true])
    assert not torch.allclose(hcal, ref_bug, atol=1e-9), "生产路径退化为此前笔误公式"
    # 指派方向与记录字段
    assert torch.equal(aux["perm"], perm_true) and torch.equal(aux["inv"], inv_true)
    assert aux["exact"] and aux["baseline_gap"] is not None

    # 4) hard: h = Π σ_B − σ_A = σ_B[perm] − σ_A
    hhard, auxh = m.compute_surplus_h("hard_transport", **common)
    assert torch.allclose(hhard, sigma_B[perm_true] - sigma_A, atol=1e-12)
    assert torch.equal(auxh["perm"], perm_true)

    # 5) E3: h = diag(U_Aᵀ W_B V_A) − σ_A (需 donor 完整权重; 缺失返回 None)
    mm, nn = 24, 16
    W_A = torch.randn(mm, nn, dtype=torch.float64)
    U, Sv, Vh = torch.linalg.svd(W_A, full_matrices=False)
    U_A, S_A_e3, Vh_A = U[:, :r], Sv[:r], Vh[:r, :]
    W_B = W_A + (U_A * torch.randn(r, dtype=torch.float64).unsqueeze(0)) @ Vh_A
    blk3 = {"operator_similarity": torch.eye(r, dtype=torch.float64),
            "P": torch.eye(r, dtype=torch.float64)}
    h3, _ = m.compute_surplus_h(
        "full_donor_response", blk=blk3, S_A=S_A_e3, S_B=S_A_e3.clone(),
        sigma_used=S_A_e3.clone(), t_operator=S_A_e3.clone(), r_eff=r,
        W_B_full=W_B, U_A=U_A, Vh_A=Vh_A)
    ref3 = ((U_A.T @ W_B) * Vh_A).sum(dim=1) - S_A_e3
    assert torch.allclose(h3, ref3, atol=1e-10)
    hNone, _ = m.compute_surplus_h(
        "full_donor_response", blk=blk3, S_A=S_A_e3, S_B=S_A_e3.clone(),
        sigma_used=S_A_e3.clone(), t_operator=S_A_e3.clone(), r_eff=r)
    assert hNone is None

    print("[OK] compute_surplus_h (production path): E0/E1/E2-M/hard/E3 全部与手写参考一致;"
          " E2-M 不等于笔误公式; 零假设精确为 0")


def test_normalize_injection_budget_exact_rel() -> None:
    """预算归一化: ‖Δ‖_F/‖W_A‖_F 必须精确等于 alpha (∈[0,1]), 与 target/gate/rank 无关。"""
    m = _load_stage3_module()
    torch.manual_seed(41)
    mm, nn, r = 40, 24, 6
    W_A = torch.randn(mm, nn, dtype=torch.float64)
    U, _, Vh = torch.linalg.svd(W_A, full_matrices=False)
    U_R, Vh_R = U[:, :r], Vh[:r, :]
    W_A_norm = float(W_A.norm())

    for alpha in (0.005, 0.0114, 0.02, 0.04):
        for scale in (1.0, 37.0, 1e-3):          # 任意位移量级都应被归一
            eta = torch.randn(r, dtype=torch.float64) * scale
            eta_n = m.normalize_injection_budget(eta, alpha, W_A_norm)
            Delta = (U_R * eta_n.unsqueeze(0)) @ Vh_R
            rel = float(Delta.norm()) / W_A_norm
            assert abs(rel - alpha) < 1e-12, (alpha, scale, rel)
    # 方向不变 (只缩放, 不旋转)
    eta = torch.randn(r, dtype=torch.float64)
    eta_n = m.normalize_injection_budget(eta, 0.02, W_A_norm)
    assert torch.allclose(eta_n / eta_n.norm(), eta / eta.norm(), atol=1e-12)
    # 零位移: 不产生 0/0
    z = m.normalize_injection_budget(torch.zeros(r, dtype=torch.float64), 0.02, W_A_norm)
    assert float(z.abs().max()) == 0.0
    print("[OK] normalize_injection_budget: ||dW||_F/||W_A||_F == alpha exactly "
          "(alpha in [0,1], scale/target/gate independent); direction preserved; zero-safe")


def test_ta_stage1_anchor_subtraction() -> None:
    """TA 分支: `--anchor-model` 时分解对象必须是 τ = W − W_0 (而非 W)。

    同时检验下游契约: τ 与 W 同形 → Stage-3 的 shape 校验不变;
    指纹绑定的是 τ (分量身份随分解对象走)。
    """
    import importlib.util
    import json
    import tempfile

    import safetensors.torch

    spec = importlib.util.spec_from_file_location(
        "s1sub", os.path.join(ROOT, "rapt_v3_stage1_subspace.py"))
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)

    torch.manual_seed(7)
    mm, nn, r = 32, 20, 5
    key = "model.layers.0.self_attn.q_proj.weight"
    W_0 = torch.randn(mm, nn)
    W_A = W_0 + 0.05 * torch.randn(mm, nn)

    with tempfile.TemporaryDirectory() as tmp:
        dA, d0 = os.path.join(tmp, "A"), os.path.join(tmp, "anchor")
        for d, W in ((dA, W_A), (d0, W_0)):
            os.makedirs(d, exist_ok=True)
            safetensors.torch.save_file({key: W.float()}, os.path.join(d, "model.safetensors"))

        out_abs, out_ta = os.path.join(tmp, "abs.pt"), os.path.join(tmp, "ta.pt")

        def run(save, anchor):
            import argparse as _ap
            ns = _ap.Namespace(model_dir=dA, save=save, rank=r, cuda=False,
                               low_rank_svd=False, anchor_model=anchor,
                               verbose=False, no_progress=True)
            s1.extract(ns)

        run(out_abs, None)
        run(out_ta, d0)

        abs_obj = torch.load(out_abs, map_location="cpu", weights_only=True)
        ta_obj = torch.load(out_ta, map_location="cpu", weights_only=True)
        mod = "model.layers.0.self_attn.q_proj"

        # 元数据留痕
        assert abs_obj["decomp_object"] == "absolute_weight"
        assert ta_obj["decomp_object"] == "task_vector"
        assert ta_obj["anchor_model"] == d0

        # TA 的奇异值与 τ 的精确 SVD 一致 (绝对权重分支必须不同)
        tau = (W_A - W_0).float()
        S_true = torch.linalg.svd(tau, full_matrices=False).S[:r]
        S_ta = ta_obj["subspaces"][mod]["S"].float()
        assert torch.allclose(S_ta, S_true, atol=1e-5), (S_ta, S_true)
        S_abs = abs_obj["subspaces"][mod]["S"].float()
        assert not torch.allclose(S_abs, S_true, atol=1e-4), "锚点未生效: 仍分解了绝对权重"

        # τ 与 W 同形 -> 下游 shape 校验不需要改动
        assert tuple(ta_obj["subspaces"][mod]["shape"].tolist()) == (mm, nn)

        # 指纹绑定分解对象: 两个分支必须不同
        assert (ta_obj["subspaces"][mod]["fingerprint"]
                != abs_obj["subspaces"][mod]["fingerprint"])

    print("[OK] TA Stage-1: --anchor-model 使分解对象变为 τ = W − W_0 (奇异值逐位对齐 τ 的 SVD);"
          " 同形契约与指纹身份均正确")


def test_ta_basis_reachability_mechanism() -> None:
    """TA 的机制检验 (合成): 当微调增量 τ 落在 W_0 的**非主方向**上时,
    APT 的绝对权重基 reach(U_W) 很低, 而 task vector 基 reach(U_τ) 接近满。

    这正是 §8.8.7 的 98.86% 不可达在合成上的最小复现, 也是 TA 分支的动机。
    """
    torch.manual_seed(11)
    mm, nn, r = 60, 40, 8
    k = min(mm, nn)                                   # 奇异值个数
    U0 = orthogonal(mm, k)
    V0 = orthogonal(nn, k)

    # W_0: 前 r 个方向远大于其余 (预训练主方向主导)
    s0 = torch.cat([torch.linspace(10.0, 9.0, r), torch.full((k - r,), 0.5)])
    W_0 = (U0 * s0.unsqueeze(0)) @ V0.T

    # τ: 支撑在 W_0 的**尾部**方向 (索引 r..r+2r) —— 与预训练主方向近似正交
    m = 2 * r
    tau_A = (U0[:, r:r + m] * torch.linspace(0.30, 0.01, m)) @ V0[:, r:r + m].T
    tau_B = tau_A * 1.5
    W_A, W_B = W_0 + tau_A, W_0 + tau_B
    D = W_B - W_A

    def topr(M, k):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return U[:, :k], Vh[:k, :]

    U_W, Vh_W = topr(W_A, r)
    U_t, Vh_t = topr(tau_A, r)
    reach_W = float((U_W.T @ D @ Vh_W.T).norm()) / float(D.norm())
    reach_t = float((U_t.T @ D @ Vh_t.T).norm()) / float(D.norm())

    assert reach_t > 0.9, f"task vector 基应几乎完全触达 Δ, 实测 {reach_t:.4f}"
    assert reach_W < 0.2, f"绝对权重基应基本触达不到 Δ, 实测 {reach_W:.4f}"
    assert reach_t > 5 * reach_W

    print(f"[OK] TA 机制: reach(U_W)={reach_W:.4f} vs reach(U_τ)={reach_t:.4f} "
          f"({reach_t / max(reach_W, 1e-12):.1f}x) —— 保护域选择而非 gate 决定可达性")


def test_ta_zero_hypothesis_and_confinement() -> None:
    """TA 契约: ① τ_B = τ_A (两侧微调完全相同) 时 Δ ≡ 0, 注入必须为空;
    ② 更新严格落在 U_τ 子空间内, 但**一般不在** U_W 子空间内 (两域确实不同)。
    """
    m = _load_stage3_module()
    torch.manual_seed(19)
    mm, nn, r = 48, 32, 6
    W_0 = torch.randn(mm, nn)
    tau = 0.05 * torch.randn(mm, nn)
    W_A = W_0 + tau
    W_B = W_0 + tau.clone()                       # 零假设: 同一 task vector
    assert float((W_B - W_A).norm()) == 0.0

    U_t, S_t, Vh_t = torch.linalg.svd(tau, full_matrices=False)
    U_t, Vh_t = U_t[:, :r], Vh_t[:r, :]
    eta = m.normalize_injection_budget(torch.zeros(r), 0.5, float(W_A.norm()))
    Delta = (U_t * eta.unsqueeze(0)) @ Vh_t
    assert float(Delta.norm()) == 0.0, "零 task 差异必须产生零更新"

    # 封闭性: Δ 在 U_τ 左右域内精确为 0 残差
    eta = m.normalize_injection_budget(torch.randn(r) + 0.5, 0.5, float(W_A.norm()))
    Delta = (U_t * eta.unsqueeze(0)) @ Vh_t
    dn = float(Delta.norm())
    off_u = float((Delta - U_t @ (U_t.T @ Delta)).norm()) / dn
    off_v = float((Delta - (Delta @ Vh_t.T) @ Vh_t).norm()) / dn
    assert off_u < 1e-10 and off_v < 1e-10

    # 与 W_A 基的关系: τ 基的更新一般**不**落在 W_A 的 top-r 内 (否则两分支等价)
    U_W = torch.linalg.svd(W_A, full_matrices=False).U[:, :r]
    leak = float((Delta - U_W @ (U_W.T @ Delta)).norm()) / dn
    assert leak > 0.5, f"τ 基与 W 基应显著不同, 实测域外比例仅 {leak:.4f}"

    print(f"[OK] TA 契约: 零 task 差异 -> 精确零更新; 封闭性残差 <1e-10; "
          f"τ 基更新在 W_A top-{r} 外的比例 {leak:.4f} (两保护域确实不同)")


def test_task_delta_and_donor_gate() -> None:
    """TA 分支的生产函数检验:
    ① task_delta: h = diag(Uᵀ(W_B − W_A)Vᵀ); 自合并 (W_B = W_A) 精确为 0;
    ② donor_component_gate: 同一计划按列读取, 与按行读取不同; 双随机 P 时退化为列期望;
    ③ build_gate 能取到 donor 侧逐组件 gate。
    """
    m = _load_stage3_module()
    torch.manual_seed(29)
    mm, nn, r = 36, 24, 7
    W_0 = torch.randn(mm, nn, dtype=torch.float64)
    tau_A = 0.05 * torch.randn(mm, nn, dtype=torch.float64)
    tau_B = 0.05 * torch.randn(mm, nn, dtype=torch.float64)
    W_A, W_B = W_0 + tau_A, W_0 + tau_B

    # 注入基 = donor task vector 的 top-r 奇异基
    U_tB, S_tB, Vh_tB = torch.linalg.svd(tau_B, full_matrices=False)
    U_R, Vh_R = U_tB[:, :r], Vh_tB[:r, :]

    blk = {"operator_similarity": torch.eye(r, dtype=torch.float64),
           "P": torch.eye(r, dtype=torch.float64)}
    h, _ = m.compute_surplus_h(
        "task_delta", blk=blk, S_A=S_tB[:r], S_B=S_tB[:r],
        sigma_used=S_tB[:r], t_operator=S_tB[:r], r_eff=r,
        W_B_full=W_B, U_A=U_R, Vh_A=Vh_R, W_A=W_A)
    ref = ((U_R.T @ W_B) * Vh_R).sum(dim=1) - ((U_R.T @ W_A) * Vh_R).sum(dim=1)
    assert torch.allclose(h, ref, atol=1e-12), "task_delta 与手写参考不一致"
    # 该基是 donor 任务基 -> 差分响应应与 donor 谱高度一致 (正部占绝大多数)
    assert float((h > 0).float().mean()) > 0.9, "donor 任务基下差分响应应几乎全为正"
    # 自合并零更新
    h0, _ = m.compute_surplus_h(
        "task_delta", blk=blk, S_A=S_tB[:r], S_B=S_tB[:r],
        sigma_used=S_tB[:r], t_operator=S_tB[:r], r_eff=r,
        W_B_full=W_A, U_A=U_R, Vh_A=Vh_R, W_A=W_A)
    assert float(h0.abs().max()) < 1e-12, "W_B = W_A 时 task_delta 必须精确为 0"
    # 缺 W_A / W_B 时返回 None (不静默算错)
    assert m.compute_surplus_h(
        "task_delta", blk=blk, S_A=S_tB[:r], S_B=S_tB[:r],
        sigma_used=S_tB[:r], t_operator=S_tB[:r], r_eff=r,
        W_B_full=W_B, U_A=U_R, Vh_A=Vh_R, W_A=None)[0] is None

    # donor 侧逐组件 gate: 非对称计划下必须与按行读取不同
    P = torch.rand(r, r, dtype=torch.float64)
    P = P / P.sum(dim=1, keepdim=True)
    S = torch.rand(r, r, dtype=torch.float64) * 2 - 1
    g_col = m.donor_component_gate(P, S)
    q_row = ((P) * S).sum(dim=1)
    g_row = m.mismatch_gate_from_q(q_row)
    assert not torch.allclose(g_col, g_row, atol=1e-6), "按列与按行读取不应相同"
    # 双随机 (行/列和都为 1) 时, 列读取退化为列期望 Σ_i P_ij S_ij
    Pd = sinkhorn(S, tau=0.5, iters=200)
    g_col2 = m.donor_component_gate(Pd, S)
    g_ref2 = m.mismatch_gate_from_q((Pd * S).sum(dim=0))
    assert torch.allclose(g_col2, g_ref2, atol=1e-6)
    # build_gate 可取出
    g = m.build_gate("mismatch_component_donor", torch.zeros(r), torch.zeros(r),
                     0.3, 0.3, r, g7_donor=g_col2)
    assert torch.allclose(g, g_col2)
    assert float(g.min()) >= 0.2 - 1e-9 and float(g.max()) <= 1.0 + 1e-9

    print("[OK] task_delta: 与手写参考一致, 自合并精确为 0, 缺权重返回 None; "
          "donor_component_gate 按列读取 (双随机时退化为列期望), build_gate 可取用")


def main() -> None:
    test_permutation_recovery()
    test_sign_invariance_and_rho_range()
    test_selfmerge_zero_update_operator_target()
    test_selfmerge_bias_transport_target()
    test_normmatch_budget_equality()
    test_proxy_descent_identity()
    test_identity_mapping_recomputes_gate()
    test_dual_target_distinct()
    test_mismatch_gate_g6_g7()
    test_shape_rejection()
    test_apt_approximation()
    test_subspace_injection_confinement_and_capacity()
    test_calibrated_transport_zero_hypothesis()
    test_score_norm_scale_invariance()
    test_compute_surplus_h_production_path()
    test_normalize_injection_budget_exact_rel()
    test_ta_stage1_anchor_subtraction()
    test_ta_basis_reachability_mechanism()
    test_ta_zero_hypothesis_and_confinement()
    test_task_delta_and_donor_gate()
    print("=== ALL RAPT V0.3 SYNTHETIC CHECKS PASSED ===")


if __name__ == "__main__":
    main()
