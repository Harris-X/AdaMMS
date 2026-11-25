#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TA_hyb —— 当前最优 RAPT V0.3 配置的独立最小实现 (从 rapt_v3_stage3_injection.py
的 20+ 个开关 / 3 阶段缓存文件中剥离出来, 只保留真正生效的那条路径)。

对应原多分支脚本里的确切配置 (SUM3=216.731, 见 §21.6 边缘效应分析,
已证明 rank/target/gate/alpha/interp_alpha 这 5 个轴全部在最优点上):
    --rank 128 --alpha 0.5 --mode hybrid --interp-alpha 0.1
    --target task_delta --gate apt --identity-mapping
    --base-subs <donor 任务向量 SVD, anchor=Qwen2-7B>

关键简化 (代码验证, 非猜测): identity_mapping=True 时,
    lambda_apt_identity = 1/(1+c), c = structural_divergence(U_A_tv, U_B_tv, I, r)
是纯闭式计算 (只需两个 task-vector 基的 U 矩阵内积 + QR 正交化), **不需要 Sinkhorn/OT**;
gate=apt 又只用这一个标量。于是原来的 "Stage-1 SVD 缓存 -> Stage-2 OT 映射缓存
-> Stage-3 读缓存注入" 三段式管线, 在这一个配置下可以合并成单文件、无缓存依赖的
一次性计算:

    对每个线性模块 (host W_A, donor W_B, 预训练锚点 W_0):
        τ_A = W_A − W_0                          host 任务向量
        τ_B = W_B − W_0                          donor 任务向量
        (U_A, ·, ·)  = SVD_r(τ_A)                 只用来算对齐度 (host 侧)
        (U_B, S_B, Vh_B) = SVD_r(τ_B)             注入基 (donor 侧)
        c   = ||U_A^T · orth(U_B) − I||_F^2 / r    两侧任务方向的结构失配 (闭式)
        λ   = 1 / (1 + c)                         APT 原义的可靠性门控 (标量)
        h   = diag(U_B^T W_B Vh_B^T) − diag(U_B^T W_A Vh_B^T)   task_delta 盈余
        d   = [h]_+                                非负锥投影 (Boyd 2014)
        Δ   = alpha · λ · (U_B ⊙ d) Vh_B
        W*  = (1 − ia)·W_A + ia·W_B + Δ            ia=interp_alpha

来源契约: SVD 截断/符号规范、非负锥投影沿用 Boyd(2014); λ=1/(1+c) 的
discrepancy 定义沿用 APT 参考稿; task_delta target 与"插值+band 叠加"的组合
是本项目 (RAPT V0.3.13+) 的本地设计, 不借公开方法之名宣称已获实证。

用法:
    python -m merge.rapt_V0_3.ta_hyb_standalone \
        --anchor-model downloaded_models/Qwen2-7B \
        --host-model   downloaded_models/Qwen2-VL-7B-Instruct \
        --donor-model  downloaded_models/llava-onevision-qwen2-7b-si-hf \
        --output-dir   merge/rapt_V0_3/merged_models_standalone
    # 其余超参数均有默认值 (即 TA_hyb 的确认最优点), 无需再传。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import resource
import shutil
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import safetensors.torch
import torch
from tqdm import tqdm

try:
    from .utils import load_weights, need_merge
    from .rapt_v3_common import canon_module_name, canon_svd_signs, orth, strip_multiway_segment
except Exception:  # 直接脚本运行
    from utils import load_weights, need_merge
    from rapt_v3_common import canon_module_name, canon_svd_signs, orth, strip_multiway_segment

EPS = 1e-8


def _fmt(v) -> str:
    s = "%g" % float(v)
    return s.replace(".", "p").replace("-", "m")


def canon_key(key: str) -> str:
    """跨架构 (Qwen2 / LLaVA-OneVision / InternVL 等) 的参数名规范化, 保留 weight/bias 后缀。"""
    return strip_multiway_segment(canon_module_name(key))


def _build_index(weights: Dict[str, torch.Tensor]):
    """canon_key -> 该 canon 下的原始 key 列表 (按遍历顺序, 通常长度为 1)。

    2026-09-17 修正: mPLUG-Owl2 的 k_proj/v_proj 在 canon 化后与另一模态分支
    (`.multiway.0` / `.multiway.1`) 撞在同一个 canon 键上 (LN 因 need_merge 排除,
    不受影响)。此前实现用 `index[canon_key(k)] = k` 做 dict 赋值, 同 canon 的多个
    分支被"后写覆盖先写", 只留下遍历顺序里最后一个分支 (实测为 `.multiway.1`),
    另一分支从未参与合并, 且两分支任务向量近乎正交 (E0 审计: 权重余弦 0.955,
    任务向量余弦仅 0.119), 被静默丢弹的分支携带的是实质不同的信息, 不是冗余。

    2026-09-17 二次修正 (讨论后定稿, 不采用"输入端取均值"): AdaMMS 的
    `combine_multiway(mode=mean)` 是在**绝对权重空间线性插值**语境下取均值, 对方向不敏感,
    均值仍是一个合法的插值目标。但 TA_hyb 的核心机制 (`structural_divergence_identity`
    给出的 `λ_apt` 模块级缩放 + donor 任务向量 SVD 基 `U_B`/`Vh_B` 决定注入方向) 是
    **方向敏感**的——若先把两个近正交 (cos=0.119) 的分支权重在输入端平均掉再做一次 SVD,
    会把这两个真正起作用的机制建立在一个不代表任一分支真实适配方向的人造"骑墙"方向上。
    (讨论中同时核实: 该结论不受 `positive_surplus_ratio≈1.0` 全正现象的影响——全正只说明
    ReLU 选择性截断这一个子机制在 Group1/Group3 上都没有实际筛选任何方向 [已在方法文档
    §26.2/§26.5 正式撤回"全正=门控失效"的旧结论], 但 `λ_apt` 与 `U_B` 两个方向敏感的机制
    仍正常起作用, 不应被输入端平均污染。)

    因此改为**输出端合并**: 碰撞的每个分支各自独立走完整条 TA_hyb 流水线 (各自的
    τ_B/SVD/λ_apt/正响应), 在 `merge_module_multi` 里把各分支的 `Delta` 相加后再写回同一
    host 位置; 插值项 `(1-ia)*W_A + ia*W_B` 中的 `W_B` 因为不涉及方向敏感计算, 用分支均值
    (与 AdaMMS 的用法一致, 这部分沿用均值没有方法论冲突)。

    2026-09-18 增补 (消融决策): 增加 `--multiway-combine mean` 模式 —— 与 AdaMMS
    `combine_multiway(mode=mean)` 完全一致的**输入端均值**: 碰撞分支先平均成单一 W_B,
    再走普通 (单分支) TA_hyb 流水线。两种模式并存、由 CLI 开关选择、实证择优:
    mean 的意义 = 与 AdaMMS 基线完全同口径 (唯一差异是更新规则);
    output_sum 的意义 = 方向敏感机制 (λ_apt/U_B) 作用于各分支真实适配方向。
    注意: 纯插值消融变体 (W_A + ia*(W_B-W_A), 无 Delta) 在两种模式下逐位一致。

    返回 (index, n_collision):
        index      : canon -> 该 canon 下的原始 key 列表 (len==1 为正常无碰撞情形)
        n_collision: 碰撞 (len>1) 的 canon 键数量 (写入 meta, 供审计)
    """
    groups: Dict[str, list] = defaultdict(list)
    for k in weights.keys():
        if need_merge(k):
            groups[canon_key(k)].append(k)
    n_collision = sum(1 for ks in groups.values() if len(ks) > 1)
    return groups, n_collision


def resolve_device(dev: str, min_free_gb: float = 8.0) -> str:
    """设备解析。

    'auto' : 仅当 CUDA 可用 **且** 空闲显存 >= min_free_gb 时用 cuda (cuSOLVER),
             否则回落 cpu。实测 (2026-09-16): GPU 与评测进程共存时, SVD kernel 需排队,
             单模块耗时被拉长到 ~4.5s 且与矩阵大小无关 (4096x4096 与 4096x11008 同为 4.5s),
             此时 GPU 反而比 6 线程 CPU 慢; 只有 GPU 真正空闲时 cuSOLVER 才能把
             45 分钟的合并压到分钟级。故用显存余量作为"是否空闲"的廉价代理指标。
    'cuda' : 强制使用 (忽略余量检查, 由用户自行保证不被抢占)。
    'cpu'  : 强制 CPU (数值口径与 CONFIRMED 配方完全一致)。
    """
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[device] --device cuda 但当前 CUDA 不可用")
        return "cuda"
    if dev == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    free_gb = torch.cuda.mem_get_info()[0] / 2 ** 30
    if free_gb < min_free_gb:
        print(f"[device] auto -> cpu: GPU 空闲显存 {free_gb:.1f}GiB < 阈值 {min_free_gb}GiB, "
              f"疑似有评测/训练占用中; 共存时 GPU SVD 需排队, 反而慢于 CPU。"
              f"若确认 GPU 已空闲或愿意等待, 可显式传 --device cuda。")
        return "cpu"
    return "cuda"


def svd_trunc(W: torch.Tensor, r: int, device: str = "cpu"):
    """rank-r 截断 SVD + 联合符号规范 (Boyd 2014 惯例, 与 rapt_v3_stage3_injection 一致)。

    device: 'cuda' 时走 cuSOLVER (比 6 线程 CPU 快 1-2 个数量级), 'cpu' 时与历史口径完全一致
    (精确截断 SVD, 不使用 lowrank 近似, 保证与 CONFIRMED 配方数值可比)。
    """
    Wf = W.float().to(device)
    r_eff = min(int(r), Wf.shape[0], Wf.shape[1])
    if r_eff <= 0:
        return None
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    U, S, Vh = U[:, :r_eff].contiguous(), S[:r_eff].contiguous(), Vh[:r_eff, :].contiguous()
    U, Vh = canon_svd_signs(U, Vh)
    return U, S, Vh


def structural_divergence_identity(U_A: torch.Tensor, U_B: torch.Tensor, r: int) -> float:
    """c = ||U_A^T orth(U_B) − I||_F^2 / r  (identity_mapping 版, M=I, 闭式无 OT)。"""
    with torch.no_grad():
        U_B_bar = orth(U_B)
        X = U_A.T @ U_B_bar
        I = torch.eye(r, device=X.device, dtype=X.dtype)
        return float(((X - I) ** 2).sum().item() / max(r, 1))


def compute_branch_delta(W0f: torch.Tensor, WAf: torch.Tensor, WBf: torch.Tensor,
                          rank: int, alpha: float,
                          conflict_filter: bool = False,
                          ortho_scale: bool = False) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """单个 donor 分支的 band 增量 `Delta` (不含插值项)。W0f/WAf/WBf 须已在目标 device 上、fp32。

    从 `merge_module` 中拆出 (2026-09-17), 供多分支碰撞场景 (`merge_module_multi`) 复用:
    每个分支独立算自己的 τ_B/SVD 基/λ_apt, 保持方向敏感计算的完整性 (见 `_build_index` 注释),
    不在算 Delta 之前对分支做任何平均。

    2026-09-18 增补 (规划 §6.3 方案 B/C 的待命实现, 默认关闭, 不改变既有行为):
      conflict_filter (方案 C): 逐方向冲突过滤 —— donor 第 i 个奇异方向 u_i 与 host 任务向量
        τ_A 的 |cos| 越大 (host 已在该方向位移), 该方向注入越弱: eta_i *= (1 - |<u_i, τ_A/||τ_A||>|)。
        零新超参 (U_B 列单位正交, |cos|∈[0,1])。
      ortho_scale (方案 B): 互补度自适应强度 —— 模块级 κ_m = cos(τ_A, τ_B), eta *= (1 - κ_m)。
        g(κ)=1-κ 为零超参选择, 恰好同时命中两对校准点 (G1 κ≈0.03→0.97 全强度;
        G3 κ≈0.49→0.51 约半强度), 且逐模块计算有 224/196 个点, 无"两点定曲线"问题。
    """
    tau_A = WAf - W0f
    tau_B = WBf - W0f

    res_A = svd_trunc(tau_A, rank, WAf.device.type)
    res_B = svd_trunc(tau_B, rank, WAf.device.type)
    if res_A is None or res_B is None:
        return None, {"skipped": 1.0}
    U_A, _, _ = res_A
    U_B, S_B, Vh_B = res_B
    r = U_B.shape[1]
    U_A = U_A[:, :r]

    c = structural_divergence_identity(U_A, U_B, r)
    lam_apt = 1.0 / (1.0 + c)

    t_donor = ((U_B.T @ WBf) * Vh_B).sum(dim=1)
    t_host = ((U_B.T @ WAf) * Vh_B).sum(dim=1)
    h = t_donor - t_host
    d = h.clamp_min(0.0)

    eta = float(alpha) * lam_apt * d
    diag = {
        "lambda_apt": lam_apt,
        "h_mean": float(h.mean().item()),
        "positive_surplus_ratio": float((h > 0).float().mean().item()),
    }
    if conflict_filter:
        # overlap_i = ||u_i^T @ τ_A||_2 / ||τ_A||_F ∈ [0,1]  (τ_A 是矩阵; u_i 单位正交,
        # Σ_i ||u_i^T τ_A||² <= ||τ_A||²_F 保证单个 overlap <= 1)
        overlap = (U_B.T @ tau_A).norm(dim=1) / (tau_A.norm() + EPS)
        eta = eta * (1.0 - overlap)
        diag["conflict_overlap_mean"] = float(overlap.mean().item())
    if ortho_scale:
        kappa = float((tau_A.flatten() @ tau_B.flatten()) /
                      (tau_A.norm() * tau_B.norm() + EPS))
        eta = eta * (1.0 - kappa)
        diag["kappa_module"] = kappa
    Delta = (U_B * eta.unsqueeze(0)) @ Vh_B
    diag["delta_norm"] = float(Delta.norm().item())
    return Delta, diag


def merge_module(W0: torch.Tensor, W_A: torch.Tensor, W_B: torch.Tensor,
                  rank: int, alpha: float, interp_alpha: float,
                  device: str = "cpu",
                  conflict_filter: bool = False, ortho_scale: bool = False
                  ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """单个线性模块 (无 multiway 碰撞) 的 TA_hyb 计算。返回 (W_new, diag)。

    device='cuda' 时全部中间量 (τ、SVD、注入) 都在显存里算, 只把最终 W_new 搬回 CPU,
    因此峰值显存 ≈ 3 个模块张量 + SVD 工作区 (4096×11008 fp32 约 0.2GB/个), 与 7B 全模型无关。
    """
    W0f, WAf, WBf = W0.float().to(device), W_A.float().to(device), W_B.float().to(device)
    Delta, diag = compute_branch_delta(W0f, WAf, WBf, rank, alpha,
                                       conflict_filter=conflict_filter, ortho_scale=ortho_scale)
    if Delta is None:
        return W_A, diag
    ia = float(interp_alpha)
    W_new = (1.0 - ia) * WAf + ia * WBf + Delta
    diag["w_a_norm"] = float(WAf.norm().item())
    return W_new.to("cpu").to(W_A.dtype), diag


def merge_module_multi(W0: torch.Tensor, W_A: torch.Tensor, W_Bs: "list[torch.Tensor]",
                        rank: int, alpha: float, interp_alpha: float,
                        device: str = "cpu",
                        conflict_filter: bool = False, ortho_scale: bool = False
                        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """碰撞场景 (donor 同一 canon 键下有多个分支, 如 mPLUG 的 `.multiway.{0,1}`)。

    每个分支各自独立算 `Delta_b` (各自的 SVD 基/λ_apt, 不预先平均输入), 最终把所有分支的
    `Delta_b` 相加写回同一 host 位置; 插值项 `(1-ia)*W_A + ia*mean(W_B_b)` 中的 `mean` 不涉及
    方向敏感计算, 与 AdaMMS `combine_multiway(mode=mean)` 的用法一致, 无方法论冲突。
    见 `_build_index` 顶部注释的完整论证。
    """
    W0f, WAf = W0.float().to(device), W_A.float().to(device)
    deltas, diags = [], []
    for W_B in W_Bs:
        WBf = W_B.float().to(device)
        Delta, diag = compute_branch_delta(W0f, WAf, WBf, rank, alpha,
                                           conflict_filter=conflict_filter, ortho_scale=ortho_scale)
        if Delta is not None:
            deltas.append(Delta)
            diags.append(diag)
    if not deltas:
        return W_A, {"skipped": 1.0}
    Delta_sum = torch.stack(deltas, dim=0).sum(dim=0)
    W_B_mean = torch.stack([b.float().to(device) for b in W_Bs], dim=0).mean(dim=0)
    ia = float(interp_alpha)
    W_new = (1.0 - ia) * WAf + ia * W_B_mean + Delta_sum

    n = len(diags)
    diag = {
        "lambda_apt": sum(d["lambda_apt"] for d in diags) / n,
        "h_mean": sum(d["h_mean"] for d in diags) / n,
        "positive_surplus_ratio": sum(d["positive_surplus_ratio"] for d in diags) / n,
        "delta_norm": float(Delta_sum.norm().item()),
        "w_a_norm": float(WAf.norm().item()),
        "n_branches": n,
        "per_branch_delta_norm": [d["delta_norm"] for d in diags],
    }
    return W_new.to("cpu").to(W_A.dtype), diag


def _output_root(args: argparse.Namespace) -> str:
    base_name = osp.basename(args.host_model.rstrip(os.sep))
    tag = f"ta_hyb_standalone_r{int(args.rank)}_a{_fmt(args.alpha)}_ia{_fmt(args.interp_alpha)}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    return osp.join(args.output_dir, base_name, tag)


def _copy_side_files(base_model: str, out_root: str) -> None:
    for fn in os.listdir(base_model):
        if fn.endswith((".json", ".model", ".py", ".md")):
            src, dst = osp.join(base_model, fn), osp.join(out_root, fn)
            if not osp.exists(dst):
                shutil.copy(src, dst)


def _save_model(args: argparse.Namespace, merged: Dict[str, torch.Tensor]) -> str:
    out_root = _output_root(args)
    os.makedirs(out_root, exist_ok=True)
    st_index = osp.join(args.host_model, "model.safetensors.index.json")
    pt_index = osp.join(args.host_model, "pytorch_model.bin.index.json")
    if osp.exists(st_index) or osp.exists(pt_index):
        index_path = st_index if osp.exists(st_index) else pt_index
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = defaultdict(dict)
        for k, v in merged.items():
            if k in weight_map:
                shards[weight_map[k]][k] = v
        for shard, state in shards.items():
            if shard.endswith(".safetensors"):
                safetensors.torch.save_file(state, osp.join(out_root, shard))
            else:
                torch.save(state, osp.join(out_root, shard))
    elif osp.exists(osp.join(args.host_model, "model.safetensors")):
        safetensors.torch.save_file(merged, osp.join(out_root, "model.safetensors"))
    elif osp.exists(osp.join(args.host_model, "pytorch_model.bin")):
        torch.save(merged, osp.join(out_root, "pytorch_model.bin"))
    else:
        safetensors.torch.save_file(merged, osp.join(out_root, "model.safetensors"))
    _copy_side_files(args.host_model, out_root)
    return out_root


def run(args: argparse.Namespace) -> None:
    print("\n--- [TA_hyb standalone] task_delta + apt(identity) + hybrid ---")
    print(f"    rank={args.rank} alpha={args.alpha} interp_alpha={args.interp_alpha}  "
          f"(TA_hyb 确认最优点, 见 §21.6)")

    device = resolve_device(getattr(args, "device", "cpu"),
                           getattr(args, "gpu_min_free_gb", 8.0))
    if device == "cuda":
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"    device=cuda (cuSOLVER SVD), 当前空闲显存 "
              f"{free_b / 2**30:.1f}/{total_b / 2**30:.1f} GiB")
    else:
        print(f"    device={device} (CPU SVD; 数值口径与 CONFIRMED 配方一致)")

    weights_0 = load_weights(args.anchor_model)
    weights_A = load_weights(args.host_model)
    weights_B = load_weights(args.donor_model)

    groups_0, coll_0 = _build_index(weights_0)
    groups_B, coll_B = _build_index(weights_B)
    if coll_0:
        print(f"[Map] 锚点 canon 碰撞 {coll_0} 个 —— 当前实现不支持锚点侧多分支配对, "
              f"命中的 key 会记为 n_anchor_collision_unsupported 并跳过, 不静默猜配对")
    if coll_B:
        print(f"[Map] donor canon 碰撞 {coll_B} 个 (mPLUG k/v 的 multiway 双分支): "
              f"multiway_combine={args.multiway_combine} "
              f"(mean=输入端分支平均, 与 AdaMMS 完全同口径 / "
              f"output_sum=各分支独立流水线后 Delta 相加)")

    merged = weights_A.copy()

    n_total = n_merged = n_missing_anchor = n_missing_donor = n_shape_mismatch = 0
    n_anchor_collision_unsupported = n_multi_branch_merged = 0
    n_gpu_oom = 0
    n_1d = 0
    lam_sum = 0.0
    pos_ratio_sum = 0.0

    # 2026-09-21 修复 (与 ablation_e2_gen.py 同款 bug, 该脚本此前已修、本脚本漏修):
    # 旧实现在每个 key 处理完的 finally 里直接 pop 掉共享的 anchor/donor 键。
    # 反向 (host=mPLUG-Owl2) 存在多个 host key 映射到**同一个** anchor 键的情况,
    # 第一个分支 pop 之后第二个分支再取 -> KeyError(该 bug 使 rev 产物生成整体失败)。
    # 改为引用计数: 统计每个共享键被多少个待合并模块需要, 计数归零才释放。
    need: Dict[str, int] = defaultdict(int)
    for _k in list(weights_A.keys()):
        if not need_merge(_k):
            continue
        _ck = canon_key(_k)
        for _a in groups_0.get(_ck, []):
            need[f"0::{_a}"] += 1
        for _d in groups_B.get(_ck, []):
            need[f"B::{_d}"] += 1

    def _release(prefix: str, key_name: Optional[str], store: Dict[str, Any]) -> None:
        if key_name is None:
            return
        _k = f"{prefix}::{key_name}"
        need[_k] -= 1
        if need[_k] <= 0:
            store.pop(key_name, None)

    for key in tqdm(list(weights_A.keys()), desc="TA_hyb standalone", disable=args.no_progress):
        if not need_merge(key):
            continue
        n_total += 1
        ck = canon_key(key)
        anchor_keys = groups_0.get(ck, [])
        donor_keys = groups_B.get(ck, [])
        anchor_key0 = anchor_keys[0] if anchor_keys else None
        try:
            if not anchor_keys:
                n_missing_anchor += 1
                continue
            if not donor_keys:
                n_missing_donor += 1
                continue
            if len(anchor_keys) > 1:
                n_anchor_collision_unsupported += 1
                continue
            if anchor_key0 not in weights_0:     # 2026-09-21: 防御, 配合引用计数
                n_missing_anchor += 1
                continue
            W0 = weights_0[anchor_key0]
            W_A = weights_A[key]
            W_Bs = [weights_B[k] for k in donor_keys]
            if W0.ndim != 2 or W_A.ndim != 2 or any(wb.ndim != 2 for wb in W_Bs):
                # 2026-09-21: 1D 权重 (bias/norm) 保留 host 值, 与 ablation_e2_gen.py
                # 口径一致; 缺失该检查时 Qwen2-VL 会在 svd_trunc 里 IndexError 崩溃。
                n_1d += 1
                continue
            if W0.shape != W_A.shape or any(wb.shape != W_A.shape for wb in W_Bs):
                n_shape_mismatch += 1
                continue

            multi = len(W_Bs) > 1          # 碰撞模块标记 (计数用)
            if multi and args.multiway_combine == "mean":
                # 2026-09-18: AdaMMS 同口径的输入端均值 (消融对照), 见 _build_index 注释。
                # 均值在 fp32 下计算 (与 AdaMMS 插值口径一致), 后续 .float() 为无操作。
                W_Bs = [torch.stack([wb.float() for wb in W_Bs], dim=0).mean(dim=0)]
            use_multi = multi and args.multiway_combine != "mean"
            merge_fn = merge_module_multi if use_multi else merge_module
            merge_args = (W0, W_A, (W_Bs if use_multi else W_Bs[0]), args.rank, args.alpha, args.interp_alpha)
            merge_kw = {"conflict_filter": args.conflict_filter, "ortho_scale": args.ortho_scale}
            if device == "cuda":
                try:
                    W_new, diag = merge_fn(*merge_args, device, **merge_kw)
                except torch.cuda.OutOfMemoryError:
                    n_gpu_oom += 1
                    if n_gpu_oom == 1:
                        print(f"\n[WARN] {key}: CUDA OOM -> 本模块及后续模块回退 CPU SVD")
                    torch.cuda.empty_cache()
                    device = "cpu"
                    W_new, diag = merge_fn(*merge_args, "cpu", **merge_kw)
            else:
                W_new, diag = merge_fn(*merge_args, device, **merge_kw)
            if diag.get("skipped"):
                continue
            merged[key] = W_new
            n_merged += 1
            if multi:
                n_multi_branch_merged += 1
            lam_sum += diag["lambda_apt"]
            pos_ratio_sum += diag["positive_surplus_ratio"]
            if n_merged % 16 == 0:
                rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
                print(f"  ... {n_merged} 模块完成 (peakRSS={rss_peak:.1f}GB)", flush=True)
        finally:
            # 2026-09-18: 边算边释放三模型权重引用 (容器 cgroup 内存上限 62GB, 防 OOM 静默杀进程;
            # merged 已持有全部待保存张量的引用, 这些 pop 只删字典引用, 不改变任何数值)。
            # 2026-09-21: 共享键改为引用计数释放 (见上方 need 统计), 否则反向会因共享
            # anchor 键被提前 pop 而 KeyError。
            weights_A.pop(key, None)
            _release("0", anchor_key0, weights_0)
            for _dk in donor_keys:
                _release("B", _dk, weights_B)

    print(f"[Done] merged={n_merged}/{n_total} (multi_branch={n_multi_branch_merged})  "
          f"missing_anchor={n_missing_anchor}  missing_donor={n_missing_donor}  "
          f"shape_mismatch={n_shape_mismatch}  anchor_collision_unsupported={n_anchor_collision_unsupported}  "
          f"final_device={device}  gpu_oom_fallback={n_gpu_oom}")
    if n_merged:
        print(f"[Diag] mean lambda_apt={lam_sum / n_merged:.4f}  "
              f"mean positive_surplus_ratio={pos_ratio_sum / n_merged:.4f}")

    out_root = _save_model(args, merged)
    meta = {
        "method": "TA_hyb_standalone",
        "anchor_model": args.anchor_model,
        "host_model": args.host_model,
        "donor_model": args.donor_model,
        "rank": args.rank,
        "alpha": args.alpha,
        "interp_alpha": args.interp_alpha,
        "device": device,
        "gpu_oom_fallback": n_gpu_oom,
        "n_total": n_total,
        "n_merged": n_merged,
        "n_missing_anchor": n_missing_anchor,
        "n_missing_donor": n_missing_donor,
        "n_shape_mismatch": n_shape_mismatch,
        "n_anchor_multiway_collisions": coll_0,
        "n_donor_multiway_collisions": coll_B,
        "n_anchor_collision_unsupported": n_anchor_collision_unsupported,
        "n_multi_branch_merged": n_multi_branch_merged,
        "multiway_combine_mode": args.multiway_combine,
        "conflict_filter": args.conflict_filter,
        "ortho_scale": args.ortho_scale,
        "multiway_combine": (
            "mean (AdaMMS 同口径): 碰撞分支在输入端平均成单一 W_B (fp32), 再走普通单分支 "
            "TA_hyb 流水线; 与 AdaMMS combine_multiway(mode=mean) 完全一致。2026-09-18 增加的 "
            "消融模式, 与 output_sum 并存、实证择优。"
            if args.multiway_combine == "mean" else
            "output-level: 碰撞分支各自独立走完整 TA_hyb 流水线 "
            "(各自 SVD 基/λ_apt, 不预先平均输入), 结果 Delta 相加写回同一 host 位置; "
            "插值项 W_B 用分支均值 (不涉及方向敏感计算, 与 AdaMMS combine_multiway "
            "用法一致)。2026-09-17 讨论定稿; 2026-09-18 起与 mean 模式并存作消融对照, "
            "由 --multiway-combine 选择。方向敏感理由: λ_apt/SVD 基 U_B 作用于各分支真实 "
            "适配方向 (两分支任务向量近正交 cos=0.119)。此前实现 (2026-09-17 前) 为隐式取 "
            "遍历顺序最后一个分支, 另一分支被丢弹, 已废弃。"
        ),
        "mean_lambda_apt": (lam_sum / n_merged) if n_merged else None,
        "mean_positive_surplus_ratio": (pos_ratio_sum / n_merged) if n_merged else None,
    }
    with open(osp.join(out_root, "merge_meta_ta_hyb_standalone.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {out_root}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor-model", required=True,
                    help="预训练锚点 (task vector 零点), 如 Qwen2-7B")
    ap.add_argument("--host-model", required=True,
                    help="host, 如 Qwen2-VL-7B-Instruct; 输出以它为骨架")
    ap.add_argument("--donor-model", required=True, help="donor 专家模型")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=128, help="TA_hyb 确认最优值")
    ap.add_argument("--alpha", type=float, default=0.5, help="band 注入强度, TA_hyb 确认最优值")
    ap.add_argument("--interp-alpha", type=float, default=0.1, help="插值系数, TA_hyb 确认最优值")
    ap.add_argument("--tag-suffix", default=None)
    ap.add_argument("--multiway-combine", default="output_sum", choices=["output_sum", "mean"],
                    help="multiway 碰撞 (donor 同一 canon 下多分支, 如 mPLUG k/v) 的合并方式: "
                         "output_sum=各分支独立走完整流水线后 Delta 相加 (2026-09-17 定稿); "
                         "mean=分支输入端平均成单一 W_B 后走单分支流水线, 与 AdaMMS "
                         "combine_multiway(mode=mean) 完全同口径 (2026-09-18 消融对照)")
    ap.add_argument("--conflict-filter", action="store_true",
                    help="方案 C (规划 §6.3, 待命): 逐方向冲突过滤 —— donor 奇异方向 u_i 与 "
                         "host 任务向量 τ_A 的 |cos| 越大注入越弱, eta_i *= (1-|cos|); 零新超参")
    ap.add_argument("--ortho-scale", action="store_true",
                    help="方案 B (规划 §6.3, 待命): 互补度自适应强度 —— 模块级 "
                         "eta *= (1-cos(τ_A,τ_B)); g=1-κ 零超参且同时命中两对校准点")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="SVD/注入计算设备; auto=GPU 空闲(空闲显存>=gpu_min_free_gb)时用 cuSOLVER, "
                         "否则回落 CPU; cuda 显存不足时该模块自动回退 CPU; 数值结果与 cpu 同量级一致")
    ap.add_argument("--gpu-min-free-gb", type=float, default=8.0,
                    help="auto 模式下判定 GPU '空闲' 的空闲显存阈值 (GiB); 低于此值回落 CPU, "
                         "避免与评测进程争抢导致 SVD 排队反而更慢")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
