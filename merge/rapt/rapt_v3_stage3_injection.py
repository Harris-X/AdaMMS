#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAPT V0.3 Stage 3: target × gate 解耦的保护性注入。

核心更新 (J0, host 奇异基对角非负):
    h = target_raw - sigma_A                       # 未截断盈余 (target 决定 target_raw)
    d = [h]_+                                      # 非负锥投影 (Boyd 2014 的一维最优解)
    g = Gate(...)                                  # 可靠性衰减 (component/mean/apt/energy/none/norm_match)
    eta = alpha * g * d
    W* = W_A + U_A diag(eta) V_A^T                 # 保留 host 截断残差

target 与 gate 完全解耦:
    --target transport_spectrum  => h = P sigma_B - sigma_A        (E0, APT 原义)
    --target operator_response   => h = S sigma_B - sigma_A        (E1, 本地登记的替代 target)
    --target task_delta          => h = diag(U^T (W_B - W_A) V^T)  (TA 分支, 见下)

**TA 分支 (V0.3.13)**: 当 `--base-subs` 指向带 `--anchor-model` 生成的 Stage-1 缓存时,
注入基 U/Vh 是 **task vector** τ = W − W_0 的奇异基 (而非绝对权重 W 的奇异基)。
动机与实测见 `diag_taskvector.py` / `diag_taskvector2.py`:
  * 绝对权重基下 reach(Δ)=0.083, 且对角仅占块内能量 0.205、正部仅 0.109
    → 非负对角注入可用量 0.00145‖W_A‖ (近乎为零);
  * donor 任务基 U_τB 下 reach(Δ)=0.192, 对角占 0.906、正部 1.000
    → 可用量 0.04785‖W_A‖ (33x), 且 h 恒为正, 非负约束不再切掉信号。
此时 `--gate mismatch_component_donor` 提供按 donor 列读取的逐组件门控 (索引与注入分量一致)。

来源契约: 非负正交锥投影/二次收缩沿用 Boyd(2014); E0 沿用 APT 参考稿;
E1 与逐组件/预算匹配门控是本地登记的探索, 不借公开方法之名宣称已获实证。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import shutil
from collections import defaultdict
from typing import Dict, Optional

import safetensors.torch
import torch
from tqdm import tqdm

try:
    from .utils import load_weights, need_merge
    from .rapt_v3_common import (
        EPS,
        canon_module_from_param_key,
        canon_svd_signs,
        load_subspaces,
        load_subspaces_full,
        strip_multiway_segment,
    )
    from .rapt_v3_stage2_mapping import mismatch_gate_from_q
except Exception:  # 直接脚本运行
    from utils import load_weights, need_merge
    from rapt_v3_common import (
        EPS,
        canon_module_from_param_key,
        canon_svd_signs,
        load_subspaces,
        load_subspaces_full,
        strip_multiway_segment,
    )
    from rapt_v3_stage2_mapping import mismatch_gate_from_q

SCHEMA_VERSION = "rapt-v2.1"
TARGETS = ("transport_spectrum", "operator_response", "full_donor_response",
           "calibrated_transport", "hard_transport", "task_delta")
TGT_ABBR = {"transport_spectrum": "ot", "operator_response": "op",
            "full_donor_response": "fd", "calibrated_transport": "cal",
            "hard_transport": "hard", "task_delta": "td"}
GATES = ("component", "mean", "apt", "energy", "none", "norm_match",
         "mismatch_module", "mismatch_component", "mismatch_component_norm",
         "mismatch_component_donor", "host_overlap",
         # §24 判决门 (build_gate 里已实现, 此前漏加进 choices -> argparse 拒绝,
         # G 批次三个候选一个都没生成, 2026-09-16 04:56 发现):
         "sigma_sharp", "a_orth", "a_reverse")
# §20.2: target/gate 的默认值**单一来源** —— argparse 与"传了却无效"守卫共用同一常量,
# 避免两处各写一遍默认值而漂移 (守卫曾误用 TARGETS[0], 与真实默认值 operator_response 不一致)。
DEFAULT_TARGET = "operator_response"
DEFAULT_GATE = "component"
# §20.2: 已实验证伪 / 退化的 gate 取值。**保留仅为复现历史产物**, 新实验不应使用。
GATES_DEPRECATED = {
    # 注: `component` 是**默认值**且在 host 基下合法, 故不列入本表;
    #     TA 基下的索引错配由主流程的 [TA][Warn] 单独提示。
    "mismatch_component": "TA 基下索引错配 (host 索引 vs donor 分量); 且实测退化为常量 1/3",
    "mismatch_component_norm": "同上; 与 G7 等预算的标量对照未胜出",
    "mismatch_component_donor": "实测 q≈0 -> 退化为常量 1/3, 与 mismatch_module 等价",
}


def _fmt(v) -> str:
    s = "%g" % float(v)
    return s.replace(".", "p").replace("-", "m")


def _svd_trunc(W: torch.Tensor, r: int):
    Wf = W.float()
    r = min(int(r), Wf.shape[0], Wf.shape[1])
    if r <= 0:
        return None
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    U, S, Vh = U[:, :r].contiguous(), S[:r].contiguous(), Vh[:r, :].contiguous()
    U, Vh = canon_svd_signs(U, Vh)
    return U, S, Vh


_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


def _layer_mult(mod: str, args: argparse.Namespace) -> float:
    """§22 逐层强度: 层号 >= --layer-split 的模块, 其整份更新乘 --layer-gain。

    动机 (2026-09-16): 本项目从未让任何系数**随层变化** —— 全部 196 个模块共用一个
    (ia, alpha)。合并文献里逐层系数是最大的经验增益来源之一, 故把它作为一个
    **独立的结构自由度**暴露出来 (而非又一个全局标量)。

    未配置 (gain 或 split 为 None) 时恒返回 1.0, 与既有产物完全一致。
    """
    gain = getattr(args, "layer_gain", None)
    split = getattr(args, "layer_split", None)
    if gain is None or split is None:
        return 1.0
    m = _LAYER_RE.search(mod)
    if m is None:                      # multiway / 非 transformer 层: 不加权
        return 1.0
    return float(gain) if int(m.group(1)) >= int(split) else 1.0


def _output_root(args: argparse.Namespace) -> str:
    base_name = osp.basename(args.base_model.rstrip(os.sep))
    if args.mode == "direct":
        # Stage-3 = 插值基线: 命名只区分 Stage-2 是索引对应(idx)还是算子映射(op)
        mapping = "idx" if args.identity_mapping else "op"
        tag = f"raptv3_r{int(args.rank)}_a{_fmt(args.alpha)}_direct_{mapping}"
        if args.direct_full:
            tag += "_full"
    elif args.mode == "hybrid":
        tgt = TGT_ABBR.get(args.target, args.target[:4])
        tag = (f"raptv3_r{int(args.rank)}_a{_fmt(args.alpha)}_ia{_fmt(args.interp_alpha)}"
               f"_hyb_t{tgt}_g{args.gate}")
        if args.identity_mapping:
            tag += "_nomap"
        if getattr(args, "tau_a_mode", "full") == "avoid":
            tag += "_avA"
    else:
        tgt = TGT_ABBR.get(args.target, args.target[:4])
        tag = f"raptv3_r{int(args.rank)}_a{_fmt(args.alpha)}_t{tgt}_g{args.gate}"
        if args.identity_mapping:
            tag += "_nomap"
        if args.injection == "subspace":
            tag += "_sub"
        elif args.injection == "bidir":
            tag += "_bidir"
            if float(getattr(args, "bidir_a_weight", 1.0)) != 1.0:
                tag += f"_wa{_fmt(args.bidir_a_weight)}"
        if getattr(args, "inject_budget", "raw") == "rel":
            tag += "rel"
        # §18: Δ 低秩 bulk 项 —— 产物名必须能区分 bulk 有无与秩/系数
        if getattr(args, "bulk_delta_subs", None):
            tag += f"_bulk{int(args.bulk_rank)}ia{_fmt(args.bulk_ia)}"
        # §20: 尾部项 b —— 不写进产物名会导致 b 不同的产物撞名
        if float(getattr(args, "w_tail", 0.0)) > 0.0:
            tag += f"_wt{_fmt(args.w_tail)}"
    # §20.8 auto-config: 产物名必须能区分 (s 不同 -> 结果不同)
    if getattr(args, "auto_config", False):
        tag += f"_auto_s{_fmt(args.transfer_ratio)}"
    if getattr(args, "decomp_tag", None) == "task_vector":
        tag = "ta_" + tag          # TA 分支: 保护域来自 task vector 分解, 产物名必须可区分
    if getattr(args, "basis", None):
        tag += f"_b{args.basis}"
    if args.disable_selection:
        tag += "_nosel"
    # §22 逐层强度: 不写进名字会导致 (split,gain) 不同的产物撞名
    if getattr(args, "layer_gain", None) is not None and getattr(args, "layer_split", None) is not None:
        tag += f"_ly{int(args.layer_split)}g{_fmt(args.layer_gain)}"
        if getattr(args, "layer_scope", "all") != "all":
            tag += f"_{args.layer_scope}"     # §25.2: 作用域不同 -> 结果不同, 必须区分
    if getattr(args, "tag_suffix", None):
        tag += f"_{args.tag_suffix}"
    return osp.join(args.output_dir, base_name, tag)


def _linear_assignment(S: torch.Tensor):
    """参考匹配 Pi = argmax_{Pi in P_r} <Pi, S>  (标准线性指派, 一对一)。

    返回 (perm, inv): perm[i] = 与 host 分量 i 匹配的 donor 分量编号。
    与逐行 argmax 不同, 这里保证是一对一置换 —— 逐行 argmax 可能重复选中同一列。

    scipy 不可用时退化为贪心（仍保证一对一），并在返回值第二项标记。
    """
    n = S.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment((-S).double().numpy())
        perm = torch.zeros(n, dtype=torch.long)
        perm[torch.from_numpy(rows)] = torch.from_numpy(cols)
        exact = True
    except Exception:
        # 贪心回退: 按分数从高到低占用未使用的列
        flat = torch.argsort(S.flatten(), descending=True)
        perm = torch.full((n,), -1, dtype=torch.long)
        used_c = torch.zeros(n, dtype=torch.bool)
        used_r = torch.zeros(n, dtype=torch.bool)
        for idx in flat.tolist():
            i, j = divmod(idx, n)
            if used_r[i] or used_c[j]:
                continue
            perm[i] = j; used_r[i] = True; used_c[j] = True
            if bool(used_r.all()):
                break
        if bool((perm < 0).any()):          # 极端情形兜底
            rest_r = (~used_r).nonzero().flatten()
            rest_c = (~used_c).nonzero().flatten()
            perm[rest_r] = rest_c
        exact = False
    inv = torch.zeros_like(perm)
    inv[perm] = torch.arange(n, dtype=torch.long)
    return perm, inv, exact


def normalize_injection_budget(eta: torch.Tensor, alpha: float,
                               W_A_norm: float) -> torch.Tensor:
    """把盈余位移 eta 重标为"相对扰动预算"口径 (V0.3.12)。

    注入量 Δ = U_R diag(eta') Vh_R, 其中 U_R 列正交、Vh_R 行正交, 故
        ‖Δ‖_F = ‖eta'‖₂
    于是取 eta' = eta · (alpha · ‖W_A‖_F / ‖eta‖₂) 可得
        ‖Δ‖_F / ‖W_A‖_F = alpha      (精确, 与 target/gate/rank 无关)

    这样 alpha 成为跨配置可比的"相对扰动预算"∈[0,1], 不需要手填 α>1 去凑量级。
    eta 恒为 0 时返回全 0 (不做 0/0)。
    """
    raw = float(eta.norm().item())
    if raw <= EPS:
        return torch.zeros_like(eta)
    return eta * (float(alpha) * float(W_A_norm) / raw)


def compute_surplus_h(target, blk, S_A, S_B, sigma_used, t_operator, r_eff,
                      W_B_full=None, U_A=None, Vh_A=None, W_A=None):
    """计算盈余 h = target 谱 − host 谱。**生产路径**, 单测直接调用本函数。

    各 target 的精确定义 (变量名即文档, 防止 S_A/S_B 笔误):
      transport_spectrum   h = P σ_B − σ_A                   (E0, APT 原义)
      operator_response    h = S σ_B − σ_A                   (E1)
      calibrated_transport h = P (σ_B − Πᵀ σ_A)              (E2-M, Π = S 的线性指派)
      hard_transport       h = Π σ_B − σ_A                   (P = Π, 不含 τ)
      full_donor_response  h = diag(U_Aᵀ W_B V_A) − σ_A      (E3, 需 donor 完整权重)
      task_delta           h = diag(Uᵀ(W_B − W_A)Vᵀ)         (TA 分支, 见下)

    **task_delta (TA 分支专用)**: 沿**注入基** U/Vh 的「差分 donor 响应」。
    与 E3 的两点区别:
      ① 基底不是 host 自己的奇异基, 而是 donor 的 **task vector 基** U_τB
         (`--base-subs` 指向带 `--anchor-model` 生成的 Stage-1 缓存);
      ② 比较的是 W_B 与 W_A 在**同一基**下的响应之差, 而不是 W_B 的响应与 σ_A 之差。
    性质: W_B = W_A (自合并) 时 h ≡ 0 (精确), 不依赖 σ_A 与基是否匹配。
    实测 (24 模块, r=128, anchor=Qwen2-7B): 该基下 diag 全部为正 (pos_ratio 1.0000),
    diag 能量占块内 0.9064 —— 即 APT 的「非负对角」结构在此基下几乎无损;
    而在 host 绝对权重基下 diag 仅占 0.2051、正部仅 0.1087。

    Π 的方向约定 (与 _linear_assignment 一致):
      perm[i] = 与 host 分量 i 匹配的 donor 分量;  inv = perm⁻¹。
      因此 "donor 分量 j 对应的 host 强度" = S_A[inv[j]] (不是 S_B[inv[j]]!),
      "host 分量 i 收到的 donor 强度" = S_B[perm[i]]。

    返回 (h, aux)。E3 且缺 donor 权重时返回 (None, None), 由调用方跳过。
    aux 仅 calibrated/hard 有内容: {"perm", "inv", "exact", "baseline_gap"}。
      baseline_gap = mean|σ_B − σ_A[inv]| / mean|σ_A|, 用于观测校准基线的实际差距
      (若恒为 0, 说明谱完全相同或基线取错变量 -> 调用方计为退化)。
    """
    if target == "transport_spectrum":
        return sigma_used - S_A, {}
    if target == "operator_response":
        return t_operator - S_A, {}
    if target in ("calibrated_transport", "hard_transport"):
        S_full = blk["operator_similarity"].to(device=S_B.device, dtype=S_B.dtype)[:r_eff, :r_eff]
        perm, inv, exact = _linear_assignment(S_full)
        if target == "calibrated_transport":
            b_B = S_A.to(device=S_B.device, dtype=S_B.dtype)[inv]   # host 强度, 搬到 donor 编号
            P_plan = blk["P"].to(device=S_B.device, dtype=S_B.dtype)[:r_eff, :r_eff]
            h = P_plan @ (S_B - b_B)
            baseline_gap = float((S_B - b_B).abs().mean()
                                 / S_A.abs().mean().clamp_min(EPS))
        else:
            h = S_B[perm] - S_A.to(device=S_B.device, dtype=S_B.dtype)   # Π σ_B − σ_A
            baseline_gap = None
        return h, {"perm": perm, "inv": inv, "exact": exact,
                   "baseline_gap": baseline_gap}
    if target == "task_delta":
        # TA 分支: 同一基下两侧绝对权重的响应之差 (差分 donor 响应)。
        # 自合并 W_B = W_A 时精确为 0; 不需要 S_A 与基匹配。
        if W_B_full is None or W_A is None:
            return None, None
        dev, dt = U_A.device, U_A.dtype
        Uu = U_A.to(device=dev, dtype=dt)
        Vh = Vh_A.to(device=dev, dtype=dt)
        t_donor = ((Uu.T @ W_B_full.to(device=dev, dtype=dt)) * Vh).sum(dim=1)
        t_host = ((Uu.T @ W_A.to(device=dev, dtype=dt)) * Vh).sum(dim=1)
        return t_donor - t_host, {"exact": True,
                                  "t_donor_mean": float(t_donor.mean().item()),
                                  "t_host_mean": float(t_host.mean().item())}
    # full_donor_response (E3)
    if W_B_full is None:
        return None, None
    W_B_c = W_B_full.to(device=U_A.device, dtype=U_A.dtype)
    t_full = ((U_A.T @ W_B_c) * Vh_A.to(device=U_A.device, dtype=U_A.dtype)).sum(dim=1)
    return t_full - S_A.to(device=t_full.device, dtype=t_full.dtype), {}


def _build_donor_index(donor_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """canonical 模块名 -> donor 语言侧 2D 权重; 用于 E3 的 t_i = u_Ai^T W_B v_Ai。

    必须用 need_merge 过滤: 否则 donor 视觉塔的 vision_model...self_attn.q_proj
    经规范化后会与语言侧同名并覆盖索引 (shape 不同, 导致 E3 静默跳过)。
    """
    index: Dict[str, torch.Tensor] = {}
    for key, w in donor_weights.items():
        if key.endswith(".weight") and w.ndim == 2 and need_merge(key):
            index[canon_module_from_param_key(key)] = w
    return index


def _random_basis(W_A, r, seed):
    """同形状随机正交基 (off-manifold 对照)。"""
    g = torch.Generator().manual_seed(int(seed))
    U, _ = torch.linalg.qr(torch.randn(W_A.shape[0], r, generator=g))
    V, _ = torch.linalg.qr(torch.randn(W_A.shape[1], r, generator=g))
    return U.float(), V.T.contiguous().float()


def _donor_basis(donor_subs, blk, W_A, r):
    if donor_subs is None:
        return None
    dblk = donor_subs.get(blk.get("donor_module"))
    if dblk is None:
        return None
    U, Vh = dblk["U"].float(), dblk["Vh"].float()
    if U.shape[0] != W_A.shape[0] or Vh.shape[1] != W_A.shape[1]:
        return None
    return U[:, :r].contiguous(), Vh[:r, :].contiguous()


def _copy_side_files(base_model: str, out_root: str) -> None:
    for fn in os.listdir(base_model):
        if fn.endswith((".json", ".model", ".py", ".md")):
            src, dst = osp.join(base_model, fn), osp.join(out_root, fn)
            if not osp.exists(dst):
                shutil.copy(src, dst)


def _save_model(args: argparse.Namespace, merged: Dict[str, torch.Tensor]) -> str:
    out_root = _output_root(args)
    os.makedirs(out_root, exist_ok=True)
    st_index = osp.join(args.base_model, "model.safetensors.index.json")
    pt_index = osp.join(args.base_model, "pytorch_model.bin.index.json")
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
    elif osp.exists(osp.join(args.base_model, "model.safetensors")):
        safetensors.torch.save_file(merged, osp.join(out_root, "model.safetensors"))
    elif osp.exists(osp.join(args.base_model, "pytorch_model.bin")):
        torch.save(merged, osp.join(out_root, "pytorch_model.bin"))
    else:
        safetensors.torch.save_file(merged, osp.join(out_root, "model.safetensors"))
    _copy_side_files(args.base_model, out_root)
    return out_root


def _resolve_base_svd(mod: str, W_A: torch.Tensor, r: int, base_subs, allow_fallback: bool = False):
    """优先复用 Stage-1 的同一份 host 分解; 缺失时默认显式拒绝, 不静默重算基。

    allow_fallback 仅供工程调试; 正式实验必须为 False (缓存身份与分量顺序一致)。
    """
    if base_subs is not None and mod in base_subs:
        blk = base_subs[mod]
        U, S, Vh = blk["U"], blk["S"], blk["Vh"]
        if U.shape[0] == W_A.shape[0] and Vh.shape[1] == W_A.shape[1]:
            return U[:, :r].float(), S[:r].float(), Vh[:r, :].float(), True
    if allow_fallback:
        result = _svd_trunc(W_A, r)
        if result is not None:
            return result[0], result[1], result[2], False
    return None


def donor_component_gate(P: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """把同一运输计划按 **donor 列** 读取, 得到 donor 侧的逐组件失配 gate。

    用于注入基是 donor task vector 基的 TA 分支: 此时被修改的是 donor 分量 j,
    可靠性必须按列统计, 否则用 host 分量 i 的失配去门控 donor 分量 j (索引错配)。

        q^col_j = Σ_i (P_ij / Σ_i P_ij) · S_ij ,  g_j = 1/(1 + 2(1 − q^col_j))

    P 为行随机计划 (Stage-2 缓存口径); 若 P 恰为双随机则列和为 1, 退化为列期望。
    """
    col = P.sum(dim=0, keepdim=True).clamp_min(EPS)
    q_col = ((P / col) * S).sum(dim=0)
    return mismatch_gate_from_q(q_col)


def host_overlap_gate(U_W: torch.Tensor, Vh_W: torch.Tensor,
                      U_tB: torch.Tensor, Vh_tB: torch.Tensor):
    """候选 A（§15.3）: 逐 donor 任务方向 j 的「host 主子空间重叠度」gate。

        a_j = ‖U_Wᵀ u_τB,j‖² · ‖Vh_W v_τB,j‖²          ∈[0,1]
        g_j = a_j / max_j a_j                            (逐模块归一, g ∈ (0,1])

    含义: donor 第 j 个任务方向有多少落在 host 的 top-r 主子空间内。
    落得多 → 沿它注入对 host 既有几何的扰动小 → 安全, gate 大。

    为什么用**逐模块 max 归一**: a_j 的绝对量级很小（实测 mean≈0.0069, 随机基线
    0.0027), 直接当 gate 会把更新压到近乎为零; gate 是**相对**可靠性权重,
    故保留逐模块的排序形状, 并把最大分量归一到 1（与 G7 的 [1/5,1]、apt 的 λ_m 同类）。

    实测 (24 模块, r=128, Qwen2-VL ← LLaVA-OV): mean 0.00691 / std 0.00380 /
    min 0.00306 / max 0.03482, 随机基线 0.00267, CV 0.79 → **有区分度**（预登记判据通过）。
    """
    Au = U_W.T @ U_tB
    Av = Vh_W @ Vh_tB.T
    a_j = (Au ** 2).sum(dim=0) * (Av ** 2).sum(dim=0)
    amax = float(a_j.max().item())
    if amax <= EPS:
        return torch.zeros_like(a_j), a_j
    return (a_j / amax).clamp(0.0, 1.0), a_j


def build_gate(gate_mode, rho, d, lam_apt, lam_energy, r, g7=None, g6=None, g7_donor=None,
               g_overlap=None, sigma=None, a_raw=None):
    """返回长度 r 的 gate 向量 (g ∈ [0,1])。

    T17 新增:
      mismatch_module    (G6) = 1/(1+mean_i c_i^op)   —— 模块标量, 先平均失配再倒数
      mismatch_component (G7) = 1/(1+c_i^op)          —— 逐组件
      mismatch_component_norm (G7 的 NormMatch 对照)  —— 与 G7 等更新预算的标量
      mismatch_component_donor (G7-D)                 —— 同一计划按 donor 列读取的逐组件 gate
    其中 c_i^op = 2(1-q_i), q_i = sum_j P_ij S_ij (保留符号)。
    """
    if gate_mode == "component":
        return rho
    if gate_mode == "mean":
        return torch.full_like(rho, float(rho.mean()))
    if gate_mode == "apt":
        return torch.full_like(rho, float(lam_apt))
    if gate_mode == "energy":
        return torch.full_like(rho, float(lam_energy))
    if gate_mode == "none":
        return torch.ones_like(rho)
    if gate_mode == "mismatch_module":
        if g6 is None:
            raise ValueError("gate=mismatch_module 需要 Stage-2 缓存中的 g6_mismatch (schema rapt-v2.1)")
        return torch.full_like(rho, float(g6))
    if gate_mode == "mismatch_component":
        if g7 is None:
            raise ValueError("gate=mismatch_component 需要 Stage-2 缓存中的 g7_mismatch (schema rapt-v2.1)")
        return g7
    if gate_mode == "mismatch_component_donor":
        if g7_donor is None:
            raise ValueError("gate=mismatch_component_donor 需要 Stage-2 缓存中的 P 与 operator_similarity")
        return g7_donor
    if gate_mode == "host_overlap":
        if g_overlap is None:
            raise ValueError("gate=host_overlap 需要 --host-subs (W_A 的 Stage-1 分解)")
        return g_overlap
    if gate_mode == "mismatch_component_norm":
        if g7 is None:
            raise ValueError("gate=mismatch_component_norm 需要 Stage-2 缓存中的 g7_mismatch")
        nd = float(d.norm().item())
        if nd <= EPS:
            return torch.zeros_like(rho)
        return torch.full_like(rho, min(float((g7 * d).norm().item() / nd), 1.0))
    if gate_mode == "norm_match":
        nd = float(d.norm().item())
        if nd <= EPS:
            return torch.zeros_like(rho)
        scalar = float((rho * d).norm().item() / nd)
        return torch.full_like(rho, min(scalar, 1.0))
    # ---- §24 Stage-2 的判决门 (§17.5 的 D2/D3, 规划里列出但**从未跑过**) ----
    # 规划的 S2-B / S3-B 要求把"对齐信息"和"谱锐化"两个解释分开:
    #   sigma_sharp : 与 a_j 门控**同等程度的锐化**, 但不含任何对齐含义 -> 纯锐化对照
    #   a_orth      : 把 a_j 中与 σ_τB 同序的成分回归掉, 才是**纯对齐**信息
    #   a_reverse   : 符号检验 (§17.4 S2-C) —— 若反向更好, 说明"对齐"是负向指标
    if gate_mode == "sigma_sharp":
        if sigma is None:
            raise ValueError("gate=sigma_sharp 需要 τ_B 的奇异值 S_B (Stage-2 缓存的 S_B 字段)")
        smax = float(sigma.max().item())
        if smax <= EPS:
            return torch.zeros_like(rho)
        return (sigma / smax).clamp(0.0, 1.0)
    if gate_mode in ("a_orth", "a_reverse"):
        if a_raw is None:
            raise ValueError(f"gate={gate_mode} 需要 --host-subs (用 host_overlap_gate 得到 a_j)")
        a = a_raw.float()
        if gate_mode == "a_reverse":
            amax = float(a.max().item())
            return (1.0 - a / amax).clamp(0.0, 1.0) if amax > EPS else torch.zeros_like(rho)
        # a_orth: 逐模块单变量回归 a = β·σ + 残差, 取残差为正的部分并 max 归一。
        # 为什么用非负截断: 强调项 U diag(g⊙d) Vᵀ 的语义是"放大", gate 取负会把它变成"反向抑制",
        # 那是另一个机制 —— 这里要隔离的是"对齐信息", 不引入新机制。
        if sigma is None:
            raise ValueError("gate=a_orth 需要 τ_B 的奇异值 S_B 做正交化回归")
        s = sigma.float()
        sc = s - s.mean()
        denom = float((sc * sc).sum().item())
        beta = float(((a - a.mean()) * sc).sum().item() / denom) if denom > EPS else 0.0
        resid = (a - beta * s).clamp_min(0.0)
        rmax = float(resid.max().item())
        return (resid / rmax).clamp(0.0, 1.0) if rmax > EPS else torch.zeros_like(rho)
    raise ValueError(f"Unknown gate: {gate_mode}")


def inject(args: argparse.Namespace) -> None:
    print("\n--- [RAPT v3 Stage-3: Target x Gate Decoupled Protective Injection] ---")
    # ============ §20.4 最小接口 → 旧参数的别名映射 (不改动任何数值语义) ============
    # 推荐入口: --mode inject --injection diag, 只暴露"一个参数 = 一个机制"的名字:
    #     ΔW = a·LR_R(Δ) + b·(Δ − LR_R(Δ)) + c·U_τB diag(g⊙d_B) V_τBᵀ
    #          └ w-main ┘   └──── w-tail ────┘   └─────── w-donor ───────┘
    # 旧参数 (--bulk-ia / --alpha / --inject-budget / --bulk-budget) 保留为别名,
    # 已有队列与产物不受影响。
    if getattr(args, "budget", None) is not None:
        args.inject_budget = args.budget
        args.bulk_budget = args.budget
    if getattr(args, "w_donor", None) is not None:
        args.alpha = float(args.w_donor)
    if getattr(args, "w_main", None) is not None:
        if not args.bulk_delta_subs:
            raise ValueError("--w-main 需要 --bulk-delta-subs 定义主/尾分界 R (即 LR_R(Δ) 的来源)")
        args.bulk_ia = float(args.w_main)
    if getattr(args, "w_main", None) is not None or float(getattr(args, "w_tail", 0.0)) > 0.0:
        print(f"[§20.4] 三项分解: a={args.bulk_ia} (LR_R(Δ))  b={getattr(args, 'w_tail', 0.0)} "
              f"(Δ−LR_R(Δ))  c={args.alpha} (donor band), R={args.bulk_rank}, "
              f"budget={getattr(args, 'bulk_budget', 'raw')}")
    # ---- §20.8 auto-config: 只留一个人类参数 s, 其余由机制导出 ----
    if getattr(args, "auto_config", False):
        s = float(args.transfer_ratio)
        if not (0.0 < s < 1.0):
            raise ValueError(f"--transfer-ratio 必须在 (0,1), 实测 s>0.125 待验证, 当前 {s}")
        if not args.donor_model:
            raise ValueError("--auto-config 需要 --donor-model 提供 Δ = W_B − W_A 用于定 c")
        # a = b = s  (无主/尾偏好) => 基础搬运项恰为 s·Δ, 主/尾分界 R 无关
        args.mode = "hybrid"
        args.interp_alpha = s
        args.identity_mapping = True          # 与 TA_hyb 同口径 (绕开 _donor_basis)
        args.auto_c = True
        args.alpha = 0.0                      # 显式置 0: c 由机制算, 手填值不参与;
        #   留在产物名里会是误导 (名字里的 _a0_ 表示"未使用", 真标识是 _auto_s{s})
        print(f"[§20.8] auto-config: s={s} (唯一人类参数); "
              f"a=b=s -> 基础搬运 = {s}·Δ (R 无关); "
              f"c 逐模块由等范数原则 c := s·‖Δ‖/‖band‖ 自动定")
    # =============================================================================

    # ================= §20.2 参数合法性守卫 (消除"传了却无效") =================
    # 设计原则 (用户 2026-09-15 提出): 每个符号与步骤都要有直观意义, 且不得重复。
    # 此前 --target/--gate 在 4 个分支被**静默忽略** (只在输出打 [未使用]),
    # 用户传了参数却没有效果 —— 改为直接报错。
    _target_used = (args.mode == "hybrid") or (args.mode == "inject" and args.injection == "diag")
    if not _target_used:
        if args.target != DEFAULT_TARGET:
            raise ValueError(
                f"--target={args.target} 在 mode={args.mode}/injection={args.injection} 下**不会生效** "
                f"(该分支不使用盈余 h)。请改用 mode=inject --injection diag, 或 mode=hybrid; "
                f"若确实不需要 target, 请显式保留默认值 --target {DEFAULT_TARGET}。"
            )
        if args.gate != DEFAULT_GATE:
            raise ValueError(
                f"--gate={args.gate} 在 mode={args.mode}/injection={args.injection} 下**不会生效** "
                f"(该分支不使用可靠性门控)。请改用 mode=inject --injection diag, 或 mode=hybrid; "
                f"若确实不需要 gate, 请显式保留默认值 --gate {DEFAULT_GATE}。"
            )
    if args.gate in GATES_DEPRECATED:
        print(f"[§20.2][Warn] --gate={args.gate} 已被实验证伪/退化: {GATES_DEPRECATED[args.gate]}; "
              f"仅应用于复现历史产物, 新实验请改用 apt / mean / energy / none / mismatch_module。")
    if getattr(args, "basis", None) == "donor" and not args.donor_subs:
        raise ValueError(
            "--basis donor 需要 --donor-subs 提供 donor 子空间; "
            "缺它会静默跳过全部模块 (产物退化为 W_A 副本)。")
    if float(getattr(args, "w_tail", 0.0)) > 0.0 and not args.bulk_delta_subs:
        raise ValueError(
            "--w-tail > 0 需要 --bulk-delta-subs 提供 LR_R(Δ); "
            "尾部定义为 Δ − LR_R(Δ), 缺 bulk 则无从定义主/尾分界。")
    if float(getattr(args, "w_tail", 0.0)) > 0.0 and not args.donor_model:
        raise ValueError("--w-tail > 0 需要 --donor-model 提供 Δ = W_B − W_A。")
    # ========================================================================
    if args.gate == "norm_match" and args.disable_selection:
        raise ValueError("G5(NormMatch) 需要非负 d=[h]_+, 不能与 --disable-selection(signed) 混用")
    if args.mode == "direct":
        if not args.donor_subs:
            raise ValueError("--mode direct (Stage-3 插值基线) 需要 --donor-subs 提供 donor rank-r 基")
        print("[Note] mode=direct: Stage-3 退化为插值 W*=(1-a)W_A + a*U_B diag(sigma_used) Vh_B; target/gate 不生效")

    weights_A = load_weights(args.base_model)
    stage2_obj = torch.load(args.stage2, map_location="cpu", weights_only=True)
    modules = stage2_obj.get("modules", stage2_obj)
    stage2_meta = stage2_obj.get("meta", {})
    base_subs, base_meta = load_subspaces_full(args.base_subs)
    donor_subs, donor_meta = load_subspaces_full(args.donor_subs) if args.donor_subs else (None, {})
    host_subs = load_subspaces(args.host_subs) if getattr(args, "host_subs", None) else None
    tau_a_subs = load_subspaces(args.tau_a_subs) if getattr(args, "tau_a_subs", None) else None
    bulk_subs = load_subspaces(args.bulk_delta_subs) if getattr(args, "bulk_delta_subs", None) else None
    if bulk_subs is not None:
        print(f"[§18] Δ 低秩 bulk 项: ia={args.bulk_ia}, rank≤{args.bulk_rank} "
              f"(来自 {args.bulk_delta_subs})")
    # ---- §24.9 防呆: --rank 在 inject/hybrid 模式下**不生效** ----
    # 主循环里 `r = int(blk["rank_A"])` —— 实际秩来自 Stage-2 缓存, 而不是 --rank。
    # 实测后果: `--rank 512` 配 `stage2_dual_r128_*.pt`, 生成物与 `--rank 128` **逐字节相同**
    # (MD5 一致), 却因产物名不同而白跑一次筛查 (2026-09-16 A2/B2 各白跑一次, §24.10)。
    # 注意: inject 模式走同一条 `r = blk["rank_A"]`, 故同样不生效 —— 两种模式都要警告。
    if args.mode in ("hybrid", "inject") and modules:
        _ranks = {int(v.get("rank_A", -1)) for v in modules.values()}
        if _ranks and (len(_ranks) > 1 or args.rank not in _ranks):
            print(f"[§24.9][Warn] --rank={args.rank} 在 mode={args.mode} 下**不生效**: "
                  f"实际秩取自 Stage-2 缓存的 rank_A={sorted(_ranks)}。"
                  f"要换秩必须重新生成与目标秩匹配的 Stage-1/Stage-2 缓存。")
    if args.injection == "bidir":
        if tau_a_subs is None:
            raise ValueError("--injection bidir (BTI, §17.6) 需要 --tau-a-subs 提供 τ_A 分解")
        print(f"[BTI] 联合基注入: τ_B (--base-subs) + τ_A (--tau-a-subs, "
              f"w_A={args.bidir_a_weight}); 放弃非负锥与门控 (见 §17.6)")
    if args.gate == "host_overlap":
        if host_subs is None:
            raise ValueError("gate=host_overlap (候选 A) 需要 --host-subs 提供 W_A 的 Stage-1 分解; "
                             "注意 TA 分支下 --base-subs 是 τ_B 分解, 二者不同")
        print(f"[候选A] host 主子空间基来自 {args.host_subs} ({len(host_subs)} 模块)")
    base_decomp = base_meta.get("decomp_object", "absolute_weight")
    args.decomp_tag = base_decomp
    if base_decomp == "task_vector":
        print(f"[TA] 保护域 = task vector 基 (τ = W − {base_meta.get('anchor_model')}); "
              f"注入仍作用于 W_A, 但受限方向来自微调方向而非预训练主方向")
        if args.donor_subs and donor_meta.get("decomp_object") != "task_vector":
            print("[TA][Warn] donor 子空间不是 task vector 分解 —— 两侧分解对象不一致, "
                  "S_ij 的语义不可比")
        # 索引防呆: 逐组件 gate 的索引口径必须与注入分量一致。
        #   component(ρ) / mismatch_component(G7) / mismatch_component_norm 的 i 索引 **host 分量**;
        #   而注入基为 donor 基时被修改的是 **donor 分量 j** —— 照搬会造成索引错配。
        #   标量 gate (apt/mean/energy/none/mismatch_module) 与索引无关, 不受影响。
        if args.gate in ("component", "mismatch_component", "mismatch_component_norm"):
            print(f"[TA][Warn] --gate {args.gate} 是 host 索引口径, 而注入基为 donor 基; "
                  f"请改用 --gate mismatch_component_donor 或标量 gate "
                  f"(apt/mean/energy/none/mismatch_module)")
        if args.target != "task_delta":
            print(f"[TA][Warn] 保护域是 task vector 基, 但 target={args.target} "
                  f"仍按绝对权重谱定义盈余 —— 二者口径不一致, 建议 --target task_delta")
    donor_index = None
    need_donor_weights = (args.target in ("full_donor_response", "task_delta")
                          or args.direct_full
                          or args.injection in ("subspace", "bidir")
                          or args.mode == "hybrid"
                          or float(getattr(args, "w_tail", 0.0)) > 0.0)
    if need_donor_weights:
        if not args.donor_model:
            raise ValueError("E3 / --direct-full / --injection subspace 需要 --donor-model 提供完整 donor 权重")
        donor_index = _build_donor_index(load_weights(args.donor_model))
        print(f"[donor] full-weight index built: {len(donor_index)} language-side modules")
    merged = weights_A.copy()

    n_linear = n_used = n_reused = n_missing = n_svd_fallback = 0
    pos_cnt = tot_cnt = gate_zero = gate_cnt = 0
    perm_fallback = perm_cnt = 0
    perm_agree_sum = 0.0
    calib_degenerate = 0
    calib_gap_sum = 0.0
    calib_gap_cnt = 0
    gate_sum = 0.0
    h_sum = 0.0
    ideal_delta_sq = stored_delta_sq = 0.0
    off_ratio_max = 0.0
    cos_delta_sum = 0.0
    cos_delta_cnt = 0
    bulk_rel_sum = 0.0
    bulk_rel_cnt = 0
    tail_rel_sum = 0.0
    tail_rel_cnt = 0
    c_auto_sum = 0.0
    c_auto_cnt = 0
    per_module: Dict[str, Dict[str, float]] = {}

    for key in tqdm(list(weights_A.keys()), desc="RAPT v3 Injection", disable=bool(args.no_progress)):
        if not (need_merge(key) and key.endswith(".weight") and weights_A[key].ndim == 2):
            continue
        n_linear += 1
        mod = canon_module_from_param_key(key)
        lam = _layer_mult(mod, args)          # §22 逐层强度 (未配置时恒为 1.0)
        # §25.2 作用域: 逐层强度乘在"整份更新"还是只乘某一项。
        #   机制上 Δ(位移) 与 band(donor 知识) 是两个正交来源, L1 只测了"同时乘",
        #   这里把它们拆开 (L1a=只乘 band, L1b=只乘 Δ)。
        _scope = getattr(args, "layer_scope", "all")
        lam_ia = lam if _scope in ("all", "ia") else 1.0
        lam_band = lam if _scope in ("all", "band") else 1.0
        blk = modules.get(mod)
        if blk is None and ".multiway." in mod:
            blk = modules.get(strip_multiway_segment(mod))
        if blk is None:
            continue

        W_A = weights_A[key].float()
        r = int(blk["rank_A"])
        resolved = _resolve_base_svd(mod, W_A, r, base_subs)
        if resolved is None:
            # 正式实验要求复用同一 host 分解; 缺失即报错, 不静默重算基。
            n_missing += 1
            continue
        U_A, S_A, Vh_A, reused = resolved
        n_reused += int(reused)
        n_svd_fallback += int(not reused)

        r_eff = min(r, S_A.numel())
        U_A, S_A, Vh_A = U_A[:, :r_eff], S_A[:r_eff], Vh_A[:r_eff, :]

        similarity = blk["operator_similarity"].float()[:r_eff, :r_eff]
        S_B = blk["S_B"].float().view(-1)[:r_eff]
        sigma_hat = blk["sigma_hat_B"].float().view(-1)[:r_eff]
        t_operator = blk["t_operator"].float().view(-1)[:r_eff]

        if args.identity_mapping:
            # P = I: 所有依赖 P 的量按 I 重算, 不读 M 版缓存。
            rho = torch.diagonal(similarity).clamp(0.0, 1.0).contiguous()
            sigma_used = S_B
            lam_apt = float(blk["lambda_apt_identity"])
            q_raw = torch.diagonal(similarity).clamp(-1.0, 1.0).contiguous()
            g7 = mismatch_gate_from_q(q_raw)
            g6 = float(mismatch_gate_from_q(q_raw.mean()).item())
        else:
            rho = blk["rho_component"].float().view(-1)[:r_eff]
            sigma_used = sigma_hat
            lam_apt = float(blk["lambda_apt"])
            if "g7_mismatch" not in blk:
                raise KeyError(
                    f"Stage-2 缓存缺少 T17 字段 (q_raw/g7_mismatch); "
                    f"请用 rapt_v3_stage2_mapping 重新生成缓存 (schema {SCHEMA_VERSION})"
                )
            q_raw = blk["q_raw"].float().view(-1)[:r_eff]
            g7 = blk["g7_mismatch"].float().view(-1)[:r_eff]
            g6 = float(blk["g6_mismatch"])
        # donor 侧逐组件 gate: 同一运输计划按**列**读取 (TA 分支注入 donor 基时用)。
        # 惰性计算 —— 未选用该 gate 时不读 P, 避免对缺 P 字段的旧缓存产生新依赖。
        # identity_mapping 时 P = I, 行/列读取退化为同一个量。
        g7_donor = None
        if args.gate == "mismatch_component_donor":
            if args.identity_mapping:
                g7_donor = g7
            else:
                if "P" not in blk:
                    raise KeyError("gate=mismatch_component_donor 需要 Stage-2 缓存中的 P 字段")
                g7_donor = donor_component_gate(
                    blk["P"].float()[:r_eff, :r_eff], similarity)
        # 候选 A (host_overlap): 逐 donor 任务方向与 host 主子空间的重叠度。
        # 需要 W_A 的 Stage-1 分解 (--host-subs), 与注入基 (--base-subs) 不同。
        g_overlap, overlap_raw_mean, a_raw = None, None, None
        # §24: a_orth / a_reverse 同样需要 a_j, 故共用这一块计算
        if args.gate in ("host_overlap", "a_orth", "a_reverse"):
            hb = host_subs.get(mod)
            if hb is None and ".multiway." in mod:
                hb = host_subs.get(strip_multiway_segment(mod))
            if hb is None:
                raise KeyError(f"--host-subs 缺少模块 {mod}")
            U_W = hb["U"].float()[:, :r_eff].to(device=U_A.device, dtype=U_A.dtype)
            Vh_W = hb["Vh"].float()[:r_eff, :].to(device=U_A.device, dtype=U_A.dtype)
            g_overlap, a_raw = host_overlap_gate(U_W, Vh_W, U_A, Vh_A)
            overlap_raw_mean = float(a_raw.mean().item())
        lam_energy = float(blk["lambda_energy"])

        W_B_full = None
        if donor_index is not None:
            W_B_full = donor_index.get(mod)
            if W_B_full is None and ".multiway." in mod:
                W_B_full = donor_index.get(strip_multiway_segment(mod))
            if W_B_full is not None and W_B_full.shape != W_A.shape:
                W_B_full = None

        # ---- 盈余 h 的计算统一走 compute_surplus_h (生产函数, 有直接单测) ----
        h, t_aux = compute_surplus_h(
            args.target, blk, S_A, S_B, sigma_used, t_operator, r_eff,
            W_B_full=W_B_full, U_A=U_A, Vh_A=Vh_A, W_A=W_A,
        )
        if h is None:
            n_missing += 1
            continue
        if t_aux:
            if "perm" in t_aux:              # 仅 calibrated/hard 会带指派诊断
                if not t_aux["exact"]:
                    perm_fallback += 1
                perm_agree_sum += float(
                    (t_aux["perm"] == torch.arange(r_eff)).float().mean())
                perm_cnt += 1
            gap = t_aux.get("baseline_gap")
            if gap is not None:
                calib_gap_sum += gap
                calib_gap_cnt += 1
                if gap < 1e-12:          # 谱完全无差 -> 校准盈余恒 0, 记为退化
                    calib_degenerate += 1
        d = h if args.disable_selection else h.clamp_min(0.0)

        gate = build_gate(args.gate, rho, d, lam_apt, lam_energy, r_eff,
                          g7=g7, g6=g6, g7_donor=g7_donor, g_overlap=g_overlap,
                          sigma=S_B, a_raw=a_raw)

        # 分支用途标记: mode=direct 与 injection=subspace 都**不使用** target/gate,
        # 若照常上报会把未参与计算的默认值当成有效诊断 (V0.3.2 修正)。
        gate_used = (args.mode == "hybrid") or (args.mode == "inject" and args.injection == "diag")
        target_used = gate_used

        if args.mode == "direct":
            if args.direct_full:
                # 插值基线 (alpha ∈ [0,1]): W* = (1-a)W_A + a*D_hat
                #   --identity-mapping => D_hat = W_B          (纯插值, = AdaMMS 口径)
                #   算子映射          => D_hat = W_B + U_B diag(sigma_hat - sigma_B) V_B^T
                #                                        (donor 的 top-r 谱被运输谱替换, 其余不动)
                # 这样 Stage-2 的修改直接体现在"被插值的对象"上, 且更新量级与纯插值同口径。
                if W_B_full is None:
                    n_missing += 1
                    continue
                D_hat = W_B_full.float()
                if not args.identity_mapping:
                    basis = _donor_basis(donor_subs, blk, W_A, r_eff)
                    if basis is None:
                        n_missing += 1
                        continue
                    U_R, Vh_R = basis
                    D_hat = D_hat + (U_R * (sigma_used - S_B).unsqueeze(0)) @ Vh_R
                W_new = (1.0 - float(args.alpha)) * W_A + float(args.alpha) * D_hat
                eta = torch.zeros_like(d)
            else:
                basis = _donor_basis(donor_subs, blk, W_A, r_eff)
                if basis is None:
                    n_missing += 1
                    continue
                U_R, Vh_R = basis
                W_new = (1.0 - float(args.alpha)) * W_A + float(args.alpha) * (U_R * sigma_used.unsqueeze(0)) @ Vh_R
                eta = torch.zeros_like(d)
        elif args.mode == "hybrid":
            # 组合方案: 插值 (负责方向性迁移) + 对齐保持的保护性谱补强 (负责 MMMU 类增益)
            #   W* = (1-ia) W_A + ia * D_hat + alpha * U_A diag(g ⊙ d) V_A^T
            #   ia = --interp-alpha (0..1), alpha = 注入系数 (0..1)
            if W_B_full is None:
                n_missing += 1
                continue
            ia = float(args.interp_alpha) * lam_ia   # §22/§25.2 逐层强度: 稠密位移项
            D_hat = W_B_full.float()
            if not args.identity_mapping:
                basis = _donor_basis(donor_subs, blk, W_A, r_eff)
                if basis is None:
                    n_missing += 1
                    continue
                U_R, Vh_R = basis
                D_hat = D_hat + (U_R * (sigma_used - S_B).unsqueeze(0)) @ Vh_R
            if getattr(args, "auto_c", False):
                # ---- §20.8 等范数原则: c := s·‖Δ‖_F / ‖band(未门控)‖_F ----
                # 语义: "额外要的 donor 专长, 与整体搬运 s·Δ 的量级相当"。
                # 逐模块计算 (不同模块的 ‖Δ‖/‖band‖ 比不同), 而非全局常数。
                band_ung = (U_A * d.unsqueeze(0)) @ Vh_A
                bn = float(band_ung.norm().item())
                dn = float((W_B_full.float() - W_A).norm().item())
                c_mod = (ia * dn / bn) if bn > EPS else 0.0
                c_auto_sum += c_mod
                c_auto_cnt += 1
                eta = c_mod * gate * d
                del band_ung
            else:
                eta = float(args.alpha) * lam_band * gate * d
            Delta = (U_A * eta.unsqueeze(0)) @ Vh_A
            W_new = (1.0 - ia) * W_A + ia * D_hat + Delta
            # ---- §25: τ_A 项只减主方向 (--tau-a-mode trunc) ----
            # 机制形式 (来自 §24.9): W* = W_A − ia·τ_A + (ia+c′)·LR_r(τ_B) + ia·tail(τ_B)。
            # 其中 −ia·τ_A 是对 host 任务向量的**全谱**相减。直觉 (§25.1):
            #   「远离 host 任务向量时, 只应远离它的主方向; 其尾部方向是噪声/背景,
            #     不该被减去」。trunc 模式把相减项换成 −ia·LR_r(τ_A):
            #   W* = W_A + ia·(τ_B − τ_A) + ia·tail(τ_A) + band
            # 实现: 在标准结果上再加回 ia·tail(τ_A) (≡ −(−ia·τ_A 的尾部))。
            if getattr(args, "tau_a_mode", "full") == "avoid":
                # §25.1 结构变体: 位移避开 host 任务的主方向。
                # 实际语义 (无 anchor 可用, 只能这样实现):
                #   W* = (1−ia)W_A + ia·(W_B − LR_r(τ_A)) + band
                #      = W_A + ia·(Δ − LR_r(τ_A)) + band
                # 直觉: 「位移方向不要落在 host 任务的主方向上 —— 那些方向是 host 已掌握的
                #   任务结构, 扰动它们既破坏 host 又不带来新知识」。
                # 注意与 APT 的 U_W 保护的区别: 这里是 U_τA (host **任务**基, reach 0.610),
                #   不是预训练主基 (reach 0.083, §8.8.7 已判否)。
                # 对照数据点: duA128c0p5 (把 τ_A 主方向**加**进 band) = 214.482 (中性偏负)。
                ta = tau_a_subs.get(mod) if tau_a_subs else None
                if ta is None and ".multiway." in mod and tau_a_subs:
                    ta = tau_a_subs.get(strip_multiway_segment(mod))
                if ta is None:
                    raise KeyError(f"--tau-a-mode trunc 需要 --tau-a-subs 提供 {mod} 的 τ_A 分解")
                r_ta = min(r_eff, int(ta["S"].numel()))
                U_ta = ta["U"].float()[:, :r_ta].to(device=W_A.device, dtype=W_A.dtype)
                S_ta = ta["S"].float()[:r_ta].to(device=W_A.device, dtype=W_A.dtype)
                Vh_ta = ta["Vh"].float()[:r_ta, :].to(device=W_A.device, dtype=W_A.dtype)
                LR_ta = (U_ta * S_ta.unsqueeze(0)) @ Vh_ta
                W_new = (1.0 - ia) * W_A + ia * (D_hat - LR_ta) + Delta
            ideal_delta_sq += float(Delta.pow(2).sum().item())
        elif args.injection == "bidir":
            # ---- BTI (§17.6): 联合基**有符号**对角注入 ----
            #   W* = W_A + alpha * [ U_τB diag(d_B) V_τB^T + w_A * U_τA diag(diag Z_A) V_τA^T ]
            #   d_B     = [diag(U_τB^T Δ V_τB^T)]_+     (实测 pos_ratio=1.0000 → 恒全正)
            #   diag Z_A =  diag(U_τA^T Δ V_τA^T)       (实测 pos_ratio=0.0000 → 恒全负, **有符号保留**)
            #
            # 与 APT 的差别 (§17.6(e)): 放弃非负锥 (τ 基下 pos_ratio=1.0, 非负约束**冗余**而非有害)
            # 与可靠性门控 (实测有害: 门控+加倍预算 = 199.103)。
            # 实测 cos(ΔW, Δ) = 0.7581, 是所有结构化方案中最高 (E0 0.014 / TA 0.268 / dense 1.0)。
            if W_B_full is None:
                n_missing += 1
                continue
            D = W_B_full.float() - W_A
            a_subs = tau_a_subs.get(mod)
            if a_subs is None and ".multiway." in mod:
                a_subs = tau_a_subs.get(strip_multiway_segment(mod))
            if a_subs is None:
                raise KeyError(f"--tau-a-subs 缺少模块 {mod}")
            U_A2 = a_subs["U"].float()[:, :r_eff].to(device=U_A.device, dtype=U_A.dtype)
            Vh_A2 = a_subs["Vh"].float()[:r_eff, :].to(device=U_A.device, dtype=U_A.dtype)
            dB = torch.diagonal(U_A.T @ D @ Vh_A.T).clamp_min(0.0)
            dA = torch.diagonal(U_A2.T @ D @ Vh_A2.T)               # 有符号, 不截断
            Delta = (U_A * dB.unsqueeze(0)) @ Vh_A
            if float(args.bidir_a_weight) != 0.0:
                Delta = Delta + float(args.bidir_a_weight) * ((U_A2 * dA.unsqueeze(0)) @ Vh_A2)
            if getattr(args, "inject_budget", "raw") == "rel":
                dn0 = float(Delta.norm().item())
                Delta = (Delta * (float(args.alpha) * float(W_A.norm().item()) / dn0)
                         if dn0 > EPS else torch.zeros_like(Delta))
            else:
                Delta = float(args.alpha) * Delta
            W_new = W_A + Delta
            eta = dB
            ideal_delta_sq += float(Delta.pow(2).sum().item())
        elif args.injection == "subspace":
            # 子空间内插值: 把 donor-host 差异投影到 host top-r 子空间对后整体注入 (可旋转, 保留块内结构)
            if W_B_full is None:
                n_missing += 1
                continue
            Z = U_A.T @ (W_B_full.float() - W_A) @ Vh_A.T
            Delta = float(args.alpha) * lam_band * (U_A @ Z @ Vh_A)
            W_new = W_A + Delta
            eta = Z.diagonal().clone()
            ideal_delta_sq += float(Delta.pow(2).sum().item())
        else:
            basis_choice = args.basis or "host"
            if basis_choice == "donor":
                sel = _donor_basis(donor_subs, blk, W_A, r_eff)
            elif basis_choice == "random":
                sel = _random_basis(W_A, r_eff, args.seed)
            else:
                sel = (U_A, Vh_A)
            if sel is None:
                n_missing += 1
                continue
            U_R, Vh_R = sel
            eta = float(args.alpha) * lam_band * gate * d
            if getattr(args, "inject_budget", "raw") == "rel":
                # ---- 预算归一化注入 (V0.3.12) ----
                # 问题: α 作为"盈余位移系数"时量纲随 target 变化 ——
                #   E0(τ=0.15) α=1 给出 rel≈0.0114, 而校准后的 target 同一 α=1 只有 0.0023,
                #   于是为了对齐量级不得不手填 α=4.9, 超出 [0,1] 契约且不可解释。
                # 解决: 让 α 表示**相对扰动预算** rel = ‖Δ‖_F/‖W_A‖_F (∈[0,1]), 物理缩放由方法算:
                #   ‖U_R diag(η) Vh_R‖_F = ‖η‖₂   (U_R 列正交, Vh_R 行正交)
                #   => Δ = α · ‖W_A‖_F · U_R diag(η/‖η‖₂) Vh_R,  即 rel ≡ α。
                # 这样 α 与 target/gate/rank 全部解耦, 跨配置可比, 且始终在 [0,1] 内。
                eta = normalize_injection_budget(
                    eta, float(args.alpha), float(W_A.norm().item()))
            Delta = (U_R * eta.unsqueeze(0)) @ Vh_R
            W_new = W_A + Delta
            ideal_delta_sq += float(Delta.pow(2).sum().item())

        # ---- §18: Δ 低秩 bulk 项 (替换稠密插值项; 与上面的 band 注入相加) ----
        if bulk_subs is not None:
            db = bulk_subs.get(mod)
            if db is None and ".multiway." in mod:
                db = bulk_subs.get(strip_multiway_segment(mod))
            if db is None:
                raise KeyError(f"--bulk-delta-subs 缺少模块 {mod}")
            # 注意: bulk 的秩**不受 --rank 约束** (--rank 管的是 band 注入基的秩)。
            # 早期版本写成 min(bulk_rank, r_eff, ...) 导致 --bulk-rank 512 被 --rank 128 截断。
            rb = min(int(args.bulk_rank), int(db["S"].numel()))
            U_b = db["U"].float()[:, :rb].to(device=W_A.device, dtype=W_A.dtype)
            S_b = db["S"].float()[:rb].to(device=W_A.device, dtype=W_A.dtype)
            Vh_b = db["Vh"].float()[:rb, :].to(device=W_A.device, dtype=W_A.dtype)
            bulk_raw = (U_b * S_b.unsqueeze(0)) @ Vh_b
            if getattr(args, "bulk_budget", "raw") == "rel":
                # 逐模块把 bulk 归一到目标相对预算, 使 bulk 与稠密插值项
                # `ia·Δ` 的 per-module rel 完全一致 (可比性最强)。
                bn = float(bulk_raw.norm().item())
                bulk = (bulk_raw * (float(args.bulk_ia) * float(W_A.norm().item()) / bn)
                        if bn > EPS else torch.zeros_like(bulk_raw))
            else:
                bulk = float(args.bulk_ia) * bulk_raw
            # ---- §20 新增: Δ 的**尾部**项 b·(Δ − LR_R(Δ)) ----
            # 把更新写成三个**互不冗余**的正交项 (§20.1):
            #     ΔW = a·LR_R(Δ) + b·(Δ − LR_R(Δ)) + c·U_τB diag(g⊙d_B) V_τBᵀ
            # 其中 LR_R(Δ) 与 Δ−LR_R(Δ) **按 SVD 截断定义正交**, band 落在 U_τB
            # (对 Δ 的 reach 仅 0.249, 与 U_Δ 的 0.647 非嵌套) —— 三项确实张成三个方向。
            # `b` 是本项目**从未独立调过**的自由度: 既有实验只落在 b=a (dense) 或 b=0 (纯 bulk)。
            # 语义: 尾部只占 ‖Δ‖ 的 11%, 但 C1(b=0, 216.095) 与 TA_hyb(b=0.0031, 216.731)
            # 差 0.64 —— 影响远超其范数占比, 值得单独扫。
            tail = None
            if float(getattr(args, "w_tail", 0.0)) > 0.0:
                if W_B_full is None:
                    raise RuntimeError(
                        "--w-tail > 0 需要 Δ = W_B − W_A, 请提供 --donor-model (或 --direct-full)")
                D_full = W_B_full.to(device=W_A.device, dtype=W_A.dtype) - W_A
                tail_raw = D_full - bulk_raw              # Δ − LR_R(Δ), 与 bulk_raw 正交
                tn = float(tail_raw.norm().item())
                if tn > EPS:
                    if getattr(args, "bulk_budget", "raw") == "rel":
                        tail = tail_raw * (float(args.w_tail) * float(W_A.norm().item()) / tn)
                    else:
                        tail = float(args.w_tail) * tail_raw
                    W_new = W_new + tail
                    tail_rel_sum += float(tail.norm().item()) / max(float(W_A.norm().item()), EPS)
                    tail_rel_cnt += 1
                    ideal_delta_sq += float(tail.pow(2).sum().item())
                del D_full, tail_raw

            W_new = W_new + bulk
            bulk_rel_sum += float(bulk.norm().item()) / max(float(W_A.norm().item()), EPS)
            bulk_rel_cnt += 1
            ideal_delta_sq += float(bulk.pow(2).sum().item())

        merged[key] = W_new.to(weights_A[key].dtype)
        stored_delta_sq += float((merged[key].float() - W_A).pow(2).sum().item())
        n_used += 1

        pos_cnt += int((h > 0).sum().item())
        tot_cnt += r_eff
        gate_zero += int((gate <= EPS).sum().item())
        gate_sum += float(gate.sum().item())
        h_sum += float(h.sum().item())
        gate_cnt += r_eff

        if args.mode != "direct":
            Dn = float(Delta.norm().item())
            if Dn > EPS:
                off_u = float((Delta - U_A @ (U_A.T @ Delta)).norm().item()) / Dn
                off_v = float((Delta - (Delta @ Vh_A.T) @ Vh_A).norm().item()) / Dn
                off_ratio_max = max(off_ratio_max, off_u, off_v)
            # §17.6(b): cos(ΔW, Δ) —— 本模型对上迄今唯一与分数单调对应的权重侧量。
            # 需要 donor 完整权重 (W_B_full); 缺失时跳过。
            if W_B_full is not None:
                # 用**总更新量** (含 §18 bulk 项) 与 Δ 比, 而非只看 band 的 Delta
                Dtot = W_new.float() - W_A
                Dtn = float(Dtot.norm().item())
                Ddiff = W_B_full.float() - W_A
                ddn = float(Ddiff.norm().item())
                if Dtn > EPS and ddn > EPS:
                    cos_delta_sum += float((Dtot * Ddiff).sum().item()) / (Dtn * ddn)
                    cos_delta_cnt += 1

        per_module[mod] = {
            "rank": r_eff,
            "target": args.target if target_used else None,
            "gate": args.gate if gate_used else None,
            "target_used": bool(target_used),
            "gate_used": bool(gate_used),
            "q_mean": float(q_raw.mean().item()),
            "g6_mismatch": float(g6),
            "g7_mismatch_mean": float(g7.mean().item()),
            "g7_donor_mean": (float(g7_donor.mean().item()) if g7_donor is not None else None),
            "host_overlap_raw_mean": overlap_raw_mean,
            "positive_surplus_ratio": float((h > 0).float().mean().item()) if target_used else None,
            "relative_spectral_surplus": float(d.sum().item() / S_A.sum().clamp_min(EPS).item()) if target_used else None,
            "gate_mean": float(gate.mean().item()) if gate_used else None,
            "gate_zero_ratio": float((gate <= EPS).float().mean().item()) if gate_used else None,
            "eta_l1": float(eta.abs().sum().item()),
            "relative_perturbation": float((W_new - W_A).norm().item() / W_A.norm().clamp_min(EPS).item()),
        }

    # ---- 防呆 (2026-09-14 22:45, 必须在写盘**之前**): 0 更新的产物是 W_A 的完整副本,
    # 会静默污染筛查队列。实际发生过: hybrid 模式漏传 --donor-subs 且未设 --identity-mapping
    # -> `_donor_basis` 返回 None -> 196 个模块全部 continue -> 产物 = base,
    # 实测 SUM3=212.192 与 base 212.19 完全一致, 白跑 20 分钟筛查才发现。
    if n_linear > 0 and n_used == 0:
        raise RuntimeError(
            f"Stage-3 未更新任何模块 (updated=0/{n_linear}, missing_base_svd={n_missing}); "
            f"产物将是 W_A 的完整副本, 已中止写盘。常见原因: "
            f"① hybrid 模式缺 --donor-subs 且未设 --identity-mapping; "
            f"② --base-subs 与 --base-model 的模块名对不上; "
            f"③ Stage-2 缓存与 Stage-1 不是同一次分解。"
        )
    if n_linear > 0 and n_used < 0.5 * n_linear:
        print(f"[Warn] 只有 {n_used}/{n_linear} 个模块被更新 —— 产物与 W_A 高度接近, "
              f"请核对 --base-subs / --donor-subs / Stage-2 缓存是否匹配")

    out_root = _save_model(args, merged)
    stored_vs_ideal = (stored_delta_sq ** 0.5) / max(ideal_delta_sq ** 0.5, EPS)
    meta = {
        "method": "RAPT-V0.3-ProtectiveInjection",
        "code_version": "rapt_V0_3",
        "schema_version": SCHEMA_VERSION,
        "base_model": args.base_model,
        "stage2": args.stage2,
        "stage2_code_version": stage2_meta.get("code_version"),
        "base_subs": args.base_subs,
        "donor_subs": args.donor_subs,
        "host_subs": getattr(args, "host_subs", None),
        "donor_model": args.donor_model,
        "base_decomp_object": base_decomp,
        "base_anchor_model": base_meta.get("anchor_model"),
        "donor_decomp_object": donor_meta.get("decomp_object", "absolute_weight"),
        "donor_anchor_model": donor_meta.get("anchor_model"),
        "decomp_tag": getattr(args, "decomp_tag", None),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "layer_split": getattr(args, "layer_split", None),
        "layer_gain": getattr(args, "layer_gain", None),
        "layer_scope": getattr(args, "layer_scope", "all"),
        "tau_a_mode": getattr(args, "tau_a_mode", "full"),
        "target": args.target,
        "gate": args.gate,
        "target_used": bool(target_used),
        "gate_used": bool(gate_used),
        "mode": args.mode,
        "inject_budget": getattr(args, "inject_budget", "raw"),
        "interp_alpha": float(args.interp_alpha),
        "injection": args.injection,
        "tau_a_subs": getattr(args, "tau_a_subs", None),
        "bidir_a_weight": float(getattr(args, "bidir_a_weight", 1.0)),
        "bulk_delta_subs": getattr(args, "bulk_delta_subs", None),
        "bulk_ia": float(getattr(args, "bulk_ia", 0.0)),
        "bulk_budget": getattr(args, "bulk_budget", "rel"),
        "bulk_rank": int(getattr(args, "bulk_rank", 0)),
        "bulk_rel_mean": (bulk_rel_sum / bulk_rel_cnt) if bulk_rel_cnt else None,
        "w_tail": float(getattr(args, "w_tail", 0.0)),
        "tail_rel_mean": (tail_rel_sum / tail_rel_cnt) if tail_rel_cnt else None,
        "auto_config": bool(getattr(args, "auto_config", False)),
        "transfer_ratio": float(getattr(args, "transfer_ratio", 0.0)),
        "c_auto_mean": (c_auto_sum / c_auto_cnt) if c_auto_cnt else None,
        "cos_delta_mean": (cos_delta_sum / cos_delta_cnt) if cos_delta_cnt else None,
        "direct_full": bool(args.direct_full),
        "basis": args.basis or "host",
        "tag_suffix": getattr(args, "tag_suffix", None),
        "perm_cnt": perm_cnt,
        "perm_fallback_greedy": perm_fallback,
        "perm_identity_agreement": (perm_agree_sum / perm_cnt) if perm_cnt else None,
        "calib_degenerate_modules": calib_degenerate,
        "calib_baseline_gap_mean": (calib_gap_sum / calib_gap_cnt) if calib_gap_cnt else None,
        "identity_mapping": bool(args.identity_mapping),
        "disable_selection": bool(args.disable_selection),
        "seed": int(args.seed),
        "linear_layers": n_linear,
        "updated_layers": n_used,
        "reused_subspaces": n_reused,
        "missing_base_svd": n_missing,
        "positive_surplus_ratio": (pos_cnt / max(1, tot_cnt)) if target_used else None,
        "gate_mean": (gate_sum / max(1, gate_cnt)) if gate_used else None,
        "gate_zero_ratio": (gate_zero / max(1, gate_cnt)) if gate_used else None,
        "h_mean": (h_sum / max(1, tot_cnt)) if target_used else None,
        "stored_over_ideal_delta_norm": stored_vs_ideal,
        "off_subspace_ratio_max": off_ratio_max,
        "per_module": per_module,
    }
    with open(osp.join(out_root, "merge_meta_rapt_v2.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Done] RAPT v3 Stage-3 -> {out_root}; updated={n_used}/{n_linear} missing_base_svd={n_missing}")
    if base_decomp == "task_vector":
        print(f"  [TA] 保护域 = task vector 基 (anchor={base_meta.get('anchor_model')})")
    _gm = "n/a(未使用)" if meta["gate_mean"] is None else f"{meta['gate_mean']:.4f}"
    _ps = "n/a(未使用)" if meta["positive_surplus_ratio"] is None else f"{meta['positive_surplus_ratio']:.4f}"
    print(
        f"  target={args.target}{'' if target_used else ' [未使用]'}, "
        f"gate={args.gate}{'' if gate_used else ' [未使用]'}, alpha={args.alpha}, "
        f"gate_mean={_gm}, positive_surplus_ratio={_ps}, "
        f"off_subspace_max={off_ratio_max:.2e}, stored/ideal={stored_vs_ideal:.4f}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAPT v3 Stage-3 target x gate decoupled injection")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--base-subs", default=None)
    ap.add_argument("--host-subs", dest="host_subs", default=None,
                    help="候选 A (gate=host_overlap) 需要: W_A 的 Stage-1 分解 (host 主子空间 U_W/V_W)。"
                         "TA 分支下 --base-subs 是 τ_B 分解, 二者是不同对象, 必须分别给出")
    ap.add_argument("--donor-subs", dest="donor_subs", default=None)
    ap.add_argument("--donor-model", dest="donor_model", default=None,
                    help="donor 完整权重目录 (仅 --target full_donor_response / E3 需要)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.5)
    # ---- §22 逐层强度 (本项目此前**从无任何系数随层变化**) ----
    ap.add_argument("--layer-split", dest="layer_split", type=int, default=None,
                    help="§22: 层号 >= 该值的模块, 其整份更新乘 --layer-gain; "
                         "不传则所有模块同强度 (与既有产物一致)")
    ap.add_argument("--layer-gain", dest="layer_gain", type=float, default=None,
                    help="§22: 后半层的更新缩放系数 (>1 加强, <1 减弱); 需与 --layer-split 同时给出")
    ap.add_argument("--layer-scope", dest="layer_scope",
                    choices=["all", "ia", "band"], default="all",
                    help="§25.2: 逐层强度的作用域。all=整份更新 (L1 的行为); "
                         "ia=只乘稠密位移项 ia·Δ; band=只乘 donor 知识项 (band)。"
                         "机制上两项来源不同, L1 只测过'同时乘'")
    ap.add_argument("--inject-budget", dest="inject_budget", default="raw",
                    choices=["raw", "rel"],
                    help="raw: alpha 直接乘盈余位移 (旧口径); "
                         "rel: alpha 表示相对扰动预算 ||dW||_F/||W_A||_F in [0,1], 缩放由方法计算")
    ap.add_argument("--target", choices=list(TARGETS), default=DEFAULT_TARGET)
    ap.add_argument("--gate", choices=list(GATES), default=DEFAULT_GATE)
    ap.add_argument("--mode", choices=["inject", "direct", "hybrid"], default="inject",
                    help="Stage-3 合并方式: inject=host 奇异基保护性注入(默认, V0.3 主路径); "
                         "direct=插值基线; hybrid=插值 + 保护性注入")
    ap.add_argument("--interp-alpha", dest="interp_alpha", type=float, default=0.1,
                    help="hybrid 模式的插值系数 (0..1), 默认 0.1")
    ap.add_argument("--basis", choices=["host", "donor", "random"], default=None)
    ap.add_argument("--identity-mapping", dest="identity_mapping", action="store_true")
    ap.add_argument("--disable-selection", action="store_true")
    ap.add_argument("--injection", choices=["diag", "subspace", "bidir"], default="diag",
                    help="diag=host 奇异基对角注入(默认); subspace=把 donor-host 差异投影到 host top-r 子空间对后整体注入; "
                         "bidir=BTI(§17.6): 联合基有符号对角, W*=W_A+a*(U_τB diag(d_B) V_τB^T + w_A*U_τA diag(diag Z_A) V_τA^T), "
                         "需要 --tau-a-subs 提供 τ_A 分解")
    ap.add_argument("--tau-a-subs", dest="tau_a_subs", default=None,
                    help="BTI (--injection bidir) 需要的 τ_A 分解 (Stage-1 --anchor-model 产物)")
    ap.add_argument("--bidir-a-weight", dest="bidir_a_weight", type=float, default=1.0,
                    help="BTI 中 τ_A 项的相对权重 (默认 1.0 = 原设计; 设 0 则退化为纯 τ_A 有符号对角)")
    # ---- §25: 位移避开 host 任务主方向 (机制直觉: 保护 host 已掌握的任务结构) ----
    ap.add_argument("--tau-a-mode", dest="tau_a_mode", choices=["full", "avoid"], default="full",
                    help="§25: full = 位移 Δ 原样 (现有行为); "
                         "avoid = 位移避开 LR_r(τ_A) 的主方向, W* = W_A + ia·(Δ − LR_r(τ_A)) + band "
                         "(需 --tau-a-subs)")
    # ---- §18: Δ 低秩 bulk 项 (可加在 band 注入之上, 用于替换稠密插值项) ----
    ap.add_argument("--bulk-delta-subs", dest="bulk_delta_subs", default=None,
                    help="§18: Δ = W_B − W_A 的 Stage-1 分解 (用 --minus-model 生成)。"
                         "指定后额外加 ia·LR_r(Δ) 作为 bulk 项, 与 band 注入相加")
    ap.add_argument("--bulk-ia", dest="bulk_ia", type=float, default=0.0,
                    help="§18 bulk 系数: raw 口径下 bulk = bulk_ia · LR_r(Δ); "
                         "bulk_budget=rel 时 bulk_ia 是**目标相对预算** "
                         "(逐模块使 ‖bulk‖/‖W_A‖ ≡ bulk_ia)")
    ap.add_argument("--bulk-budget", dest="bulk_budget", choices=["raw", "rel"], default="rel",
                    help="§18 bulk 的预算口径; rel(默认) 逐模块范数匹配, 与稠密插值的 rel 完全可比")
    ap.add_argument("--bulk-rank", dest="bulk_rank", type=int, default=512,
                    help="§18 bulk 使用的秩 (受 Δ 缓存秩上限约束)")
    # ---- §20: 把更新写成三个互不冗余的正交项后, 显式暴露从未调过的 `b` ----
    ap.add_argument("--w-tail", dest="w_tail", type=float, default=0.0,
                    help="§20: Δ **尾部**项 b·(Δ − LR_R(Δ)) 的系数。与 --bulk-ia 同口径 "
                         "(bulk_budget=rel 时也是相对预算)。语义: '细节搬多少'。"
                         "b=0 即纯低秩 bulk (C1); 取 b=a 的配比即退化为稠密插值。"
                         "需要 --donor-model 提供 Δ。")
    # ---- §20.4 最小接口: 三个"一个参数 = 一个机制"的新名字, 旧参数保留为别名 ----
    ap.add_argument("--w-main", dest="w_main", type=float, default=None,
                    help="§20.4 最小接口: Δ **主方向**项 a·LR_R(Δ) 的系数 (= --bulk-ia)。"
                         "语义: '大方向搬多少'。需要 --bulk-delta-subs 定义主/尾分界 R。")
    ap.add_argument("--w-donor", dest="w_donor", type=float, default=None,
                    help="§20.4 最小接口: **额外** donor 任务带 c·U_τB diag(g⊙d_B) V_τBᵀ 的系数 "
                         "(= --alpha)。语义: '多要多少 donor 专长'。")
    ap.add_argument("--budget", dest="budget", choices=["raw", "rel"], default=None,
                    help="§20.4 最小接口: 统一口径 (= --inject-budget 与 --bulk-budget)。"
                         "rel 时各系数表示 ‖项‖/‖W_A‖, 跨配置可比。**口径不是建模参数。**")
    # ---- §20.8 auto-config: 用户只给一个 s, c 由机制自动定 ----
    ap.add_argument("--auto-config", dest="auto_config", action="store_true",
                    help="§20.8 自动配置: 只暴露唯一人类参数 --transfer-ratio s; "
                         "donor 强调强度 c 由**等范数原则**逐模块自动定 "
                         "c := s·‖Δ‖_F/‖band‖_F (使两项量级相当)。"
                         "等价于 --mode hybrid --interp-alpha s + 自动 c。"
                         "若 a=b=s (无主/尾偏好) 则主/尾分界 R 无关, 故无需 Δ 缓存。")
    ap.add_argument("--transfer-ratio", dest="transfer_ratio", type=float, default=0.1,
                    help="§20.8 唯一人类参数 s: Δ 的搬运比例 (默认 0.1, 与 AdaMMS 同值同义)。")
    ap.add_argument("--direct-full", dest="direct_full", action="store_true",
                    help="direct 模式使用完整 donor 权重 W*=(1-a)W_A+a W_B (与 AdaMMS 插值口径一致)")
    ap.add_argument("--allow-svd-fallback", dest="allow_svd_fallback", action="store_true",
                    help="仅调试用: 缺失 host 缓存时现场重算 SVD (正式实验禁止, 会破坏分量身份)")
    ap.add_argument("--tag-suffix", dest="tag_suffix", default=None,
                    help="附加到产物目录名的后缀, 用于区分不同 Stage-2 温度等同结构变体")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    inject(parse_args())
