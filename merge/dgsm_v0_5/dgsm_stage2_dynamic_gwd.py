#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DGSM-TEFM Stage-2: Dynamic Gromov Subspace Mapping + Alignment

核心: 在静态 GWD 之前/过程中，对 donor 子空间 U_B 施加一个可学习注意力映射 M = softmax((U_A W_q)(U_B W_k)^T / sqrt(r))，
通过最小化映射后 GWD(C_A, C_{B'}) + 正则，实现子空间动态自适应，对齐后输出 π, 动态映射矩阵 M, 以及 psi 编码。

特点:
  * 数据独立; 仅参数空间; 可选小步数 (dynamic-steps) 近似优化 W_q, W_k。
  * 若 --dynamic-steps=0 则退化为原 GSF Stage-2 (保持复现能力)。

输出结构: { 'modules': { name: { 'pi','gwd_cost','M','psi_A','psi_B','S_A','S_B','rank_A','rank_B','lambda_est'? } }, 'meta':{...}}
"""
from __future__ import annotations
import argparse, math, os
from typing import Dict, Optional, Tuple
import torch
from tqdm import tqdm
try:
    from .utils import load_weights, need_merge  # noqa: F401
except Exception:
    pass

# 尝试 POT
_HAVE_POT=False
try:
    import ot  # type: ignore
    from ot.gromov import gromov_wasserstein2, gromov_wasserstein, entropic_gromov_wasserstein
    _HAVE_POT=True
except Exception:
    _HAVE_POT=False

EPS=1e-8

# =============================================
# Entropic / Differentiable GW helpers
# =============================================

def _einsum_cost(L: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Compute GW cost = sum_{i,j,k,l} L_{i,j,k,l} pi_{i,k} pi_{j,l}
    Using einsum for clarity. L shape: (r,r,r,r), pi: (r,r).
    """
    return torch.einsum('ijkl,ik,jl->', L, pi, pi)

def _entropic_gw_torch(CA: torch.Tensor, CB: torch.Tensor, eps: float, iters: int = 20, sinkhorn_it: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable entropic GW (approx) in pure torch.

    Simplified iterative scheme:
      1. Initialize uniform pi.
      2. Loop T times: build 4D cost tensor L=(CA_ij - CB_kl)^2.
      3. Compute gradient surrogate G_{i,k} = 2 * sum_{j,l} L_{i,j,k,l} * pi_{j,l}.
      4. Update pi_raw = exp(-G/eps); perform a few Sinkhorn scalings to enforce approx uniform marginals.
      5. (Optional) Early stability if change small (skipped for simplicity / determinism).
    Return (pi, cost) where cost is differentiable wrt CB (and thus UB params).

    NOTE: O(r^4) memory; advisable to keep r<=128.
    """
    r = CA.shape[0]
    device = CA.device
    pi = torch.full((r, r), 1.0 / (r * r), device=device, dtype=CA.dtype, requires_grad=False)
    for _ in range(iters):
        L = (CA[:, :, None, None] - CB[None, None, :, :]) ** 2  # (r,r,r,r)
        # gradient surrogate G (factor 2 is conventional; can omit)
        # G_{i,k} = 2 * sum_{j,l} L_{i,j,k,l} pi_{j,l}
        G = 2.0 * torch.einsum('ijkl,jl->ik', L, pi)
        K = torch.exp(-G / max(eps, 1e-6))  # (r,r)
        # Sinkhorn for uniform marginals
        u = torch.ones(r, device=device, dtype=CA.dtype)
        v = torch.ones(r, device=device, dtype=CA.dtype)
        for _s in range(sinkhorn_it):
            u = (1.0 / r) / (K @ v).clamp_min(EPS)
            v = (1.0 / r) / (K.T @ u).clamp_min(EPS)
        pi = torch.diag(u) @ K @ torch.diag(v)
        pi = pi / pi.sum().clamp_min(EPS)
    # Final cost (differentiable): reuse last L to avoid recompute
    cost = _einsum_cost(L, pi)
    return pi, cost

def _load_subspaces(path:str)->Dict[str,Dict[str,torch.Tensor]]:
    obj=torch.load(path,map_location='cpu'); return obj.get('subspaces',obj)

def _softmax(x: torch.Tensor, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return (x.exp() / x.exp().sum(dim=dim, keepdim=True).clamp_min(EPS))

def _pairwise_sqdist(U: torch.Tensor, S: Optional[torch.Tensor], mode:str)->torch.Tensor:
    assert U.ndim==2
    d_out,r = U.shape
    if mode not in ('u','us','usn'): mode='u'
    if (mode=='u') or (S is None):
        X = U.T
    else:
        w = torch.sqrt(S.clamp_min(EPS))
        X = (U * w.unsqueeze(0)).T
        if mode=='usn':
            X = X / X.norm(dim=1, keepdim=True).clamp_min(EPS)
    xx=(X*X).sum(dim=1,keepdim=True)
    D=xx+xx.T-2*(X@X.T)
    return D.clamp_min(0.)

def _gwd_cost(CA:torch.Tensor, CB:torch.Tensor, pi:torch.Tensor)->float:
    """Vectorized GW cost under square loss.
    cost = Σ_{i,j,k,l} (CA_{i,j} - CB_{k,l})^2 * pi_{i,k} * pi_{j,l}
    """
    # Shapes: CA (r,r), CB (r,r), pi (r,r)
    L = (CA[:, :, None, None] - CB[None, None, :, :]).pow(2)  # (r,r,r,r)
    # Weight tensor factorizes: w_{i,j,k,l} = pi_{i,k} * pi_{j,l}
    cost = (L * (pi[:, None, :, None] * pi[None, :, None, :])).sum().item()
    return float(cost)

def _lambda_from_cost(cost:float,gamma:float,cost_scale:float)->float:
    c_norm=min(1.0, cost/max(EPS,cost_scale))
    return 1.0/(1.0+math.exp(-gamma*(1.0-c_norm)))

def _compute_psi(S:torch.Tensor, gwd_cost:float)->torch.Tensor:
    p=_softmax(S)
    bar_s=(p*S).sum(); bar_h=-(p*(p.clamp_min(EPS).log())).sum(); bar_d=torch.tensor(float(gwd_cost))
    # 保证同 device
    bar_d = bar_d.to(device=bar_s.device, dtype=bar_s.dtype)
    return torch.stack([bar_s,bar_h,bar_d])

def _sinkhorn_uniform(cost:torch.Tensor, reg:float, iters:int=50)->torch.Tensor:
    n,m=cost.shape
    K=torch.exp(-cost/reg)
    u=torch.full((n,),1.0/n,dtype=K.dtype,device=K.device)
    v=torch.full((m,),1.0/m,dtype=K.dtype,device=K.device)
    for _ in range(iters):
        u = (1.0/n)/(K@v).clamp_min(EPS)
        v = (1.0/m)/(K.T@u).clamp_min(EPS)
    pi = torch.diag(u)@K@torch.diag(v)
    return pi/pi.sum().clamp_min(EPS)

def _approx_gw(CA:torch.Tensor, CB:torch.Tensor, steps:int, reg:float, tol:float, patience:int, verbose:bool)->Tuple[torch.Tensor,float,int]:
    rA,rB=CA.shape[0],CB.shape[0]
    # 保证所有中间张量与 CA 在同一 device / dtype
    device = CA.device; dtype = CA.dtype
    pi=torch.full((rA,rB),1.0/(rA*rB), device=device, dtype=dtype)
    prev=None; bad=0; best=float('inf'); best_pi=pi.clone()
    CA_b=CA[:,:,None,None].to(device=device, dtype=dtype)
    CB_b=CB[None,None,:,:].to(device=device, dtype=dtype)
    L=(CA_b-CB_b).pow(2)  # [rA,rA,rB,rB] on device
    for it in range(steps):
        M=(L*(pi[None,:,None,:])).sum(dim=(1,3))  # [rA,rB]
        M=M-M.min()
        pi=_sinkhorn_uniform(M,reg,50)
        c=_gwd_cost(CA,CB,pi)
        if c<best-1e-9: best=c; best_pi=pi.clone()
        if verbose and (it%5==0 or it==steps-1): print(f"    [GW it {it:02d}] cost={c:.6f}{'*' if c==best else ''}")
        if prev is not None:
            rel=(prev-c)/max(EPS,prev)
            if rel<tol: bad+=1
            else: bad=0
            if bad>=patience:
                if verbose: print(f"    [GW EarlyStop] it={it} rel={rel:.3e}")
                return best_pi,best,it+1
        prev=c
    return best_pi,best,steps


def _pot_pi(CA: torch.Tensor, CB: torch.Tensor, method: str, eps: float, max_iter: int, verbose: bool = False) -> Tuple[torch.Tensor, float]:
    """使用 POT 计算 GW 传输计划。
    method: 'entropic' | 'gw'
    返回 (pi_torch, cost_val)
    """
    r = CA.shape[0]
    device, dtype = CA.device, CA.dtype
    p = torch.full((r,), 1.0 / r, device=device, dtype=dtype)
    q = torch.full((r,), 1.0 / r, device=device, dtype=dtype)
    try:
        if method == 'entropic':
            # POT entropic 返回 transport 矩阵
            pi = entropic_gromov_wasserstein(CA, CB, p, q, loss_fun='square_loss', epsilon=float(eps), max_iter=int(max_iter))
            if not torch.is_tensor(pi):
                pi = torch.tensor(pi, device=device, dtype=dtype)
            cost_val = _gwd_cost(CA, CB, pi.detach())
            return pi, float(cost_val)
        else:
            # 经典 GW：用 gromov_wasserstein 获取 transport，再用 torch 计算 cost
            pi = gromov_wasserstein(CA, CB, p, q, loss_fun='square_loss', max_iter=int(max_iter))
            if not torch.is_tensor(pi):
                pi = torch.tensor(pi, device=device, dtype=dtype)
            cost_val = _gwd_cost(CA, CB, pi.detach())
            return pi, float(cost_val)
    except Exception as e:
        if verbose:
            print(f"[POT] GPU fallback due to error: {e}")
        # CPU fallback：numpy -> torch
        CA_np = CA.detach().cpu().numpy(); CB_np = CB.detach().cpu().numpy()
        p_np = (torch.full((r,), 1.0 / r)).numpy(); q_np = p_np
        try:
            if method == 'entropic':
                pi_np = entropic_gromov_wasserstein(CA_np, CB_np, p_np, q_np, loss_fun='square_loss', epsilon=float(eps), max_iter=int(max_iter))
            else:
                pi_np = gromov_wasserstein(CA_np, CB_np, p_np, q_np, loss_fun='square_loss', max_iter=int(max_iter))
            pi_t = torch.tensor(pi_np, device=device, dtype=dtype)
            return pi_t, float(_gwd_cost(CA, CB, pi_t.detach()))
        except Exception as ee:
            if verbose:
                print(f"[POT] CPU fallback failed: {ee}")
            pi_u = torch.full((r, r), 1.0 / (r * r), device=device, dtype=dtype)
            return pi_u, float(_gwd_cost(CA, CB, pi_u))

# 动态映射优化 --------------------------------------------------------------

def _init_Wq_Wk(d:int,r:int,device)->Tuple[torch.Tensor,torch.Tensor]:
    # 初始化为接近单位 (d x r) 截断
    Wq=torch.eye(d, device=device)[:,:r].contiguous().clone().requires_grad_(True)
    Wk=torch.eye(d, device=device)[:,:r].contiguous().clone().requires_grad_(True)
    return Wq,Wk

def _attention_map(Ua:torch.Tensor, Ub:torch.Tensor, Wq:torch.Tensor, Wk:torch.Tensor)->torch.Tensor:
    # Ua,Ub: [d_out,r]; Wq,Wk: [d_out,r]; 计算 Q=(Ua^T Wq)?? 这里采用原文: Q=Ua Wq, K=Ub Wk
    # Ua: d_out x r; Wq: d_out x r  -> 维度不匹配: 需要 Wq 是 r x r? 重新定义: 我们改为 Wq,Wk: r x r (在子空间上做可学习线性)
    # 调整: 在调用前先投影 U 列空间 => U_col = U ; shape d_out x r, 我们学习 r x r 映射
    raise NotImplementedError

# 为保持清晰, 我们采用子空间内部映射 (r x r), 简化计算并避免 d_out^2 复杂度。
# 重写初始化与注意力函数:

def _init_Wq_Wk_subspace(r:int,device)->Tuple[torch.Tensor,torch.Tensor]:
    Wq=torch.eye(r, device=device).clone().requires_grad_(True)
    Wk=torch.eye(r, device=device).clone().requires_grad_(True)
    return Wq,Wk

def _attention_M(Ua:torch.Tensor, Ub:torch.Tensor, Wq:torch.Tensor, Wk:torch.Tensor)->torch.Tensor:
    # Ua,Ub: d_out x r, treat columns as basis vectors; we form Q = Ua @ Wq (d_out x r), K = Ub @ Wk (d_out x r)
    Q = Ua @ Wq  # d_out x r
    K = Ub @ Wk  # d_out x r
    # 将列向量视为 token: 取列维度 r 作为 length => 转置后 (r, d_out)
    Qc = Q.transpose(0,1)  # r x d_out
    Kc = K.transpose(0,1)  # r x d_out
    att = (Qc @ Kc.T) / math.sqrt(Qc.shape[1])  # r x r
    M = _softmax(att, dim=-1)  # 行 softmax: 每个源基底到目标加权
    return M  # r x r, 右乘 Ub^T 等价于基底混合

def _apply_dynamic(Ub:torch.Tensor, M:torch.Tensor)->torch.Tensor:
    # Ub: d_out x r, M: r x r -> Ub' = Ub @ M^T  (让 M_{i,j} 表示“源 i -> 目标 j”)
    return Ub @ M.T

@torch.no_grad()
def _diagnose_lambda(costs, args):
    if args.gamma is None or args.cost_scale is None: return None
    vals=[_lambda_from_cost(c,args.gamma,args.cost_scale) for c in costs]
    if len(vals)==0: return None
    return min(vals), sum(vals)/len(vals), max(vals)


def stage2(args: argparse.Namespace):
    print("\n--- [DGSM Stage-2: Dynamic Gromov Subspace Mapping + GWD Alignment] ---")
    subsA=_load_subspaces(args.subs_a); subsB=_load_subspaces(args.subs_b)
    # 诊断：检查 A/B 子空间键的交并情况，避免因为前缀不同造成遗漏
    keysA=set(subsA.keys()); keysB=set(subsB.keys())
    inter=sorted(keysA & keysB)
    onlyA=sorted(keysA - keysB)
    onlyB=sorted(keysB - keysA)
    if onlyA or onlyB:
        print(f"[Stage-2][Warn] A/B 子空间键不完全一致: A={len(keysA)} B={len(keysB)} 交集={len(inter)} A-Only={len(onlyA)} B-Only={len(onlyB)}")
        # 打印前若干个以便快速定位（避免刷屏）
        if onlyA:
            print("  [Only in A] 示例:", onlyA[:10])
        if onlyB:
            print("  [Only in B] 示例:", onlyB[:10])
        print("  提示: Stage-1 已对键名规范化为 'model.layers.*'，若仍有不一致，请检查 Stage-1 need_merge/规范化逻辑或 rank 设置。")
    modules=inter
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    # 统一 POT 开关，避免后面 meta 中取最后一次循环的局部变量
    pot_enabled_global = (((getattr(args, 'pot', 'auto') != 'off') and _HAVE_POT) or getattr(args, 'use_pot', False))
    out_mod={}; costs=[]; lambdas=[]
    # 默认显示进度条；可通过 --no-progress 关闭（见 parse_args）
    it = tqdm(modules, desc='Stage-2 DGSM', disable=bool(getattr(args, 'no_progress', False)))
    for name in it:
        blkA,blkB = subsA[name], subsB[name]
        UA,SA = blkA['U'].float().to(device), blkA['S'].float().to(device)
        UB,SB = blkB['U'].float().to(device), blkB['S'].float().to(device)
        rA,rB = SA.shape[0], SB.shape[0]
        # 自适应 rank: 若提供 rank_threshold, 则选择最小 r 满足累计奇异值能量比例>=阈值
        r_max = min(rA, rB, args.max_rank)
        if getattr(args, 'rank_threshold', None) is not None and 0.0 < args.rank_threshold <= 1.0:
            def pick_r(S: torch.Tensor, thr: float, rcap: int) -> int:
                ssum = S[:rcap].sum()
                if ssum.item() <= 0:
                    return min(32, rcap)
                csum = 0.0
                r_sel = 1
                for i in range(min(rcap, S.shape[0])):
                    csum += float(S[i].item())
                    if csum / float(ssum.item()) >= thr:
                        r_sel = i + 1
                        break
                return max(1, min(r_sel, rcap))
            r_selA = pick_r(SA, float(args.rank_threshold), r_max)
            r_selB = pick_r(SB, float(args.rank_threshold), r_max)
            r = min(r_selA, r_selB)
        else:
            r = r_max
        UA,SA = UA[:,:r], SA[:r]
        UB,SB = UB[:,:r], SB[:r]
        CA = _pairwise_sqdist(UA, SA, args.dist_mode)
        CB = _pairwise_sqdist(UB, SB, args.dist_mode)

        # 动态映射优化 --------------------------------------------------
        M_final=None
        if args.dynamic_steps>0:
            Wq,Wk = _init_Wq_Wk_subspace(r, device)
            opt = torch.optim.Adam([Wq,Wk], lr=args.dynamic_lr)
            best_cost=None; best_state=None
            for step in range(args.dynamic_steps):
                opt.zero_grad()
                with torch.enable_grad():
                    M = _attention_M(UA, UB, Wq, Wk)  # r x r
                    UB_prime = _apply_dynamic(UB, M)  # d_out x r
                    CBp = _pairwise_sqdist(UB_prime, SB, args.dist_mode)
                    # 以近似 GW 作为统一的模型选择指标（与最终评估一致）
                    pi_tmp, gw_metric, _ = _approx_gw(CA.detach(), CBp.detach(), steps=min(4,args.iters), reg=args.sink_reg, tol=args.tol, patience=args.patience, verbose=False)
                    if args.dyn_loss == 'frob':
                        # Frobenius 代理 (简单 / 稳定)
                        loss_core = (CA - CBp).pow(2).mean()
                    elif args.dyn_loss == 'entropic':
                        # 可微 entropic GW 近似（小步数，保证速度）
                        eps = args.entropic_eps
                        inner_it = args.entropic_iters
                        _, cost_ent = _entropic_gw_torch(CA, CBp, eps=eps, iters=inner_it, sinkhorn_it=args.entropic_sinkhorn)
                        loss_core = cost_ent
                    else:  # hybrid: 混合两者
                        eps = args.entropic_eps
                        inner_it = args.entropic_iters
                        _, cost_ent = _entropic_gw_torch(CA, CBp, eps=eps, iters=inner_it, sinkhorn_it=args.entropic_sinkhorn)
                        frob = (CA - CBp).pow(2).mean()
                        loss_core = (1.0 - args.dyn_mix_alpha) * frob + args.dyn_mix_alpha * cost_ent
                    loss = loss_core
                    # 加正则
                    reg = args.dynamic_reg * ((Wq - torch.eye(r, device=device)).pow(2).sum() + (Wk - torch.eye(r, device=device)).pow(2).sum())
                    loss = loss + reg
                loss.backward()
                opt.step()
                if args.dynamic_report and step % max(1,args.dynamic_steps//5)==0:
                    print(f"  [DynMap {name}] step={step} loss={loss.item():.6f} gw≈{gw_metric:.6f}")
                # 使用 gw_metric (近似 GW 或 entropic cost) 选择最优映射
                if best_cost is None or gw_metric < best_cost:
                    best_cost=gw_metric; best_state=(Wq.detach().clone(), Wk.detach().clone())
            if best_state is not None:
                Wq,Wk = best_state
                with torch.no_grad():
                    M_final = _attention_M(UA, UB, Wq, Wk)  # r x r final
                    UB = _apply_dynamic(UB, M_final)
                    CB = _pairwise_sqdist(UB, SB, args.dist_mode)
        # 静态或动态后的正式 GW ------------------------------------------
        pot_enabled = pot_enabled_global
        if pot_enabled and _HAVE_POT:
            method = 'entropic' if getattr(args, 'pot_method', 'gw') == 'entropic' or (getattr(args, 'pot_entropic_eps', None) is not None) else 'gw'
            eps = float(getattr(args, 'pot_eps', getattr(args, 'pot_entropic_eps', 0.5)))
            max_it = int(getattr(args, 'pot_max_iter', 50))
            pi, gwd_cost = _pot_pi(CA, CB, method=method, eps=eps, max_iter=max_it, verbose=bool(args.verbose))
        else:
            pi, gwd_cost, _ = _approx_gw(CA, CB, args.iters, args.sink_reg, args.tol, args.patience, args.verbose)

        psi_A = _compute_psi(SA, gwd_cost).cpu(); psi_B = _compute_psi(SB, gwd_cost).cpu()
        rec = {
            'gwd_cost': torch.tensor(gwd_cost),
            'pi': pi.cpu(),
            'psi_A': psi_A,
            'psi_B': psi_B,
            'S_A': SA.cpu(), 'S_B': SB.cpu(),
            'rank_A': torch.tensor(r), 'rank_B': torch.tensor(r),
        }
        if M_final is not None:
            rec['M'] = M_final.cpu()
        if args.gamma is not None and args.cost_scale is not None:
            lam_est = _lambda_from_cost(gwd_cost, args.gamma, args.cost_scale)
            rec['lambda_est'] = torch.tensor(lam_est)
            lambdas.append(lam_est)
        out_mod[name]=rec; costs.append(gwd_cost)
        if args.verbose:
            src = 'POT' if (args.use_pot and _HAVE_POT) else 'Approx'
            dyn = 'Dyn' if M_final is not None else 'Static'
            lam_str = (f"lam={round(float(rec['lambda_est']),4)}" if 'lambda_est' in rec else '')
            print(f"[DGSM-{src}] {dyn} {name}: cost={gwd_cost:.6f} r={r} {lam_str}")
    meta = {
        'subs_a': args.subs_a,
        'subs_b': args.subs_b,
        'dynamic_steps': args.dynamic_steps,
        'dynamic_lr': args.dynamic_lr,
        'dynamic_reg': args.dynamic_reg,
        'use_pot': bool(pot_enabled_global and _HAVE_POT),
        'dist_mode': args.dist_mode,
        'cost_min': float(min(costs)) if costs else None,
        'cost_mean': float(sum(costs)/len(costs)) if costs else None,
        'cost_max': float(max(costs)) if costs else None,
        'n_modules': len(out_mod)
    }
    # 仅在提供时记录 gamma / cost_scale，避免 None 出现在元数据中
    if args.gamma is not None:
        meta['gamma'] = args.gamma
    if args.cost_scale is not None:
        meta['cost_scale'] = args.cost_scale
    # 仅当提供了 gamma/cost_scale 且有统计时，再写入 lambda_*，避免出现 None
    if (args.gamma is not None) and (args.cost_scale is not None) and len(lambdas)>0:
        meta.update({
            'lambda_min': min(lambdas),
            'lambda_mean': (sum(lambdas)/len(lambdas)),
            'lambda_max': max(lambdas)
        })
    torch.save({'modules':out_mod,'meta':meta}, args.save)
    print(f"[Done] DGSM Stage-2 saved -> {args.save} modules={len(out_mod)}")
    if costs:
        print(f"  cost(min/mean/max)=({meta['cost_min']:.4f}/{meta['cost_mean']:.4f}/{meta['cost_max']:.4f})")
    if lambdas:
        print(f"  lambda(min/mean/max)=({meta['lambda_min']:.4f}/{meta['lambda_mean']:.4f}/{meta['lambda_max']:.4f})")


def parse_args():
    ap=argparse.ArgumentParser(description='DGSM Stage-2 Dynamic GWD Alignment')
    ap.add_argument('--subs-a', required=True)
    ap.add_argument('--subs-b', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--dist-mode', default='us', choices=['u','us','usn'])
    # POT 选项（与 DSGA-ATF 保持一致）
    ap.add_argument('--pot', type=str, default='auto', choices=['auto','on','off'], help='优先使用 POT (auto=若安装则用)')
    ap.add_argument('--pot-method', type=str, default='gw', choices=['entropic','gw'], help='POT 的 GW 类型 (默认更锐利的 gw)')
    ap.add_argument('--pot-eps', type=float, default=0.5, help='POT entropic 的 epsilon')
    ap.add_argument('--pot-max-iter', type=int, default=50, help='POT 最大迭代步数')
    # 兼容旧参数（可选）
    ap.add_argument('--use-pot', action='store_true')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--sink-reg', type=float, default=0.05)
    ap.add_argument('--tol', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--gamma', type=float, default=None)
    ap.add_argument('--cost-scale', type=float, default=None)
    ap.add_argument('--max-rank', type=int, default=128, help='对齐时截断最多使用的 r (兼顾内存)')
    ap.add_argument('--rank-threshold', type=float, default=None, help='自适应 rank 阈值 ∈(0,1]，按累计奇异值能量选择 r')
    # 动态映射相关
    ap.add_argument('--dynamic-steps', type=int, default=0, help='>0 启用子空间注意力映射优化步数')
    ap.add_argument('--dynamic-lr', type=float, default=5e-3)
    ap.add_argument('--dynamic-reg', type=float, default=1e-3)
    ap.add_argument('--dynamic-report', action='store_true')
    ap.add_argument('--dyn-loss', default='hybrid', choices=['frob','entropic','hybrid'], help='动态阶段优化目标: Frobenius / 可微 entropic / 二者混合')
    ap.add_argument('--dyn-mix-alpha', type=float, default=0.2, help='hybrid 模式下 entropic 的权重系数 ∈[0,1]')
    ap.add_argument('--entropic-eps', dest='entropic_eps', type=float, default=0.1, help='动态可微 entropic GW 的温度/epsilon')
    ap.add_argument('--entropic-iters', dest='entropic_iters', type=int, default=5, help='动态阶段 entropic GW 迭代次数')
    ap.add_argument('--entropic-sinkhorn', dest='entropic_sinkhorn', type=int, default=5, help='每次 entropic GW 内部 Sinkhorn 步数')
    # 兼容旧参数：若提供则覆盖 pot-eps 并启用 entropic
    ap.add_argument('--pot-entropic-eps', type=float, default=None, help='兼容旧参数：设置则等价启用 POT entropic，覆盖 --pot-eps')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--no-progress', action='store_true', help='关闭模块级进度条')
    return ap.parse_args()

if __name__=='__main__':
    stage2(parse_args())
