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

# 尝试 POT
_HAVE_POT=False
try:
    import ot  # type: ignore
    from ot.gromov import gromov_wasserstein2, gromov_wasserstein
    _HAVE_POT=True
except Exception:
    _HAVE_POT=False

EPS=1e-8

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
    rA,rB=CA.shape[0],CB.shape[0]
    cost=0.0
    for i in range(rA):
        CA_i=CA[i]; pi_i=pi[i]
        for j in range(rA):
            dA=CA_i[j]; pi_j=pi[j]
            diff2=(dA-CB).pow(2)
            cost += (pi_i.unsqueeze(1)*pi_j.unsqueeze(0)*diff2).sum().item()
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
    modules=sorted(set(subsA.keys()) & set(subsB.keys()))
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out_mod={}; costs=[]; lambdas=[]
    for name in modules:
        blkA,blkB = subsA[name], subsB[name]
        UA,SA = blkA['U'].float().to(device), blkA['S'].float().to(device)
        UB,SB = blkB['U'].float().to(device), blkB['S'].float().to(device)
        rA,rB = SA.shape[0], SB.shape[0]
        # 统一 rank 以进行动态映射: 取 r = min(rA,rB,args.max_rank)
        r = min(rA,rB,args.max_rank)
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
                    # NOTE: 这里使用近似 GW 得到的标量 cost_tmp 是 "离散" 求解结果, 与 Wq/Wk 不可微（无梯度路径）。
                    # 为了让动态映射真正可学习, 需要一个可微代理损失；暂时加入一个简单 Frobenius 代理 cost_proxy。
                    pi_tmp, cost_tmp, _ = _approx_gw(CA.detach(), CBp.detach(), steps=min(4,args.iters), reg=args.sink_reg, tol=args.tol, patience=args.patience, verbose=False)
                    # Frobenius 代理 (可微): 差分距离矩阵直接匹配
                    cost_proxy = ( (CA - CBp).pow(2).mean() )
                    loss = cost_proxy
                    # 加正则
                    reg = args.dynamic_reg * ((Wq - torch.eye(r, device=device)).pow(2).sum() + (Wk - torch.eye(r, device=device)).pow(2).sum())
                    loss = loss + reg
                loss.backward()
                opt.step()
                if args.dynamic_report and step % max(1,args.dynamic_steps//5)==0:
                    print(f"  [DynMap {name}] step={step} loss={loss.item():.6f} gw_cost≈{cost_tmp:.6f}")
                # 用近似 GW cost 仍作为度量挑选最优 M (虽然梯度来自 proxy)
                if best_cost is None or cost_tmp < best_cost:
                    best_cost=cost_tmp; best_state=(Wq.detach().clone(), Wk.detach().clone())
            if best_state is not None:
                Wq,Wk = best_state
                with torch.no_grad():
                    M_final = _attention_M(UA, UB, Wq, Wk)  # r x r final
                    UB = _apply_dynamic(UB, M_final)
                    CB = _pairwise_sqdist(UB, SB, args.dist_mode)
        # 静态或动态后的正式 GW ------------------------------------------
        if args.use_pot and _HAVE_POT:
            try:
                p = torch.full((r,),1.0/r,dtype=torch.float64); q = torch.full((r,),1.0/r,dtype=torch.float64)
                cost_val, pi_np = gromov_wasserstein2(CA.double().cpu().numpy(), CB.double().cpu().numpy(), p.numpy(), q.numpy(), 'square_loss', log=False, armijo=False, return_transport=True)
                pi = torch.from_numpy(pi_np).float()
                gwd_cost = float(cost_val)
            except Exception:
                pi, gwd_cost, _ = _approx_gw(CA, CB, args.iters, args.sink_reg, args.tol, args.patience, args.verbose)
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
            print(f"[DGSM-{src}] {dyn} {name}: cost={gwd_cost:.6f} r={r} {'lam='+str(round(lam_est,4)) if 'lambda_est' in rec else ''}")
    meta = {
        'subs_a': args.subs_a,
        'subs_b': args.subs_b,
        'dynamic_steps': args.dynamic_steps,
        'dynamic_lr': args.dynamic_lr,
        'dynamic_reg': args.dynamic_reg,
        'use_pot': bool(args.use_pot and _HAVE_POT),
        'dist_mode': args.dist_mode,
        'gamma': args.gamma,
        'cost_scale': args.cost_scale,
        'cost_min': float(min(costs)) if costs else None,
        'cost_mean': float(sum(costs)/len(costs)) if costs else None,
        'cost_max': float(max(costs)) if costs else None,
        'lambda_min': (min(lambdas) if lambdas else None),
        'lambda_mean': (sum(lambdas)/len(lambdas) if lambdas else None),
        'lambda_max': (max(lambdas) if lambdas else None),
        'n_modules': len(out_mod)
    }
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
    ap.add_argument('--use-pot', action='store_true')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--sink-reg', type=float, default=0.05)
    ap.add_argument('--tol', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--gamma', type=float, default=None)
    ap.add_argument('--cost-scale', type=float, default=None)
    ap.add_argument('--max-rank', type=int, default=128, help='对齐时截断最多使用的 r (兼顾内存)')
    # 动态映射相关
    ap.add_argument('--dynamic-steps', type=int, default=0, help='>0 启用子空间注意力映射优化步数')
    ap.add_argument('--dynamic-lr', type=float, default=5e-3)
    ap.add_argument('--dynamic-reg', type=float, default=1e-3)
    ap.add_argument('--dynamic-report', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()

if __name__=='__main__':
    stage2(parse_args())
