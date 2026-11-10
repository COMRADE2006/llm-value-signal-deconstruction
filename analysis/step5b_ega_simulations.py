# -*- coding: utf-8 -*-
"""step5b: EGA模拟证据集 —— 实现验证 / L0校准 / 功效扫描 / L2解析封箱。

依赖: numpy, scikit-learn, scipy, pandas(经step5_ega)。运行:
    python analysis/step5b_ega_simulations.py                # 全部(功率扫描约85分钟)
    python analysis/step5b_ega_simulations.py --parts ge2017,l2analytic   # 指定部分
结果: data/results/old/ega_simulations.json (增量落盘, 断点续跑: 已完成部分自动跳过)

四个模拟的结论(2026-10-09实跑):
1) ge2017复刻  : N=5000/4因子/φ=0.7 → 10/10恢复4社区ARI=1.00, 管线实现正确
2) L0校准      : p=12/N=22下, 真4因子(含聚合降噪)100%空网(检验力0), 单因子假阳0~6%
                 → 真实L0的29边/3社区非噪声伪影, 但"空网"在此N下无区分力
3) 功效扫描    : N=22全条件空网(γ=0另有10~15%假阳); γ=0.5可靠窗口极窄:
                 B(+g)恰在N=100恢复100%且假阳0%; C(单因子)N=200假阳100%(γ任意)
                 → EGA存在"空网失明"与"稠密假阳"双失效模式
4) L2解析封箱  : 1773题三种ridge收缩下|偏相关|max=0.031 ≪ γ=0门槛0.495
                 → 空网为EBIC全局最优, L2层glasso无需实跑
"""
import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, r'd:\competitionresearch\cognitivesafety\实验2')
sys.path.insert(0, r'd:\competitionresearch\cognitivesafety\实验2\analysis')
import config
from step5_ega import load_matrices, shrink_corr, glasso_ebic, detect_communities

OUT_PATH = config.DATA_DIR / "results" / "old" / "ega_simulations.json"
ALPHAS = np.logspace(-3, np.log10(2.0), 12)  # 固定λ路径: 0.001~2.0对数12点


def _load_out():
    """增量落盘: 读已有结果, 断点续跑。"""
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {"date": datetime.now().isoformat(),
                     "note": "EGA模拟证据集; 管线=step5_ega.glasso_ebic(G&E2017复刻10/10)"}}


def _save(out):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[落盘] {OUT_PATH}", flush=True)


# ---------- 1) G&E 2017 实现验证 ----------

def ge2017_replication(reps=10):
    """复刻Golino & Epskamp(2017): 4因子×10题, 载荷U(0.70,0.90), φ=0.7, N=5000。
    其报告EGA此条件100%准确 → 本管线应接近reps/reps。"""
    rng = np.random.default_rng(7)
    p, k, n = 40, 4, 5000
    blocks = np.repeat(np.arange(k), p // k)
    n_ok, dims, aris = 0, [], []
    for _ in range(reps):
        lam = rng.uniform(0.70, 0.90, p)
        Cf = np.full((k, k), 0.7)
        np.fill_diagonal(Cf, 1.0)
        F = np.linalg.cholesky(Cf) @ rng.standard_normal((k, n))
        X = lam[:, None] * F[blocks] + np.sqrt(1 - lam**2)[:, None] * rng.standard_normal((p, n))
        R, _ = shrink_corr(X, "lw")
        best, _ = glasso_ebic(R, n, gamma=0.5)
        if best is None or best["edges"] == 0:
            dims.append(0)
            continue
        labels, _, _ = detect_communities(best["P"])
        lab, blk = labels[labels >= 0], blocks[labels >= 0]
        nc = len(set(lab)) if len(lab) else 0
        dims.append(nc)
        if nc > 1:
            aris.append(float(adjusted_rand_score(lab, blk)))
        n_ok += int(nc == k)
    print(f"[ge2017] {n_ok}/{reps}恢复4社区, ARI={[round(a, 2) for a in aris]}", flush=True)
    return {"reps": reps, "recovered_4comm": n_ok, "dims": dims, "ari": aris}


# ---------- 2) L0情形校准(p=12, N=22) ----------

def l0_calibration(reps=50):
    """校准真实L0(12维均值节点/22观测): 同等p/N下真4因子能否恢复? 假阳性多少?
    D1/D2=4因子(单变量/每节点30题聚合), E1/E2=单因子假阳性参照。"""
    rng = np.random.default_rng(2027)
    p, k, lam, n = 12, 4, 0.75, 22
    n_agg = 30

    def gen(cond, agg):
        blocks = np.repeat(np.arange(k), p // k)
        Cf = np.full((k, k), 0.5)
        np.fill_diagonal(Cf, 1.0)
        F = np.linalg.cholesky(Cf) @ rng.standard_normal((k, n))
        if cond == "single":
            Xn = lam * rng.standard_normal((1, n)) + np.sqrt(1 - lam**2) * rng.standard_normal((p, n))
            blocks = np.zeros(p, dtype=int)
        else:
            Xn = lam * F[blocks] + np.sqrt(1 - lam**2) * rng.standard_normal((p, n))
        if not agg:
            return Xn, blocks
        g = rng.standard_normal((n_agg, p, n))
        return (Xn[None] / np.sqrt(n_agg) + 0.3 * g).mean(axis=0), blocks

    res = {}
    for cond, agg, tag in [("multi", False, "D1单变量"), ("multi", True, "D2聚合30"),
                           ("single", False, "E1单变量"), ("single", True, "E2聚合30")]:
        ncs, aris, empt = [], [], 0
        for _ in range(reps):
            X, blocks = gen(cond, agg)
            R, _ = shrink_corr(X, "sample")
            best, _ = glasso_ebic(R, n, gamma=0.5, alphas=ALPHAS)
            if best is None or best["edges"] == 0:
                empt += 1
                ncs.append(0)
                continue
            labels, _, _ = detect_communities(best["P"])
            lab, blk = labels[labels >= 0], blocks[labels >= 0]
            nc = len(set(lab)) if len(lab) else 0
            ncs.append(nc)
            if cond == "multi" and nc > 1:
                aris.append(float(adjusted_rand_score(lab, blk)))
        nc = np.array(ncs)
        res[tag] = {"empty_pct": empt / reps, "nc_median": float(np.median(nc)),
                    "ari_median": float(np.median(aris)) if aris else None,
                    "false_positive_pct": float(np.mean(nc > 1)) if cond == "single" else None}
        print(f"[l0校准] {tag}: 空网{empt / reps:.0%} 社区中位{np.median(nc):.0f}", flush=True)
    return res


# ---------- 3) 功效扫描(p=39, 3条件×4N×2γ) ----------

def power_scan(reps=20):
    """EGA恢复已知结构所需观测数N。A=4因子(跨相关0.5) B=+共识g(0.4) C=单因子(假阳)。"""
    rng = np.random.default_rng(2026)
    p, k, lam = 39, 4, 0.75

    def gen(n_obs, cond):
        if cond == "C":
            return (lam * rng.standard_normal((1, n_obs))
                    + np.sqrt(1 - lam**2) * rng.standard_normal((p, n_obs))), np.zeros(p, dtype=int)
        blocks = np.arange(p) % k
        Cf = np.full((k, k), 0.5)
        np.fill_diagonal(Cf, 1.0)
        F = np.linalg.cholesky(Cf) @ rng.standard_normal((k, n_obs))
        g = 0.4 if cond == "B" else 0.0
        X = lam * F[blocks] + g * rng.standard_normal((1, n_obs))
        return X + np.sqrt(max(1 - lam**2 - g**2, 0.05)) * rng.standard_normal((p, n_obs)), blocks

    rows = []
    for cond in ["A", "B", "C"]:
        for n_obs in [22, 50, 100, 200]:
            for gamma in [0.0, 0.5]:
                t0 = time.time()
                ncs, aris, empt = [], [], 0
                for _ in range(reps):
                    X, blocks = gen(n_obs, cond)
                    R, _ = shrink_corr(X, "lw")
                    best, _ = glasso_ebic(R, n_obs, gamma=gamma, alphas=ALPHAS)
                    if best is None or best["edges"] == 0:
                        empt += 1
                        ncs.append(0)
                        continue
                    labels, _, _ = detect_communities(best["P"])
                    lab, blk = labels[labels >= 0], blocks[labels >= 0]
                    nc = len(set(lab)) if len(lab) else 0
                    ncs.append(nc)
                    if cond in "AB" and nc > 1:
                        aris.append(float(adjusted_rand_score(lab, blk)))
                nc = np.array(ncs)
                row = {"cond": cond, "N": n_obs, "gamma": gamma,
                       "empty_pct": empt / reps, "nc_median": float(np.median(nc)),
                       "ari_median": float(np.median(aris)) if aris else None,
                       "false_positive_pct": float(np.mean(nc > 1)) if cond == "C" else None}
                rows.append(row)
                print(f"[power] {cond}|N={n_obs}|γ={gamma}: 空{empt / reps:.0%} "
                      f"社区中位{np.median(nc):.0f} ({time.time() - t0:.0f}s)", flush=True)
                _save({**_load_out(), "power_scan": rows})  # 每格落盘防中断
    return rows


# ---------- 4) L2解析封箱(p=1773 ≫ N=22) ----------

def l2_analytic():
    """空网最优性解析论证: 单边似然增益上界(N/2)ln(1/(1-ρ²)) vs EBIC罚。
    ridge收缩后相关阵正定可逆 → 偏相关解析可算。超门槛边数=0 ⟹ 空网全局最优。"""
    X, meta, _ = load_matrices("old")
    p, n = X.shape
    res = {"p": p, "n": n, "configs": {}}
    for gamma in [0.0, 0.5]:
        pen = np.log(n) + 2 * gamma * np.log(p * (p - 1) / 2)
        res[f"rho_star_gamma{gamma}"] = float(np.sqrt(1 - np.exp(-2 * pen / n)))
        print(f"[l2解析] γ={gamma}: 罚={pen:.2f}nat/边 → 门槛ρ*={res[f'rho_star_gamma{gamma}']:.3f}",
              flush=True)
    for lam in [0.3, 0.5, 0.7]:
        Xc = X - X.mean(axis=1, keepdims=True)
        Xs = Xc / np.maximum(Xc.std(axis=1, keepdims=True), 1e-12)
        R = (1 - lam) * (Xs @ Xs.T) / n + lam * np.eye(p)
        P = np.linalg.inv(R)
        d = np.sqrt(np.diag(P))
        PC = -P / np.outer(d, d)
        np.fill_diagonal(PC, 0)
        apc = np.abs(PC[np.triu_indices(p, 1)])
        cfg = {"apc_max": float(apc.max()), "apc_q999": float(np.quantile(apc, .999)),
               "edges_over_threshold_gamma0": int((apc > res["rho_star_gamma0.0"]).sum()),
               "edges_over_threshold_gamma0.5": int((apc > res["rho_star_gamma0.5"]).sum())}
        res["configs"][f"ridge:{lam}"] = cfg
        print(f"[l2解析] ridge:{lam}: |ρ_p|max={apc.max():.3f} "
              f"超门槛边(γ=0)={cfg['edges_over_threshold_gamma0']}", flush=True)
    return res


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="EGA模拟证据集(断点续跑, 已完成部分跳过)")
    ap.add_argument("--parts", default="ge2017,l0cal,power,l2analytic",
                    help="逗号分隔: ge2017,l0cal,power,l2analytic")
    ap.add_argument("--reps-ge", type=int, default=10)
    ap.add_argument("--reps-l0", type=int, default=50)
    ap.add_argument("--reps-power", type=int, default=20)
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    out = _load_out()
    if "ge2017" in parts and "ge2017_replication" not in out:
        out["ge2017_replication"] = ge2017_replication(args.reps_ge)
        _save(out)
    if "l0cal" in parts and "l0_calibration" not in out:
        out["l0_calibration"] = l0_calibration(args.reps_l0)
        _save(out)
    if "power" in parts:
        out["power_scan"] = power_scan(args.reps_power)  # 内部每格落盘
    if "l2analytic" in parts and "l2_analytic" not in out:
        out["l2_analytic"] = l2_analytic()
        _save(out)
    print("\n完成。已有部分:", ", ".join(k for k in out if k != "meta"), flush=True)


if __name__ == "__main__":
    main()
