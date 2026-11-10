# -*- coding: utf-8 -*-
"""
第一步：测量主体一致性检验（22 列公因子分析，总方案v2 §4 第一步）。

操作:
  - KMO + Bartlett 球形检验（取样适切性）
  - 平行分析（自实现，随机正态对照，95 分位准则）
  - Velicer MAP（偏相关平方均值最小准则，自实现）
  - EFA（minres，不旋转看第一公因子 + varimax 旋转看结构）
  - 一致性判定: 第一公因子解释方差充分（默认阈值 40%）→ 题目聚类有正当性

敏感性: --on-only 只用 11 个 on 列（控 on/off 非独立灌水）

输入: data/results/{source}/scored_matrix.csv + active_items.csv（仅活跃题）
输出: data/results/{source}/step1_efa_loadings.csv（旋转后载荷）
      data/results/{source}/step1_result.json + stdout 报告

用法: python analysis/step1_consistency.py --source agree|old [--on-only] [--threshold 0.40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def paf(R, m, max_iter=100, tol=1e-6):
    """主轴因子法（Principal Axis Factoring），自实现。
    R: 相关矩阵, m: 因子数。返回载荷矩阵 L (p×m)。
    注：factor_analyzer 包与 sklearn 1.8 的 check_array 不兼容，故自实现（口径等价）。"""
    p = R.shape[0]
    try:  # 初始共同度 = SMC（多元相关平方）
        smc = 1 - 1 / np.diag(np.linalg.pinv(R))
    except np.linalg.LinAlgError:
        smc = np.full(p, 0.5)
    h2 = np.clip(smc, 0.01, 0.99)
    L = None
    for _ in range(max_iter):
        R_h = R.copy()
        np.fill_diagonal(R_h, h2)
        eigval, eigvec = np.linalg.eigh(R_h)
        order = np.argsort(eigval)[::-1][:m]
        L = eigvec[:, order] * np.sqrt(np.clip(eigval[order], 0, None))
        h2_new = np.clip((L ** 2).sum(axis=1), 0.01, 0.99)
        if np.max(np.abs(h2_new - h2)) < tol:
            break
        h2 = h2_new
    return L


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax 旋转（标准实现）。返回旋转后载荷。"""
    p_, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        L = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p_) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return Phi @ R


def parallel_analysis(corr, n_obs, n_iter=1000, pct=95, seed=config.SEED):
    """平行分析：实际特征值 vs 随机正态数据特征值分布。返回 (实际特征值, 随机pct分位线, PA因子数)。"""
    p = corr.shape[0]
    rng = np.random.default_rng(seed)
    eig_null = np.empty((n_iter, p))
    for i in range(n_iter):
        data = rng.standard_normal((n_obs, p))
        r = np.corrcoef(data, rowvar=False)
        eig_null[i] = np.sort(np.linalg.eigvalsh(r))[::-1]
    thresh = np.percentile(eig_null, pct, axis=0)
    eig_real = np.sort(np.linalg.eigvalsh(corr))[::-1]
    n_factors = int(np.sum(eig_real > thresh))
    return eig_real, thresh, n_factors


def velicer_map(corr):
    """Velicer MAP：保留 m 个成分后的偏相关平方均值取最小。返回 (MAP因子数, 序列)。"""
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    p = corr.shape[0]
    seq = []
    for m in range(0, p):
        r_m = (eigvec[:, :m] * eigval[:m]) @ eigvec[:, :m].T
        r_m[np.diag_indices(p)] = 1.0
        try:
            inv = np.linalg.pinv(r_m)
            d = np.sqrt(np.maximum(np.diag(inv), 1e-12))
            pc = -inv / np.outer(d, d)
        except np.linalg.LinAlgError:
            seq.append(np.inf)
            continue
        off = ~np.eye(p, dtype=bool)
        seq.append(float(np.mean(pc[off] ** 2)))
    seq = np.array(seq)
    # MAP 准则取序列最小值（排除 m=0 之前的平凡段，从 m=1 起比较则含 m=0 起点全序列）
    k_map = int(np.argmin(seq))
    return k_map, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--on-only", action="store_true", help="只用 on 列（控 on/off 非独立性）")
    ap.add_argument("--threshold", type=float, default=0.40, help="第一公因子解释方差一致性阈值")
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    scored = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0)
    active = pd.read_csv(res_dir / "active_items.csv")
    mat = scored.loc[active["rule_id"]].astype(float)
    if args.on_only:
        mat = mat[[c for c in mat.columns if c.endswith("_on")]]
    mat = mat.dropna(axis=0, thresh=max(2, mat.shape[1] // 2))  # 行级 NaN 防御
    X = mat.values
    n, p = X.shape

    corr = np.corrcoef(np.nan_to_num(X - np.nanmean(X, axis=0)), rowvar=False)

    # 取样适切性
    kmo_all, kmo_model = calculate_kmo(mat.fillna(mat.mean()))
    chi2, p_bart = calculate_bartlett_sphericity(mat.fillna(mat.mean()))

    # 平行分析 + MAP
    eig_real, pa_thresh, k_pa = parallel_analysis(corr, n)
    k_map, map_seq = velicer_map(corr)

    # Kaiser（特征值>1，作参考基线）
    k_kaiser = int(np.sum(eig_real > 1.0))

    # 共识区间
    k_set = sorted({k_kaiser, k_pa, k_map})
    k_consensus = int(np.median(k_set))

    # EFA（自实现 PAF：不旋转第一公因子 → 一致性；varimax 旋转后 → 模型因子结构）
    n_factors = max(k_pa, 1)
    L_unrot = paf(corr, n_factors)
    first_pv = float((L_unrot[:, 0] ** 2).sum() / corr.shape[0])  # 第一公因子方差解释率

    L_rot = varimax(L_unrot)
    loadings = pd.DataFrame(L_rot, index=mat.columns,
                            columns=[f"F{i+1}" for i in range(n_factors)])
    loadings.to_csv(res_dir / "step1_efa_loadings.csv", encoding="utf-8-sig")

    consistent = (not np.isnan(first_pv)) and first_pv >= args.threshold
    result = {
        "source": args.source,
        "on_only": bool(args.on_only),
        "n_items": int(n), "n_columns": int(p),
        "kmo": float(kmo_model),
        "bartlett_p": float(p_bart),
        "n_factors_kaiser": k_kaiser,
        "n_factors_parallel": k_pa,
        "n_factors_map": k_map,
        "n_factors_consensus_median": k_consensus,
        "first_factor_variance_ratio": first_pv,
        "consistency_threshold": args.threshold,
        "consistent": bool(consistent),
        "verdict": "",
    }
    if consistent:
        result["verdict"] = (
            f"✅ 测量主体一致性成立: 第一公因子解释方差 {first_pv*100:.1f}% ≥ {args.threshold*100:.0f}%，"
            f"22(或11)个模型-模式列存在共享测量基准，题目聚类正当"
        )
    else:
        result["verdict"] = (
            f"⚠ 一致性不足: 第一公因子解释方差 {first_pv*100:.1f}% < {args.threshold*100:.0f}%，"
            f"模型间分化明显——经验维度需按模型簇分别报告（见载荷矩阵）"
        )
    with open(res_dir / "step1_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"[source={args.source}] 第一步: 测量主体一致性检验 (on_only={args.on_only})")
    print(f"样本: {n} 活跃题 × {p} 模型列")
    print(f"KMO = {kmo_model:.3f} (>0.6 可做因子分析) | Bartlett p = {p_bart:.2e}")
    print(f"\n因子数三准则: Kaiser(λ>1) = {k_kaiser} | 平行分析 = {k_pa} | Velicer MAP = {k_map}")
    print(f"共识区间: [{min(k_set)}, {max(k_set)}]，中位数 {k_consensus}")
    print(f"\n第一公因子方差解释率 = {first_pv*100:.1f}%")
    print(f"\n{result['verdict']}")
    print(f"\n旋转后载荷已存: {res_dir / 'step1_efa_loadings.csv'}（每列模型在因子上的归属→模型分化结构）")


if __name__ == "__main__":
    main()
