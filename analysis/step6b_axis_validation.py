# -*- coding: utf-8 -*-
"""
step6b 轴真实性验证：三项独立检验，回答"轴是不是随机噪声"

V1 标签置换零分布：把12维标签在题目间随机打乱1000次，重算η²。
   若真实η²=0.084远超打乱分布 → 维度分组效应不是巧合。
   （关键：机会水平η²≈(k-1)/(n-1)≈0.006，需用数字堵住"0.084很小"的质疑）

V2 对半稳定性：题目随机对半分，各自独立走完整管线（自己的中心化+SVD，
   在前5个方向中选η²最大的作为内容方向），比较两半的12维均值剖面相关。
   噪声数据的两半各自挑出的"方向"其维度剖面不会相关。

V3 维度排序稳定性：维度内有放回重采样1000次，12维排序的Spearman自相关
   + 各维度留在原符号侧的比例。
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center, eta_squared

BASE = config.DATA_DIR / "results" / "old"
rng = np.random.default_rng(20261010)


def pick_content_direction(Xh, cv_h, top=5):
    """对半数据的独立管线: 中心化→SVD→前top个方向中η²最大者。
    返回 (载荷向量, 该方向η²)。"""
    U, s, Vt = np.linalg.svd(double_center(Xh), full_matrices=False)
    best_v, best_eta = None, -1.0
    for k in range(min(top, U.shape[1])):
        eta = eta_squared(U[:, k], cv_h)
        if eta > best_eta:
            best_eta, best_v = eta, U[:, k]
    return best_v, best_eta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["old", "agree", "schwartz"], default="old")
    args = ap.parse_args()
    BASE = config.DATA_DIR / "results" / args.source

    X, meta, cols = load_matrices(args.source)
    cv = meta["core_value"].values
    n = len(cv)
    dims = pd.unique(cv)

    # 全数据内容方向载荷：取该数据源MP谱检验的"内容效应"方向（与step6一致）
    with open(BASE / "mp_spectrum_test.json", encoding="utf-8") as f:
        mp = json.load(f)
    content = [i for i in mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    k_axis = max(content, key=lambda i: i["eta2_core_value_control_direction"])["component"] - 1
    print(f"轴方向: {args.source}数据的MP方向{k_axis + 1}（内容效应）")

    U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
    v2 = U[:, k_axis]
    eta_real = eta_squared(v2, cv)
    dim_mean_full = pd.Series(v2).groupby(cv).mean()

    # ---- V1 标签置换零分布 ----
    print("V1 标签置换零分布: 1000次...")
    eta_null = np.array([eta_squared(v2, rng.permutation(cv)) for _ in range(1000)])
    v1_p = float((eta_null >= eta_real).mean())
    print(f"  真实η²={eta_real:.4f} | 置换零分布: 均值{eta_null.mean():.4f} "
          f"最大{eta_null.max():.4f} | p={v1_p}")
    print(f"  机会水平(理论(k-1)/(n-1))={(len(dims)-1)/(n-1):.4f} → 真实值是机会水平的"
          f"{eta_real/((len(dims)-1)/(n-1)):.1f}倍")

    # ---- V2 对半稳定性 ----
    print("\nV2 对半稳定性: 200次独立对半...")
    rs = []
    for b in range(200):
        idx = rng.permutation(n)
        A, B = idx[:n // 2], idx[n // 2:]
        vA, _ = pick_content_direction(X[A], cv[A])
        vB, _ = pick_content_direction(X[B], cv[B])
        mA = pd.Series(vA).groupby(cv[A]).mean()
        mB = pd.Series(vB).groupby(cv[B]).mean()
        # 两半的维度剖面相关（符号全局翻转不影响|r|）。
        # 稀少维度（如活跃题仅3题的刺激）可能整维落入另一半，取公共维度计算
        common = mA.index.intersection(mB.index)
        rs.append(abs(np.corrcoef(mA.loc[common].values, mB.loc[common].values)[0, 1]))
    rs = np.array(rs)
    print(f"  两半维度剖面|相关|: 中位数{np.median(rs):.3f} "
          f"(IQR {np.percentile(rs,25):.3f}~{np.percentile(rs,75):.3f}) "
          f"| >0.7的比例: {(rs>0.7).mean():.0%} | >0.9: {(rs>0.9).mean():.0%}")

    # ---- V3 维度排序bootstrap稳定性 ----
    print("\nV3 维度排序bootstrap: 1000次...")
    rhos, side = [], {d: 0.0 for d in dims}
    for b in range(1000):
        idxs = []
        for d in dims:
            pool = np.where(cv == d)[0]
            idxs.append(rng.choice(pool, size=len(pool), replace=True))
        boot = np.concatenate(idxs)
        m = pd.Series(v2[boot]).groupby(cv[boot]).mean()
        rho = spearmanr(m.values, dim_mean_full.values).statistic
        rhos.append(rho)
        for d in dims:
            if np.sign(m[d]) == np.sign(dim_mean_full[d]):
                side[d] += 1
    rhos = np.array(rhos)
    n_boot = len(rhos)
    side = {d: p / n_boot for d, p in side.items()}
    print(f"  排序Spearman自相关: 中位数{np.median(rhos):.3f} "
          f"(5%~95%: {np.percentile(rhos,5):.3f}~{np.percentile(rhos,95):.3f})")
    print("  各维度留在原符号侧的比例:")
    side_sorted = sorted(side.items(), key=lambda kv: dim_mean_full[kv[0]])
    for d, p in side_sorted:
        print(f"    {d}: {p:.1%}")

    out = {
        "V1_label_permutation": {
            "eta_real": float(eta_real), "null_mean": float(eta_null.mean()),
            "null_max": float(eta_null.max()), "p": v1_p,
            "chance_level": float((len(dims) - 1) / (n - 1)),
            "times_above_chance": float(eta_real / ((len(dims) - 1) / (n - 1)))},
        "V2_split_half": {
            "n_reps": 200, "median_abs_r": float(np.median(rs)),
            "iqr": [float(np.percentile(rs, 25)), float(np.percentile(rs, 75))],
            "frac_gt_0.7": float((rs > 0.7).mean()),
            "frac_gt_0.9": float((rs > 0.9).mean())},
        "V3_bootstrap_ordering": {
            "median_spearman": float(np.median(rhos)),
            "pct5_95": [float(np.percentile(rhos, 5)), float(np.percentile(rhos, 95))],
            "side_stay_rate": {d: float(p) for d, p in side.items()}},
        "dimension_ordering": {d: float(v) for d, v in dim_mean_full.sort_values().items()},
    }
    with open(BASE / "axis_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {BASE/'axis_validation.json'}")


if __name__ == "__main__":
    main()
