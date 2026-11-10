# -*- coding: utf-8 -*-
"""
step6c 语义轴的推理模式控制检验（论文第5章稳健性素材）

背景：全数据方向2（语义轴）的模型载荷被推理模式占据（负端几乎全是_on、
正端几乎全是_off），需要检验轴是否为推理模式伪影。

检验：分别在纯on(11列)/纯off(11列)子集内独立走管线（双重中心化→SVD→
前5方向中η²最大者），看内容效应是否独立存在、12维排序是否与全数据轴同构。

结论（2026-10-10，见data/results/old/axis_mode_control.json）：
  off子集: η²=0.0928, p<0.001, 与全数据轴|ρ|=0.881 → 独立恢复同构轴
  on子集:  η²=0.0666, p<0.001, 与全数据轴|ρ|=0.517 → 同方向但被稀释
  → 轴不是推理模式伪影；off（直觉响应）里轴更干净
  → 两个方法学发现：①推理模式系统性移动模型在价值轴上的位置（评估必须
    固定/报告推理模式）；②推理稀释价值轴信号
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

rng = np.random.default_rng(20261010)


def content_direction_in_subset(Xs, cvs, top=5):
    """子集内独立管线：中心化→SVD→前top方向中与12维η²最大者"""
    U, s, Vt = np.linalg.svd(double_center(Xs), full_matrices=False)
    best_v, best_eta, best_k = None, -1.0, None
    for k in range(min(top, U.shape[1])):
        eta = eta_squared(U[:, k], cvs)
        if eta > best_eta:
            best_eta, best_v, best_k = eta, U[:, k], k
    return best_v, best_eta, best_k


def perm_p(eta_real, v, cvs, n_perm=1000):
    null = [eta_squared(v, rng.permutation(cvs)) for _ in range(n_perm)]
    return float((np.array(null) >= eta_real).mean()), float(np.max(null))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["old", "agree", "schwartz"], default="old")
    args = ap.parse_args()
    BASE = config.DATA_DIR / "results" / args.source

    X, meta, cols = load_matrices(args.source)
    cv = meta["core_value"].values
    is_on = np.array([c.endswith("_on") for c in cols])

    # 全数据参考轴：取该数据源MP谱检验中"内容效应"方向（而非硬编码方向2）
    with open(BASE / "mp_spectrum_test.json", encoding="utf-8") as f:
        mp = json.load(f)
    content = [i for i in mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    k_ref = max(content, key=lambda i: i["eta2_core_value_control_direction"])["component"] - 1
    print(f"参考轴: {args.source}数据的MP方向{k_ref + 1}（内容效应）")

    U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
    v2 = U[:, k_ref]
    full_order = pd.Series(v2).groupby(cv).mean()

    # 模型端：轴上的模型排序（含bar图）
    loadings = Vt[k_ref, :]
    neg_pole = full_order.sort_values().index[:3].tolist()
    pos_pole = full_order.sort_values().index[-3:].tolist()
    print(f"语义轴上的模型排序（负端={neg_pole} ←→ 正端={pos_pole}）：\n")
    model_table = []
    for i in np.argsort(loadings):
        bar = "█" * int(abs(loadings[i]) * 100)
        sign = "-" if loadings[i] < 0 else "+"
        print(f"  {cols[i]:<24} {loadings[i]:+.3f}  {sign}{bar}")
        model_table.append({"model": cols[i], "loading": float(loadings[i])})

    # 子集控制检验
    results = {"model_loadings_on_axis": model_table}
    orders = {}
    for name, mask in [("on_only", is_on), ("off_only", ~is_on)]:
        v, eta, k = content_direction_in_subset(X[:, mask], cv)
        p, null_max = perm_p(eta, v, cv)
        sub_order = pd.Series(v).groupby(cv).mean()
        orders[name] = sub_order
        rho = spearmanr(sub_order.values, full_order.values).statistic
        results[name] = {
            "best_direction_index": int(k), "eta2": float(eta),
            "perm_p": p, "null_max_eta2": null_max,
            "spearman_vs_full_axis": float(rho),
            "abs_spearman_vs_full_axis": float(abs(rho)),
            "dim_ordering": {d: float(x) for d, x in sub_order.sort_values().items()}}
        print(f"\n[{name}] 方向#{k}: η²={eta:.4f} 置换p={p:.3f} (零分布max={null_max:.4f}) "
              f"与全数据轴排序|ρ|={abs(rho):.3f}")
        print("   12维排序:", " < ".join(sub_order.sort_values().index.tolist()))

    # 子集间互相关（按维度名对齐，避免索引错位）
    rho_on_off = spearmanr(orders["on_only"].values, orders["off_only"].values).statistic
    results["spearman_aligned"] = {
        "on_vs_full": float(spearmanr(orders["on_only"].values, full_order.values).statistic),
        "off_vs_full": float(spearmanr(orders["off_only"].values, full_order.values).statistic),
        "on_vs_off": float(rho_on_off),
        "abs_on_vs_off": float(abs(rho_on_off))}
    print(f"\non子集 vs off子集 12维排序互相关: ρ={rho_on_off:+.3f} (|ρ|={abs(rho_on_off):.3f})")
    print("（SVD方向符号任意，|ρ|才是结构一致性）")

    with open(BASE / "axis_mode_control.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {BASE/'axis_mode_control.json'}")


if __name__ == "__main__":
    main()
