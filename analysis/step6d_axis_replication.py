# -*- coding: utf-8 -*-
"""
step6d 轴复现检验：新数据（盲评认同度）内容方向 vs 旧数据（符合度）语义轴的全排序对比

问题：旧数据发现的"社会-人际 ←→ 个人-制度"语义轴，在新Prompt数据上是否复现？
判据（用户已确认）：
  新轴12维排序与旧轴排序的 |Spearman ρ|：≥0.7 强复现 / 0.4~0.7 部分复现 / <0.4 未复现
  参照系：旧数据自身item-bootstrap重采样的排序自相关中位数0.951（step6b）

方法：
  1. 旧轴：旧数据确定性重算（双重中心化→SVD→方向2→回归掉正反向→12维残差均值排序）
     （旧json写于dimension_ordering_full字段加入之前，故重算；SVD确定性，无随机性）
  2. 新轴：读 agree/mp_spectrum_test.json 的内容效应方向（verdict=内容效应）的
     dimension_ordering_full
  3. 对齐12维名称 → Spearman ρ（符号任意，|ρ|为结构一致性；按相关符号对齐极点后
     统计极点成员重叠数）
  4. 附加：新数据剔商汤on列变体（mp_spectrum_test_drop.json，若存在）的轴 vs 新主轴、
     vs 旧轴 —— NaN列威胁封口

输出：data/results/agree/axis_replication.json
运行：python analysis/step6d_axis_replication.py
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
from analysis.step6_mp_spectrum import double_center

OLD_DIR = config.DATA_DIR / "results" / "old"
NEW_DIR = config.DATA_DIR / "results" / "agree"


def old_axis_ordering():
    """旧数据语义轴的完整12维排序（确定性重算，与step6旧运行同流程）。"""
    with open(OLD_DIR / "mp_spectrum_test.json", encoding="utf-8") as f:
        old_mp = json.load(f)
    # 内容效应方向 = verdict=内容效应（旧数据为component 2，即索引1）
    content = [i for i in old_mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    k_idx = content[0]["component"] - 1  # 0-based奇异方向索引
    eta_old = content[0]["eta2_core_value_control_direction"]

    X, meta, _ = load_matrices("old")
    is_rev = (meta["direction"].values == "negative").astype(float)
    cv = meta["core_value"].values
    U, _, _ = np.linalg.svd(double_center(X), full_matrices=False)
    v = U[:, k_idx]
    # 回归掉正反向（与step6一致）
    beta = np.polyfit(is_rev, v, 1)
    resid = v - np.polyval(beta, is_rev)
    order = pd.Series(resid).groupby(cv).mean().sort_values()
    return order, eta_old, k_idx + 1


def new_axis_ordering(json_path):
    """读MP结果json中内容效应方向的完整排序。返回 (order, eta2, component, n_content)。"""
    with open(json_path, encoding="utf-8") as f:
        mp = json.load(f)
    content = [i for i in mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    if not content:
        return None, None, None, 0
    best = max(content, key=lambda i: i["eta2_core_value_control_direction"])
    order = pd.Series(best["dimension_ordering_full"]).sort_values()
    return order, best["eta2_core_value_control_direction"], best["component"], len(content)


def compare(order_a, order_b, label_a, label_b):
    """两轴排序对比：Spearman(按维度名对齐)、极点重叠。返回dict。"""
    dims = sorted(set(order_a.index) & set(order_b.index))
    a = order_a.reindex(dims).values
    b = order_b.reindex(dims).values
    rho = float(spearmanr(a, b).statistic)
    sign = 1.0 if rho >= 0 else -1.0
    # 按相关符号对齐b方向后，比较极点成员（各取|排序|前6为两端）
    b_aligned = b * sign
    n_half = len(dims) // 2
    a_neg = set(pd.Series(a, index=dims).sort_values().index[:n_half])
    b_neg = set(pd.Series(b_aligned, index=dims).sort_values().index[:n_half])
    overlap = len(a_neg & b_neg)
    verdict = ("强复现(ρ≥0.7)" if abs(rho) >= 0.7 else
               "部分复现(0.4≤ρ<0.7)" if abs(rho) >= 0.4 else "未复现(ρ<0.4)")
    return {
        "comparison": f"{label_a} vs {label_b}",
        "spearman_rho": rho, "abs_rho": abs(rho),
        "pole_overlap": f"{overlap}/{n_half}",
        "verdict": verdict,
        "order_a": {d: float(v) for d, v in order_a.reindex(dims).items()},
        "order_b_aligned": {d: float(v) for d, v in zip(dims, b_aligned)},
    }


def contrast_test(source, weights, n_perm=1000, seed=20261012):
    """交叉对照检验：把别处 derived 的轴权重(维度→数值)作为固定对照投射到本数据。

    逻辑：c_i = weights[题目i的维度] → 回归掉正反向 → c_r；
    投影强度 SS = Σ_j (Xc[:,j]·c_r)²/||c_r||²（模型空间22维全部计入）；
    零分布：维度标签在题目间随机置换1000次（保持对照结构，破坏题目-维度对应）。
    显著 → 该对照轴在本数据的交互结构中真实存在（比SVD再发现更直接的证实性检验）。
    返回 (var_share, p, null_median_share)。
    """
    rng_local = np.random.default_rng(seed)
    X, meta, _ = load_matrices(source)
    cv = meta["core_value"].values
    is_rev = (meta["direction"].values == "negative").astype(float)
    Xc = double_center(X)

    def ss_of(cv_arr):
        c = np.array([weights.get(d, 0.0) for d in cv_arr])
        beta = np.polyfit(is_rev, c, 1)
        c_r = c - np.polyval(beta, is_rev)
        nr = float(np.linalg.norm(c_r))
        if nr < 1e-12:
            return 0.0
        t = Xc.T @ c_r / nr          # 22维模型投影
        return float(t @ t)          # SS_contrast

    ss_real = ss_of(cv)
    ss_total = float((Xc ** 2).sum())
    null = np.array([ss_of(rng_local.permutation(cv)) for _ in range(n_perm)])
    share_real = ss_real / ss_total
    share_null = null / ss_total
    p = float((null >= ss_real).mean())
    return share_real, p, float(np.median(share_null))


def main():
    old_order, eta_old, old_comp = old_axis_ordering()
    new_order, eta_new, new_comp, n_content = new_axis_ordering(NEW_DIR / "mp_spectrum_test.json")

    print("=" * 70)
    print("轴复现检验：旧(符合度)语义轴 vs 新(盲评认同度)内容方向")
    print("=" * 70)
    print(f"旧轴: 方向{old_comp}, 控制正反向后η²={eta_old:.4f}")
    print(f"  12维排序(负端→正端): {' < '.join(old_order.index.tolist())}")
    print(f"\n新轴: 方向{new_comp}, 控制正反向后η²={eta_new:.4f} (内容方向数={n_content})")
    print(f"  12维排序(负端→正端): {' < '.join(new_order.index.tolist())}")

    res = {"old": {"component": old_comp, "eta2": eta_old,
                   "ordering": {d: float(v) for d, v in old_order.items()}},
           "new": {"component": new_comp, "eta2": eta_new,
                   "ordering": {d: float(v) for d, v in new_order.items()}}}

    main_cmp = compare(old_order, new_order, "旧轴", "新轴")
    res["main_comparison"] = main_cmp
    print(f"\n----- 主对比 -----")
    print(f"Spearman ρ = {main_cmp['spearman_rho']:+.3f} (|ρ|={main_cmp['abs_rho']:.3f})")
    print(f"极点成员重叠: {main_cmp['pole_overlap']}（两端各6维中相同维数）")
    print(f"判定: {main_cmp['verdict']}")
    print(f"参照: 旧数据自身bootstrap排序自相关中位数0.951")

    # 逐维对照表（符号对齐后）
    print(f"\n{'维度':<6}{'旧轴位':>8}{'新轴位':>8}   变化")
    old_rank = old_order.rank()
    new_rank = pd.Series(main_cmp["order_b_aligned"]).rank()
    for d in old_order.index:
        print(f"{d:<6}{old_rank[d]:>8.0f}{new_rank[d]:>8.0f}   {new_rank[d] - old_rank[d]:+.0f}")

    # 剔商汤on列变体（若已跑）
    drop_path = NEW_DIR / "mp_spectrum_test_drop.json"
    if drop_path.exists():
        drop_order, eta_drop, drop_comp, _ = new_axis_ordering(drop_path)
        if drop_order is not None:
            cmp1 = compare(new_order, drop_order, "新主轴", "剔商汤轴")
            cmp2 = compare(old_order, drop_order, "旧轴", "剔商汤轴")
            res["drop_sensenova"] = {"eta2": eta_drop, "component": drop_comp,
                                     "vs_new_main": cmp1, "vs_old": cmp2}
            print(f"\n----- 剔商汤on列变体 (方向{drop_comp}, η²={eta_drop:.4f}) -----")
            print(f"vs 新主轴: |ρ|={cmp1['abs_rho']:.3f}, 极点重叠{cmp1['pole_overlap']}")
            print(f"vs 旧轴:   |ρ|={cmp2['abs_rho']:.3f}, 极点重叠{cmp2['pole_overlap']}")

    # ---- 交叉对照检验（证实性，比SVD再发现更直接）----
    print("\n----- 交叉对照检验（轴权重固定投射，维度标签置换1000次）-----")
    old_w = {d: float(v) for d, v in old_order.items()}
    new_w = {d: float(v) for d, v in new_order.items()}
    tests = [
        ("旧轴权重→旧数据(方法学阳性对照)", "old", old_w),
        ("旧轴权重→新数据(复现检验)", "agree", old_w),
        ("新轴权重→新数据(方法学阳性对照)", "agree", new_w),
        ("新轴权重→旧数据(反向复现)", "old", new_w),
    ]
    res["contrast_tests"] = {}
    for label, src, w in tests:
        share, p, null_med = contrast_test(src, w)
        verdict = "显著存在" if p < 0.001 else ("边际" if p < 0.05 else "不显著")
        res["contrast_tests"][label] = {"var_share": share, "perm_p": p,
                                        "null_median_share": null_med, "verdict": verdict}
        print(f"  {label}: 方差占比={share*100:.3f}% (零分布中位{null_med*100:.3f}%) "
              f"p={p:.3f} → {verdict}")

    with open(NEW_DIR / "axis_replication.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {NEW_DIR / 'axis_replication.json'}")


if __name__ == "__main__":
    main()
