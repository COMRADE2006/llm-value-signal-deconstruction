# -*- coding: utf-8 -*-
"""
稳健性追加检验（2026-10-08 讨论补强，总方案 v2 配套）。

四组检验:
  A. 马蹄效应诊断: MDS2 ~ MDS1 + MDS1² 二次拟合 R²（高=马蹄伪影，环形解释作废）
     + 梯度检验（MDS1/MDS2 × 子主题支持度）+ 环形正式判据（相似度×角距离衰减相关）
  B. 维度可区分性: PERMANOVA 维度对检验（题目级，双重中心化空间，66 对）
     + 维度质心 bootstrap 95% 椭圆图（可视化参考，正式结论以 PERMANOVA 为准）
  C. 经验类标签的武器三（公平版）: on 半边聚类生成标签 → off 半边做正反判别检验，
     与 12 维标签在 off 半边的同款检验并排对照（同题同列，公平比较）
  D. 纯度置换基线: step3 维度纯度 vs 聚类标签打乱零分布

输入: data/results/{source}/ 已有输出（mds_subtopics_coords.csv / centered_orig.csv /
      scored_matrix.csv / active_items.csv / step2_labels_orig.csv）
输出: data/results/{source}/robustness_result.json,
      robustness_permanova.csv, robustness_dim_ellipse.png

用法: python analysis/robustness_extra.py --source agree|old [--n-perm 1000] [--n-boot 1000] [--kmax 15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import dist_1_minus_corr, double_center_arr, ward_best_k

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def paired_r(p, r, k):
    """按分组标签配对的正反相关（weapon3 同款逻辑）。"""
    ps, rs = [], []
    for v in np.unique(k):
        pv, rv = p[k == v].mean(), r[k == v].mean()
        if not (np.isnan(pv) or np.isnan(rv)):
            ps.append(pv)
            rs.append(rv)
    if len(ps) < 3:
        return np.nan, len(ps)
    return float(np.corrcoef(ps, rs)[0, 1]), len(ps)


def section_a_horseshoe(res_dir, active, scored):
    """A. 马蹄效应诊断 + 环形判据。"""
    coords = pd.read_csv(res_dir / "mds_subtopics_coords.csv", index_col=0)
    cent = pd.read_csv(res_dir / "centered_orig.csv", index_col=0)
    sub_map = active.set_index("rule_id")["derived_values"].fillna("").astype(str).str.split("|")

    # 子主题支持度（计分后题目均分 → 子主题均值）
    item_mean = np.nanmean(scored.values.astype(float), axis=1)
    sub_sup = {}
    for rid, m in zip(scored.index, item_mean):
        for s in sub_map.get(rid, []):
            if s:
                sub_sup.setdefault(s, []).append(m)
    support = pd.Series({s: np.mean(v) for s, v in sub_sup.items()})
    support = support.reindex(coords.index)

    x, y = coords.iloc[:, 0].values, coords.iloc[:, 1].values
    # 二次拟合（马蹄签名: MDS2 是 MDS1 的二次函数）
    coef = np.polyfit(x, y, 2)
    pred = np.polyval(coef, x)
    r2_quad = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    # 梯度检验
    r_m1_sup = float(np.corrcoef(x, support.values)[0, 1])
    r_m2_sup = float(np.corrcoef(y, support.values)[0, 1])
    # 环形判据：相似度随角距离衰减
    sub_rows, sub_ids = [], []
    for rid in cent.index:
        for s in sub_map.get(rid, []):
            if s:
                sub_ids.append(s)
                sub_rows.append(cent.loc[rid].values.astype(float))
    sub_mean = pd.DataFrame(sub_rows, index=sub_ids, columns=cent.columns).groupby(level=0).mean()
    sub_mean = sub_mean.reindex(coords.index)
    Xs = np.nan_to_num(sub_mean.values)
    xc = Xs - Xs.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(xc, axis=1, keepdims=True)
    nrm[nrm == 0] = 1e-12
    corr = (xc / nrm) @ (xc / nrm).T
    ang = np.arctan2(y - y.mean(), x - x.mean())
    iu = np.triu_indices(len(coords), k=1)
    angd = np.abs(ang[iu[0]] - ang[iu[1]])
    angd = np.minimum(angd, 2 * np.pi - angd)
    r_decay = float(np.corrcoef(corr[iu], angd)[0, 1])
    # 去循环化：控制欧氏距离后的角距离偏相关（MDS 构造保证相似×欧氏负相关，
    # 偏相关才检验"角距离携带欧氏距离之外的结构信息"= 真环形证据）
    eucd = np.sqrt((x[iu[0]] - x[iu[1]]) ** 2 + (y[iu[0]] - y[iu[1]]) ** 2)
    sim = corr[iu]
    r_se = np.corrcoef(sim, eucd)[0, 1]
    r_ae = np.corrcoef(angd, eucd)[0, 1]
    denom = np.sqrt(max((1 - r_se ** 2) * (1 - r_ae ** 2), 1e-12))
    r_decay_partial = float((r_decay - r_se * r_ae) / denom)

    horseshoe = bool(r2_quad > 0.5 and abs(r_m1_sup) > 0.5)
    circumplex = bool(r_decay_partial < -0.3)
    verdict = (
        "⚠ 马蹄伪影主导: 二次拟合 R²={:.2f} 且 MDS1×支持度 r={:+.2f} —— 环形/扇形形态是一维支持度梯度弯曲的伪影，不能作环形解释"
        .format(r2_quad, r_m1_sup) if horseshoe else
        ("有真角度衰减证据（偏相关 r={:+.2f}，探索性），可谨慎讨论构型但不得宣称验证环形"
         .format(r_decay_partial) if circumplex else
         "无明确形态结论: 既无马蹄强签名也无超出距离信息的角度结构，构型按纯描述性报告"))
    return {"quad_fit_r2": float(r2_quad), "r_mds1_support": r_m1_sup, "r_mds2_support": r_m2_sup,
            "r_similarity_angular_dist": r_decay, "r_similarity_angular_partial": r_decay_partial,
            "r_similarity_euclidean": float(r_se),
            "horseshoe": horseshoe, "circumplex_evidence": circumplex,
            "verdict": verdict}, (r2_quad, r_m1_sup, r_decay, verdict)


def _ssw(A, g):
    n1, n2 = int(g.sum()), int((~g).sum())
    return A[np.ix_(g, g)].sum() / (2 * n1) + A[np.ix_(~g, ~g)].sum() / (2 * n2)


def permanova_pair(A, g, n_perm, rng):
    """两组 PERMANOVA（Anderson 2001 距离平方和分解）。返回 (F, p, 效应量R²)。"""
    N = A.shape[0]
    ss_t = A.sum() / (2 * N)
    ss_w = _ssw(A, g)
    ss_a = ss_t - ss_w
    f_obs = ss_a / (ss_w / (N - 2)) if ss_w > 1e-12 else np.inf
    r2 = float(ss_a / ss_t) if ss_t > 1e-12 else np.nan
    cnt = 0
    for _ in range(n_perm):
        gp = rng.permutation(g)
        ssw_p = _ssw(A, gp)
        ssa_p = ss_t - ssw_p
        f_p = ssa_p / (ssw_p / (N - 2)) if ssw_p > 1e-12 else np.inf
        if f_p >= f_obs:
            cnt += 1
    return float(f_obs), (cnt + 1) / (n_perm + 1), r2


def section_b_permanova(res_dir, active, n_perm, n_boot, kmax):
    """B. 维度可区分性: PERMANOVA 66 对 + 质心 bootstrap 椭圆。"""
    rng = np.random.default_rng(config.SEED)
    cent = pd.read_csv(res_dir / "centered_orig.csv", index_col=0)
    meta = active.set_index("rule_id").loc[cent.index]
    X = np.nan_to_num(cent.values.astype(float))
    D2 = squareform(pdist(X, "sqeuclidean"))
    dims = meta["core_value"].unique().tolist()

    rows = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            idx = np.where((meta["core_value"] == dims[i]) | (meta["core_value"] == dims[j]))[0]
            g = (meta["core_value"].values[idx] == dims[i])
            if g.sum() < 5 or (~g).sum() < 5:
                continue
            f, p, r2 = permanova_pair(D2[np.ix_(idx, idx)], g, n_perm, rng)
            rows.append({"dim1": dims[i], "dim2": dims[j], "n1": int(g.sum()), "n2": int((~g).sum()),
                         "pseudo_F": f, "perm_p": p, "effect_r2": r2})
        if (i + 1) % 4 == 0:
            print(f"    PERMANOVA 进度 {i+1}/{len(dims)} 维")
    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(res_dir / "robustness_permanova.csv", index=False, encoding="utf-8-sig")
    n_sig = int((perm_df["perm_p"] < 0.05).sum())
    n_bonf = int((perm_df["perm_p"] < 0.05 / len(perm_df)).sum())
    undist = perm_df[perm_df["perm_p"] >= 0.05].sort_values("perm_p")

    # 质心 bootstrap 椭圆（题目重采样 → MDS 坐标均值）
    coords = pd.read_csv(res_dir / "mds_subtopics_coords.csv", index_col=0)
    sub_map = meta["derived_values"].fillna("").astype(str).str.split("|")
    item_pos = {}
    for rid in cent.index:
        pts = [coords.loc[s].values for s in sub_map.get(rid, []) if s and s in coords.index]
        if pts:
            item_pos[rid] = np.mean(pts, axis=0)
    ip = pd.DataFrame(item_pos).T
    ip = ip.reindex(cent.index)
    fig, ax = plt.subplots(figsize=(11, 9))
    ellipses = {}
    for v in dims:
        rids = meta.index[(meta["core_value"] == v).values]
        rids = [r for r in rids if r in ip.index and not np.isnan(ip.loc[r]).any()]
        if len(rids) < 10:
            continue
        base = ip.loc[rids].values
        boots = np.array([base[rng.integers(0, len(base), len(base))].mean(axis=0) for _ in range(n_boot)])
        mu = boots.mean(axis=0)
        cov = np.cov(boots.T)
        val, vec = np.linalg.eigh(cov)
        # 95% 椭圆（χ²(2) 0.95 = 5.991），退化方向给最小半径
        val = np.maximum(val, 1e-8)
        theta = np.linspace(0, 2 * np.pi, 64)
        circle = np.array([np.cos(theta), np.sin(theta)]).T * np.sqrt(5.991 * val)
        poly = mu + circle @ vec.T
        ellipses[v] = MplPath(poly)
        ax.plot(poly[:, 0], poly[:, 1], alpha=0.8)
        ax.annotate(v, mu, fontsize=10, fontweight="bold", ha="center", va="center")
    # 两两椭圆重叠（可视化参考）
    vs = list(ellipses.keys())
    overlap_pairs = [f"{vs[i]}~{vs[j]}" for i in range(len(vs)) for j in range(i + 1, len(vs))
                     if ellipses[vs[i]].intersects_path(ellipses[vs[j]], filled=True)]
    ax.set_title(f"12 维质心 bootstrap 95% 椭圆（n_boot={n_boot}）\n"
                 f"椭圆重叠仅为可视化参考，正式可区分性检验见 PERMANOVA")
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    fig.tight_layout()
    fig.savefig(res_dir / "robustness_dim_ellipse.png", dpi=150)

    return {"n_pairs": int(len(perm_df)), "n_sig_raw": n_sig, "n_sig_bonferroni": n_bonf,
            "sig_rate_raw": float(n_sig / len(perm_df)),
            "effect_r2_median": float(perm_df["effect_r2"].median()),
            "effect_r2_max": float(perm_df["effect_r2"].max()),
            "undistinguishable_pairs_raw": undist[["dim1", "dim2", "perm_p"]].to_dict("records")[:15],
            "n_ellipse_overlaps": len(overlap_pairs), "ellipse_overlap_pairs": overlap_pairs}, perm_df


def section_c_emp_weapon3(res_dir, active, scored, n_perm, kmax):
    """C. 经验类标签武器三（公平版）: on 半边定标签，off 半边检验。
    注意: 配对正反相关要求 ≥3 组；轮廓最优 K=2 时无定义，
    故强制在 K∈{4,6,8} 各测一次（测试分辨率稳健性），另报轮廓最优 K 供参照。"""
    on_cols = [c for c in scored.columns if c.endswith("_on")]
    off_cols = [c for c in scored.columns if c.endswith("_off")]
    Xh = scored[on_cols].values.astype(float)
    keep = np.sum(~np.isnan(Xh), axis=1) >= max(6, len(on_cols) - 3)
    Xc = np.nan_to_num(double_center_arr(Xh[keep]))
    D = dist_1_minus_corr(Xc)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform as _sq
    Z = linkage(_sq(D, checks=False), method="ward")
    _, sil, k_sil = ward_best_k(D, kmax)

    meta = active.set_index("rule_id").loc[scored.index[keep]]
    direction = meta["direction"].values
    is_pos = direction == "positive"

    def run_discriminate(group_key):
        rows = []
        for c in off_cols:
            col = scored.loc[scored.index[keep], c].values
            for g in np.unique(group_key):
                m = group_key == g
                rows.append({"model": c, "group": g,
                             "pos_mean": col[m & is_pos].mean() if (m & is_pos).sum() else np.nan,
                             "neg_mean": col[m & ~is_pos].mean() if (m & ~is_pos).sum() else np.nan})
        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return {"r_same": np.nan, "perm_p": np.nan, "null_mean": np.nan, "n_groups": 0}
        p, r, k = df["pos_mean"].values, df["neg_mean"].values, df["group"].values
        r_same, n_g = paired_r(p, r, k)
        rng = np.random.default_rng(config.SEED)
        null = []
        for _ in range(n_perm):
            nr, _ = paired_r(p, r, rng.permutation(k))
            null.append(nr)
        null = np.array([v for v in null if not np.isnan(v)])
        return {"r_same": r_same, "perm_p": float(np.mean(null >= r_same)) if len(null) else np.nan,
                "null_mean": float(np.mean(null)) if len(null) else np.nan, "n_groups": n_g}

    out = {"silhouette_k_on_half": int(k_sil), "n_items": int(keep.sum()),
           "empirical_by_k": {}, "core12": run_discriminate(meta["core_value"].values)}
    for k_test in (4, 6, 8):
        lab = fcluster(Z, t=k_test, criterion="maxclust")
        out["empirical_by_k"][str(k_test)] = run_discriminate(lab)

    candidates = [(v["r_same"], k) for k, v in out["empirical_by_k"].items()
                  if not np.isnan(v["r_same"]) and not np.isnan(v["perm_p"])]
    emp_best = max(candidates) if candidates else (np.nan, None)
    v12 = out["core12"]
    dim_r = v12["r_same"]
    emp_sig = candidates and emp_best[1] is not None and out["empirical_by_k"][str(emp_best[1])]["perm_p"] < 0.05
    dim_sig = (not np.isnan(v12["perm_p"])) and v12["perm_p"] < 0.05
    if emp_sig and (not dim_sig or emp_best[0] > dim_r):
        verdict = (f"经验类(最优K={emp_best[1]}) r={emp_best[0]:+.3f} (p<0.05) 显著超过置换零假设"
                   f"且强于12维标签 r={dim_r:+.3f} —— 经验类有样本外正反互证结构（on→off）")
    elif emp_sig:
        verdict = f"经验类 r={emp_best[0]:+.3f} 显著但未超过12维标签 r={dim_r:+.3f}"
    else:
        verdict = (f"经验类(最优K={emp_best[1] if emp_best[1] else 'NA'}) r={emp_best[0]:+.3f} 未超过自身置换零假设"
                   f"（p≥0.05，注意低K时零假设本身偏高，该检验欠功效）；12维标签 r={dim_r:+.3f} (p={v12['perm_p']:.3f})"
                   f" —— 两套标签均无显著正反互证结构")
    out["verdict"] = verdict
    return out


def section_d_purity_perm(res_dir, active, n_perm):
    """D. 纯度置换基线: 维度多数类纯度 vs 标签打乱零分布。"""
    lab = pd.read_csv(res_dir / "step2_labels_orig.csv")
    lab.columns = ["rule_id", "cluster"]
    df = active.merge(lab, on="rule_id")
    obs = df.groupby("core_value")["cluster"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    obs_mean = float(obs.mean())
    rng = np.random.default_rng(config.SEED)
    null = []
    for _ in range(n_perm):
        df["_p"] = rng.permutation(df["cluster"].values)
        null.append(df.groupby("core_value")["_p"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean())
    null = np.array(null)
    pval = float(np.mean(null >= obs_mean))
    return {"observed_mean_purity": obs_mean, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "perm_p": pval,
            "verdict": (f"纯度 {obs_mean:.3f} 超出置换 p95（p={pval:.3f}）→ 类构成高于机遇水平"
                        if pval < 0.05 else
                        f"纯度 {obs_mean:.3f} 未超出置换分布（p={pval:.3f}）→ 类构成与机遇水平无异")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--kmax", type=int, default=15)
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    active = pd.read_csv(res_dir / "active_items.csv")
    scored = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0).loc[active["rule_id"]]

    print("===== A. 马蹄效应诊断 =====")
    a_res, a_disp = section_a_horseshoe(res_dir, active, scored)
    print(f"  二次拟合 R²={a_res['quad_fit_r2']:.3f}, MDS1×支持度 r={a_res['r_mds1_support']:+.3f}, "
          f"相似度×角距离 r={a_res['r_similarity_angular_dist']:+.3f}")
    print(f"  {a_res['verdict']}")

    print("===== B. 维度可区分性（PERMANOVA + 椭圆）=====")
    b_res, perm_df = section_b_permanova(res_dir, active, args.n_perm, args.n_boot, args.kmax)
    print(f"  66 对中可区分: 原始 p<0.05 共 {b_res['n_sig_raw']} 对, Bonferroni 后 {b_res['n_sig_bonferroni']} 对")
    print(f"  效应量 R²: 中位数 {b_res['effect_r2_median']:.3f}, 最大 {b_res['effect_r2_max']:.3f}"
          f"（N=1773 检验过功效，显著性看 Bonferroni，实质看效应量）")
    print(f"  椭圆重叠对数: {b_res['n_ellipse_overlaps']}（参考）")
    if b_res["undistinguishable_pairs_raw"]:
        print("  不可区分对（p≥0.05，响应空间中维度边界模糊）:")
        for r in b_res["undistinguishable_pairs_raw"][:8]:
            print(f"    {r['dim1']} ~ {r['dim2']} (p={r['perm_p']:.3f})")

    print("===== C. 经验类标签的武器三（公平版）=====")
    c_res = section_c_emp_weapon3(res_dir, active, scored, args.n_perm, args.kmax)
    print(f"  on 半边轮廓最优 K={c_res['silhouette_k_on_half']}（<3 组时配对相关无定义，强制测 K=4/6/8）")
    for k, v in c_res["empirical_by_k"].items():
        print(f"  经验类 K={k}: r={v['r_same']:+.3f} (p={v['perm_p']:.3f}, 零假设均值={v['null_mean']:+.3f})")
    v12 = c_res["core12"]
    print(f"  12 维标签:    r={v12['r_same']:+.3f} (p={v12['perm_p']:.3f}, 零假设均值={v12['null_mean']:+.3f})")
    print(f"  {c_res['verdict']}")

    print("===== D. 纯度置换基线 =====")
    d_res = section_d_purity_perm(res_dir, active, args.n_perm)
    print(f"  {d_res['verdict']}")

    result = {"source": args.source, "n_perm": args.n_perm, "n_boot": args.n_boot,
              "horseshoe": a_res, "permanova": b_res, "emp_weapon3": c_res, "purity_perm": d_res}
    with open(res_dir / "robustness_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已存: {res_dir / 'robustness_result.json'}")


if __name__ == "__main__":
    main()
