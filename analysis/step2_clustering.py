# -*- coding: utf-8 -*-
"""
第二步：题目层聚类主分析 + 子主题层交叉验证（总方案v2 §4 第二步）。

三通道各聚一次（2026-10-08 用户确认：保留强度和方向，分三次聚类互为稳健性对照）:
  orig = 计分后原始分（强度+方向全保留，主通道）
  I    = IR-Tree 强度通道（丢方向）
  P    = IR-Tree 乘积通道（方向×强度）

每通道:
  - 输入: centered_{ch}.csv（双重中心化后；NaN 填 0 = 无交互信息）
  - 距离: 1 − Pearson 相关（题目×题目）
  - Ward 层次聚类 + 轮廓系数扫 K=2..15
  - 子抽样稳定性: 重复抽 70% 题 → Ward(K*) → 与基准划分算 ARI（均值±sd）
  - 难度因子诊断: PCA 第一主成分与题目均分(raw scored)的相关 >0.3 且解释方差>30% → 警告

通道间稳健性: 三通道基准划分两两 ARI。

子主题层交叉验证（仅 orig 主通道）:
  - 法A: 50子主题×22列行均值 → 双重中心化 → 1−Pearson → Ward
  - 法C: 50子主题两两 RV 系数（矩阵相关，Böckenholt/RV 标准）→ 1−RV → Ward
  - 双法 ARI 一致性 + 法A与题目层划分的子主题纯度

输入: data/results/{source}/centered_*.csv + scored_matrix.csv + active_items.csv
输出: data/results/{source}/step2_labels_{ch}.csv, step2_silhouette_{ch}.csv,
      step2_subtopic_labels.csv, step2_subtopic_rv_labels.csv, step2_result.json

用法: python analysis/step2_clustering.py --source agree|old [--n-sub 300] [--sub-frac 0.7]
      [--channels orig,I,P] [--kmax 15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def dist_1_minus_corr(X):
    """1 − Pearson 相关距离矩阵。X: n×p（NaN 已填 0，即中心化后无交互=0）。"""
    xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(xc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    xc = xc / norm
    corr = xc @ xc.T
    return np.clip(1.0 - corr, 0.0, 2.0)


def ward_best_k(D, kmax):
    """Ward + 轮廓扫 K。返回 (基准标签dict, 轮廓表, K*)。"""
    Z = linkage(squareform(D, checks=False), method="ward")
    sils = []
    labels = {}
    for k in range(2, min(kmax, D.shape[0] - 1) + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        s = silhouette_score(D, lab, metric="precomputed")
        sils.append({"k": k, "silhouette": float(s)})
        labels[k] = lab
    sil_df = pd.DataFrame(sils)
    k_star = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    return labels, sil_df, k_star


def subsample_stability(X, base_lab, k, n_sub, sub_frac, seed):
    """子抽样稳定性: 抽 sub_frac 题 → Ward(k) → 与基准划分（同子样本）ARI。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = max(50, int(n * sub_frac))
    aris = []
    for _ in range(n_sub):
        idx = rng.choice(n, size=m, replace=False)
        D_sub = dist_1_minus_corr(X[idx])
        Z = linkage(squareform(D_sub, checks=False), method="ward")
        lab = fcluster(Z, t=k, criterion="maxclust")
        aris.append(adjusted_rand_score(base_lab[idx], lab))
    return float(np.mean(aris)), float(np.std(aris))


def double_center_arr(X, n_iter=3):
    """双重中心化（numpy 版，preprocess.py 同逻辑）。"""
    mask = ~np.isnan(X)
    x = np.where(mask, X, 0.0)
    for _ in range(n_iter):
        cnt = mask.sum(axis=1, keepdims=True)
        row_m = x.sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
        xt = np.where(mask, x - row_m, 0.0)
        cnt_c = mask.sum(axis=0, keepdims=True)
        col_m = xt.sum(axis=0, keepdims=True) / np.maximum(cnt_c, 1)
        grand = xt.sum() / max(mask.sum(), 1)
        x = np.where(mask, x - row_m - col_m + grand, 0.0)
    return np.where(mask, x, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old", "schwartz"], default="agree")
    ap.add_argument("--channels", default="orig,I,P", help="逗号分隔: orig,D,I,P")
    ap.add_argument("--n-sub", type=int, default=300, help="子抽样稳定性次数")
    ap.add_argument("--sub-frac", type=float, default=0.7)
    ap.add_argument("--kmax", type=int, default=15)
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    channels = [c.strip() for c in args.channels.split(",")]
    active = pd.read_csv(res_dir / "active_items.csv")
    scored = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0).loc[active["rule_id"]]

    result = {"source": args.source, "n_items": int(len(active)), "channels": {}, "cross_channel_ari": {},
              "difficulty_check": {}, "subtopic": {}}
    base_labels = {}

    for ch in channels:
        print(f"\n----- 通道 {ch} -----")
        cent = pd.read_csv(res_dir / f"centered_{ch}.csv", index_col=0).loc[active["rule_id"]]
        X = np.nan_to_num(cent.values.astype(float))  # NaN=无交互信息=0

        D = dist_1_minus_corr(X)
        labels, sil_df, k_star = ward_best_k(D, args.kmax)
        lab_df = pd.DataFrame({"rule_id": cent.index, f"cluster_k{k_star}": labels[k_star]})
        lab_df.to_csv(res_dir / f"step2_labels_{ch}.csv", index=False, encoding="utf-8-sig")
        sil_df.to_csv(res_dir / f"step2_silhouette_{ch}.csv", index=False, encoding="utf-8-sig")
        base_labels[ch] = labels[k_star]

        ari_m, ari_sd = subsample_stability(X, labels[k_star], k_star, args.n_sub, args.sub_frac, config.SEED)
        result["channels"][ch] = {
            "k_star": k_star,
            "silhouette_at_k": float(sil_df.loc[sil_df["k"] == k_star, "silhouette"].iloc[0]),
            "subsample_ari_mean": ari_m, "subsample_ari_sd": ari_sd,
        }
        print(f"  K* = {k_star} (轮廓 {result['channels'][ch]['silhouette_at_k']:.3f})，"
              f"子抽样 ARI = {ari_m:.3f} ± {ari_sd:.3f} (n_sub={args.n_sub})")

        # 难度因子诊断（对中心化后矩阵：若 PC1 仍与题目均分高相关 → 中心化未扣净难度残余）
        pc1_load = np.linalg.svd(X, full_matrices=False)[2][0]
        pc1_score = (X - X.mean(axis=0)) @ pc1_load
        item_mean = np.nanmean(scored.loc[cent.index].values.astype(float), axis=1)
        r_diff = float(np.corrcoef(pc1_score, item_mean)[0, 1])
        evr = float(np.var(pc1_score) / np.sum(np.var(X, axis=0)))
        result["difficulty_check"][ch] = {"pc1_itemmean_r": r_diff, "pc1_evr": evr,
                                          "warning": bool(abs(r_diff) > 0.3 and evr > 0.30)}
        print(f"  难度因子(中心化后残余): PC1×题目均分 r={r_diff:+.3f}, PC1解释方差={evr*100:.1f}%"
              + ("  ⚠ 中心化未扣净难度因子，聚类或仍被支持度基线影响" if abs(r_diff) > 0.3 and evr > 0.30 else ""))

    # 通道间稳健性（两两 ARI）
    chs = list(base_labels.keys())
    for i in range(len(chs)):
        for j in range(i + 1, len(chs)):
            ari = float(adjusted_rand_score(base_labels[chs[i]], base_labels[chs[j]]))
            result["cross_channel_ari"][f"{chs[i]}_vs_{chs[j]}"] = ari
            print(f"通道间 ARI[{chs[i]} vs {chs[j]}] = {ari:.3f}")

    # ---------- 对半拆列复制（样本外复制检验，2026-10-08 补强）----------
    # on 半边列聚类 vs off 半边列聚类：两个聚类过程互不见对方半边 → ARI 是真样本外一致性
    if "orig" in channels:
        cent_orig = pd.read_csv(res_dir / "centered_orig.csv", index_col=0).loc[active["rule_id"]]
        base_series = pd.Series(base_labels["orig"], index=cent_orig.index)
        scored_act = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0).loc[cent_orig.index]
        halves = {"on_half": [c for c in scored_act.columns if c.endswith("_on")],
                  "off_half": [c for c in scored_act.columns if c.endswith("_off")]}
        hl, half_info = {}, {}
        for hname, hcols in halves.items():
            Xh = scored_act[hcols].values.astype(float)
            # 半边内最少有效列（容忍 sensenova_off 这类缺口，要求 ≥ 半边列数−3）
            keep_h = np.sum(~np.isnan(Xh), axis=1) >= max(6, len(hcols) - 3)
            Xc = np.nan_to_num(double_center_arr(Xh[keep_h]))
            labs_all, sil_h, k_h = ward_best_k(dist_1_minus_corr(Xc), args.kmax)
            hl[hname] = pd.Series(labs_all[k_h], index=scored_act.index[keep_h])
            half_info[hname] = {"k": int(k_h), "n_items": int(keep_h.sum()),
                                "best_silhouette": float(sil_h["silhouette"].max())}
        common = hl["on_half"].index.intersection(hl["off_half"].index)
        ari_on_off = float(adjusted_rand_score(hl["on_half"][common], hl["off_half"][common]))
        ari_on_full = float(adjusted_rand_score(hl["on_half"], base_series[hl["on_half"].index]))
        ari_off_full = float(adjusted_rand_score(hl["off_half"], base_series[hl["off_half"].index]))
        result["split_half"] = {"on_half": half_info["on_half"], "off_half": half_info["off_half"],
                                "n_common_items": int(len(common)),
                                "ari_on_vs_off": ari_on_off,
                                "ari_on_vs_full": ari_on_full, "ari_off_vs_full": ari_off_full,
                                "verdict": ("✅ 通过: 样本外复制 ARI≥0.6，经验类划分跨模式稳定"
                                            if ari_on_off >= 0.6 else
                                            f"⚠ 未通过: 样本外复制 ARI={ari_on_off:.3f}<0.6，"
                                            f"经验类划分依赖全列信息，不足以作为新维度")}
        print(f"\n对半拆列复制: on半边 K={half_info['on_half']['k']}(n={half_info['on_half']['n_items']}), "
              f"off半边 K={half_info['off_half']['k']}(n={half_info['off_half']['n_items']}), "
              f"共同题 {len(common)}")
        print(f"  ARI[on vs off] = {ari_on_off:.3f}（样本外复制一致性，≥0.6 通过）")
        print(f"  ARI[on vs 全列] = {ari_on_full:.3f}, ARI[off vs 全列] = {ari_off_full:.3f}")
        print(f"  {result['split_half']['verdict']}")

    # ---------- 子主题层交叉验证（orig 主通道）----------
    if "orig" in channels and "derived_values" in active.columns:
        sub_map = active.set_index("rule_id")["derived_values"].astype(str).str.split("|")
        cent = pd.read_csv(res_dir / "centered_orig.csv", index_col=0).loc[active["rule_id"]]
        sub_rows, sub_ids, sub_rids = [], [], []
        for rid in cent.index:
            subs_of_item = sub_map.get(rid, [])
            for s in subs_of_item:
                if s:
                    sub_ids.append(s)
                    sub_rids.append(rid)
                    sub_rows.append(cent.loc[rid].values.astype(float))
        sub_df = pd.DataFrame(sub_rows, index=sub_ids, columns=cent.columns)
        sub_mean = sub_df.groupby(level=0).mean()  # 50子主题 × 22列
        Xm = double_center_arr(sub_mean.values)
        Xm = np.nan_to_num(Xm)
        Dm = dist_1_minus_corr(Xm)
        labels_all_m, sil_m, k_m = ward_best_k(Dm, args.kmax)
        labels_m = labels_all_m[k_m]

        # RV 系数法（法C）：行对齐 = 22 个模型观测行，子主题题目为变量列。
        # RV 合法构造：W_s = 中心化后 (题目×22)^T @ (题目×22) = 22×22 模型协变异矩阵，
        # RV(i,j) = tr(W_i W_j) / sqrt(tr(W_i²) tr(W_j²)) —— 比较两子主题的模型间协变异结构。
        subs = sub_df.index.unique().tolist()
        n_sub_t = len(subs)
        W = {}
        for s in subs:
            A = np.nan_to_num(sub_df.loc[[s]].values.astype(float))  # n_s × 22
            Ac = A - A.mean(axis=0)
            W[s] = Ac.T @ Ac
        rv_mat = np.eye(n_sub_t)
        for a in range(n_sub_t):
            for b in range(a + 1, n_sub_t):
                Wi, Wj = W[subs[a]], W[subs[b]]
                num = np.trace(Wi @ Wj)
                den = np.sqrt(np.trace(Wi @ Wi) * np.trace(Wj @ Wj))
                rv_mat[a, b] = rv_mat[b, a] = float(num / den) if den > 1e-12 else 0.0
        rv_dist = np.clip(1.0 - rv_mat, 0.0, 2.0)
        np.fill_diagonal(rv_dist, 0.0)
        labels_all_rv, sil_rv, k_rv = ward_best_k(rv_dist, args.kmax)
        labels_rv = labels_all_rv[k_rv]

        ari_dual = float(adjusted_rand_score(labels_m, labels_rv))

        # 题目层划分 → 子主题纯度（法A vs 题目层 orig K*）
        base_lab = dict(zip(cent.index, base_labels["orig"]))
        pair_df = pd.DataFrame({"subtopic": sub_ids, "rule_id": sub_rids})
        purity_rows = []
        for s in subs:
            rids = pair_df.loc[pair_df["subtopic"] == s, "rule_id"].tolist()
            labs = pd.Series([base_lab.get(r, -1) for r in rids])
            vc = labs[labs >= 0].value_counts(normalize=True)
            purity_rows.append({"subtopic": s, "n_items": int(len(labs)),
                                "majority_cluster": int(vc.index[0]) if len(vc) else -1,
                                "purity": float(vc.iloc[0]) if len(vc) else np.nan})
        pur = pd.DataFrame(purity_rows)

        pd.DataFrame({"subtopic": subs, f"cluster_k{k_m}": labels_m}).to_csv(
            res_dir / "step2_subtopic_labels.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"subtopic": subs, f"cluster_k{k_rv}": labels_rv}).to_csv(
            res_dir / "step2_subtopic_rv_labels.csv", index=False, encoding="utf-8-sig")
        pur.to_csv(res_dir / "step2_subtopic_purity.csv", index=False, encoding="utf-8-sig")

        result["subtopic"] = {
            "n_subtopics": int(n_sub_t),
            "k_pearson": int(k_m), "k_rv": int(k_rv),
            "dual_method_ari": ari_dual,
            "purity_mean": float(pur["purity"].mean()),
        }
        print(f"\n子主题层: K(Pearson)={k_m}, K(RV)={k_rv}, 双法ARI={ari_dual:.3f}, "
              f"题目层→子主题纯度均值={pur['purity'].mean():.3f}")

    with open(res_dir / "step2_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已存: {res_dir / 'step2_result.json'}")


if __name__ == "__main__":
    main()
