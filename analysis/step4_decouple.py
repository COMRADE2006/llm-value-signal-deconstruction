# -*- coding: utf-8 -*-
"""
第四步：解耦分析（总方案v3 §3解剖 + §4解耦，2026-10-09，v3全面替代v2）。

四模块:
  1. variance_budget  方差预算审计: 总方差 = 题目主效应(支持度基线) + 模型主效应(作答风格)
                       + 交互残差(分化信息)；+ 模型级中点压缩指标（Neutral Mask 2606.09735 对接）
  2. channel_locate   生成严格 IR-Tree D_strict/M/E 敏感性通道（Böckenholt 2012; Zhang & Wang 2020）
                       + 两套通道（D/I/P 主, D_strict/M/E 敏感性）EVR1 + PC1 住址相关
  3. stat_decouple    统计解耦: 剥前 k∈{1,2,3} 个主成分 → 残差重聚类（轮廓/子抽样稳定性）
                       + 对半拆列复制（半边各自独立 中心化→剥PC1→聚类, 样本外 ARI）
                       + 轮廓置换检验（行内置换破坏题目间相关, 仅 k=1）
  4. score_decouple   计分解耦: 7 通道聚类质量对比 + 模型间分化（中心化列相关均值）

预注册结局判定（防 HARKing, v3 §5）:
  A 结构恢复  : orig 剥1 对半 ARI ≥ 0.6  → 12维结构被一维基线遮蔽
  B 部分恢复  : 0.4 ≤ ARI < 0.6，或仅强度/中点通道恢复 → 计分处方
  C 不恢复    : 全通道 ARI < 0.4        → 低维坐实, 换题型必要性

输入: data/results/{source}/scored_matrix.csv, active_items.csv, centered_{orig,D,I,P}.csv
输出: data/results/{source}/decouple_result.json
      + decouple_variance_budget.csv, decouple_midpoint.csv
      + chan_Dstrict/M/E.csv, centered_Dstrict/M/E.csv
      + decouple_channel_evr.csv, decouple_residual_labels.csv, decouple_channel_compare.csv

用法: python analysis/step4_decouple.py --source agree|old [--kmax 15] [--n-sub 300] [--n-perm 50]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import (dist_1_minus_corr, ward_best_k,
                              subsample_stability, double_center_arr)

# 主分析通道（preprocess.py 已实现）+ 敏感性通道（本脚本生成）
MAIN_CHANNELS = ["orig", "D", "I", "P"]
STRICT_CHANNELS = ["Dstrict", "M", "E"]
ALL_CHANNELS = MAIN_CHANNELS + STRICT_CHANNELS


# ---------------------------------------------------------------- 通道生成
def channel_matrix(scored, ch):
    """从计分后矩阵生成指定通道矩阵（DataFrame，同 index/columns）。

    orig    原始计分（强度+方向）
    D       方向三值: X*>=4→+1, ==3→0, <=2→−1（preprocess.py 同口径）
    I       强度: |X*−3|/2 ∈ {0, 0.5, 1}（混合中点/极端信息）
    P       乘积: D × I
    Dstrict 严格方向: X*>=4→+1, <=2→−1, ==3→NaN（中点无方向）
    M       中点倾向: X*==3→1, 否则 0（NaN 保持）
    E       极端倾向: |X*−3|==2→1, ==1→0, ==3 或 NaN→NaN（条件于离开中点）
    """
    v = scored.values.astype(float)
    if ch == "orig":
        out = v.copy()
    elif ch == "D":
        out = np.where(np.isnan(v), np.nan,
                       np.where(v >= 4, 1.0, np.where(v == 3, 0.0, -1.0)))
    elif ch == "I":
        out = np.abs(v - 3.0) / 2.0
    elif ch == "P":
        d = np.where(np.isnan(v), np.nan,
                     np.where(v >= 4, 1.0, np.where(v == 3, 0.0, -1.0)))
        out = d * (np.abs(v - 3.0) / 2.0)
    elif ch == "Dstrict":
        out = np.full_like(v, np.nan)
        out[~np.isnan(v) & (v >= 4)] = 1.0
        out[~np.isnan(v) & (v <= 2)] = -1.0
    elif ch == "M":
        out = np.full_like(v, np.nan)
        mv = ~np.isnan(v)
        out[mv] = (v[mv] == 3).astype(float)
    elif ch == "E":
        out = np.full_like(v, np.nan)
        ev = ~np.isnan(v) & (v != 3)
        out[ev] = (np.abs(v[ev] - 3) == 2).astype(float)
    else:
        raise ValueError(f"未知通道: {ch}")
    return pd.DataFrame(out, index=scored.index, columns=scored.columns)


def centered_channel(scored, ch, min_valid_ch=6):
    """通道矩阵 → 行过滤 → 双重中心化 → NaN 填 0。返回 (X, valid_ids)。

    行过滤: 该通道有效值 ≥ min_valid_ch。Dstrict/E 的中点格为 NaN，
    全中点题（所有模型都选 3）在此通道无信息，必须剔除，否则全 0 行
    会在距离矩阵对角线产生非 0 值并污染聚类。"""
    ch_df = channel_matrix(scored, ch)
    nvalid = ch_df.notna().sum(axis=1)
    keep = nvalid >= min_valid_ch
    X = np.nan_to_num(double_center_arr(ch_df.loc[keep].values.astype(float)))
    return X, ch_df.index[keep]


def safe_dist(X):
    """1−Pearson 距离 + 对角线归零（sklearn 要求预计算距离对角=0；
    全 0 行经 norm 保护后自相关=0 → 对角=1，需显式归零）。"""
    D = dist_1_minus_corr(X)
    np.fill_diagonal(D, 0.0)
    return D


def row_center(X):
    """行中心化（题向量在模型空间去均值；剥 PC1 前的预处理）。"""
    return X - X.mean(axis=1, keepdims=True)


def svd_top(Xc):
    """行中心化矩阵的 SVD，返回 (Vt, 奇异值数组)。"""
    return np.linalg.svd(Xc, full_matrices=False)


# ---------------------------------------------------------------- 模块1: 方差预算审计
def impute_iterative(df, n_iter=5):
    """迭代行/列均值填充（仅用于方差分解，不进聚类管线）。返回(填充矩阵, 填充率)。"""
    x = df.values.astype(float).copy()
    mask = np.isnan(x)
    if mask.any():
        col_m = np.nanmean(x, axis=0)
        for j in range(x.shape[1]):
            x[mask[:, j], j] = col_m[j]
        for _ in range(n_iter):
            row_m = x.mean(axis=1, keepdims=True)
            x = np.where(mask, row_m, x)
            col_m = x.mean(axis=0, keepdims=True)
            x = np.where(mask, col_m, x)
    return x, float(mask.mean())


def variance_budget(scored):
    """两向方差分解。对任意完整矩阵为代数恒等式:
    SS_total = SS_item + SS_model + SS_interaction（主效应均值=0 → 交叉项为0）。"""
    X, fill_rate = impute_iterative(scored)
    grand = X.mean()
    a = X.mean(axis=1) - grand  # 题目主效应（支持度基线）
    b = X.mean(axis=0) - grand  # 模型主效应（作答风格）
    ss_total = float(((X - grand) ** 2).sum())
    ss_item = float(X.shape[1] * (a ** 2).sum())
    ss_model = float(X.shape[0] * (b ** 2).sum())
    ss_inter = ss_total - ss_item - ss_model
    return {
        "n_items": int(X.shape[0]), "n_cols": int(X.shape[1]),
        "fill_rate": fill_rate,
        "ss_item_pct": ss_item / ss_total * 100,
        "ss_model_pct": ss_model / ss_total * 100,
        "ss_interaction_pct": ss_inter / ss_total * 100,
    }


def midpoint_compression(scored):
    """模型级中点压缩指标（Neutral Mask 对接）。计分 X*=6−X 不改变 3 与 |X−3|。"""
    v = scored.values.astype(float)
    rows = []
    for j, col in enumerate(scored.columns):
        x = v[:, j]
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            continue
        rows.append({
            "model_col": col,
            "n_valid": int(len(valid)),
            "midpoint_rate": float(np.mean(valid == 3)),
            "sd": float(np.std(valid, ddof=1)),
            "mean": float(np.mean(valid)),
        })
    allv = v[~np.isnan(v)]
    ref = {"global_midpoint_rate": float(np.mean(allv == 3)),
           "global_sd": float(np.std(allv, ddof=1))}
    return pd.DataFrame(rows), ref


# ---------------------------------------------------------------- 模块2: 通道定位
def channel_locate(res_dir, scored):
    """两套通道 EVR1 + 题目间平均相关 + PC1 住址相关；敏感性通道矩阵落盘。

    各通道独立做行过滤（Dstrict/E 的全中点题剔除），住址相关在两通道
    有效题交集上计算。"""
    pc1_store, rows = {}, []
    for ch in ALL_CHANNELS:
        X, ids = centered_channel(scored, ch)
        Xc = row_center(X)
        _, S, Vt = svd_top(Xc)
        evr1 = float(S[0] ** 2 / np.sum(S ** 2))
        pc1_store[ch] = pd.Series(Xc @ Vt[0], index=ids)
        # 题目间平均相关（非对角均值）：1 − mean(D)
        D = safe_dist(X)
        iu = np.triu_indices(D.shape[0], 1)
        mean_r = float(1.0 - D[iu].mean())
        rows.append({"channel": ch, "n_valid_items": int(len(ids)),
                     "evr1": evr1, "mean_interitem_r": mean_r})
    # PC1 住址: orig 的 PC1 得分与各通道 PC1 得分在有效题交集上的相关
    for r in rows:
        ch = r["channel"]
        if ch == "orig":
            r["pc1_r_vs_orig"] = 1.0
        else:
            common = pc1_store["orig"].index.intersection(pc1_store[ch].index)
            r["pc1_r_vs_orig"] = float(np.corrcoef(pc1_store["orig"][common],
                                                   pc1_store[ch][common])[0, 1])
    evr_df = pd.DataFrame(rows)
    evr_df.to_csv(res_dir / "decouple_channel_evr.csv", index=False, encoding="utf-8-sig")

    # 敏感性通道矩阵落盘（作为 preprocess 输出的补充，供复查/后续复用）
    for ch in STRICT_CHANNELS:
        ch_df = channel_matrix(scored, ch)
        ch_df.to_csv(res_dir / f"chan_{ch}.csv", encoding="utf-8-sig")
        pd.DataFrame(np.nan_to_num(double_center_arr(ch_df.values.astype(float))),
                     index=ch_df.index, columns=ch_df.columns
                     ).to_csv(res_dir / f"centered_{ch}.csv", encoding="utf-8-sig")
    return evr_df


# ---------------------------------------------------------------- 对半拆列（统计解耦版/纯通道版）
def _half_matrix(scored_act, hcols):
    """半边原始矩阵 + 覆盖过滤（容忍 sensenova_off 类缺口，同 step2 口径）。"""
    Xh = scored_act[hcols].values.astype(float)
    keep_h = np.sum(~np.isnan(Xh), axis=1) >= max(6, len(hcols) - 3)
    return Xh, keep_h


def half_split_ari(scored_act, kmax, k_strip=0, seed=None):
    """对半拆列复制。k_strip=0 → 纯通道聚类；k_strip>=1 → 半边各自独立
    中心化→剥前k_strip个主成分→聚类（统计解耦的样本外验收）。"""
    halves = {"on_half": [c for c in scored_act.columns if c.endswith("_on")],
              "off_half": [c for c in scored_act.columns if c.endswith("_off")]}
    hl, info = {}, {}
    for hname, hcols in halves.items():
        Xh, keep_h = _half_matrix(scored_act, hcols)
        Xc = row_center(np.nan_to_num(double_center_arr(Xh[keep_h])))
        if k_strip >= 1:
            _, _, Vt = svd_top(Xc)
            Vk = Vt[:k_strip]
            Xc = Xc - (Xc @ Vk.T) @ Vk  # 剥半边自己的前 k 个主方向
        labs, sil, k_h = ward_best_k(safe_dist(Xc), kmax)
        hl[hname] = pd.Series(labs[k_h], index=scored_act.index[keep_h])
        info[hname] = {"k": int(k_h), "n_items": int(keep_h.sum()),
                       "best_silhouette": float(sil["silhouette"].max())}
    common = hl["on_half"].index.intersection(hl["off_half"].index)
    ari = float(adjusted_rand_score(hl["on_half"][common], hl["off_half"][common]))
    return ari, info, int(len(common))


# ---------------------------------------------------------------- 模块3: 统计解耦
def stat_decouple(ch, scored, res_dir, kmax, n_sub, n_perm, seed):
    """剥前 k∈{1,2,3} 个主成分 → 残差重聚类 + 对半复制 + 置换检验(仅k=1)。

    输入通道矩阵与对半拆列均从同一通道的计分矩阵出发，保证口径一致。"""
    cent = pd.read_csv(res_dir / f"centered_{ch}.csv", index_col=0).loc[scored.index]
    X = np.nan_to_num(cent.values.astype(float))
    Xc = row_center(X)
    _, S, Vt = svd_top(Xc)
    out = {"evr1_before": float(S[0] ** 2 / np.sum(S ** 2)), "strips": {}}
    ch_raw = channel_matrix(scored, ch)  # 对半拆列用同通道原始矩阵
    for k_strip in (1, 2, 3):
        Vk = Vt[:k_strip]
        Xr = Xc - (Xc @ Vk.T) @ Vk
        D = safe_dist(Xr)
        labels, sil_df, k_star = ward_best_k(D, kmax)
        sil_k = float(sil_df.loc[sil_df["k"] == k_star, "silhouette"].iloc[0])
        ari_m, ari_sd = subsample_stability(Xr, labels[k_star], k_star, n_sub, 0.7, seed)
        entry = {
            "k_star": int(k_star), "silhouette": sil_k,
            "subsample_ari_mean": float(ari_m), "subsample_ari_sd": float(ari_sd),
        }
        # 残差里剩余的最大方向强度（剥后还剩多强的一维结构）
        _, S_r, _ = svd_top(row_center(Xr))
        entry["evr1_after"] = float(S_r[0] ** 2 / np.sum(S_r ** 2))
        # 对半拆列复制（半边各自独立剥 k 个）
        ari_half, info_half, n_common = half_split_ari(ch_raw, kmax, k_strip=k_strip)
        entry["split_half_ari"] = ari_half
        entry["split_half"] = info_half
        entry["split_half_n_common"] = n_common
        if k_strip == 1:
            # 置换检验: 行内置换破坏题目间相关（保留每题边缘分布）→ 轮廓零分布
            rng = np.random.default_rng(seed)
            null_sils = []
            for _ in range(n_perm):
                Xp = np.array([rng.permutation(r) for r in Xr])
                _, silp, _ = ward_best_k(safe_dist(Xp), kmax)
                null_sils.append(float(silp["silhouette"].max()))
            entry["perm_null_sil_mean"] = float(np.mean(null_sils))
            entry["perm_p"] = float((np.sum(np.array(null_sils) >= sil_k) + 1) / (n_perm + 1))
            # 残差聚类标签落盘（k=1 主分析，仅 orig 通道）
            if ch == "orig":
                pd.DataFrame({"rule_id": cent.index,
                              f"residual_cluster_k{k_star}": labels[k_star]}
                             ).to_csv(res_dir / "decouple_residual_labels.csv",
                                      index=False, encoding="utf-8-sig")
        out["strips"][f"strip{k_strip}"] = entry
        print(f"    剥{k_strip}个: K*={k_star}, 轮廓={sil_k:.3f}, "
              f"子抽样ARI={ari_m:.3f}±{ari_sd:.3f}, 对半ARI={ari_half:.3f}, "
              f"剥后EVR1={entry['evr1_after']*100:.1f}%"
              + (f", 置换p={entry['perm_p']:.3f}" if k_strip == 1 else ""))
    return out


# ---------------------------------------------------------------- 模块4: 计分解耦
def score_decouple(scored, kmax, n_sub, seed):
    """7 通道聚类质量（K*/轮廓/子抽样/对半ARI）+ 模型间分化（中心化列相关均值）。

    各通道独立做行过滤（Dstrict/E 剔除全中点等无信息题），n_valid 记录在结果中。"""
    rows = []
    for ch in ALL_CHANNELS:
        X, valid_ids = centered_channel(scored, ch)
        D = safe_dist(X)
        labels, sil_df, k_star = ward_best_k(D, kmax)
        sil_k = float(sil_df.loc[sil_df["k"] == k_star, "silhouette"].iloc[0])
        ari_m, ari_sd = subsample_stability(X, labels[k_star], k_star, n_sub, 0.7, seed)
        # 模型间分化: 中心化矩阵列两两相关均值（越低=模型在该通道分歧越大）
        C = np.corrcoef(X.T)
        iu = np.triu_indices(C.shape[0], 1)
        vals = C[iu]
        vals = vals[~np.isnan(vals)]
        mean_col_r = float(vals.mean()) if len(vals) else np.nan
        # 该通道原始计分矩阵的对半拆列（纯通道聚类，不剥PC1）
        ch_raw = channel_matrix(scored, ch)
        ari_half, _, _ = half_split_ari(ch_raw, kmax, k_strip=0)
        rows.append({
            "channel": ch, "n_valid_items": int(len(valid_ids)),
            "k_star": int(k_star), "silhouette": sil_k,
            "subsample_ari_mean": float(ari_m), "subsample_ari_sd": float(ari_sd),
            "split_half_ari": ari_half,
            "mean_col_corr_models": mean_col_r,
        })
        print(f"    {ch:8s}: n={len(valid_ids)}, K*={k_star}, 轮廓={sil_k:.3f}, "
              f"子抽样ARI={ari_m:.3f}, 对半ARI={ari_half:.3f}, 模型间平均相关={mean_col_r:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old", "schwartz"], default="agree")
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--n-sub", type=int, default=300, help="子抽样稳定性次数")
    ap.add_argument("--n-perm", type=int, default=50, help="置换检验次数（仅剥1个时）")
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    active = pd.read_csv(res_dir / "active_items.csv")
    scored = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0).loc[active["rule_id"]]
    n_items = len(scored)
    print("=" * 70)
    print(f"[source={args.source}] 解耦分析（v3 §3解剖 + §4解耦），活跃题 {n_items}，"
          f"列 {scored.shape[1]}")

    # ---------- 模块1: 方差预算审计 ----------
    print("\n----- 模块1: 方差预算审计 -----")
    vb = variance_budget(scored)
    mid_df, mid_ref = midpoint_compression(scored)
    vb_row = pd.DataFrame([vb])
    vb_row.to_csv(res_dir / "decouple_variance_budget.csv", index=False, encoding="utf-8-sig")
    mid_df.to_csv(res_dir / "decouple_midpoint.csv", index=False, encoding="utf-8-sig")
    print(f"  题目主效应(支持度基线) = {vb['ss_item_pct']:.1f}%")
    print(f"  模型主效应(作答风格)   = {vb['ss_model_pct']:.1f}%")
    print(f"  交互残差(分化信息)     = {vb['ss_interaction_pct']:.1f}%   (NaN填充率 {vb['fill_rate']*100:.1f}%)")
    print(f"  中点压缩(Neutral Mask): 全局中点率 {mid_ref['global_midpoint_rate']*100:.1f}%, "
          f"全局SD {mid_ref['global_sd']:.3f}; "
          f"模型中点率范围 [{mid_df['midpoint_rate'].min()*100:.1f}%, {mid_df['midpoint_rate'].max()*100:.1f}%], "
          f"模型SD范围 [{mid_df['sd'].min():.3f}, {mid_df['sd'].max():.3f}]")

    # ---------- 模块2: 通道定位 ----------
    print("\n----- 模块2: 通道定位（D/I/P 主 + Dstrict/M/E 敏感性）-----")
    evr_df = channel_locate(res_dir, scored)
    for _, r in evr_df.iterrows():
        print(f"  {r['channel']:8s}: EVR1={r['evr1']*100:5.1f}%, "
              f"题目间平均r={r['mean_interitem_r']:+.3f}, PC1住址r={r['pc1_r_vs_orig']:+.3f}")

    # ---------- 模块3: 统计解耦（主: orig; 对照: D）----------
    stat_out = {}
    for ch in ["orig", "D"]:
        print(f"\n----- 模块3: 统计解耦（通道 {ch}，剥PC1后残差重聚类）-----")
        stat_out[ch] = stat_decouple(ch, scored, res_dir, args.kmax, args.n_sub,
                                     args.n_perm, config.SEED)
        print(f"  （剥前 EVR1 = {stat_out[ch]['evr1_before']*100:.1f}%）")

    # ---------- 模块4: 计分解耦 ----------
    print("\n----- 模块4: 计分解耦（7通道聚类质量 + 模型间分化）-----")
    compare_df = score_decouple(scored, args.kmax, args.n_sub, config.SEED)
    compare_df.to_csv(res_dir / "decouple_channel_compare.csv", index=False, encoding="utf-8-sig")

    # ---------- 汇总 + 预注册结局判定 ----------
    ari_main = stat_out["orig"]["strips"]["strip1"]["split_half_ari"]
    valid_sh = compare_df.dropna(subset=["split_half_ari"])
    best_channel = (valid_sh.loc[valid_sh["split_half_ari"].idxmax()]
                    if len(valid_sh) else compare_df.iloc[0])
    best_ch_ari = float(best_channel["split_half_ari"])
    if ari_main >= 0.6:
        verdict = ("✅ 结局A 结构恢复: orig剥1对半ARI≥0.6 → 12维结构被一维基线遮蔽，"
                   "解耦后可见（诊断+解耦方法贡献）")
    elif ari_main >= 0.4 or best_ch_ari >= 0.6:
        verdict = (f"🔶 结局B 部分恢复: orig剥1对半ARI={ari_main:.3f}, "
                   f"最佳通道({best_channel['channel']})对半ARI={best_ch_ari:.3f} "
                   "→ 方向通道坍缩、强度/中点通道分化，计分处方为规范贡献")
    else:
        verdict = (f"❌ 结局C 不恢复: orig剥1对半ARI={ari_main:.3f}, "
                   f"最佳通道({best_channel['channel']})对半ARI={best_ch_ari:.3f} "
                   "→ 低维坐实，换题型必要性论证（对接题型-可测性规律）")

    result = {
        "source": args.source, "n_items": int(n_items),
        "variance_budget": vb, "midpoint_ref": mid_ref,
        "channel_evr": evr_df.to_dict(orient="records"),
        "stat_decouple": stat_out,
        "score_decouple": compare_df.to_dict(orient="records"),
        "verdict": verdict,
    }
    with open(res_dir / "decouple_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)

    print("\n" + "=" * 70)
    print("解耦前后对比（orig 通道）:")
    s2 = res_dir / "step2_result.json"
    old_sh_ari = None
    if s2.exists():
        with open(s2, encoding="utf-8") as f:
            s2j = json.load(f)
        old = s2j["channels"]["orig"]
        old_sh_ari = s2j.get("split_half", {}).get("ari_on_vs_off")
        print(f"  解耦前:  K*={old['k_star']}, 轮廓={old['silhouette_at_k']:.3f}, "
              f"子抽样ARI={old['subsample_ari_mean']:.3f}"
              + (f", 对半ARI={old_sh_ari:.3f}" if old_sh_ari is not None else ""))
    st1 = stat_out["orig"]["strips"]["strip1"]
    print(f"  剥1后:   K*={st1['k_star']}, 轮廓={st1['silhouette']:.3f}, "
          f"子抽样ARI={st1['subsample_ari_mean']:.3f}, 对半ARI={st1['split_half_ari']:.3f}"
          + (f", 置换p={st1['perm_p']:.3f}" if "perm_p" in st1 else ""))
    print(f"\n预注册结局判定: {verdict}")
    print(f"结果已存: {res_dir / 'decouple_result.json'}")


if __name__ == "__main__":
    main()
