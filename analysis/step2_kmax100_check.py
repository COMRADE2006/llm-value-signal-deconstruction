# -*- coding: utf-8 -*-
"""K上限敏感性核查：聚类数上限从15放宽到100，核查第3章表2全部数字是否依赖2~15的人为设置。
只读既有数据、不覆盖任何结果文件；输出 step2_kmax100_check.json。
若K*仍为15且各指标不变 → 论文只需改"扫描范围"表述；若K*改变 → 表2数字全部更新。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import dist_1_minus_corr, double_center_arr, ward_best_k, subsample_stability

KMAX = 100
res = Path(config.DATA_DIR) / "results" / "agree"
active = pd.read_csv(res / "active_items.csv")
scored = pd.read_csv(res / "scored_matrix.csv", index_col=0).loc[active["rule_id"]]
out = {"kmax": KMAX, "channels": {}, "orig_detail": {}, "half_split": {}, "subtopic": {}}
t0 = time.time()

# ---------- 1) 三通道自由选优（K=2~100） ----------
base = {}
for ch in ("orig", "I", "P"):
    cent = pd.read_csv(res / f"centered_{ch}.csv", index_col=0).loc[active["rule_id"]]
    X = np.nan_to_num(cent.values.astype(float))
    labels, sil, k_star = ward_best_k(dist_1_minus_corr(X), KMAX)
    sil.to_csv(res / f"check_sil_k100_{ch}.csv", index=False)
    base[ch] = (labels, sil, k_star)
    out["channels"][ch] = {
        "k_star": int(k_star),
        "sil_at_kstar": float(sil["silhouette"].max()),
        "sil_at_12": float(sil.loc[sil["k"] == 12, "silhouette"].iloc[0]),
        "sil_at_15": float(sil.loc[sil["k"] == 15, "silhouette"].iloc[0]),
        "top5": sil.nlargest(5, "silhouette")[["k", "silhouette"]].to_dict("records"),
    }
    c = out["channels"][ch]
    print(f"通道 {ch}: K*={c['k_star']} sil={c['sil_at_kstar']:.4f} "
          f"(K=12: {c['sil_at_12']:.4f}, K=15: {c['sil_at_15']:.4f})")
for a, b in (("orig", "I"), ("orig", "P"), ("I", "P")):
    v = float(adjusted_rand_score(base[a][0][base[a][2]], base[b][0][base[b][2]]))
    out["channels"][f"ari_{a}_vs_{b}"] = v
    print(f"ARI[{a} vs {b}] = {v:.3f}")

# ---------- 2) orig主通道：纯度与子抽样（自由K* 与 强设K=12 两套） ----------
labels, sil, k_star = base["orig"]
cent = pd.read_csv(res / "centered_orig.csv", index_col=0).loc[active["rule_id"]]
X = np.nan_to_num(cent.values.astype(float))
meta = active.set_index("rule_id").loc[cent.index]
core12 = meta["core_value"].values

# 子主题多标签对（复刻step2协议）
sub_map = meta["derived_values"].astype(str).str.split("|")
pair_rows = []
for rid, subs in zip(cent.index, sub_map):
    for s in subs:
        if s:
            pair_rows.append((s, rid))
pair_df = pd.DataFrame(pair_rows, columns=["subtopic", "rule_id"])
rid_to_pos = {r: i for i, r in enumerate(cent.index)}

def purity_subtopic(lab):
    vals = []
    for s in pair_df["subtopic"].unique():
        rids = pair_df.loc[pair_df["subtopic"] == s, "rule_id"]
        l = pd.Series([lab[rid_to_pos[r]] for r in rids])
        l = l[l >= 0]
        if len(l):
            vals.append(l.value_counts(normalize=True).iloc[0])
    return float(np.mean(vals))

def purity_core(lab):
    vals = []
    for k in pd.unique(core12):
        l = pd.Series(lab[core12 == k])
        l = l[l >= 0]
        if len(l):
            vals.append(l.value_counts(normalize=True).iloc[0])
    return float(np.mean(vals))

for tag, k_fix in (("free_k", k_star), ("forced_12", 12)):
    lab = labels[k_fix]
    ari_m, ari_sd = subsample_stability(X, lab, k_fix, 300, 0.7, config.SEED)
    out["orig_detail"][tag] = {
        "k": int(k_fix),
        "ari_vs_core12": float(adjusted_rand_score(lab, pd.factorize(core12)[0])),
        "purity_core12": purity_core(lab),
        "purity_subtopic": purity_subtopic(lab),
        "subsample_ari_mean": ari_m, "subsample_ari_sd": ari_sd,
    }
    d = out["orig_detail"][tag]
    print(f"orig[{tag}] K={d['k']}: ARI(vs12维)={d['ari_vs_core12']:.3f} "
          f"纯度(12维)={d['purity_core12']:.3f} 纯度(子主题)={d['purity_subtopic']:.3f} "
          f"子抽样ARI={ari_m:.3f}±{ari_sd:.3f}")

# ---------- 3) 对半拆列复制（on/off 半边各自 K=2~100） ----------
halves = {"on_half": [c for c in scored.columns if c.endswith("_on")],
          "off_half": [c for c in scored.columns if c.endswith("_off")]}
hl = {}
for hname, hcols in halves.items():
    Xh = scored[hcols].values.astype(float)
    keep = np.sum(~np.isnan(Xh), axis=1) >= max(6, len(hcols) - 3)
    Xc = np.nan_to_num(double_center_arr(Xh[keep]))
    labs_h, sil_h, k_h = ward_best_k(dist_1_minus_corr(Xc), KMAX)
    hl[hname] = pd.Series(labs_h[k_h], index=scored.index[keep])
    out["half_split"][hname] = {"k": int(k_h), "n_items": int(keep.sum()),
                                "best_silhouette": float(sil_h["silhouette"].max())}
common = hl["on_half"].index.intersection(hl["off_half"].index)
out["half_split"]["ari_on_vs_off"] = float(adjusted_rand_score(hl["on_half"][common], hl["off_half"][common]))
print(f"对半: on半K={out['half_split']['on_half']['k']}, off半K={out['half_split']['off_half']['k']}, "
      f"ARI={out['half_split']['ari_on_vs_off']:.3f}")

# ---------- 4) 子主题层双法（法A Pearson / 法C RV，K=2~100） ----------
sub_rows, sub_ids = [], []
for rid, subs in zip(cent.index, sub_map):
    for s in subs:
        if s:
            sub_ids.append(s)
            sub_rows.append(cent.loc[rid].values.astype(float))
sub_df = pd.DataFrame(sub_rows, index=sub_ids, columns=cent.columns)
sub_mean = sub_df.groupby(level=0).mean()
Xm = np.nan_to_num(double_center_arr(sub_mean.values))
labels_m, sil_m, k_m = ward_best_k(dist_1_minus_corr(Xm), KMAX)

subs = sub_df.index.unique().tolist()
W = {}
for s in subs:
    A = np.nan_to_num(sub_df.loc[[s]].values.astype(float))
    Ac = A - A.mean(axis=0)
    W[s] = Ac.T @ Ac
n_sub = len(subs)
rv = np.eye(n_sub)
for i in range(n_sub):
    for j in range(i + 1, n_sub):
        Wi, Wj = W[subs[i]], W[subs[j]]
        den = np.sqrt(np.trace(Wi @ Wi) * np.trace(Wj @ Wj))
        rv[i, j] = rv[j, i] = float(np.trace(Wi @ Wj) / den) if den > 1e-12 else 0.0
rv_dist = np.clip(1.0 - rv, 0.0, 2.0)
np.fill_diagonal(rv_dist, 0.0)
labels_rv, sil_rv, k_rv = ward_best_k(rv_dist, KMAX)
out["subtopic"] = {"k_pearson": int(k_m), "k_rv": int(k_rv),
                   "dual_method_ari": float(adjusted_rand_score(labels_m[k_m], labels_rv[k_rv]))}
print(f"子主题双法: K(Pearson)={k_m}, K(RV)={k_rv}, ARI={out['subtopic']['dual_method_ari']:.3f}")

out["elapsed_sec"] = round(time.time() - t0, 1)
with open(res / "step2_kmax100_check.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print(f"\n完成，耗时{out['elapsed_sec']}s → step2_kmax100_check.json")
