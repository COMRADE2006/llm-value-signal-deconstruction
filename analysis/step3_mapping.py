# -*- coding: utf-8 -*-
"""
第三步：经验维度 vs 12 维框架对照表（总方案v2 §4 第三步）。

对题目层聚类结果（orig 主通道基准划分），统计每个经验类的内容构成:
  - 经验类 × 12 维交叉表（题数 + 行占比 + 卡方独立性检验）
  - 经验类 × Top 子主题（每类前 8 个子主题构成）
  - 12 维 → 经验类映射（每维度多数类 + 纯度 → "哪些维度在 LLM 响应层面可合并"）
  - 跨类子主题清单（题目散布在 ≥2 个经验类的子主题 → 边界模糊证据）

输入: data/results/{source}/step2_labels_orig.csv + active_items.csv
输出: data/results/{source}/step3_cluster_x_dim.csv, step3_cluster_x_subtopic.csv,
      step3_dim_mapping.csv, step3_cross_subtopics.csv, step3_result.json

用法: python analysis/step3_mapping.py --source agree|old [--channel orig]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--channel", default="orig", help="用哪个通道的划分（默认 orig 主通道）")
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    lab = pd.read_csv(res_dir / f"step2_labels_{args.channel}.csv")
    lab.columns = ["rule_id", "cluster"]
    active = pd.read_csv(res_dir / "active_items.csv")
    df = active.merge(lab, on="rule_id")
    df["subtopics"] = df["derived_values"].fillna("").astype(str).str.split("|")

    # ---------- 经验类 × 12 维 ----------
    ct = pd.crosstab(df["cluster"], df["core_value"])
    ct_share = ct.div(ct.sum(axis=1), axis=0)
    ct_out = ct.astype(str) + " (" + (ct_share * 100).round(1).astype(str) + "%)"
    ct_out.to_csv(res_dir / "step3_cluster_x_dim.csv", encoding="utf-8-sig")

    chi2, p_chi, dof, _ = chi2_contingency(ct.values)

    # ---------- 经验类 × Top 子主题 ----------
    rows = []
    for c, g in df.groupby("cluster"):
        sub_cnt = {}
        for subs in g["subtopics"]:
            for s in subs:
                if s:
                    sub_cnt[s] = sub_cnt.get(s, 0) + 1
        top = sorted(sub_cnt.items(), key=lambda x: -x[1])[:8]
        rows.append({"cluster": c, "n_items": len(g),
                     "top_subtopics": "; ".join(f"{s}({n})" for s, n in top)})
    pd.DataFrame(rows).to_csv(res_dir / "step3_cluster_x_subtopic.csv", index=False, encoding="utf-8-sig")

    # ---------- 12 维 → 经验类映射（哪些维度可合并）----------
    dim_rows = []
    for v, g in df.groupby("core_value"):
        vc = g["cluster"].value_counts()
        majority, purity = int(vc.index[0]), float(vc.iloc[0] / len(g))
        # 该维度题目在各经验类的分布（映射明细）
        dist = {f"cluster_{k}": int(n) for k, n in vc.items()}
        dim_rows.append({"core_value": v, "n_items": len(g),
                         "majority_cluster": majority, "purity": round(purity, 3), **dist})
    dim_map = pd.DataFrame(dim_rows).sort_values("purity")
    dim_map.to_csv(res_dir / "step3_dim_mapping.csv", index=False, encoding="utf-8-sig")

    # 同一多数类的维度组 → 可合并候选
    merge_groups = {}
    for _, r in dim_map.iterrows():
        merge_groups.setdefault(r["majority_cluster"], []).append(
            f"{r['core_value']}(纯度{r['purity']:.2f})")

    # ---------- 跨类子主题 ----------
    sub_rows = []
    for s in set(x for subs in df["subtopics"] for x in subs if x):
        sub_df = df[df["subtopics"].apply(lambda x: s in x)]
        n_clust = sub_df["cluster"].nunique()
        if n_clust >= 2 and len(sub_df) >= 10:
            vc = sub_df["cluster"].value_counts(normalize=True)
            sub_rows.append({"subtopic": s, "n_items": len(sub_df), "n_clusters": int(n_clust),
                             "majority_cluster": int(vc.index[0]), "majority_share": round(float(vc.iloc[0]), 3)})
    cross = pd.DataFrame(sub_rows).sort_values("n_clusters", ascending=False)
    cross.to_csv(res_dir / "step3_cross_subtopics.csv", index=False, encoding="utf-8-sig")

    result = {
        "source": args.source, "channel": args.channel,
        "n_clusters": int(df["cluster"].nunique()),
        "chi2": float(chi2), "chi2_p": float(p_chi), "chi2_dof": int(dof),
        "dim_purity_mean": float(dim_map["purity"].mean()),
        "merge_candidates": {str(k): v for k, v in merge_groups.items()},
        "n_cross_subtopics": int(len(cross)),
    }
    with open(res_dir / "step3_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- 报告 ----------
    print("=" * 70)
    print(f"[source={args.source}] 第三步: 经验类({df['cluster'].nunique()}) × 12维对照")
    print(f"卡方独立性: χ²={chi2:.1f}, df={dof}, p={p_chi:.2e}"
          + ("（经验类与维度来源显著关联）" if p_chi < 0.05 else "（⚠ 经验类与维度来源无显著关联）"))
    print(f"\n维度纯度（0=完全散布, 1=整体落入同一经验类）均值: {dim_map['purity'].mean():.3f}")
    print("纯度最低的维度（LLM 响应层面边界最模糊）:")
    for _, r in dim_map.head(5).iterrows():
        print(f"  {r['core_value']:6s} 纯度={r['purity']:.2f} 多数类=cluster_{r['majority_cluster']} n={r['n_items']}")
    print(f"\n可合并候选（同一多数经验类的维度组）:")
    for k, v in merge_groups.items():
        if len(v) >= 2:
            print(f"  cluster_{k}: {' + '.join(v)}")
    print(f"\n跨类子主题 {len(cross)} 个（题目散布≥2经验类），已存 step3_cross_subtopics.csv")


if __name__ == "__main__":
    main()
