# -*- coding: utf-8 -*-
"""
武器三：正反题分相关（先锋把关，判别维度特异性）。

逻辑: 若 12 维框架在 LLM 响应层面有维度特异性，则同一维度的正向题均分与
反向题均分（计分后，高分均=认同该价值）应正相关；跨维配对作对照。

方法:
  - 每模型每维度: 正向题均分 P[v,c]、反向题均分 R[v,c]（计分后，NaN 跳过）
  - 同维相关: 12维 × 22列 = 264 个 (P, R) 点的 Pearson r
  - 置换检验: 打乱 R 的维度标签 10000 次 → 跨维对照分布 → 经验 p 值
  - 敏感性(--center-model): 每模型列扣除列均值（控 acquiescence 基线）后重算

判读:
  r_same 显著 > r_cross 分布 → 维度特异性存在，后续聚类出的"维度"有意义
  r_same 不高于跨维分布 → 维度特异性不存在 → 先锋止损警告

输入: data/results/{source}/scored_matrix.csv + data/final_items_v2.jsonl
输出: data/results/{source}/weapon3_posneg.csv（每维每模型正反均值）
      data/results/{source}/weapon3_result.json（检验结果）+ stdout 报告

用法: python analysis/weapon3_posneg.py --source agree|old [--center-model] [--n-perm 10000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--center-model", action="store_true", help="扣模型列均值（控 acquiescence）后重算")
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    scored = pd.read_csv(res_dir / "scored_matrix.csv", index_col=0)

    items = []
    with open(config.FINAL_ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    meta = pd.DataFrame([{
        "rule_id": it["rule_id"],
        "direction": it.get("direction", "positive"),
        "core_value": it.get("core_value", ""),
    } for it in items]).set_index("rule_id")
    meta = meta.loc[scored.index]

    # 模型列中心化选项（控 acquiescence）
    mat = scored - scored.mean(axis=0) if args.center_model else scored

    # 每模型每维度的正/反向均分
    dims = meta["core_value"].unique().tolist()
    cols = mat.columns.tolist()
    rows = []
    for c in cols:
        col = mat[c]
        for v in dims:
            m_v = (meta["core_value"] == v).values
            m_pos = m_v & (meta["direction"] == "positive").values
            m_neg = m_v & (meta["direction"] == "negative").values
            rows.append({
                "model_mode": c, "core_value": v,
                "pos_mean": col[m_pos].mean(),
                "neg_mean": col[m_neg].mean(),
                "n_pos": int(m_pos.sum()), "n_neg": int(m_neg.sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "weapon3_posneg.csv", index=False, encoding="utf-8-sig")

    valid = df.dropna(subset=["pos_mean", "neg_mean"])
    p_all = valid["pos_mean"].values
    r_all = valid["neg_mean"].values
    key = valid["core_value"].values

    def paired_r(p, r, k):
        """按维度标签配对的正反相关（同维 P-R 点对）。"""
        ps, rs = [], []
        for v in np.unique(k):
            pv = p[k == v].mean()
            rv = r[k == v].mean()
            if not (np.isnan(pv) or np.isnan(rv)):
                ps.append(pv)
                rs.append(rv)
        return np.corrcoef(ps, rs)[0, 1], len(ps)

    # 逐模型同维相关（每模型 12 个维度点，P vs R）
    per_model = []
    for c in valid["model_mode"].unique():
        sub = valid[valid["model_mode"] == c]
        r_m, n_m = paired_r(sub["pos_mean"].values, sub["neg_mean"].values, sub["core_value"].values)
        per_model.append({"model_mode": c, "r_same_dim": r_m, "n_dims": n_m})
    pm = pd.DataFrame(per_model)

    # pooled 同维相关（维度层配对）
    r_same, n_pair = paired_r(p_all, r_all, key)

    # 置换检验：打乱 R 的维度标签（跨维对照分布）
    rng = np.random.default_rng(config.SEED)
    n_perm = args.n_perm
    null_r = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(key)
        null_r[i], _ = paired_r(p_all, r_all, perm)
    p_val = float(np.mean(null_r >= r_same))

    result = {
        "source": args.source,
        "center_model": bool(args.center_model),
        "n_valid_cells": int(len(valid)),
        "r_same_dim_pooled": float(r_same),
        "n_dim_pairs": int(n_pair),
        "r_same_dim_by_model_mean": float(pm["r_same_dim"].mean()),
        "r_same_dim_by_model_sd": float(pm["r_same_dim"].std()),
        "null_mean": float(np.mean(null_r)),
        "null_p95": float(np.percentile(null_r, 95)),
        "perm_p_value": p_val,
        "verdict": "",
    }
    # 判读：同维相关显著高于跨维置换分布（单尾 5%）
    if p_val < 0.05 and r_same > np.percentile(null_r, 95):
        result["verdict"] = (
            f"✅ 维度特异性存在: 同维正反相关 r={r_same:.3f} 显著高于跨维置换分布 "
            f"(p={p_val:.4f}, 置换均值={np.mean(null_r):.3f})，后续聚类出的维度有意义"
        )
    else:
        result["verdict"] = (
            f"⚠ 维度特异性不成立: 同维正反相关 r={r_same:.3f} 未显著高于跨维置换分布 "
            f"(p={p_val:.4f})——12 维框架在 LLM 响应层面无正反互证结构，聚类结果需谨慎解读"
        )
    with open(res_dir / "weapon3_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"[source={args.source}] 武器三: 正反题分相关 (center_model={args.center_model})")
    print(f"有效单元: {len(valid)} (模型×维度), 涉及 {valid['model_mode'].nunique()} 列 × {len(dims)} 维")
    print(f"\n逐模型同维相关 (每模型 12 维度点 P vs R):")
    for _, row in pm.iterrows():
        print(f"  {row['model_mode']:34s} r={row['r_same_dim']:+.3f}")
    print(f"\npooled 同维相关: r={r_same:+.3f} (n={n_pair} 维度)")
    print(f"跨维置换分布:   mean={np.mean(null_r):+.3f}, p95={np.percentile(null_r, 95):+.3f}, n_perm={n_perm}")
    print(f"\n{result['verdict']}")
    print(f"结果已存: {res_dir / 'weapon3_result.json'}")


if __name__ == "__main__":
    main()
