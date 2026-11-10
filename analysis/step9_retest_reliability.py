# -*- coding: utf-8 -*-
"""
step9 (T3) 复测信度检验：同一批题隔期重跑，模型答案稳不稳？

数据：data/raw_resp_agree_retest/{alias}_agree-off.jsonl（200题×3模型，off模式）
      vs 原全量采集 data/raw_resp_agree/{alias}_agree-off.jsonl（同题、同prompt、同参数）

指标（按模型分别 + 合并）：
  - 完全一致率（两次调用打出同一分数的比例）
  - ±1分一致率（容许一格漂移）
  - 二次加权κ（quadratic weighted kappa，计分误差的加权一致性）
  - Pearson r / Spearman ρ（两次调用的分数相关）
  - 均值漂移（重测均值-原均值，检验系统性偏移）

输出: data/results/agree/retest_reliability.json
用法: python analysis/step9_retest_reliability.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.build_matrix import parse_score

BASE = config.DATA_DIR / "results" / "agree"
RETEST_DIR = config.DATA_DIR / "raw_resp_agree_retest"
MODELS = ["doubao-seed-2-1-pro", "kimi-k2.6", "ds-v4-pro"]


def load_scores(path, retest=False):
    """读jsonl → {rule_id: score}（parse_score：非1-5数字一律NaN）。"""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["rule_id"]] = parse_score(r.get("raw_response"))
    return out


def reliability_stats(orig, retest):
    """两列分数（同一rule_id对齐）的信度指标。"""
    ids = sorted(set(orig) & set(retest))
    a = np.array([orig[i] for i in ids])
    b = np.array([retest[i] for i in ids])
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]          # 成对删除NaN
    n = len(a)
    if n == 0:
        return {"n_pairs": 0}
    exact = float((a == b).mean())
    within1 = float((np.abs(a - b) <= 1).mean())
    # 二次加权κ（分数1-5有序量表）
    try:
        kw = float(cohen_kappa_score(a.astype(int), b.astype(int),
                                     weights="quadratic"))
    except Exception:
        kw = float("nan")
    return {
        "n_pairs": n,
        "exact_agreement": exact,
        "within1_agreement": within1,
        "quadratic_weighted_kappa": kw,
        "pearson_r": float(pearsonr(a, b)[0]),
        "spearman_rho": float(spearmanr(a, b)[0]),
        "mean_orig": float(a.mean()), "mean_retest": float(b.mean()),
        "mean_shift": float(b.mean() - a.mean()),
    }


def main():
    print("=" * 62)
    print("step9 (T3) 复测信度检验（200题 × 3模型 × off模式）")
    print("=" * 62)
    out = {}
    all_a, all_b = [], []
    for alias in MODELS:
        orig = load_scores(config.AGREE_RESP_DIR / f"{alias}_agree-off.jsonl")
        retest = load_scores(RETEST_DIR / f"{alias}_agree-off.jsonl")
        st = reliability_stats(orig, retest)
        out[alias] = st
        # 合并池（收集对齐的成对分数）
        ids = sorted(set(orig) & set(retest))
        for i in ids:
            va, vb = orig[i], retest[i]
            if not (np.isnan(va) or np.isnan(vb)):
                all_a.append(va); all_b.append(vb)
        print(f"\n== {alias} == 可比对 {st['n_pairs']} 题")
        for k in ["exact_agreement", "within1_agreement",
                  "quadratic_weighted_kappa", "pearson_r", "spearman_rho"]:
            print(f"  {k:<28} {st[k]:.4f}")
        print(f"  {'mean_shift(重测-原)':<24} {st['mean_shift']:+.4f}")

    # 合并
    a, b = np.array(all_a), np.array(all_b)
    pooled = {
        "n_pairs": len(a),
        "exact_agreement": float((a == b).mean()),
        "within1_agreement": float((np.abs(a - b) <= 1).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(
            a.astype(int), b.astype(int), weights="quadratic")),
        "pearson_r": float(pearsonr(a, b)[0]),
        "spearman_rho": float(spearmanr(a, b)[0]),
        "mean_shift": float(b.mean() - a.mean()),
    }
    out["pooled"] = pooled
    print(f"\n== 合并（3模型×{len(a)//3}题） ==")
    for k in ["exact_agreement", "within1_agreement",
              "quadratic_weighted_kappa", "pearson_r", "spearman_rho"]:
        print(f"  {k:<28} {pooled[k]:.4f}")

    out["note"] = ("复测与原采集同prompt/同参数/同模型版本，间隔数日；"
                   "完全一致率+二次加权κ为论文实验设置引用值")
    path = BASE / "retest_reliability.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
