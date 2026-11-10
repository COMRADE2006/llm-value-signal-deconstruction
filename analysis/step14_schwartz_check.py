# -*- coding: utf-8 -*-
"""
step14 Schwartz框架结构检验（表2数字一站式产出，内部操作）：

三个变体（2026-11-04用户裁决：7维/10维各跑一次，择优定稿）：
  dim10_full     全部10维标签（含权力/享乐/刺激），全部活跃题，强设K=10
  dim7_full      剔除题量<30的权力/享乐/刺激，强设K=7
  dim7_balanced  7维等量下采样（每维取最小维度题量，重复10次），敏感性核查——
                 预答"ARI低是标签失衡伪影"的质疑（是否写入论文由用户另行定夺）

每变体产出表2三行：
  行1 簇—维度标签一致性: 三通道(orig/I/P) Ward强设K vs schwartz标签 ARI + 置换机遇基线(p95)
  行2 子抽样稳定性: 70%题目重抽300次，重聚类(强设K) vs 基准划分 ARI（仅full变体）
  行3 跨推理模式对半复制: on/off半边各自中心化→强设K → ARI（仅full变体）

依赖: results/schwartz/ 由 step13_schwartz_merge.py 生成
用法: python analysis/step14_schwartz_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step2_clustering import dist_1_minus_corr, ward_best_k, subsample_stability, double_center_arr

RES = Path(config.DATA_DIR) / "results" / "schwartz"
BIG7 = ["仁慈", "普世主义", "安全", "成就", "从众", "自我导向", "传统"]


def forced_ari(X, codes, k, rng, n_perm=1000):
    """强设K聚类 vs 标签的ARI + 置换机遇基线p95。"""
    labs_all, _, _ = ward_best_k(dist_1_minus_corr(X), k)
    lab = labs_all[k]
    ari = float(adjusted_rand_score(lab, codes))
    null = [adjusted_rand_score(lab, rng.permutation(codes)) for _ in range(n_perm)]
    p95 = float(np.percentile(null, 95))
    return ari, p95, bool(ari > p95)


def full_variant(name, item_ids, k, cents, scored, core, rng):
    """表2三行检验。"""
    out = {"n_items": len(item_ids), "k": k, "channels": {}}
    codes = pd.factorize(core.loc[item_ids].values)[0]

    # 行1：三通道 ARI + 机遇基线
    for ch in ("orig", "I", "P"):
        X = np.nan_to_num(cents[ch].loc[item_ids].values.astype(float))
        ari, p95, exc = forced_ari(X, codes, k, rng)
        out["channels"][ch] = {"ari": ari, "null_p95": p95, "exceed": exc}
        print(f"[{name}][{ch}] K={k} ARI={ari:.3f}  机遇p95={p95:.4f}  超出: {'是' if exc else '否'}")

    # 行2：子抽样稳定性（orig通道）
    X = np.nan_to_num(cents["orig"].loc[item_ids].values.astype(float))
    labs_all, _, _ = ward_best_k(dist_1_minus_corr(X), k)
    ari_m, ari_sd = subsample_stability(X, labs_all[k], k, 300, 0.7, config.SEED)
    out["subsample"] = {"ari_mean": ari_m, "ari_sd": ari_sd, "n_sub": 300, "frac": 0.7}
    print(f"[{name}][子抽样] ARI = {ari_m:.3f} ± {ari_sd:.3f}")

    # 行3：跨推理模式对半复制
    halves = {"on": [c for c in scored.columns if c.endswith("_on")],
              "off": [c for c in scored.columns if c.endswith("_off")]}
    hl = {}
    sub_scored = scored.loc[item_ids]
    for hname, hcols in halves.items():
        Xh = sub_scored[hcols].values.astype(float)
        keep_h = np.sum(~np.isnan(Xh), axis=1) >= max(6, len(hcols) - 3)
        Xc = np.nan_to_num(double_center_arr(Xh[keep_h]))
        labs_h, _, _ = ward_best_k(dist_1_minus_corr(Xc), k)
        hl[hname] = pd.Series(labs_h[k], index=sub_scored.index[keep_h])
    common = hl["on"].index.intersection(hl["off"].index)
    ari_h = float(adjusted_rand_score(hl["on"][common], hl["off"][common]))
    out["half_split"] = {"n_common": int(len(common)), "ari_on_vs_off": ari_h}
    print(f"[{name}][对半复制] 共同{len(common)}题 ARI = {ari_h:.3f}")
    return out


def balanced_variant(cents, core, n_draws=10, n_perm=500):
    """7维等量下采样敏感性核查：每维取最小题量，重复n_draws次，报告ARI均值±SD。"""
    per_dim = {d: core.index[core == d] for d in BIG7}
    n_min = min(len(v) for v in per_dim.values())
    res = {ch: [] for ch in ("orig", "I", "P")}
    p95s = []
    for d in range(n_draws):
        rng_d = np.random.default_rng(config.SEED + d)
        ids = np.concatenate([rng_d.choice(per_dim[dim], size=n_min, replace=False) for dim in BIG7])
        codes = pd.factorize(core.loc[ids].values)[0]
        for ch in ("orig", "I", "P"):
            X = np.nan_to_num(cents[ch].loc[ids].values.astype(float))
            ari, p95, _ = forced_ari(X, codes, len(BIG7), rng_d, n_perm)
            res[ch].append(ari)
            if ch == "orig":
                p95s.append(p95)
    out = {
        "n_draws": n_draws, "per_dim_n": int(n_min), "n_items": int(n_min * len(BIG7)),
        "ari_mean": {ch: float(np.mean(v)) for ch, v in res.items()},
        "ari_sd": {ch: float(np.std(v)) for ch, v in res.items()},
        "null_p95_mean": float(np.mean(p95s)),
    }
    for ch in ("orig", "I", "P"):
        print(f"[dim7_balanced][{ch}] ARI = {out['ari_mean'][ch]:.3f} ± {out['ari_sd'][ch]:.3f}"
              f"（{n_draws}次等量抽样，每次{out['n_items']}题）")
    print(f"[dim7_balanced] 机遇p95均值 = {out['null_p95_mean']:.4f}")
    return out


def main():
    active = pd.read_csv(RES / "active_items.csv", encoding="utf-8-sig")
    scored = pd.read_csv(RES / "scored_matrix.csv", index_col=0)
    core = active.set_index("rule_id")["core_value"]

    cents = {ch: pd.read_csv(RES / f"centered_{ch}.csv", index_col=0) for ch in ("orig", "I", "P")}
    rng = np.random.default_rng(config.SEED)

    all_ids = list(active["rule_id"])
    big_ids = [r for r in all_ids if core[r] in BIG7]
    print(f"变体A dim10_full: {len(all_ids)}题 K=10 | 变体B dim7_full: {len(big_ids)}题 K=7\n")

    out = {"variants": {}}
    out["variants"]["dim10_full"] = full_variant("dim10_full", all_ids, 10, cents, scored, core, rng)
    print()
    out["variants"]["dim7_full"] = full_variant("dim7_full", big_ids, 7, cents, scored, core, rng)
    print()
    out["variants"]["dim7_balanced"] = balanced_variant(cents, core)

    with open(RES / "table2_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总（orig通道） ===")
    for name in ("dim10_full", "dim7_full"):
        v = out["variants"][name]
        c = v["channels"]["orig"]
        print(f"{name:14s} K={v['k']:2d} n={v['n_items']}  ARI={c['ari']:.3f} (p95={c['null_p95']:.4f})  "
              f"子抽样={v['subsample']['ari_mean']:.3f}±{v['subsample']['ari_sd']:.3f}  "
              f"对半={v['half_split']['ari_on_vs_off']:.3f}")
    b = out["variants"]["dim7_balanced"]
    print(f"dim7_balanced  K=7  n={b['n_items']}  ARI={b['ari_mean']['orig']:.3f}±{b['ari_sd']['orig']:.3f} "
          f"(p95={b['null_p95_mean']:.4f})")
    print(f"\n已写入 {RES / 'table2_check.json'}")


if __name__ == "__main__":
    main()
