# -*- coding: utf-8 -*-
"""3.3.1块结构强度区间估计（等效性检验口径）。
交互残差中同维题目对与异维题目对的平均相关及其差值Δr，题目层自助1000次95%CI；
另计算关联法检出所需观测数（80%功效、双尾α=0.05，Fisher z）。
输出 data/results/agree/block_strength_ci.json（3.3.1表3与正文的数字来源）。
"""
import json
import sys
import time
from math import atanh, ceil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

res = Path(config.DATA_DIR) / "results" / "agree"
active = pd.read_csv(res / "active_items.csv")
cent = pd.read_csv(res / "centered_orig.csv", index_col=0).loc[active["rule_id"]]
X = np.nan_to_num(cent.values.astype(float))
core = active.set_index("rule_id").loc[cent.index, "core_value"].values
n, m = X.shape
t0 = time.time()

# 题目×题目交互相关矩阵（与step2聚类同口径：nan→0后相关）
R = np.corrcoef(X)
iu0, iu1 = np.triu_indices(n, 1)


def stats(idx):
    """给定题目重抽样索引，返回(同维平均r, 异维平均r, 差值)；剔除重复抽到同一原题的对。"""
    core_b = core[idx]
    same = core_b[iu0] == core_b[iu1]
    dup = idx[iu0] == idx[iu1]
    vals = R[idx[iu0], idx[iu1]]
    s = vals[same & ~dup]
    c = vals[~same & ~dup]
    return s.mean(), c.mean(), s.mean() - c.mean()


# 点估计
same_r, cross_r, delta = stats(np.arange(n))
r_all = R[iu0, iu1]

# 题目层自助（B=1000）
rng = np.random.default_rng(config.SEED)
B = 1000
boot = np.array([stats(rng.integers(0, n, n)) for _ in range(B)])
q = lambda a: [float(x) for x in np.quantile(a, [0.025, 0.975])]

# 关联法所需观测数：N = (z_{α/2}+z_β)^2 / atanh(r)^2 + 3
zz = (1.959964 + 0.841621) ** 2
need = {str(r): ceil(zz / atanh(r) ** 2 + 3) for r in (0.5, 0.3, 0.1, 0.05, 0.01)}

out = {
    "n_items": int(n), "n_models": int(m),
    "same_dim_mean_r": float(same_r), "cross_dim_mean_r": float(cross_r),
    "delta_same_minus_cross": float(delta),
    "same_ci95": q(boot[:, 0]), "cross_ci95": q(boot[:, 1]), "delta_ci95": q(boot[:, 2]),
    "mean_r_all_pairs": float(r_all.mean()), "mean_abs_r_all_pairs": float(np.abs(r_all).mean()),
    "se_single_r": float(1 / np.sqrt(m - 3)),
    "n_needed_80power_alpha05": need,
    "bootstrap_B": B, "seed": int(config.SEED), "elapsed_sec": round(time.time() - t0, 1),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
(res / "block_strength_ci.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
