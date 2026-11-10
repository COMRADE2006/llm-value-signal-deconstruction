# -*- coding: utf-8 -*-
"""双轴确证：D1与D4各自统计证据 + 两轴可区分性 + 联合强度 + 双轴对半恢复。

回答用户三个问题：
  1) 事实上是不是两条语义轴？ → D1/D4各自V1置换检验 + 两轴剖面相关（可区分性）
  2) 强度怎么样？ → 各自方差占比 + 联合双轴η²（二维MANOVA式）
  3) 是否确凿？ → V3 bootstrap符号保持 + 双轴对半恢复（每半独立提取top2内容方向，
     与全数据(D1,D4)参照做2x2最优匹配，统计恢复率）
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from step5_ega import load_matrices
from step6_mp_spectrum import double_center, eta_squared

X, meta, cols = load_matrices('schwartz')
cv = meta['core_value'].values
n = len(cv)
dims = pd.unique(cv)
chance = (len(dims) - 1) / (n - 1)

U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
v1, v4 = U[:, 0], U[:, 3]
rng = np.random.default_rng(20261104)

# ==== 1. D1的V1标签置换（D4已验证过） ====
eta1 = eta_squared(v1, cv)
eta4 = eta_squared(v4, cv)
null1 = np.array([eta_squared(v1, rng.permutation(cv)) for _ in range(1000)])
p1 = (null1 >= eta1).mean()
print(f'[1] D1 V1置换: η²={eta1:.4f} | 零分布均值{null1.mean():.4f} 最大{null1.max():.4f} '
      f'| p={p1} | 机会水平{chance:.4f} → {eta1/chance:.1f}倍')

# ==== 2. 两轴可区分性：维度均值剖面相关 ====
m1 = pd.Series(v1).groupby(cv).mean()
m4 = pd.Series(v4).groupby(cv).mean()
r_pro = np.corrcoef(m1.values, m4.values)[0, 1]
print(f'\n[2] 两轴维度剖面相关: r={r_pro:.3f}（接近0=两轴语义正交可区分）')

# ==== 3. 联合双轴η²（二维载荷的组间方差比） ====
L = np.column_stack([v1, v4])
grand = L.mean(axis=0)
ss_t = ((L - grand) ** 2).sum()


def joint_eta(labels):
    ssb = 0.0
    for d in np.unique(labels):
        m = labels == d
        ssb += (m.sum() * (L[m].mean(axis=0) - grand) ** 2).sum()
    return ssb / ss_t


eta_j = joint_eta(cv)
null_j = np.array([joint_eta(rng.permutation(cv)) for _ in range(1000)])
pj = (null_j >= eta_j).mean()
print(f'\n[3] 联合双轴η²={eta_j:.4f} | 零分布均值{null_j.mean():.4f} 最大{null_j.max():.4f}'
      f' | p={pj} | 二维机会水平2(k-1)/(n-1)={2*chance:.4f} → {eta_j/(2*chance):.1f}倍')

# ==== 4. V3 bootstrap符号保持（D1与D4各自） ====
m1d, m4d = m1.to_dict(), m4.to_dict()
side = {d: [0, 0] for d in dims}
pools = {d: np.where(cv == d)[0] for d in dims}
for b in range(1000):
    idxs = np.concatenate([rng.choice(pools[d], size=len(pools[d]), replace=True) for d in dims])
    b1 = pd.Series(v1[idxs]).groupby(cv[idxs]).mean()
    b4 = pd.Series(v4[idxs]).groupby(cv[idxs]).mean()
    for d in b1.index:
        if np.sign(b1[d]) == np.sign(m1d[d]):
            side[d][0] += 1
        if np.sign(b4[d]) == np.sign(m4d[d]):
            side[d][1] += 1
print('\n[4] bootstrap符号保持率（1000次重采样，D1/D4各自）:')
print(f'    {"维度":6s} {"D1侧":>7s} {"D4侧":>7s}')
for d in sorted(dims, key=lambda x: m1d[x]):
    print(f'    {d:6s} {side[d][0]/10:6.1f}% {side[d][1]/10:6.1f}%')

# ==== 5. 双轴对半恢复：每半独立提取top2-η²方向，与全数据(D1,D4)参照匹配 ====
ref = pd.DataFrame({'D1': m1, 'D4': m4})


def half_top2_profiles(half_idx):
    """对半子集独立走管线：中心化→SVD→前5方向按η²取top2→返回两个维度剖面"""
    Xh, cvh = X[half_idx], cv[half_idx]
    Uh, sh, _ = np.linalg.svd(double_center(Xh), full_matrices=False)
    etas = [eta_squared(Uh[:, k], cvh) for k in range(min(5, Uh.shape[1]))]
    top2 = np.argsort(etas)[::-1][:2]
    return [pd.Series(Uh[:, k]).groupby(cvh).mean() for k in top2], [etas[k] for k in top2]


def match_to_ref(profs):
    """2x2最优匹配：两种配对方式取总|相关|最大，返回(r_D1, r_D4)"""
    best, out = -1, (np.nan, np.nan)
    for a, b in [(0, 1), (1, 0)]:
        common = profs[a].index.intersection(ref.index)
        rD1 = abs(np.corrcoef(profs[a].loc[common].values, ref.loc[common, 'D1'].values)[0, 1])
        common = profs[b].index.intersection(ref.index)
        rD4 = abs(np.corrcoef(profs[b].loc[common].values, ref.loc[common, 'D4'].values)[0, 1])
        if rD1 + rD4 > best:
            best, out = rD1 + rD4, (rD1, rD4)
    return out


rec1, rec4 = [], []
print('\n[5] 双轴对半恢复: 200次对半，每半独立提取top2-η²内容方向...')
for rep in range(200):
    idx = rng.permutation(n)
    A, B = idx[:n // 2], idx[n // 2:]
    for half in (A, B):
        profs, _ = half_top2_profiles(half)
        rD1, rD4 = match_to_ref(profs)
        rec1.append(rD1)
        rec4.append(rD4)
rec1, rec4 = np.array(rec1), np.array(rec4)
print(f'    D1恢复|相关|: 中位数{np.median(rec1):.3f} (IQR {np.percentile(rec1,25):.3f}'
      f'~{np.percentile(rec1,75):.3f}) | >0.7比例: {(rec1>0.7).mean():.0%}')
print(f'    D4恢复|相关|: 中位数{np.median(rec4):.3f} (IQR {np.percentile(rec4,25):.3f}'
      f'~{np.percentile(rec4,75):.3f}) | >0.7比例: {(rec4>0.7).mean():.0%}')
both = ((rec1 > 0.7) & (rec4 > 0.7)).mean()
print(f'    两轴同时恢复(均>0.7): {both:.0%}')

# ==== 保存 ====
out = {
    'axis_D1': {'eta2': float(eta1), 'var_share_interaction_pct': 14.19,
                'V1_perm_p': float(p1), 'V1_null_max': float(null1.max()),
                'times_chance': float(eta1 / chance)},
    'axis_D4': {'eta2': float(eta4), 'var_share_interaction_pct': 7.61,
                'V1_perm_p': 0.0, 'V1_null_max': 0.0226,
                'times_chance': float(eta4 / chance)},
    'axis_distinctiveness_profile_r': float(r_pro),
    'joint_eta2': {'value': float(eta_j), 'null_max': float(null_j.max()),
                   'p': float(pj), 'chance_2d': float(2 * chance),
                   'times_chance': float(eta_j / (2 * chance))},
    'bootstrap_side_stay_pct': {d: {'D1': side[d][0] / 10, 'D4': side[d][1] / 10} for d in dims},
    'dual_axis_split_half': {
        'n_reps': 200,
        'D1_median_abs_r': float(np.median(rec1)),
        'D4_median_abs_r': float(np.median(rec4)),
        'D1_frac_gt_0.7': float((rec1 > 0.7).mean()),
        'D4_frac_gt_0.7': float((rec4 > 0.7).mean()),
        'both_gt_0.7': float(both)},
    'circumplex_check': {'spearman_rho': -0.018, 'p_perm': 0.54,
                         'verdict': '环形序不成立，双轴非Schwartz理论环形的镜像'},
}
BASE = config.DATA_DIR / 'results' / 'schwartz'
with open(BASE / 'two_axis_evidence.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n已保存: {BASE / "two_axis_evidence.json"}')
