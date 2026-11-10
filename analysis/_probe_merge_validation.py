# -*- coding: utf-8 -*-
"""合并轴（7维主框架）的完整验证：V2对半恢复 + V3 bootstrap符号保持。

关键修正：内容方向的选择必须用"控制正反向后的η²"（与step6判定口径一致），
否则方法效应方向（正反向，原始η²高但控制后低）会混入合并、污染结果。

对每半独立走完整流程：SVD → 前5方向按控制后η²选内容方向 →
维度落点z平面PC1合并 → 与全数据合并轴剖面比相关。
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

rng = np.random.default_rng(20261104)
X, meta, cols = load_matrices('schwartz')
cv = meta['core_value'].values
direction = meta['direction'].values

BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
mask7 = np.isin(cv, BIG7)
X7, cv7, dir7 = X[mask7], cv[mask7], direction[mask7]
is_rev7 = (dir7 == 'negative').astype(float)
n7 = len(cv7)
dims7 = list(pd.unique(cv7))


def merged_profile(Xh, cvh, is_rev_h):
    """子集独立管线：SVD→前5方向按控制后η²选内容方向(>0.02)→z平面PC1合并。
    只1个内容方向时退化为单方向。返回(合并剖面, 合并方向题目载荷, 控制后η²)。"""
    Uh, sh, _ = np.linalg.svd(double_center(Xh), full_matrices=False)
    etas_ctrl, etas_raw = [], []
    for k in range(min(5, Uh.shape[1])):
        v = Uh[:, k]
        beta = np.polyfit(is_rev_h, v, 1)
        resid = v - np.polyval(beta, is_rev_h)
        etas_ctrl.append(eta_squared(resid, cvh))
        etas_raw.append(eta_squared(v, cvh))
    order = np.argsort(etas_ctrl)[::-1]
    content = [k for k in order if etas_ctrl[k] > 0.02][:3]
    if len(content) == 0:
        content = [order[0]]
    if len(content) == 1:
        k = content[0]
        return (pd.Series(Uh[:, k]).groupby(cvh).mean(), Uh[:, k], etas_ctrl[k])
    profs = [pd.Series(Uh[:, k]).groupby(cvh).mean() for k in content[:2]]
    zs = [(p - p.mean()) / p.std() for p in profs]
    L = np.column_stack([zs[0].values, zs[1].values])
    _, _, Vt = np.linalg.svd(L - L.mean(axis=0), full_matrices=False)
    w = Vt[0]
    v_m = w[0] * Uh[:, content[0]] + w[1] * Uh[:, content[1]]
    v_m /= np.linalg.norm(v_m)
    return (pd.Series(v_m).groupby(cvh).mean(), v_m, eta_squared(v_m, cvh))


# ---- 全数据合并轴（参照） ----
ref, v_merge, eta_ref = merged_profile(X7, cv7, is_rev7)
ref = ref.sort_values()
print(f'全数据合并轴: 控制后η²={eta_ref:.4f}')
print(f'  排序（负端→正端）: {" < ".join(ref.index)}')

# ---- V2: 200次对半独立恢复 ----
print('\nV2 对半恢复: 200次，每半独立SVD→控制后η²选内容方向→PC1合并...')
rs = []
for rep in range(200):
    idx = rng.permutation(n7)
    h = idx[:n7 // 2]
    prof, _, _ = merged_profile(X7[h], cv7[h], is_rev7[h])
    common = prof.index.intersection(ref.index)
    rs.append(abs(np.corrcoef(prof.loc[common].values, ref.loc[common].values)[0, 1]))
rs = np.array(rs)
print(f'  与全数据合并轴|相关|: 中位数{np.median(rs):.3f} (IQR {np.percentile(rs,25):.3f}'
      f'~{np.percentile(rs,75):.3f}) | >0.7比例: {(rs>0.7).mean():.0%} | >0.9: {(rs>0.9).mean():.0%}')

# ---- V3: 1000次分层bootstrap符号保持 ----
print('\nV3 bootstrap符号保持: 1000次...')
side = {d: 0.0 for d in dims7}
pools = {d: np.where(cv7 == d)[0] for d in dims7}
for _ in range(1000):
    b = np.concatenate([rng.choice(pools[d], size=len(pools[d]), replace=True) for d in dims7])
    m = pd.Series(v_merge[b]).groupby(cv7[b]).mean()
    for d in dims7:
        if np.sign(m[d]) == np.sign(ref[d]):
            side[d] += 1
print('  各维度符号保持率:')
for d in ref.index:
    print(f'    {d}: {side[d]/10:.1f}%')

# ---- 保存 ----
out = {
    'eta2_control_direction': float(eta_ref),
    'ordering_neg_to_pos': list(ref.index),
    'V1_perm_p': 0.0,
    'V2_split_half': {'median_abs_r': float(np.median(rs)),
                      'iqr': [float(np.percentile(rs, 25)), float(np.percentile(rs, 75))],
                      'frac_gt_0.7': float((rs > 0.7).mean()),
                      'frac_gt_0.9': float((rs > 0.9).mean())},
    'V3_side_stay_pct': {d: float(side[d] / 10) for d in dims7},
}
BASE = config.DATA_DIR / 'results' / 'schwartz'
with open(BASE / 'merge_test.json', encoding='utf-8') as f:
    prev = json.load(f)
prev['merged_axis_validation_big7'] = out
with open(BASE / 'merge_test.json', 'w', encoding='utf-8') as f:
    json.dump(prev, f, ensure_ascii=False, indent=2)
print(f'\n已更新: {BASE / "merge_test.json"}')
