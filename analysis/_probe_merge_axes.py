# -*- coding: utf-8 -*-
"""合并检验：两条内容轴能否合并为一条？（用户直觉的统计裁决）

逻辑：若两轴"其实是一条"，则取维度落点在(D1,D4)平面的PC1作为最佳单轴近似（合并轴），
其正交残差方向（PC2）的内容η²应落在置换噪声带内；
若残差η²显著超噪声 → 第二轴承载独立内容（成就反向分离/权力极端负载），不能无损合并。

在两个口径下各做一次：全数据10维 + 7维主框架（BIG7，论文口径）。
"""
import sys
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


def merge_test(vA, vB, cvx, label):
    """vA=方差轴(类D1), vB=主内容轴(类D4)。返回合并轴与残差轴的检验结果。"""
    dims = list(pd.unique(cvx))
    mA = pd.Series(vA).groupby(cvx).mean()
    mB = pd.Series(vB).groupby(cvx).mean()
    # z标准化使两轴平权（回答"维度怎么排"而非"方差多大"）
    zA = (mA - mA.mean()) / mA.std()
    zB = (mB - mB.mean()) / mB.std()
    L = np.column_stack([zA.loc[dims].values, zB.loc[dims].values])
    Lc = L - L.mean(axis=0)
    # 维度落点的PC1 → 合并权重w（vA、vB正交等范数 → 组合后仍正交）
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False)
    w = Vt[0]
    v_merge = w[0] * vA + w[1] * vB
    v_resid = -w[1] * vA + w[0] * vB

    eta_m = eta_squared(v_merge, cvx)
    eta_r = eta_squared(v_resid, cvx)
    null_m = np.array([eta_squared(v_merge, rng.permutation(cvx)) for _ in range(1000)])
    null_r = np.array([eta_squared(v_resid, rng.permutation(cvx)) for _ in range(1000)])
    p_m, p_r = (null_m >= eta_m).mean(), (null_r >= eta_r).mean()

    print(f'\n{"="*70}\n[{label}] PC1权重: 方差轴×{w[0]:+.3f} + 主内容轴×{w[1]:+.3f}')
    print(f'  合并轴: η²={eta_m:.4f} | 零分布max={null_m.max():.4f} | p={p_m}')
    print(f'  残差轴: η²={eta_r:.4f} | 零分布max={null_r.max():.4f} | p={p_r}')
    verdict = ('残差显著 → 不能无损合并（第二轴有独立内容信号）' if p_r < 0.05
               else '残差不显著 → 单轴近似足够，可合并为一条')
    print(f'  裁决: {verdict}')

    order_m = pd.Series(v_merge).groupby(cvx).mean().sort_values()
    order_r = pd.Series(v_resid).groupby(cvx).mean().sort_values()
    print(f'  合并轴维度排序（负端→正端）: {" < ".join(order_m.index)}')
    print(f'  残差轴维度排序（负端→正端）: {" < ".join(order_r.index)}')
    return {'w': w.tolist(), 'eta_merge': float(eta_m), 'p_merge': float(p_m),
            'eta_resid': float(eta_r), 'p_resid': float(p_r),
            'merge_order': list(order_m.index), 'resid_order': list(order_r.index),
            'verdict': verdict}


out = {}
# ---- 口径1: 全数据10维（vA=方向1, vB=方向4）----
U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
out['full_10d'] = merge_test(U[:, 0], U[:, 3], cv, '全数据10维')

# ---- 口径2: 7维主框架（BIG7子集独立SVD）----
BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
mask7 = np.isin(cv, BIG7)
U7, s7, _ = np.linalg.svd(double_center(X[mask7]), full_matrices=False)
out['big7'] = merge_test(U7[:, 0], U7[:, 3], cv[mask7], '7维主框架(BIG7)')

import json
BASE = config.DATA_DIR / 'results' / 'schwartz'
with open(BASE / 'merge_test.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n已保存: {BASE / "merge_test.json"}')
