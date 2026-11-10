# -*- coding: utf-8 -*-
"""探查：10个Schwartz维度在（方向1×方向4）双轴平面上的落位与环形序一致性。

问题：双语义轴张成的二维平面上，维度排布是否镜像Schwartz理论环形？
方法：维度均值→(D1,D4)坐标→观测角；理论环形角（Schwartz十价值圆环序）；
      成对距离的Mantel式Spearman相关 + 标签置换检验。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from step5_ega import load_matrices
from step6_mp_spectrum import double_center

X, meta, cols = load_matrices('schwartz')
cv = meta['core_value'].values
U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)

v1 = U[:, 0]   # 方向1（方差最大，14.2%）
v4 = U[:, 3]   # 方向4（η²最大，7.6%）

m1 = pd.Series(v1).groupby(cv).mean()
m4 = pd.Series(v4).groupby(cv).mean()

# 各维度双轴坐标（z标准化便于比较）
z1 = (m1 - m1.mean()) / m1.std()
z4 = (m4 - m4.mean()) / m4.std()

dims = ['权力', '成就', '享乐', '刺激', '自我导向', '普世主义', '仁慈', '传统', '从众', '安全']
print('维度        D1(z)   D4(z)   观测角(°)')
for d in dims:
    ang = np.degrees(np.arctan2(z4[d], z1[d])) % 360
    print(f'{d:6s} {z1[d]:+7.3f} {z4[d]:+7.3f} {ang:8.1f}')

# Schwartz理论环形角（相邻30°一等分；享乐在成就与刺激之间按Schwartz经典放置）
theory_ang = {'权力': 15, '成就': 45, '享乐': 75, '刺激': 105, '自我导向': 135,
              '普世主义': 175, '仁慈': 215, '传统': 255, '从众': 295, '安全': 335}

# Mantel式：理论成对环形距离 vs 观测成对平面欧氏距离
from itertools import combinations
pairs = list(combinations(dims, 2))
theo_d, obs_d = [], []
for a, b in pairs:
    da = min(abs(theory_ang[a] - theory_ang[b]), 360 - abs(theory_ang[a] - theory_ang[b]))
    theo_d.append(da)
    obs_d.append(np.hypot(z1[a] - z1[b], z4[a] - z4[b]))
rho, p = spearmanr(theo_d, obs_d)
print(f'\n环形序一致性（理论角距 vs 观测平面距离, Spearman）: ρ={rho:.3f}, p={p:.4f}')

# 标签置换零分布
rng = np.random.default_rng(20261104)
null = []
theo_arr = np.array(theo_d)
for _ in range(2000):
    perm = rng.permutation(obs_d)
    r, _ = spearmanr(theo_arr, perm)
    null.append(r)
null = np.array(null)
p_perm = (null >= rho).mean()
print(f'置换检验: 零分布均值{null.mean():.3f}, p={p_perm:.4f}')
print(f'判定: {"环形序成立" if p_perm < 0.05 else "环形序不成立"}')
