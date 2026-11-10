# -*- coding: utf-8 -*-
"""补打7维子集两条内容方向的维度剖面，并与全数据D1/D4剖面并排比较。"""
import sys
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
U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
m1 = pd.Series(U[:, 0]).groupby(cv).mean()
m4 = pd.Series(U[:, 3]).groupby(cv).mean()

BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
mask7 = np.isin(cv, BIG7)
X7, cv7 = X[mask7], cv[mask7]
is_rev7 = (meta['direction'].values[mask7] == 'negative').astype(float)
U7, s7, _ = np.linalg.svd(double_center(X7), full_matrices=False)

p7_1 = pd.Series(U7[:, 0]).groupby(cv7).mean()   # 7维方向1（η²=0.0246）
p7_4 = pd.Series(U7[:, 3]).groupby(cv7).mean()   # 7维方向4（η²=0.0410）

# z标准化便于横向比较
def z(sc):
    return (sc - sc.mean()) / sc.std()

cmp = pd.DataFrame({
    '7维方向1(z)': z(p7_1),
    '7维方向4(z)': z(p7_4),
    '全数据D1(z)': z(m1).loc[BIG7],
    '全数据D4(z)': z(m4).loc[BIG7],
}).loc[BIG7]
print('7维子集两条内容方向 vs 全数据D1/D4（z标准化）:')
print(cmp.round(3).to_string())
print()
print('相关矩阵:')
print(cmp.corr().round(3).to_string())
print()
# 7维方向4与全数据D4的相关
r_4_vs_D4 = np.corrcoef(z(p7_4).values, z(m4).loc[BIG7].values)[0, 1]
r_4_vs_D1 = np.corrcoef(z(p7_4).values, z(m1).loc[BIG7].values)[0, 1]
r_1_vs_D1 = np.corrcoef(z(p7_1).values, z(m1).loc[BIG7].values)[0, 1]
r_1_vs_D4 = np.corrcoef(z(p7_1).values, z(m4).loc[BIG7].values)[0, 1]
print(f'7维方向1 vs 全数据D1: r={r_1_vs_D1:+.3f} | vs D4: r={r_1_vs_D4:+.3f}')
print(f'7维方向4 vs 全数据D4: r={r_4_vs_D4:+.3f} | vs D1: r={r_4_vs_D1:+.3f}')
