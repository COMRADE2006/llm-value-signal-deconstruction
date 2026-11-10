# -*- coding: utf-8 -*-
"""诊断：7维子集（BIG7主框架）上的轴结构 + D4弱恢复机制。

问题1: 论文主框架是7维（剔除权力15/享乐11/刺激3），全数据双轴(D1/D4)的区分
       主要靠权力(D1极端负载)与成就(两轴符号相反)。剔除后7维上是几条内容轴？
       - 若合并为一条 → 主框架下单轴叙事成立，双轴是10维全谱的现象
       - 若仍两条     → 双轴叙事在7维下也成立
问题2: 对半恢复中D4偏弱(0.597)的机制：top2-η²选择是否被方法方向挤占？
       用top3重算恢复率：若D4在top3中高比例出现，弱恢复是排序竞争而非结构缺失。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from step5_ega import load_matrices
from step6_mp_spectrum import double_center, eta_squared

X, meta, cols = load_matrices('schwartz')
cv = meta['core_value'].values
n = len(cv)
U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
v1, v4 = U[:, 0], U[:, 3]
m1 = pd.Series(v1).groupby(cv).mean()
m4 = pd.Series(v4).groupby(cv).mean()
rng = np.random.default_rng(20261104)

BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']

# ==== 问题1: 7维子集上的轴结构 ====
mask7 = np.isin(cv, BIG7)
X7, cv7 = X[mask7], cv[mask7]
is_rev7 = (meta['direction'].values[mask7] == 'negative').astype(float)
n7 = mask7.sum()
print(f'=' * 70)
print(f'[问题1] 7维子集（BIG7）: {n7}题 × {X7.shape[1]}模型')
U7, s7, _ = np.linalg.svd(double_center(X7), full_matrices=False)
var7 = s7 ** 2 / (s7 ** 2).sum()

ref1 = m1.loc[BIG7]  # 全数据D1剖面的7维部分（剔除权力/享乐/刺激）
ref4 = m4.loc[BIG7]

print(f'{"方向":>4s} {"方差%":>6s} {"η²":>7s} {"控制后η²":>8s} {"KW-p":>9s} '
      f'{"r(D1)":>7s} {"r(D4)":>7s}  判定')
for k in range(5):
    v = U7[:, k]
    beta = np.polyfit(is_rev7, v, 1)
    resid = v - np.polyval(beta, is_rev7)
    eta = eta_squared(v, cv7)
    eta_c = eta_squared(resid, cv7)
    p_c = kruskal(*[resid[cv7 == c] for c in pd.unique(cv7)]).pvalue
    prof = pd.Series(resid).groupby(cv7).mean()
    # 关键：必须按公共维度index对齐（groupby输出顺序≠BIG7列表顺序，直接.values会错位）
    common = prof.index.intersection(ref1.index)
    rD1 = np.corrcoef(prof.loc[common].values, ref1.loc[common].values)[0, 1]
    rD4 = np.corrcoef(prof.loc[common].values, ref4.loc[common].values)[0, 1]
    verdict = ('内容' if (eta_c > 0.02 and p_c < 0.001) else
               ('方法' if abs(np.corrcoef(v, is_rev7)[0, 1]) > 0.3 else '其他'))
    print(f'{k+1:4d} {var7[k]*100:6.1f} {eta:7.4f} {eta_c:8.4f} {p_c:9.2e} '
          f'{rD1:+7.3f} {rD4:+7.3f}  {verdict}')

# 7维内容方向的标签置换（V1式）
print('\n7维各内容方向标签置换1000次:')
for k in range(5):
    v = U7[:, k]
    beta = np.polyfit(is_rev7, v, 1)
    resid = v - np.polyval(beta, is_rev7)
    eta_c = eta_squared(resid, cv7)
    if not (eta_c > 0.02):
        continue
    null = np.array([eta_squared(resid, rng.permutation(cv7)) for _ in range(1000)])
    print(f'  方向{k+1}: 控制后η²={eta_c:.4f} | 零分布max={null.max():.4f} '
          f'| p={(null >= eta_c).mean()}')

# ==== 问题2: D4弱恢复机制（top3诊断） ====
print(f'=' * 70)
print('[问题2] 对半top3诊断: 200次对半，每半前5方向按η²排序取top3')


def half_profiles(half_idx):
    Xh, cvh = X[half_idx], cv[half_idx]
    Uh, sh, _ = np.linalg.svd(double_center(Xh), full_matrices=False)
    etas = [eta_squared(Uh[:, k], cvh) for k in range(min(5, Uh.shape[1]))]
    order = np.argsort(etas)[::-1][:3]
    return [pd.Series(Uh[:, k]).groupby(cvh).mean() for k in order], [etas[k] for k in order]


def ref_match(prof, ref):
    common = prof.index.intersection(ref.index)
    if len(common) < 4:
        return np.nan
    return abs(np.corrcoef(prof.loc[common].values, ref.loc[common].values)[0, 1])


hit1_top2, hit4_top2, hit1_top3, hit4_top3 = 0, 0, 0, 0
n_half = 0
for rep in range(200):
    idx = rng.permutation(n)
    A, B = idx[:n // 2], idx[n // 2:]
    for half in (A, B):
        profs, _ = half_profiles(half)
        r1s = [ref_match(p, m1) for p in profs]
        r4s = [ref_match(p, m4) for p in profs]
        n_half += 1
        if r1s[0] > 0.7:
            hit1_top2 += 1
        if r4s[0] > 0.7 or r4s[1] > 0.7:
            hit4_top2 += 1
        if max(r1s) > 0.7:
            hit1_top3 += 1
        if max(r4s) > 0.7:
            hit4_top3 += 1
print(f'  D1: top2中出现(与参照|相关|>0.7): {hit1_top2/n_half:.0%} | top3中: {hit1_top3/n_half:.0%}')
print(f'  D4: top2中出现: {hit4_top2/n_half:.0%} | top3中: {hit4_top3/n_half:.0%}')
print('  （top2含第1、第2；top3含前3。若D4的top3远高于top2 → 弱恢复源于方向排序竞争）')
