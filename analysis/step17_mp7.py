# -*- coding: utf-8 -*-
"""step17 七维主框架MP谱检验（论文正式口径）。

与step6完全同管线（双重中心化→SVD→列洗牌零分布1000次→Bonferroni→方向判定），
数据限制为七维主框架1198活跃题（BIG7=仁慈/普世主义/安全/成就/从众/自我导向/传统）。
权力/享乐/刺激（活跃题15/11/3）已在题库构造阶段剔除（2.1声明），不入任何分析。

输出: data/results/schwartz/mp_spectrum_test_7d.json + mp_spectrum_7d.png
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center, eta_squared

N_PERM = 1000
SEED = 42
BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']

X_all, meta_all, model_cols = load_matrices('schwartz')
mask = meta_all['core_value'].isin(BIG7).values
X = X_all[mask]
meta = meta_all.loc[mask].reset_index(drop=True)
n_items, n_models = X.shape
direction = meta['direction'].values
is_rev = (direction == 'negative').astype(float)
core_value = meta['core_value'].values
print(f'七维主框架: {n_items}题 × {n_models}模型观测 | 反向{int(is_rev.sum())} 正向{int((1-is_rev).sum())}')

rng = np.random.default_rng(SEED)

# ---- 1. 真实谱 ----
Xc = double_center(X)
U, s_real, Vt = np.linalg.svd(Xc, full_matrices=False)
k_nonzero = min(n_items, n_models) - 1
var_share = s_real ** 2 / (s_real ** 2).sum()
print(f'顶部5个方差占比: {np.round(var_share[:5] * 100, 1)}%')

# ---- 2. 列洗牌零分布 ----
print(f'置换零分布: {N_PERM}次（交互项Xc逐列洗牌）...')
null_spec = np.zeros((N_PERM, k_nonzero))
for b in range(N_PERM):
    Xp = Xc.copy()
    for j in range(n_models):
        Xp[:, j] = Xp[rng.permutation(n_items), j]
    null_spec[b] = np.linalg.svd(Xp, compute_uv=False)[:k_nonzero]
    if (b + 1) % 200 == 0:
        print(f'  {b + 1}/{N_PERM}')

null_med = np.median(null_spec, axis=0)
null_p95 = np.percentile(null_spec, 95, axis=0)
s_real_k = s_real[:k_nonzero]
p_vals = (null_spec >= s_real_k[None, :]).sum(axis=0) / N_PERM
bonf = 0.05 / k_nonzero
sig_flags = (p_vals < bonf) & (p_vals <= 1.0 / N_PERM + 1e-9)
n_sig = 0
for k in range(k_nonzero):
    if sig_flags[k]:
        n_sig += 1
    else:
        break
print(f'逐位置p值(Bonferroni α={bonf:.5f}): 顶部10个 p={np.round(p_vals[:10], 4)}')
print(f'连续显著顶部奇异值个数: {n_sig}')

# ---- 3. 方向解读 ----
from scipy.stats import kruskal, mannwhitneyu

interpretations = []
for k in range(n_sig):
    v_item = U[:, k]
    u_model = Vt[k]
    r_dir = float(np.corrcoef(v_item, is_rev)[0, 1])
    eta_cv = eta_squared(v_item, core_value)
    beta = np.polyfit(is_rev, v_item, 1)
    resid = v_item - np.polyval(beta, is_rev)
    eta_cv_resid = eta_squared(resid, core_value)
    groups_resid = [resid[core_value == c] for c in pd.unique(core_value)]
    p_cv_resid = float(kruskal(*groups_resid).pvalue)
    p_dir = float(mannwhitneyu(v_item[is_rev == 1], v_item[is_rev == 0]).pvalue)
    top_models = pd.Series(u_model, index=model_cols).abs().nlargest(3)
    dim_order = pd.Series(resid).groupby(core_value).mean().sort_values()
    verdict = ('方法效应(正反向)' if abs(r_dir) > 0.3 and p_dir < 0.001 else
               ('内容效应(7维)' if eta_cv_resid > 0.02 and p_cv_resid < 0.001 else '其他结构'))
    interpretations.append({
        'component': k + 1, 'singular_value': float(s_real[k]),
        'var_share_pct': float(var_share[k] * 100),
        'biserial_r_direction': r_dir, 'p_direction': p_dir,
        'eta2_core_value': eta_cv,
        'eta2_core_value_control_direction': eta_cv_resid,
        'p_core_value_control_direction': p_cv_resid,
        'top_model_loadings': {str(kk): float(vv) for kk, vv in top_models.items()},
        'dimension_ordering_full': {d: float(v) for d, v in dim_order.items()},
        'verdict': verdict})
    print(f'\n  方向{k + 1}: s={s_real[k]:.2f} (方差{var_share[k]*100:.1f}%)')
    print(f'    正反向r={r_dir:+.3f} (p={p_dir:.2e}) | 控制后η²={eta_cv_resid:.4f} (p={p_cv_resid:.2e})')
    if verdict == '内容效应(7维)':
        lo = dim_order.index[:3].tolist()
        hi = dim_order.index[-3:].tolist()
        interpretations[-1]['dimension_semantics'] = {'negative_pole': lo, 'positive_pole': hi}
        print(f'    语义轴: {lo} ←→ {hi}')
    print(f'    判定: {verdict} | 高负载: '
          f'{", ".join(f"{kk}({vv:+.2f})" for kk, vv in top_models.items())}')

content_sig = [i for i in interpretations if i['verdict'] == '内容效应(7维)']
method_sig = [i for i in interpretations if i['verdict'] == '方法效应(正反向)']
conclusion = (f'七维框架下检出{len(content_sig)}个内容效应方向 + {len(method_sig)}个方法效应方向'
              f'（顶部{n_sig}个方向显著超列洗牌零分布）')
print(f'\n===== 结论 =====\n{conclusion}')

# ---- 4. 落盘+图 ----
base = config.DATA_DIR / 'results' / 'schwartz'
out = {
    'data': {'n_items': int(n_items), 'n_models': int(n_models),
             'n_reverse': int(is_rev.sum()), 'n_positive': int((1 - is_rev).sum()),
             'framework': 'BIG7（权力/享乐/刺激已于题库构造阶段剔除）'},
    'params': {'n_perm': N_PERM, 'seed': SEED, 'bonferroni_alpha': bonf},
    'spectrum_real': s_real[:k_nonzero].tolist(),
    'spectrum_null_median': null_med[:k_nonzero].tolist(),
    'spectrum_null_p95': null_p95[:k_nonzero].tolist(),
    'p_values': p_vals[:k_nonzero].tolist(),
    'n_significant_top': int(n_sig),
    'interpretations': interpretations,
    'conclusion': conclusion,
}
with open(base / 'mp_spectrum_test_7d.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
                     'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(9, 5.5))
ks = np.arange(1, k_nonzero + 1)
ax.fill_between(ks, null_med, null_p95, alpha=0.2, color='gray', label='零分布中位数~95分位（列洗牌）')
ax.plot(ks, null_med, 'k--', lw=1, label='零分布中位数')
ax.plot(ks, s_real_k, 'o-', color='#1f77b4', ms=4, lw=1.5, label='真实交互项谱')
if n_sig > 0:
    ax.plot(ks[:n_sig], s_real_k[:n_sig], 'o', color='red', ms=9, mfc='none', mew=2,
            label=f'显著超噪声（前{n_sig}个）')
ax.set_xlabel('奇异值序号（交互项谱）')
ax.set_ylabel('奇异值')
ax.set_title(f'交互项噪声等价性检验（MP谱检验）：真实谱 vs 列洗牌零分布\n'
             f'（七维主框架 {n_items}题×{n_models}模型观测，双重中心化后）')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(base / 'mp_spectrum_7d.png', dpi=300)
print(f'\n已保存: mp_spectrum_test_7d.json / mp_spectrum_7d.png')
