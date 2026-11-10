# -*- coding: utf-8 -*-
"""补跑实验①②（spec第4节）：改稿前的数据准备。

实验① 3.3.1块结构区间估计：交互残差题目间同维/异维平均相关、Δr及95%CI
       （7维主口径1198题 + 10维稳健口径1227题，题目层bootstrap）
实验② 5.2合并轴模型坐标：22观测在自主—秩序轴的z分数、推理开/关位移、
       on/off子样本独立轴与全轴的Spearman、厂商分布、合并轴方差占比
输出: data/results/schwartz/paperB_patch_experiments.json
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
n_items, n_models = X.shape
Xc = double_center(X)

BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
out = {}

# ================= 实验① 块结构区间估计 =================
print('=' * 70)
print('[实验①] 块结构区间估计（同维/异维平均相关、Δr及95%CI）')


def block_stats(Xc_sub, cv_sub, label, n_boot=500):
    """题目间相关矩阵 → 同维/异维平均相关 → Δr bootstrap CI（题目层重采样）。"""
    n = len(cv_sub)
    R = np.corrcoef(Xc_sub)                     # 题目×题目Pearson相关
    iu = np.triu_indices(n, 1)
    same = cv_sub[iu[0]] == cv_sub[iu[1]]
    r_same = R[iu][same].mean()
    r_diff = R[iu][~same].mean()
    # bootstrap：题目层重采样（每维内重采样，保持维度结构），重算Δr
    deltas = []
    dims = np.unique(cv_sub)
    pools = {d: np.where(cv_sub == d)[0] for d in dims}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pools[d], len(pools[d]), replace=True) for d in dims])
        cvb = cv_sub[idx]
        Rb = R[np.ix_(idx, idx)]
        iub = np.triu_indices(len(idx), 1)
        sb = cvb[iub[0]] == cvb[iub[1]]
        d = Rb[iub][sb].mean() - Rb[iub][~sb].mean()
        deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f'  [{label}] n={n} | 同维平均r={r_same:.3f} 异维={r_diff:.3f} '
          f'Δr={r_same-r_diff:.3f} 95%CI=[{lo:.3f}, {hi:.3f}]')
    return {'n_items': int(n), 'r_same': float(r_same), 'r_diff': float(r_diff),
            'delta_r': float(r_same - r_diff), 'ci95': [float(lo), float(hi)]}


mask7 = np.isin(cv, BIG7)
out['exp1_block_structure'] = {
    'big7': block_stats(Xc[mask7], cv[mask7], '7维主口径'),
    'dim10': block_stats(Xc, cv, '10维全量'),
}

# ================= 实验② 合并轴模型坐标 =================
print('=' * 70)
print('[实验②] 自主—秩序合并轴：模型坐标与推理形态')

X7, cv7, dir7 = X[mask7], cv[mask7], meta['direction'].values[mask7]
is_rev7 = (dir7 == 'negative').astype(float)
n7 = len(cv7)
U7, s7, Vt7 = np.linalg.svd(double_center(X7), full_matrices=False)
var7 = s7 ** 2 / (s7 ** 2).sum()

# 合并轴（与step7c完全同构：方向1+方向4等权PC1，符号规约=自主端负/秩序端正）
profs = [pd.Series(U7[:, k]).groupby(cv7).mean() for k in (0, 3)]
zs = [(p - p.mean()) / p.std() for p in profs]
L = np.column_stack([zs[0].values, zs[1].values])
_, _, VtL = np.linalg.svd(L - L.mean(axis=0), full_matrices=False)
w = VtL[0]
v_m = w[0] * U7[:, 0] + w[1] * U7[:, 3]
v_m /= np.linalg.norm(v_m)
# 模型空间载荷 = 同一组合作用于V矩阵列，符号随题目端同步
u_m = w[0] * Vt7[0, :] + w[1] * Vt7[3, :]
m = pd.Series(v_m).groupby(cv7).mean()
if m['传统'] < m['自我导向']:        # 统一规约：自主端负、秩序端正
    v_m, u_m = -v_m, -u_m
    m = -m

# 合并轴方差占比：正交方向能量加权和（w0²+w1²=1）
var_merge = w[0] ** 2 * var7[0] + w[1] ** 2 * var7[3]
eta_m = eta_squared(v_m, cv7)
beta = np.polyfit(is_rev7, v_m, 1)
eta_m_ctrl = eta_squared(v_m - np.polyval(beta, is_rev7), cv7)
inter_share = 0.218  # 3.3.3节交互成分占总方差
print(f'  合并轴: 交互方差占比={var_merge*100:.1f}% (方向1 {var7[0]*100:.1f}% + 方向4 '
      f'{var7[3]*100:.1f}% 等权) | 折合总方差≈{var_merge*inter_share*100:.2f}%')
print(f'  η²={eta_m:.4f} 控制正反向后={eta_m_ctrl:.4f}')
print(f'  维度排序（自主端→秩序端）: {" < ".join(m.sort_values().index)}')

# 22观测z分数（口径与旧文一致：loading×√n，模型列单位范数）
z = pd.Series(u_m * np.sqrt(n_models), index=cols).sort_values()
print('\n  模型观测z分数（自主端←→秩序端）:')
for c, v in z.items():
    print(f'    {c:28s} {v:+.2f}')

# 推理位移（同一模型on/off的z差；正值=向秩序端移动）
bases = sorted({c[:-3] for c in cols if c.endswith('_on')})
displ = {b: float(z[b + '_on'] - z[b + '_off']) for b in bases
         if (b + '_on') in z.index and (b + '_off') in z.index}
print('\n  推理开启位移（z_on − z_off，正值=向秩序端移动）:')
for b, d in sorted(displ.items(), key=lambda kv: -abs(kv[1])):
    print(f'    {b:24s} {d:+.2f} (off {z[b+"_off"]:+.2f} → on {z[b+"_on"]:+.2f})')

# on/off子样本独立轴与全轴的一致性（7维口径，控制正反向后η²选方向+合并）
def subset_axis(idxs, label):
    Xs, cvs = X7[:, idxs], cv7
    is_rev_s = is_rev7  # 题目维度不变
    U_s, s_s, Vt_s = np.linalg.svd(double_center(Xs), full_matrices=False)
    etas = []
    for k in range(min(5, U_s.shape[1])):
        v = U_s[:, k]
        b_ = np.polyfit(is_rev_s, v, 1)
        etas.append(eta_squared(v - np.polyval(b_, is_rev_s), cvs))
    order = np.argsort(etas)[::-1]
    content = [k for k in order if etas[k] > 0.02][:2]
    if len(content) == 1:      # 内容方向不足2个时退化用单方向（仍可与全轴比ρ）
        k = content[0]
        mm = pd.Series(U_s[:, k]).groupby(cvs).mean()
        if mm['传统'] < mm['自我导向']:
            mm = -mm
        rho = spearmanr(mm.values, m.loc[mm.index].values).statistic
        print(f'  [{label}] 仅1个内容方向(η²={etas[k]:.4f})，单方向与全轴 ρ={rho:+.3f}')
        return None, {'rho_single': float(rho), 'eta_single': float(etas[k]),
                      'etas_ctrl_top5': [float(e) for e in etas[:5]]}
    if len(content) == 0:
        print(f'  [{label}] 无内容方向: etas={np.round(etas[:5], 4)}')
        return None, {'note': '无内容方向', 'etas_ctrl_top5': [float(e) for e in etas[:5]]}
    ps = [pd.Series(U_s[:, k]).groupby(cvs).mean() for k in content]
    zs_ = [(p - p.mean()) / p.std() for p in ps]
    L_ = np.column_stack([zs_[0].values, zs_[1].values])
    _, _, Vt_ = np.linalg.svd(L_ - L_.mean(axis=0), full_matrices=False)
    w_ = Vt_[0]
    vm_ = w_[0] * U_s[:, content[0]] + w_[1] * U_s[:, content[1]]
    vm_ /= np.linalg.norm(vm_)
    mm = pd.Series(vm_).groupby(cvs).mean()
    if mm['传统'] < mm['自我导向']:
        mm = -mm
    return mm, etas


from scipy.stats import spearmanr
on_idx = [i for i, c in enumerate(cols) if c.endswith('_on')]
off_idx = [i for i, c in enumerate(cols) if c.endswith('_off')]
sub_out = {}
for idxs, lab in [(off_idx, 'off_only'), (on_idx, 'on_only')]:
    mm, extra = subset_axis(idxs, lab)
    if mm is None:
        sub_out[lab] = extra
        continue
    rho = spearmanr(mm.values, m.loc[mm.index].values).statistic
    print(f'  [{lab}] 独立合并轴与全轴维度排序Spearman ρ={rho:+.3f}')
    sub_out[lab] = {'rho_vs_full': float(rho),
                    'ordering': list(mm.sort_values().index),
                    'etas_ctrl_top5': extra}
if 'off_only' in sub_out and 'on_only' in sub_out and 'ordering' in sub_out.get('off_only', {}) \
        and 'ordering' in sub_out.get('on_only', {}):
    a = sub_out['off_only']['ordering']
    b = sub_out['on_only']['ordering']
    sub_out['off_vs_on_rho'] = float(abs(spearmanr(a, b).statistic))

# 汇总
out['exp2_merged_axis_models'] = {
    'var_share_interaction_pct': float(var_merge * 100),
    'var_share_total_pct': float(var_merge * inter_share * 100),
    'eta2': float(eta_m), 'eta2_control_direction': float(eta_m_ctrl),
    'dim_ordering_autonomy_to_order': list(m.sort_values().index),
    'model_z': {c: float(v) for c, v in z.items()},
    'reasoning_displacement': {b: float(d) for b, d in displ.items()},
    'subset_checks': sub_out,
    'w': [float(w[0]), float(w[1])],
}

BASE = config.DATA_DIR / 'results' / 'schwartz'
with open(BASE / 'paperB_patch_experiments.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n已保存: {BASE / "paperB_patch_experiments.json"}')
