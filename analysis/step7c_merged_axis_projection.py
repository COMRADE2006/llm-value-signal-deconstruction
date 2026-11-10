# -*- coding: utf-8 -*-
"""论文正式图：7维主框架在合并语义轴上的投影（旧图6同款森林图版式）。

合并轴 = 7维子集SVD方向1与方向4的PC1等权组合（w=±0.707，见merge_test.json），
成长端（自我导向·仁慈）←→ 保守端（传统·安全），控制正反向后η²=0.0565，
置换p=0.0，对半恢复中位0.916，符号保持100%维度：自我导向/仁慈/安全/传统。

红色 = bootstrap 95%CI不含零的确凿锚点。仅7个BIG7维度（权力/享乐/刺激已剔除）。
输出: data/results/schwartz/figures/fig_merged_axis_projection.png/pdf + merged_axis_dim_ci.json
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center, eta_squared

BASE = config.DATA_DIR / "results" / "schwartz"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})
C_SIG, C_NS = "#C44E52", "#5B7DB1"

# ---- 数据：7维子集 + 合并轴 ----
X, meta, cols = load_matrices("schwartz")
cv = meta["core_value"].values
BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
mask7 = np.isin(cv, BIG7)
X7, cv7 = X[mask7], cv[mask7]
is_rev7 = (meta["direction"].values[mask7] == "negative").astype(float)
n7 = len(cv7)
dims7 = list(pd.unique(cv7))
counts = pd.Series(cv7).value_counts()

# 7维SVD：方向1与方向4是两条内容方向（r=1.000/0.998复现全数据D1/D4）
U7, s7, _ = np.linalg.svd(double_center(X7), full_matrices=False)
profs = [pd.Series(U7[:, k]).groupby(cv7).mean() for k in (0, 3)]
zs = [(p - p.mean()) / p.std() for p in profs]
L = np.column_stack([zs[0].values, zs[1].values])
_, _, Vt = np.linalg.svd(L - L.mean(axis=0), full_matrices=False)
w = Vt[0]
v_merge = w[0] * U7[:, 0] + w[1] * U7[:, 3]
v_merge /= np.linalg.norm(v_merge)

# 符号规约：让"成长端为负、保守端为正"（自我导向最负、传统最正）
m = pd.Series(v_merge).groupby(cv7).mean()
if m["传统"] < m["自我导向"]:
    v_merge = -v_merge
    m = -m

# 控制正反向后的η²（口径与step6一致）
beta = np.polyfit(is_rev7, v_merge, 1)
eta_ctrl = eta_squared(v_merge - np.polyval(beta, is_rev7), cv7)
dim_mean = m.sort_values()
print(f"合并轴控制后η²={eta_ctrl:.4f} | 排序: {' < '.join(dim_mean.index)}")

# ---- 分层bootstrap 1000次 → 95%CI ----
rng = np.random.default_rng(20261104)
pools = {d: np.where(cv7 == d)[0] for d in dims7}
boot = {d: [] for d in dims7}
for _ in range(1000):
    b = np.concatenate([rng.choice(pools[d], size=len(pools[d]), replace=True) for d in dims7])
    mb = pd.Series(v_merge[b]).groupby(cv7[b]).mean()
    for d in dims7:
        boot[d].append(mb[d])
lo = {d: np.percentile(boot[d], 2.5) for d in dims7}
hi = {d: np.percentile(boot[d], 97.5) for d in dims7}
ci_excl_zero = {d: (lo[d] > 0 or hi[d] < 0) for d in dims7}
print("CI不含零（确凿锚点）:", [d for d, t in ci_excl_zero.items() if t])

# ---- 森林图（旧图6版式） ----
order = dim_mean.index.tolist()
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7.2, 5.4))
xr = dim_mean.max() - dim_mean.mean()
for yi, d in zip(y, order):
    anchor = ci_excl_zero[d]
    ax.errorbar(dim_mean[d], yi,
                xerr=[[dim_mean[d] - lo[d]], [hi[d] - dim_mean[d]]],
                fmt="o", ms=9 if anchor else 6,
                color=C_SIG if anchor else C_NS,
                ecolor=C_SIG if anchor else "#9BB0CC",
                elinewidth=2.2 if anchor else 1.2, capsize=4, zorder=3)
    mid = (dim_mean.min() + dim_mean.max()) / 2
    off = 0.06 * xr if dim_mean[d] >= mid else -0.06 * xr
    ax.text(dim_mean[d] + off, yi, d, ha="left" if off > 0 else "right",
            va="center", fontsize=12.5,
            fontweight="bold" if anchor else "normal",
            color=C_SIG if anchor else "#333333")
ax.axvline(0, color="black", lw=1.0, alpha=0.7)
ax.set_yticks(y)
ax.set_yticklabels([f"{d} (n={counts[d]})" for d in order], fontsize=11)
ax.set_xlabel("维度在语义轴上的投影（均值载荷，误差棒=bootstrap 95%CI）\n"
              "Axis projection (mean loading; error bars = bootstrap 95% CI)", fontsize=10.5)
ax.set_xlim(dim_mean.min() - 0.30 * xr, dim_mean.max() + 0.30 * xr)
ax.spines[["top", "right"]].set_visible(False)
ax.text(dim_mean.min() - 0.24 * xr, len(order) - 0.4, "◀ 成长端（自我导向·仁慈）",
        fontsize=11, color=C_SIG, fontweight="bold")
ax.text(dim_mean.max() + 0.24 * xr, 0.4, "保守端（传统·安全）▶",
        fontsize=11, color=C_SIG, fontweight="bold", ha="right")
fig.suptitle("图4 7个价值观维度在合并语义轴上的投影（红色=95%CI不含零的确凿锚点）\n"
             "Fig. 4 Projection of 7 value dimensions onto the merged semantic axis "
             "(red = 95% CI excludes zero)",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig_merged_axis_projection.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[已保存] figures/fig_merged_axis_projection.png/.pdf")

# ---- CI落盘 ----
ci_out = {d: {"mean": float(dim_mean[d]), "ci95": [float(lo[d]), float(hi[d])],
              "anchor": bool(ci_excl_zero[d]), "n_items": int(counts[d])} for d in order}
ci_out["_meta"] = {"eta2_control_direction": float(eta_ctrl),
                   "ordering_neg_to_pos": order,
                   "w_direction1": float(abs(w[0])), "w_direction4": float(abs(w[1])),
                   "pole_neg": "成长端（自我导向·仁慈）", "pole_pos": "保守端（传统·安全）"}
with open(BASE / "merged_axis_dim_ci.json", "w", encoding="utf-8") as f:
    json.dump(ci_out, f, ensure_ascii=False, indent=2)
print("[已保存] merged_axis_dim_ci.json")
