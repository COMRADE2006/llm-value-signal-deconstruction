# -*- coding: utf-8 -*-
"""双轴投影图：10个Schwartz维度在两条内容语义轴上的位置（旧图6的两面板版）。

轴A = MP方向4（主内容轴，η²=0.040）：传统/安全/成就 ←→ 仁慈/享乐/自我导向
轴B = MP方向1（方差最大轴，η²=0.039）：权力/安全/传统 ←→ 成就/自我导向/仁慈
红色 = bootstrap 95%CI不含零的确凿锚点；y轴标签括号内为该维度活跃题数，
稀少维度（权力15/享乐11/刺激3，非主框架BIG7）以灰色显示。

输出: data/results/schwartz/figures/fig_two_axis_projection.png/pdf + two_axis_dim_ci.json
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
from analysis.step6_mp_spectrum import double_center

BASE = config.DATA_DIR / "results" / "schwartz"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})
C_SIG, C_NS = "#C44E52", "#5B7DB1"   # 确凿锚点/未确证
BIG7 = {'仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统'}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] figures/{name}.png/.pdf")


# ---- 数据与两条轴 ----
X, meta, cols = load_matrices("schwartz")
cv = meta["core_value"].values
n_items = len(cv)
dims = pd.unique(cv)
counts = pd.Series(cv).value_counts()

U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
AXES = [("A", U[:, 3], "轴A 主内容方向（保守↔成长，MP方向4）",
         "◀ 保守端（传统·安全）", "成长端（自我导向·仁慈·享乐）▶"),
        ("B", U[:, 0], "轴B 方差最大方向（MP方向1）",
         "◀ 权力·安全·传统端", "成就·自我导向·仁慈端▶")]

# ---- 分层bootstrap 1000次 → 每轴每维95%CI ----
rng = np.random.default_rng(20261104)
pools = {d: np.where(cv == d)[0] for d in dims}
boot = {a[0]: {d: [] for d in dims} for a in AXES}
for _ in range(1000):
    b = np.concatenate([rng.choice(pools[d], size=len(pools[d]), replace=True) for d in dims])
    cvb = cv[b]
    for tag, v, *_ in AXES:
        m = pd.Series(v[b]).groupby(cvb).mean()
        for d in dims:
            boot[tag][d].append(m[d])

# ---- 双面板森林图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.6))
ci_out = {}
for ax, (tag, v, subtitle, pole_neg, pole_pos) in zip(axs, AXES):
    dim_mean = pd.Series(v).groupby(cv).mean().sort_values()
    order = dim_mean.index.tolist()
    y = np.arange(len(order))
    xr = dim_mean.max() - dim_mean.min()
    for yi, d in zip(y, order):
        lo, hi = np.percentile(boot[tag][d], [2.5, 97.5])
        anchor = (lo > 0 or hi < 0)
        big7 = d in BIG7
        ax.errorbar(dim_mean[d], yi,
                    xerr=[[dim_mean[d] - lo], [hi - dim_mean[d]]],
                    fmt="o", ms=9 if anchor else 6,
                    color=C_SIG if anchor else C_NS,
                    ecolor=C_SIG if anchor else "#9BB0CC",
                    elinewidth=2.2 if anchor else 1.2, capsize=4, zorder=3)
        off = 0.025 * xr if dim_mean[d] >= (dim_mean.min() + dim_mean.max()) / 2 else -0.025 * xr
        ax.text(dim_mean[d] + off, yi, d, ha="left" if off > 0 else "right",
                va="center", fontsize=12.5,
                fontweight="bold" if anchor else "normal",
                color=C_SIG if anchor else "#333333")
        ci_out[f"{tag}:{d}"] = {"mean": float(dim_mean[d]),
                                "ci95": [float(lo), float(hi)],
                                "anchor": bool(anchor), "n_items": int(counts[d])}
    ax.axvline(0, color="black", lw=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} (n={counts[d]})" for d in order], fontsize=10.5)
    for tick, d in zip(ax.get_yticklabels(), order):
        if d not in BIG7:
            tick.set_color("#999999")
    ax.set_xlabel("维度在轴上的投影（均值载荷，误差棒=bootstrap 95%CI）\n"
                  "Axis projection (mean loading; error bars = bootstrap 95% CI)", fontsize=10)
    m0, m1 = dim_mean.min(), dim_mean.max()
    ax.set_xlim(m0 - 0.17 * xr, m1 + 0.17 * xr)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=10)
    ax.text(m0 - 0.13 * xr, len(order) - 0.4, pole_neg, fontsize=10.5,
            color=C_SIG, fontweight="bold")
    ax.text(m1 + 0.13 * xr, 0.4, pole_pos, fontsize=10.5,
            color=C_SIG, fontweight="bold", ha="right")

fig.suptitle("图 10个Schwartz价值观维度在两条内容语义轴上的投影（红色=95%CI不含零的确凿锚点；"
             "灰色=稀少维度，非7维主框架）\n"
             "Fig. Projection of 10 Schwartz value dimensions onto the two content semantic axes "
             "(red = 95% CI excludes zero; gray = rare dimensions outside the 7-dim framework)",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "fig_two_axis_projection")

with open(BASE / "two_axis_dim_ci.json", "w", encoding="utf-8") as f:
    json.dump(ci_out, f, ensure_ascii=False, indent=2)
print("[已保存] two_axis_dim_ci.json")

# ---- 文字表（答复用） ----
print("\n=== 各维度在两轴上的位置（按轴A排序，负=左端 正=右端）===")
tab = pd.DataFrame({f"轴{t}": pd.Series(v).groupby(cv).mean() for t, v, *_ in AXES})
tab = tab.reindex(columns=["轴A", "轴B"]).sort_values("轴A")
tab["n"] = counts
ztab = (tab[["轴A", "轴B"]] - tab[["轴A", "轴B"]].mean()) / tab[["轴A", "轴B"]].std()
print(pd.concat([tab, ztab.add_suffix("(z)")], axis=1).round(4).to_string())
