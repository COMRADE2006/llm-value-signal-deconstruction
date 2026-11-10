# -*- coding: utf-8 -*-
"""诊断图：七维框架下方向1 / 方向4 / 合并轴的直观对比（回答"为什么有两个内容方向"）。

2×2面板：
  (a) 方向1维度剖面（控制正反向后载荷均值，z标准化）
  (b) 方向4维度剖面
  (c) 合并轴（等权PC1）维度剖面
  (d) D1×D4平面落点 + 合并方向箭头——成就/从众互换象限直观可见

着色规则：成就(橙)/从众(紫)=两方向间互换的维度；安全/传统(红)=秩序锚点；
自我导向/仁慈(蓝)=自主锚点；普世主义(灰)=中段。
输出: figures/diag_d1_d4_merged.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from step5_ega import load_matrices
from step6_mp_spectrum import double_center, eta_squared

BASE = config.DATA_DIR / "results" / "schwartz"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                     "axes.unicode_minus": False})

BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
COLORS = {"成就": "#E08214", "从众": "#7B52AB",
          "安全": "#C44E52", "传统": "#C44E52",
          "自我导向": "#2E7EBB", "仁慈": "#2E7EBB",
          "普世主义": "#888888"}

X, meta, cols = load_matrices("schwartz")
mask = meta["core_value"].isin(BIG7).values
X7, cv7 = X[mask], meta["core_value"].values[mask]
is_rev7 = (meta["direction"].values[mask] == "negative").astype(float)

U, s, Vt = np.linalg.svd(double_center(X7), full_matrices=False)
var = s ** 2 / (s ** 2).sum()


def orient(v):
    """符号规约：自我导向为负（自主端）。"""
    m = pd.Series(v).groupby(cv7).mean()
    return -v if m["自我导向"] > 0 else v


def ctrl(v):
    """控制正反向（回归去除方向后残差载荷，与step17判读同口径）。"""
    b = np.polyfit(is_rev7, v, 1)
    return v - np.polyval(b, is_rev7)


v1, v4 = orient(U[:, 0].copy()), orient(U[:, 3].copy())
m1 = pd.Series(ctrl(v1)).groupby(cv7).mean()
m4 = pd.Series(ctrl(v4)).groupby(cv7).mean()
eta1, eta4 = eta_squared(ctrl(v1), cv7), eta_squared(ctrl(v4), cv7)

# 合并轴：维度落点z平面的PC1（与step7c/16同协议）
z1 = (m1 - m1.mean()) / m1.std()
z4 = (m4 - m4.mean()) / m4.std()
L = np.column_stack([z1.values, z4.values])
_, _, VtL = np.linalg.svd(L - L.mean(axis=0), full_matrices=False)
w = VtL[0]
if w.sum() < 0:
    w = -w                        # 合并方向指向秩序象限（右上）
vm = w[0] * v1 + w[1] * v4
vm /= np.linalg.norm(vm)
mm = pd.Series(vm).groupby(cv7).mean()
if mm["传统"] < mm["自我导向"]:
    vm, mm = -vm, -mm
eta_m = eta_squared(ctrl(vm), cv7)
var_m = w[0] ** 2 * var[0] + w[1] ** 2 * var[3]

# 打印数字表
zm = (mm - mm.mean()) / mm.std()
tab = pd.DataFrame({"方向1(z)": z1, "方向4(z)": z4, "合并轴(z)": zm}).loc[BIG7]
print("维度落点（z标准化；方向1/4为控制正反向后载荷，合并轴为原始终载）：")
print(tab.round(3).to_string())
print(f"\n方向1: 交互方差{var[0]*100:.1f}% η²_ctrl={eta1:.4f}")
print(f"方向4: 交互方差{var[3]*100:.1f}% η²_ctrl={eta4:.4f}")
print(f"合并轴: 交互方差{var_m*100:.1f}% η²_ctrl={eta_m:.4f} | PC1权重 w=({w[0]:+.3f}, {w[1]:+.3f})")

# ---------- 画图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

panels = [(m1, f"(a) 方向1：交互方差{var[0]*100:.1f}%，η²={eta1:.3f}"),
          (m4, f"(b) 方向4：交互方差{var[3]*100:.1f}%，η²={eta4:.3f}"),
          (mm, f"(c) 合并轴：交互方差{var_m*100:.1f}%，η²={eta_m:.3f}")]
for ax, (means, title) in zip(axes.flat[:3], panels):
    z = (means - means.mean()) / means.std()
    order = z.sort_values().index
    for yi, d in enumerate(order):
        ax.scatter(z[d], yi, s=140, color=COLORS[d], zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.text(z[d], yi + 0.32, d, ha="center", fontsize=11,
                color=COLORS[d], fontweight="bold")
    ax.axvline(0, color="black", lw=1, alpha=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=11)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlim(-2.6, 2.6)
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
for lbl_ax in axes.flat[:3]:
    lbl_ax.set_xlabel("← 自主端        秩序端 →  (z载荷)", fontsize=10)

# (d) D1×D4平面
ax = axes.flat[3]
for d in BIG7:
    ax.scatter(z1[d], z4[d], s=170, color=COLORS[d], zorder=3,
               edgecolor="white", linewidth=1.2)
    off = (0.12, 0.12) if d != "成就" else (0.12, -0.22)
    ax.text(z1[d] + off[0], z4[d] + off[1], d, fontsize=11.5,
            color=COLORS[d], fontweight="bold")
arr = np.array([w[0], w[1]])
arr = arr / np.linalg.norm(arr) * 1.9
ax.annotate("", xy=arr, xytext=-arr,
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#333333", alpha=0.85))
ax.text(arr[0] * 0.98, arr[1] * 0.80, "合并方向\n(两方向的公共轴)", fontsize=10.5,
        ha="center", color="#333333")
ax.axhline(0, color="black", lw=0.8, alpha=0.4)
ax.axvline(0, color="black", lw=0.8, alpha=0.4)
ax.set_xlabel("方向1维度落点(z)  ←自主 · 秩序→", fontsize=10.5)
ax.set_ylabel("方向4维度落点(z)  ←自主 · 秩序→", fontsize=10.5)
ax.set_title("(d) 两个方向的维度落点平面：成就与从众互换象限，四锚点同侧",
             fontsize=12.5, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-2.4, 2.4)

fig.suptitle("七维主框架：方向1与方向4的直观对比与合并（诊断图）", fontsize=14,
             fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "diag_d1_d4_merged.png", dpi=200, bbox_inches="tight")
print(f"\n已保存: {FIG / 'diag_d1_d4_merged.png'}")
