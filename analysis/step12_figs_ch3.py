# -*- coding: utf-8 -*-
"""第3章新增两图（2026-11-03，用户裁决）：
图2 交互残差的题目间相关矩阵热图（题目按预设12维分组排列）→ fig_item_corr_heatmap.png
图3 平均轮廓系数随聚类数K的变化曲线（K=2~100）→ fig_silhouette_k_curve.png
协议与step2_clustering.dist_1_minus_corr完全一致：行中心化+行归一化的Pearson相关。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

res = Path(config.DATA_DIR) / "results" / "agree"
FIG_DIR = res / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})
DIMS = ["富强", "民主", "文明", "和谐", "自由", "平等",
        "公正", "法治", "爱国", "敬业", "诚信", "友善"]


def _title(fig, zh, en):
    fig.suptitle(f"{zh}\n{en}", fontsize=13, fontweight="bold", y=0.99)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================== 图2 相关矩阵热图 ==============================
def fig_corr_heatmap():
    active = pd.read_csv(res / "active_items.csv")
    cent = pd.read_csv(res / "centered_orig.csv", index_col=0).loc[active["rule_id"]]
    X = np.nan_to_num(cent.values.astype(float))

    # 与dist_1_minus_corr一致的相关定义：行中心化+行归一化后取内积
    xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(xc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    xc = xc / norm
    corr = xc @ xc.T

    # 按预设12维分组重排（组内保持原序）
    cv = active.set_index("rule_id").loc[cent.index, "core_value"].values
    order = np.concatenate([np.where(cv == d)[0] for d in DIMS])
    C = corr[np.ix_(order, order)]
    bounds = np.cumsum([0] + [int((cv == d).sum()) for d in DIMS])

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   interpolation="nearest", rasterized=True)
    # 维度分界白线
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.9)
        ax.axvline(b - 0.5, color="white", lw=0.9)
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(12)]
    ax.set_xticks(centers)
    ax.set_xticklabels(DIMS, fontsize=9, rotation=90)
    ax.set_yticks(centers)
    ax.set_yticklabels(DIMS, fontsize=9)
    ax.set_xlabel("题目（按预设价值维度分组排列）/ Items grouped by presupposed dimension", fontsize=10)
    ax.set_ylabel("题目（按预设价值维度分组排列）/ Items grouped by presupposed dimension", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("题目间Pearson相关 / Item-wise Pearson correlation", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    _title(fig, "图2 交互残差的题目间相关矩阵（题目按预设12维分组排列）",
           "Fig. 2 Item-by-item correlation matrix of interaction residuals "
           "(items grouped by the presupposed 12 dimensions)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig_item_corr_heatmap")
    print("fig_item_corr_heatmap 完成")


# ============================== 图3 轮廓系数-K曲线 ==============================
def fig_sil_curve():
    sil = pd.read_csv(res / "check_sil_k100_orig.csv")
    k = sil["k"].values
    s = sil["silhouette"].values

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(k, s, color="#3A6EA5", lw=2.0, zorder=3)
    # 假设维度数参考线
    ax.axvline(12, color="#C0392B", lw=1.2, ls="--", zorder=2)
    ax.text(12.8, s.max() * 0.30, "K=12（假设维度数）", fontsize=10,
            color="#C0392B", ha="left")
    # 端点数值标注
    ax.annotate(f"{s[0]:.2f}", xy=(k[0], s[0]), xytext=(k[0] + 1.5, s[0] - 0.022),
                fontsize=10, color="#333333")
    ax.annotate(f"{s[-1]:.2f}", xy=(k[-1], s[-1]), xytext=(k[-1] - 8, s[-1] + 0.008),
                fontsize=10, color="#333333", ha="right")
    ax.set_xlabel("聚类数 K / Number of clusters", fontsize=11)
    ax.set_ylabel("平均轮廓系数 / Average silhouette coefficient", fontsize=11)
    ax.set_xlim(0, 102)
    ax.set_ylim(0.15, 0.50)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, "图3 平均轮廓系数随聚类数K的变化（K=2~100）",
           "Fig. 3 Average silhouette coefficient versus number of clusters K (K=2–100)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "fig_silhouette_k_curve")
    print("fig_silhouette_k_curve 完成")


if __name__ == "__main__":
    fig_corr_heatmap()
    fig_sil_curve()
    print("全部完成 →", FIG_DIR)
