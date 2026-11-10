# -*- coding: utf-8 -*-
"""论文B七维口径图系重画（spec v3，2026-11-05）。

覆盖图1/2/3/4/7/8（图5=mp_spectrum_7d.png、图6=fig_merged_axis_projection、
附录诊断图=diag_d1_d4_merged 已有，本脚本不重画）。

图1 题库构成：七维2345题×正反向（final_items_v3过滤BIG7）
图2 相关矩阵热图：活跃1198题按七维分组（centered_orig）
图3 轮廓系数曲线：K=2~100（check_sil_k100_orig.csv七维版），K=7参考线
图4 方差解剖饼图：七维活跃1198题重算（75.x/3.x/21.x）
图7 轴验证：V1合并轴标签置换零分布(1000次) + V3符号保持率（merge_test.json）
图8 模型光谱：22观测合并轴z分数（paperB_patch_experiments.json exp2）

输出: data/results/schwartz/figures/fig{1,2,3,4,7,8}_*.png/pdf
用法: python analysis/step19_figs_7d.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center, eta_squared
from analysis.step4_decouple import variance_budget

BASE = config.DATA_DIR / "results" / "schwartz"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})
C_ITEM, C_MODEL, C_INTER = "#4C72B0", "#DD8452", "#55A868"
C_SIG, C_NS = "#C44E52", "#5B7DB1"
C_NULL_BAND = "#B0B0B0"
BIG7 = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']
# 图1维度顺序：按题量降序
BIG7_BY_SIZE = ['仁慈', '普世主义', '安全', '成就', '从众', '自我导向', '传统']


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] figures/{name}.png/.pdf")


def _title(fig, zh, en):
    fig.suptitle(f"{zh}\n{en}", fontsize=13, fontweight="bold", y=0.99)


# ============================== 图1 题库构成（七维×正反向，N=2345） ==============================
def fig1_items_distribution():
    items = [json.loads(l) for l in open(config.DATA_DIR / "final_items_v3.jsonl",
                                         encoding="utf-8")]
    items = [it for it in items if it["core_value"] in BIG7]
    dims = BIG7_BY_SIZE
    pos = [sum(1 for it in items if it["core_value"] == d and it["direction"] == "positive")
           for d in dims]
    neg = [sum(1 for it in items if it["core_value"] == d and it["direction"] == "negative")
           for d in dims]
    total = sum(pos) + sum(neg)
    print(f"图1: 七维题库共{total}题（应为2345）")

    x = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, pos, width=0.62, label="正向题", color=C_ITEM,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, neg, width=0.62, bottom=pos, label="负向题", color=C_SIG,
           edgecolor="black", linewidth=0.5)
    for xi, (p, n) in enumerate(zip(pos, neg)):
        ax.text(xi, p / 2, str(p), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        if n >= 30:
            ax.text(xi, p + n / 2, str(n), ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        ax.text(xi, p + n + 6, str(p + n), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylabel("题数 / Number of items", fontsize=11)
    ax.set_ylim(0, 760)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              fontsize=10.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, f"图1 题库的七维×正反向题目分布（共{total}题）",
           f"Fig. 1 Distribution of the item pool across 7 value dimensions ({total:,} items)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "fig1_items_distribution_7d")


# ============================== 图2 相关矩阵热图（七维分组，1198） ==============================
def fig2_corr_heatmap():
    active = pd.read_csv(BASE / "active_items.csv")
    active7 = active[active["core_value"].isin(BIG7)].copy()
    cent = pd.read_csv(BASE / "centered_orig.csv", index_col=0).loc[active7["rule_id"]]
    X = np.nan_to_num(cent.values.astype(float))
    print(f"图2: 七维活跃题 {len(active7)}（应为1198）")

    # 与dist_1_minus_corr一致：行中心化+行归一化的Pearson相关
    xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(xc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    xc = xc / norm
    corr = xc @ xc.T

    # 按七维分组重排（组内保持原序，顺序与图1一致）
    cv = active7.set_index("rule_id").loc[cent.index, "core_value"].values
    order = np.concatenate([np.where(cv == d)[0] for d in BIG7_BY_SIZE])
    C = corr[np.ix_(order, order)]
    bounds = np.cumsum([0] + [int((cv == d).sum()) for d in BIG7_BY_SIZE])

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   interpolation="nearest", rasterized=True)
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.9)
        ax.axvline(b - 0.5, color="white", lw=0.9)
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(7)]
    ax.set_xticks(centers)
    ax.set_xticklabels(BIG7_BY_SIZE, fontsize=9, rotation=90)
    ax.set_yticks(centers)
    ax.set_yticklabels(BIG7_BY_SIZE, fontsize=9)
    ax.set_xlabel("题目（按预设价值维度分组排列）/ Items grouped by presupposed dimension", fontsize=10)
    ax.set_ylabel("题目（按预设价值维度分组排列）/ Items grouped by presupposed dimension", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("题目间Pearson相关 / Item-wise Pearson correlation", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    _title(fig, "图2 交互残差的题目间相关矩阵（题目按预设七维分组排列）",
           "Fig. 2 Item-by-item correlation matrix of interaction residuals "
           "(items grouped by the presupposed 7 dimensions)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig2_item_corr_heatmap_7d")


# ============================== 图3 轮廓系数曲线（K=2~100，七维1198） ==============================
def fig3_sil_curve():
    sil = pd.read_csv(BASE / "check_sil_k100_orig.csv")
    k = sil["k"].values
    s = sil["silhouette"].values
    s7 = float(sil.loc[sil["k"] == 7, "silhouette"].iloc[0])
    print(f"图3: K=2→{s[0]:.3f}, K=100→{s[-1]:.3f}, K=7处{s7:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(k, s, color="#3A6EA5", lw=2.0, zorder=3)
    ax.axvline(7, color="#C0392B", lw=1.2, ls="--", zorder=2)
    ax.text(7.8, s.max() * 0.30, "K=7（假设维度数）", fontsize=10,
            color="#C0392B", ha="left")
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
    _save(fig, "fig3_silhouette_k_curve_7d")


# ============================== 图4 方差解剖饼图（七维活跃1198重算） ==============================
def fig4_variance_pie():
    active = pd.read_csv(BASE / "active_items.csv")
    active7 = active[active["core_value"].isin(BIG7)]
    scored = pd.read_csv(BASE / "scored_matrix.csv", index_col=0).loc[active7["rule_id"]]
    vb = variance_budget(scored)
    print(f"图4: 七维活跃{vb['n_items']}题方差预算 "
          f"{vb['ss_item_pct']:.1f}/{vb['ss_model_pct']:.1f}/{vb['ss_interaction_pct']:.1f}")
    # 数字落盘（供spec与论文引用）
    with open(BASE / "variance_budget_7d.json", "w", encoding="utf-8") as f:
        json.dump(vb, f, ensure_ascii=False, indent=2)

    parts = [
        ("题目主效应（共识）", vb["ss_item_pct"], C_ITEM),
        ("模型主效应（宽严风格）", vb["ss_model_pct"], C_MODEL),
        ("题目×模型交互（差异候选）", vb["ss_interaction_pct"], C_INTER),
    ]
    labels = [p[0] for p in parts]
    vals = [p[1] for p in parts]
    colors = [p[2] for p in parts]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    wedges, _, autotexts = ax.pie(
        vals, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%", pctdistance=0.68,
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=12.5, fontweight="bold", color="white"),
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.98, 0.5),
              fontsize=11, frameon=False)
    ax.set_aspect("equal")
    _title(fig, f"图4 价值观Likert响应的方差解剖（n={vb['n_items']}题×22模型观测）",
           "Fig. 4 Variance decomposition of value-related Likert responses")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig4_variance_pie_7d")


# ============================== 图7 轴验证（V1置换 + V3符号保持） ==============================
def fig7_validation():
    # ---- 重建合并轴（与step7c/step16同协议） ----
    X, meta, cols = load_matrices("schwartz")
    cv_all = meta["core_value"].values
    mask7 = np.isin(cv_all, BIG7)
    X7, cv7 = X[mask7], cv_all[mask7]
    U7, _, _ = np.linalg.svd(double_center(X7), full_matrices=False)
    profs = [pd.Series(U7[:, k]).groupby(cv7).mean() for k in (0, 3)]
    zs = [(p - p.mean()) / p.std() for p in profs]
    L = np.column_stack([zs[0].values, zs[1].values])
    _, _, Vt = np.linalg.svd(L - L.mean(axis=0), full_matrices=False)
    w = Vt[0]
    v_merge = w[0] * U7[:, 0] + w[1] * U7[:, 3]
    v_merge /= np.linalg.norm(v_merge)
    m = pd.Series(v_merge).groupby(cv7).mean()
    if m["传统"] < m["自我导向"]:
        v_merge = -v_merge
        m = -m

    # ---- V1: 标签置换零分布（1000次，固定seed） ----
    print("图7: 重算V1合并轴标签置换零分布 1000次 ...")
    rng = np.random.default_rng(20261104)
    eta_null = np.array([eta_squared(v_merge, rng.permutation(cv7)) for _ in range(1000)])
    eta_real = eta_squared(v_merge, cv7)
    chance = (7 - 1) / (len(cv7) - 1)
    print(f"  η²={eta_real:.4f}, 零分布max={eta_null.max():.4f}, "
          f"机会水平={chance:.4f}, =机会的{eta_real / chance:.1f}倍")

    # ---- V3: 符号保持率（merge_test.json） ----
    with open(BASE / "merge_test.json", encoding="utf-8") as f:
        mt = json.load(f)
    side = mt["merged_axis_validation_big7"]["V3_side_stay_pct"]
    order = mt["merged_axis_validation_big7"]["ordering_neg_to_pos"]  # 自主→秩序

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    # 左panel：V1
    ax1.hist(eta_null, bins=40, color=C_NULL_BAND, alpha=0.8,
             label="标签置换零分布（1000次）")
    ax1.axvline(eta_real, color=C_SIG, lw=2.2,
                label=f"真实效应量 η²={eta_real:.4f}")
    ax1.axvline(chance, color="#2F2F2F", lw=1.2, ls="--",
                label=f"机会水平 {chance:.4f}")
    ax1.set_xlabel("η²（七维分组效应量）/ Effect size η²", fontsize=10.5)
    ax1.set_ylabel("频次 / Count", fontsize=10.5)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax1.text(eta_real, ax1.get_ylim()[1] * 0.82,
             f"=机会水平的{eta_real / chance:.1f}倍\n(p<0.001)",
             ha="right", fontsize=10, color=C_SIG, fontweight="bold")
    ax1.set_title("(a) V1 标签置换检验 / Label permutation", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)
    # 右panel：V3
    vals = [side[d] / 100.0 for d in order]
    anchors = {d for d in order if side[d] >= 90.0}
    y = np.arange(len(order))
    ax2.barh(y, vals,
             color=[C_SIG if d in anchors else "#9BB0CC" for d in order],
             edgecolor="black", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(order, fontsize=11)
    ax2.axvline(0.5, color="black", lw=1.0, ls="--")
    ax2.text(0.505, len(order) - 0.3, "50%=掷硬币", fontsize=9.5, color="#555555")
    for yi, v in zip(y, vals):
        ax2.text(v + 0.01, yi, f"{v:.1%}", va="center", fontsize=9.5)
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel("bootstrap重采样中留在原符号侧的比例 / Side-stay rate (1000 resamples)", fontsize=10)
    ax2.set_title("(b) V3 维度符号稳定性 / Bootstrap side-stay rate", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    _title(fig, "图7 自主—秩序语义轴的两层验证：效应量远超噪声带；轴两端锚点100%符号稳定",
           "Fig. 7 Validation of the autonomy–order semantic axis: effect size exceeds noise; "
           "pole anchors 100% side-stable")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, "fig7_validation_7d")


# ============================== 图8 模型光谱（22观测合并轴z） ==============================
def fig8_model_spectrum():
    with open(BASE / "paperB_patch_experiments.json", encoding="utf-8") as f:
        pe = json.load(f)
    mz = pe["exp2_merged_axis_models"]["model_z"]
    df = pd.DataFrame({"model": list(mz.keys()), "z": list(mz.values())})
    df["alias"] = df["model"].str.rsplit("_", n=1).str[0]
    df["mode"] = df["model"].str.rsplit("_", n=1).str[1]
    df = df.sort_values("z").reset_index(drop=True)
    # 校验推理位移：10/11对向自主端
    disp = pe["exp2_merged_axis_models"]["reasoning_displacement"]
    n_neg = sum(1 for v in disp.values() if v < 0)
    print(f"图8: 推理位移向自主端 {n_neg}/11 对")

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    y = np.arange(len(df))
    colors = [C_MODEL if m == "on" else C_ITEM for m in df["mode"]]
    ax.barh(y, df["z"], color=colors, edgecolor="black", linewidth=0.4, height=0.72)
    for alias, g in df.groupby("alias"):
        if len(g) == 2:
            ax.plot(g["z"], g.index, color="#888888", lw=1.0, ls=":", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=9.5)
    ax.axvline(0, color="black", lw=1.0)
    for _, r in df.iterrows():
        if abs(r["z"]) > 2.0:
            ax.text(r["z"] + (0.12 if r["z"] > 0 else -0.12), r.name,
                    f"z={r['z']:+.2f}", va="center",
                    ha="left" if r["z"] > 0 else "right",
                    fontsize=10, fontweight="bold", color=C_SIG)
    ax.set_xlabel("模型在自主—秩序语义轴上的坐标（z分数；负=偏自主端，正=偏秩序端）\n"
                  "Model position on the autonomy–order axis (z-score; negative = autonomy end)",
                  fontsize=10.5)
    ax.set_xlim(-3.4, 3.9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_ITEM, label="thinking-off（直觉响应）"),
                       Patch(facecolor=C_MODEL, label="thinking-on（推理响应）")],
              loc="lower right", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, "图8 22个模型观测的自主—秩序语义轴光谱（虚线=同模型开/关推理的位移）",
           "Fig. 8 Spectrum of 22 model observations on the autonomy–order semantic axis "
           "(dotted = reasoning-mode shift)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig8_model_spectrum_7d")


if __name__ == "__main__":
    print("=" * 62)
    print("step19: 七维口径图系重画（图1/2/3/4/7/8）")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 62)
    fig1_items_distribution()
    fig2_corr_heatmap()
    fig3_sil_curve()
    fig4_variance_pie()
    fig7_validation()
    fig8_model_spectrum()
    print("\n全部完成。")
