# -*- coding: utf-8 -*-
"""
step7 论文六张核心图（正式数据=盲评认同度 agree）

图1 题库构成图（第3章3.1）    数据: final_items_v2.jsonl
图2 方差解剖条图/饼图（第3章）数据: decouple_variance_budget.csv
图3 MP谱噪声带图（第4章）     数据: mp_spectrum_test.json
图4 12维轴投影排序图（头牌图，第4章）数据: scored_matrix重算 + bootstrap 95%CI
图5 轴验证图（V1置换+V3保持率，第4章）数据: 重算V1零分布 + axis_validation.json
图6 22列模型轴坐标光谱图（第5章）数据: axis_mode_control.json

输出: data/results/agree/figures/fig1~fig6.png+pdf（300dpi、彩色、双语图题）
依赖: numpy, pandas, matplotlib, scipy（lll_env已装）
用法: python analysis/step7_figures.py
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.step5_ega import load_matrices
from analysis.step6_mp_spectrum import double_center, eta_squared

BASE = config.DATA_DIR / "results" / "agree"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

# 论文主色（彩色方案）
C_ITEM, C_MODEL, C_INTER = "#4C72B0", "#DD8452", "#55A868"
C_SIG, C_NS = "#C44E52", "#8C8C8C"          # 显著/不显著
C_NULL_BAND = "#B0B0B0"


def _save(fig, name):
    """双语图题已由调用方设置；保存png+pdf。"""
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] figures/{name}.png/.pdf")


def _title(fig, zh, en):
    """学报双语图题：中文一行 + 英文一行。"""
    fig.suptitle(f"{zh}\n{en}", fontsize=13, fontweight="bold", y=0.99)


# ============================== 图1 题库构成（12维×正反向） ==============================
def fig1_items_distribution():
    """论文图1（§3.1）：最终题库2388题的12维×正反向分布，学报配色。"""
    import json as _json
    items = []
    with open(config.DATA_DIR / "final_items_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(_json.loads(line))
    dims = ["富强", "民主", "文明", "和谐", "自由", "平等",
            "公正", "法治", "爱国", "敬业", "诚信", "友善"]
    pos = [sum(1 for it in items if it["core_value"] == d and it["direction"] == "positive")
           for d in dims]
    neg = [sum(1 for it in items if it["core_value"] == d and it["direction"] == "negative")
           for d in dims]

    x = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(9, 4.0))
    b1 = ax.bar(x, pos, width=0.62, label="正向题", color=C_ITEM,
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, neg, width=0.62, bottom=pos, label="反向题", color=C_SIG,
                edgecolor="black", linewidth=0.5)
    for xi, (p, n) in enumerate(zip(pos, neg)):
        ax.text(xi, p / 2, str(p), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        if n >= 30:
            ax.text(xi, p + n / 2, str(n), ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        ax.text(xi, p + n + 4, str(p + n), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=11)
    ax.set_ylabel("题数 / Number of items", fontsize=11)
    ax.set_ylim(0, 235)
    # 图例置于图外下方横排，避免与柱顶总数标注重叠
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              fontsize=10.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, "图1 最终题库的12维×正反向题目分布（共2388题）",
           "Fig. 1 Distribution of the final item pool across 12 value dimensions (2 388 items)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "fig1_items_distribution")


# ============================== 图2 方差解剖 ==============================
def fig2_variance():
    """论文图2（§3.3.3）：饼图版（fig2_variance_pie），2026-10-15用户定版饼图、条图版删除。"""
    df = pd.read_csv(BASE / "decouple_variance_budget.csv")
    parts = [
        ("题目主效应（共识）", df.loc[0, "ss_item_pct"], C_ITEM),
        ("模型主效应（宽严风格）", df.loc[0, "ss_model_pct"], C_MODEL),
        ("题目×模型交互（差异候选）", df.loc[0, "ss_interaction_pct"], C_INTER),
    ]
    labels = [p[0] for p in parts]
    vals = [p[1] for p in parts]
    colors = [p[2] for p in parts]

    # ---- 饼图（定版） ----
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
    _title(fig, "图4 价值观Likert响应的方差解剖（n=1227题×22模型观测）",
           "Fig. 4 Variance decomposition of value-related Likert responses")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig2_variance_pie")


# ============================== 图3 MP谱 ==============================
def fig3_mp_spectrum():
    with open(BASE / "mp_spectrum_test.json", encoding="utf-8") as f:
        mp = json.load(f)
    real = np.array(mp["spectrum_real"])
    med = np.array(mp["spectrum_null_median"])
    p95 = np.array(mp["spectrum_null_p95"])
    k = np.arange(1, len(real) + 1)

    verdict_style = {
        "方法效应(正反向)": ("#DD8452", "方法效应（反向题作答）"),
        "内容效应(12维)": (C_SIG, "内容效应 → 语义轴"),
        "其他结构": ("#8C8C8C", "模型特异风格"),
    }
    top5 = mp["interpretations"][:5]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(k, med, p95, color=C_NULL_BAND, alpha=0.45,
                    label="列洗牌零分布（中位数~95分位）")
    ax.plot(k, med, color="#707070", lw=1.2, ls="--", label="零分布中位数")
    ax.plot(k, real, color="#B0B0B0", lw=1.2, marker="o", ms=3.5,
            label="真实谱（其余方向）")
    # 前5个显著方向：按判定着色 + 点上方小序号（颜色编码代替文字标注，避免重叠）
    seen = set()
    for i, interp in enumerate(top5):
        color, lab = verdict_style[interp["verdict"]]
        legend_lab = lab if lab not in seen else None
        seen.add(lab)
        ax.scatter(i + 1, real[i], s=70, color=color, zorder=5,
                   edgecolor="black", linewidth=0.6,
                   label=legend_lab)
        ax.annotate(str(i + 1), xy=(i + 1, real[i]),
                    xytext=(i + 1, real[i] + 0.85), fontsize=9,
                    ha="center", fontweight="bold", color="#333333")
    ax.set_xlabel("奇异方向序号 / Singular direction index", fontsize=11)
    ax.set_ylabel("奇异值 / Singular value", fontsize=11)
    ax.set_ylim(6, 26.5)
    ax.set_xticks(k)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, "图5 MP谱噪声等价性检验：仅5个方向超出列洗牌噪声带（1000次置换）",
           "Fig. 5 MP-spectrum noise-equivalence test: 5 directions exceed the column-shuffled noise band")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "fig3_mp_spectrum")


# ============================== 图4 轴投影（头牌） ==============================
def fig4_axis_projection():
    X, meta, cols = load_matrices("agree")
    cv = meta["core_value"].values
    n, dims = len(cv), pd.unique(cv)

    # 与step6b一致：取MP谱内容效应方向（方向4，索引3）
    with open(BASE / "mp_spectrum_test.json", encoding="utf-8") as f:
        mp = json.load(f)
    content = [i for i in mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    k_axis = max(content, key=lambda i: i["eta2_core_value_control_direction"])["component"] - 1
    U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
    v2 = U[:, k_axis]
    dim_mean = pd.Series(v2).groupby(cv).mean().sort_values()

    # 分层bootstrap 1000次 → 95%CI（与step6b V3同逻辑，收集每次维度均值）
    print("图4: bootstrap 1000次计算95%CI ...")
    rng = np.random.default_rng(20261013)
    boot = {d: [] for d in dims}
    for _ in range(1000):
        idxs = [rng.choice(np.where(cv == d)[0], size=(cv == d).sum(), replace=True)
                for d in dims]
        b = np.concatenate(idxs)
        m = pd.Series(v2[b]).groupby(cv[b]).mean()
        for d in dims:
            boot[d].append(m[d])
    lo = {d: np.percentile(boot[d], 2.5) for d in dims}
    hi = {d: np.percentile(boot[d], 97.5) for d in dims}
    ci_excl_zero = {d: (lo[d] > 0 or hi[d] < 0) for d in dims}   # CI不含零=确凿锚点
    print("  CI不含零的维度:", [d for d, t in ci_excl_zero.items() if t])

    order = dim_mean.index.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for yi, d in zip(y, order):
        anchor = ci_excl_zero[d]
        ax.errorbar(dim_mean[d], yi,
                    xerr=[[dim_mean[d] - lo[d]], [hi[d] - dim_mean[d]]],
                    fmt="o", ms=9 if anchor else 6,
                    color=C_SIG if anchor else "#5B7DB1",
                    ecolor=C_SIG if anchor else "#9BB0CC",
                    elinewidth=2.2 if anchor else 1.2, capsize=4, zorder=3)
        offset = 2.6e-3 if dim_mean[d] >= 0 else -2.6e-3
        ax.text(dim_mean[d] + offset, yi, d, ha="left" if dim_mean[d] >= 0 else "right",
                va="center", fontsize=12.5,
                fontweight="bold" if anchor else "normal",
                color=C_SIG if anchor else "#333333")
    ax.axvline(0, color="black", lw=1.0, ls="-", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=11)
    ax.set_xlabel("维度在语义轴上的投影（均值载荷，误差棒=bootstrap 95%CI）\n"
                  "Axis projection (mean loading; error bars = bootstrap 95% CI)", fontsize=10.5)
    ax.set_xlim(-0.017, 0.015)
    ax.spines[["top", "right"]].set_visible(False)
    # 两端锚点区背景提示
    ax.text(-0.0165, len(order) - 0.4, "◀ 自由端", fontsize=11, color=C_SIG, fontweight="bold")
    ax.text(0.0145, 0.4, "民主·爱国端 ▶", fontsize=11, color=C_SIG, fontweight="bold", ha="right")
    _title(fig, "图6 12个价值观维度在语义轴上的投影（红色=95%CI不含零的确凿锚点）",
           "Fig. 6 Projection of 12 value dimensions onto the semantic axis (red = 95% CI excludes zero)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "fig4_axis_projection")

    # CI数值随图落盘（论文可直接引用）
    ci_out = {d: {"mean": float(dim_mean[d]), "ci95": [float(lo[d]), float(hi[d])],
                  "anchor": bool(ci_excl_zero[d])} for d in order}
    with open(BASE / "axis_dim_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci_out, f, ensure_ascii=False, indent=2)
    print("[已保存] axis_dim_ci.json")


# ============================== 图6 模型光谱 ==============================
def fig6_model_spectrum():
    with open(BASE / "axis_mode_control.json", encoding="utf-8") as f:
        mc = json.load(f)
    rows = mc["model_loadings_on_axis"]
    # z分数 = loading×√n（U列单位范数、均值≈0 → std≈1/√22；与论文口径一致：
    # kimi-k2.6_off=-2.28, doubao-seed-2-1-pro_off=+3.06）
    sq22 = float(np.sqrt(22))
    df = pd.DataFrame([{"model": r["model"], "loading": r["loading"],
                        "z": r["loading"] * sq22} for r in rows])
    df["alias"] = df["model"].str.rsplit("_", n=1).str[0]
    df["mode"] = df["model"].str.rsplit("_", n=1).str[1]
    df = df.sort_values("z").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    y = np.arange(len(df))
    colors = [C_MODEL if m == "on" else C_ITEM for m in df["mode"]]
    ax.barh(y, df["z"], color=colors, edgecolor="black", linewidth=0.4, height=0.72)
    # 同模型on/off连线（推理模式位移）
    for alias, g in df.groupby("alias"):
        if len(g) == 2:
            ax.plot(g["z"], g.index, color="#888888", lw=1.0, ls=":", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=9.5)
    ax.axvline(0, color="black", lw=1.0)
    # 离群点标注
    for _, r in df.iterrows():
        if abs(r["z"]) > 2.0:
            ax.text(r["z"] + (0.12 if r["z"] > 0 else -0.12), r.name,
                    f"z={r['z']:+.2f}", va="center",
                    ha="left" if r["z"] > 0 else "right",
                    fontsize=10, fontweight="bold", color=C_SIG)
    ax.set_xlabel("模型在语义轴上的坐标（z分数；负=偏自由端，正=偏民主·爱国端）\n"
                  "Model position on the axis (z-score; negative = liberty end)", fontsize=10.5)
    ax.set_xlim(-3.4, 3.9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_ITEM, label="thinking-off（直觉响应）"),
                       Patch(facecolor=C_MODEL, label="thinking-on（推理响应）")],
              loc="lower right", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _title(fig, "图8 22个模型观测的语义轴光谱（虚线=同模型开/关推理的位移）",
           "Fig. 8 Spectrum of 22 model observations on the semantic axis (dotted = reasoning-mode shift)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig6_model_spectrum")


# ============================== 图5 验证 ==============================
def fig5_validation():
    with open(BASE / "axis_validation.json", encoding="utf-8") as f:
        av = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # ---- 左panel：V1 标签置换零分布（重算，固定seed）----
    print("图5: 重算V1置换零分布 1000次 ...")
    X, meta, cols = load_matrices("agree")
    cv = meta["core_value"].values
    with open(BASE / "mp_spectrum_test.json", encoding="utf-8") as f:
        mp = json.load(f)
    content = [i for i in mp["interpretations"] if i["verdict"] == "内容效应(12维)"]
    k_axis = max(content, key=lambda i: i["eta2_core_value_control_direction"])["component"] - 1
    U, s, Vt = np.linalg.svd(double_center(X), full_matrices=False)
    v2 = U[:, k_axis]
    rng = np.random.default_rng(20261013)
    eta_null = np.array([eta_squared(v2, rng.permutation(cv)) for _ in range(1000)])
    eta_real = av["V1_label_permutation"]["eta_real"]

    ax1.hist(eta_null, bins=40, color=C_NULL_BAND, alpha=0.8,
             label="标签置换零分布（1000次）")
    ax1.axvline(eta_real, color=C_SIG, lw=2.2,
                label=f"真实效应量 η²={eta_real:.4f}")
    ax1.axvline(av["V1_label_permutation"]["chance_level"], color="#2F2F2F",
                lw=1.2, ls="--", label=f"机会水平 {av['V1_label_permutation']['chance_level']:.4f}")
    ax1.set_xlabel("η²（12维分组效应量）/ Effect size η²", fontsize=10.5)
    ax1.set_ylabel("频次 / Count", fontsize=10.5)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax1.text(eta_real, ax1.get_ylim()[1] * 0.82,
             f"=机会水平的{av['V1_label_permutation']['times_above_chance']:.1f}倍\n(p<0.001)",
             ha="right", fontsize=10, color=C_SIG, fontweight="bold")
    ax1.set_title("(a) V1 标签置换检验 / Label permutation", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- 右panel：V3 各维度留在原符号侧的比例 ----
    side = av["V3_bootstrap_ordering"]["side_stay_rate"]
    order = pd.Series(av["dimension_ordering"]).sort_values().index.tolist()  # 按轴排序
    vals = [side[d] for d in order]
    anchors = {d for d in order if side[d] >= 0.9}
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

    _title(fig, "图7 语义轴的两层验证：效应量远超噪声带；轴两端锚点100%符号稳定",
           "Fig. 7 Validation of the semantic axis: effect size exceeds noise; pole anchors 100% side-stable")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, "fig5_validation")


if __name__ == "__main__":
    print("=" * 62)
    print("step7: 生成论文六张核心图（正式数据 agree）")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 62)
    fig1_items_distribution()
    fig2_variance()
    fig3_mp_spectrum()
    fig4_axis_projection()
    fig5_validation()
    fig6_model_spectrum()
    print("\n全部完成。")
