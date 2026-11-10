# -*- coding: utf-8 -*-
"""
step6 MP谱检验：交互项噪声等价性检验（原理级判定12维内容信号是否存在）

背景框架（对齐诱导的测量退化，Alignment-Induced Measurement Degeneracy）：
  X_ij = g_j·l_i (共识分量) + Σ_k a_jk·λ_ik (维度特异分量) + e_ij (噪声)
  双重中心化扣除题目难度（行均值，吸收l_i·ḡ）与模型共识度（列均值，吸收g_j·l̄）后，
  交互项理论上保留：维度特异交互 + 正反向方法交互 + 噪声。

检验逻辑（噪声等价性）：
  H0：交互项 = 纯噪声（题目间无任何关联结构 → 12维内容信号不存在）
  零分布：每列独立置换（保留各列边缘分布，破坏题目间行结构）→ 同管线中心化 → SVD谱
  判定：真实谱顶部奇异值超出零分布 → 存在真实结构；显著个数 = 结构维数下界
  解读：显著方向的左奇异向量（题目空间）与 direction(方法效应) / core_value(内容效应)
        的关联 —— 若显著方向全为方法效应、内容效应全不显著，
        则"12维内容信号与噪声不可区分"获原理级证明（不是没测出来，是信息不存在）。

输出：
  data/results/{source}/mp_spectrum_test.json  （谱、p值、方向解读）
  data/results/{source}/mp_spectrum.png        （谱 vs 零分布包络图）

运行：python analysis/step6_mp_spectrum.py [--source old|agree] [--drop-col 列名]
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
from analysis.step5_ega import load_matrices  # 复用加载（NaN行均值填充，活跃题×22）

N_PERM = 1000
SEED = 42
PLOT_FONT = {"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False}


def double_center(X):
    """双重中心化：扣除题目难度（行均值）与模型共识度（列均值）。"""
    rm = X.mean(axis=1, keepdims=True)
    cm = X.mean(axis=0, keepdims=True)
    gm = X.mean()
    return X - rm - cm + gm


def eta_squared(x, groups):
    """分类变量 groups 对连续变量 x 的效应量 η²（组间SS/总SS）。"""
    df = pd.DataFrame({"x": x, "g": groups})
    gm = df["x"].mean()
    ss_total = ((df["x"] - gm) ** 2).sum()
    ss_between = df.groupby("g")["x"].apply(lambda s: len(s) * (s.mean() - gm) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["old", "agree", "schwartz"], default="old")
    ap.add_argument("--drop-col", default=None,
                    help="稳健性：剔除指定模型列（如 sensenova-flash-lite_on）")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, meta, model_cols = load_matrices(args.source)
    if args.drop_col:
        # 稳健性变体：剔除指定列（NaN最多的列等），其余流程完全一致
        if args.drop_col not in model_cols:
            raise SystemExit(f"列不存在: {args.drop_col}（可用: {model_cols}）")
        j = model_cols.index(args.drop_col)
        X = np.delete(X, j, axis=1)
        model_cols = [c for c in model_cols if c != args.drop_col]
        print(f"[稳健性] 已剔除列: {args.drop_col} → {X.shape[1]}列")
    n_items, n_models = X.shape
    direction = meta["direction"].values  # 取值: positive / negative（反向题）
    is_rev = (direction == "negative").astype(float)
    core_value = meta["core_value"].values
    print(f"数据: {n_items}题 × {n_models}模型观测 | 反向(negative): "
          f"{int(is_rev.sum())} / 正向: {int((1 - is_rev).sum())}")

    # ---- 1. 真实交互项谱 ----
    Xc = double_center(X)
    U, s_real, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_nonzero = min(n_items, n_models) - 1
    var_share = s_real ** 2 / (s_real ** 2).sum()
    print(f"\n交互项非零奇异值: {k_nonzero}个 | 顶部5个: {np.round(s_real[:5], 3)}")
    print(f"顶部5个方差占比: {np.round(var_share[:5] * 100, 1)}%")

    # ---- 2. 置换零分布（对中心化后的Xc每列独立洗牌 → 谱）----
    # 注意零模型必须与真实数据"同等中心化状态"对称：若对原始X洗牌再中心化，
    # 洗牌会抹平行均值 → 行中心化吸收不到题目难度方差 → 零谱系统性偏大（伪影）。
    # 标准做法：在交互项矩阵Xc上逐列置换，保留各列边缘分布，破坏题目间行结构。
    print(f"\n置换零分布: {N_PERM}次（交互项Xc逐列洗牌）...")
    null_spec = np.zeros((N_PERM, k_nonzero))
    for b in range(N_PERM):
        Xp = Xc.copy()
        for j in range(n_models):
            Xp[:, j] = Xp[rng.permutation(n_items), j]
        null_spec[b] = np.linalg.svd(Xp, compute_uv=False)[:k_nonzero]
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{N_PERM}")
    null_med = np.median(null_spec, axis=0)
    null_p95 = np.percentile(null_spec, 95, axis=0)

    # ---- 3. 逐位置p值（置换检验）----
    s_real_k = s_real[:k_nonzero]
    p_vals = (null_spec >= s_real_k[None, :]).sum(axis=0) / N_PERM
    # 顶部谱的单调约束下，第k个显著要求前k个都显著（嵌套结构）
    bonf = 0.05 / k_nonzero
    sig_flags = (p_vals < bonf) & (p_vals <= 1.0 / N_PERM + 1e-9)
    # 从顶部往下数连续显著的个数
    n_sig = 0
    for k in range(k_nonzero):
        if sig_flags[k]:
            n_sig += 1
        else:
            break
    print(f"\n逐位置p值(Bonferroni α={bonf:.5f}): "
          f"顶部10个 p={np.round(p_vals[:10], 4)}")
    print(f"连续显著顶部奇异值个数: {n_sig}")

    # ---- 4. 显著方向解读：方法效应 vs 内容效应 ----
    interpretations = []
    for k in range(n_sig):
        v_item = U[:, k]   # 题目空间（1773维）
        u_model = Vt[k]    # 模型空间（22维）
        r_dir = float(np.corrcoef(v_item, is_rev)[0, 1])
        eta_cv = eta_squared(v_item, core_value)
        # 混杂检查: 回归掉正反向后的12维η²（内容效应是否独立于方法效应）
        beta = np.polyfit(is_rev, v_item, 1)
        resid = v_item - np.polyval(beta, is_rev)
        eta_cv_resid = eta_squared(resid, core_value)
        # 显著性背书：Kruskal-Wallis（12维）与秩和（正反向）
        from scipy.stats import kruskal, mannwhitneyu
        groups_cv = [v_item[core_value == c] for c in pd.unique(core_value)]
        p_cv = float(kruskal(*groups_cv).pvalue)
        groups_resid = [resid[core_value == c] for c in pd.unique(core_value)]
        p_cv_resid = float(kruskal(*groups_resid).pvalue)
        p_dir = float(mannwhitneyu(v_item[is_rev == 1], v_item[is_rev == 0]).pvalue)
        top_models = pd.Series(u_model, index=model_cols).abs().nlargest(3)
        # 维度语义排序（残差均值，仅内容效应方向有意义）
        dim_order = pd.Series(resid).groupby(core_value).mean().sort_values()
        interpretations.append({
            "component": k + 1,
            "singular_value": float(s_real[k]),
            "var_share_pct": float(var_share[k] * 100),
            "biserial_r_direction": r_dir,
            "p_direction": p_dir,
            "eta2_core_value": eta_cv,
            "p_core_value": p_cv,
            "eta2_core_value_control_direction": eta_cv_resid,
            "p_core_value_control_direction": p_cv_resid,
            "top_model_loadings": {str(kk): float(vv) for kk, vv in top_models.items()},
            # 完整12维排序（残差均值，升序=负端→正端），供跨数据集轴对比
            "dimension_ordering_full": {d: float(v) for d, v in dim_order.items()},
            "verdict": ("方法效应(正反向)" if abs(r_dir) > 0.3 and p_dir < 0.001 else
                        ("内容效应(12维)" if eta_cv_resid > 0.02 and p_cv_resid < 0.001 else "其他结构")),
        })
        print(f"\n  奇异方向{k + 1}: s={s_real[k]:.2f} (方差占比{var_share[k]*100:.1f}%)")
        print(f"    正反向点二列r={r_dir:+.3f} (p={p_dir:.2e}) | 12维η²={eta_cv:.4f} (p={p_cv:.2e})")
        print(f"    控制正反向后η²={eta_cv_resid:.4f} (p={p_cv_resid:.2e})")
        if interpretations[-1]["verdict"] == "内容效应(12维)":
            lo = dim_order.index[:3].tolist()
            hi = dim_order.index[-3:].tolist()
            interpretations[-1]["dimension_semantics"] = {
                "negative_pole": lo, "positive_pole": hi}
            print(f"    语义轴: {lo} ←→ {hi}")
        print(f"    判定: {interpretations[-1]['verdict']} | 高负载模型: "
              f"{', '.join(f'{kk}({vv:+.2f})' for kk, vv in top_models.items())}")

    # ---- 5. 全局结论 ----
    content_sig = [i for i in interpretations if i["verdict"] == "内容效应(12维)"]
    method_sig = [i for i in interpretations if i["verdict"] == "方法效应(正反向)"]
    if n_sig == 0:
        conclusion = ("交互项顶部奇异值全部落入置换零分布 → 12维内容信号与噪声不可区分，"
                      "且无任何可检出的题目间结构（比EGA空网更强的原理级否定）")
    elif len(content_sig) == 0:
        conclusion = (f"交互项仅存在{len(method_sig)}个方法效应方向（正反向），"
                      "12维内容信号与噪声不可区分 → 原理级否定：扣除共识与难度后，"
                      "内容维度信息在当前模型池下不存在")
    else:
        conclusion = (f"检出{len(content_sig)}个内容效应方向 + {len(method_sig)}个方法效应方向，"
                      "内容信号存在但被共识掩盖 → 叙事调整为'存在但需更大模型池分离'")
    print(f"\n===== 全局结论 =====\n{conclusion}")

    # ---- 6. 落盘 ----
    base = config.DATA_DIR / "results" / args.source
    suffix = "_drop" if args.drop_col else ""
    out = {
        "data": {"n_items": int(n_items), "n_models": int(n_models),
                 "n_reverse": int(is_rev.sum()),
                 "n_positive": int((1 - is_rev).sum()),
                 "drop_col": args.drop_col},
        "params": {"n_perm": N_PERM, "seed": SEED, "bonferroni_alpha": bonf},
        "spectrum_real": s_real[:k_nonzero].tolist(),
        "spectrum_null_median": null_med[:k_nonzero].tolist(),
        "spectrum_null_p95": null_p95[:k_nonzero].tolist(),
        "p_values": p_vals[:k_nonzero].tolist(),
        "n_significant_top": int(n_sig),
        "interpretations": interpretations,
        "conclusion": conclusion,
    }
    with open(base / f"mp_spectrum_test{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 7. 图 ----
    plt.rcParams.update(PLOT_FONT)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks = np.arange(1, k_nonzero + 1)
    ax.fill_between(ks, null_med, null_p95,
                    alpha=0.2, color="gray", label="零分布中位数~95分位（列洗牌）")
    ax.plot(ks, null_med, "k--", lw=1, label="零分布中位数")
    ax.plot(ks, s_real_k, "o-", color="#1f77b4", ms=4, lw=1.5, label="真实交互项谱")
    if n_sig > 0:
        ax.plot(ks[:n_sig], s_real_k[:n_sig], "o", color="red", ms=9,
                mfc="none", mew=2, label=f"显著超噪声（前{n_sig}个）")
    ax.set_xlabel("奇异值序号（交互项谱）")
    ax.set_ylabel("奇异值")
    ax.set_title(f"交互项噪声等价性检验（MP谱检验）：真实谱 vs 列洗牌零分布\n"
                 f"（{n_items}题×{n_models}模型观测，双重中心化后"
                 f"{'，剔除' + args.drop_col if args.drop_col else ''}）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base / f"mp_spectrum{suffix}.png", dpi=150)
    print(f"\n已保存: {base / f'mp_spectrum_test{suffix}.json'}")
    print(f"已保存: {base / f'mp_spectrum{suffix}.png'}")


if __name__ == "__main__":
    main()
