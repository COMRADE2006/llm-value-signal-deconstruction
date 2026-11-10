# -*- coding: utf-8 -*-
"""step5_ega.py — 探索性图分析(EGA)三层网络检验: L0=12核心价值 / L1=子主题 / L2=1773题目。

方法依据:
  Golino & Epskamp (2017, PLoS ONE): EGA = glasso(EBIC-gamma=0.5选lambda) + walktrap社区检测;
    模拟显示因子相关0.7时EGA是唯一正确恢复维度的方法(传统法全坍缩)——正是本题场景
  Golino et al. (2020) / Christensen et al. (2023): bootEGA稳定性; louvain≈walktrap性能相当
统计现实(论文必须写的适配点):
  节点=题目/子主题(变量), 观测=22列(11模型×on/off) → N=22 < p
  → L0(p=12<N)样本相关合法; L1(p≈47)/L2(p=1773)需收缩: LW自动 + 固定ridge敏感性{0.3,0.5,0.7}
理论对照(两种结果都有解释力):
  单一潜在因子模型 ⇔ 精度矩阵稀疏(给定因子后条件独立) → 空网络/孤立点多 = 一维坍缩的网络签名
  多维结构 ⇔ 偏相关网络呈可复制社区
输出: data/results/{source}/ega_result.json + ega_L*_assignments.csv + ega_network_L*.png
用法: python analysis/step5_ega.py --source old --layers L0,L1,L2
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.covariance import LedoitWolf, graphical_lasso
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import config  # noqa: E402

GAMMA = 0.5          # EBIC gamma (EGAnet默认0.5)
SEED = 42
ALPHAS = np.logspace(-3, -0.3, 25)   # glasso惩罚路径(从稀到密)


# ---------- 数据准备 ----------

def load_matrices(source):
    """返回 (X: 1773×22 反向计分后已填NaN, meta: 题目元数据)。"""
    res = config.DATA_DIR / "results" / source
    active = pd.read_csv(res / "active_items.csv")
    scored = pd.read_csv(res / "scored_matrix.csv", index_col=0)
    scored = scored.loc[active["rule_id"]]
    X = scored.values.astype(float)
    # 行均值填NaN(全NaN行填3.0)
    row_mean = np.nanmean(X, axis=1)
    row_mean[np.isnan(row_mean)] = 3.0
    idx = np.where(np.isnan(X))
    X[idx] = row_mean[idx[0]]
    meta = active.set_index("rule_id")
    return X, meta, scored.columns.tolist()


def aggregate(X, meta, keys, min_n=5):
    """按keys聚合成节点×观测矩阵(组内nanmean)。返回 X_agg, node_names, node_ref(模态12维)。"""
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    df["key"] = keys
    agg = df.groupby("key").mean()
    counts = df.groupby("key").size()
    keep = counts[counts >= min_n].index
    agg = agg.loc[keep]
    # 每节点的模态核心价值(供12维对照)
    tmp = pd.DataFrame({"key": keys, "cv": meta["core_value"].values})
    node_ref = tmp.groupby("key")["cv"].agg(lambda s: s.value_counts().index[0])
    return agg.values, list(agg.index), node_ref.loc[keep].tolist(), counts.loc[keep].to_dict()


# ---------- 核心统计 ----------

def shrink_corr(X, mode):
    """X: p×n(节点×观测) → 收缩相关矩阵。mode: sample / lw / ridge:λ"""
    if mode == "sample":
        return np.corrcoef(X), None
    if mode == "lw":
        lw = LedoitWolf().fit(X.T)
        S = lw.covariance_
        d = np.sqrt(np.diag(S))
        return S / np.outer(d, d), float(lw.shrinkage_)
    lam = float(mode.split(":")[1])
    R0 = np.corrcoef(X)
    return (1 - lam) * R0 + lam * np.eye(len(R0)), lam


def glasso_ebic(R, n, alphas=None, gamma=GAMMA, max_edges_factor=3.0):
    """glasso自适应路径 + EBIC选λ。

    近奇异相关阵(强正流形)下固定α路径会撞上限产生完全图伪影:
    先几何爬坡找α_top(网络刚好变空), 再在[α_top/1000, α_top]铺25点。
    """
    p = R.shape[0]
    # 1) 爬坡找空网阈值
    a = 0.5
    for _ in range(30):
        try:
            _, Pt = graphical_lasso(R, alpha=a, max_iter=300)  # 返回(协方差估计, 精度矩阵)
            if (int(np.count_nonzero(Pt)) - p) // 2 == 0:
                break
        except Exception:
            break
        a *= 1.5
    a_top = a
    # 2) 真实路径: α_top/1000 .. α_top (对数25点), 从稀到密
    # 注意: 不设提前break — 强信号下glasso从空图跳变到稠密图, break会在评估稠密图前截断
    lo = max(a_top / 1000.0, 1e-4)
    path = np.logspace(np.log10(a_top), np.log10(lo), 25)
    best, curve = None, []
    for a in path:
        try:
            _, P = graphical_lasso(R, alpha=a, max_iter=300)  # 返回(协方差估计, 精度矩阵)
        except Exception:
            curve.append({"alpha": float(a), "edges": None, "ebic": None})
            continue
        edges = (int(np.count_nonzero(P)) - p) // 2
        sgn, logdet = np.linalg.slogdet(P)
        if sgn <= 0:
            curve.append({"alpha": float(a), "edges": edges, "ebic": None})
            continue
        L = 0.5 * n * (logdet - np.trace(R @ P))  # 高斯loglik: (n/2)(logdet(Θ) − tr(SΘ))
        ebic = -2 * L + edges * np.log(n) + 2 * gamma * edges * np.log(p * (p - 1) / 2)
        curve.append({"alpha": float(a), "edges": edges, "ebic": float(ebic)})
        if best is None or ebic < best["ebic"]:
            best = {"alpha": float(a), "edges": edges, "ebic": float(ebic), "P": P}
    if best is not None:
        best["alpha_top_empty"] = float(a_top)
    return best, curve


def detect_communities(P):
    """偏相关网络 → 社区标签(-1=孤立点)。walktrap(igraph优先) / louvain(networkx)。"""
    p = P.shape[0]
    d = np.sqrt(np.diag(P))
    W = np.abs(-P / np.outer(d, d))       # |偏相关| 边权
    A = (P != 0)
    np.fill_diagonal(A, False)
    src, dst = np.where(np.triu(A, 1))
    deg = A.sum(axis=1)
    labels = np.full(p, -1, dtype=int)
    if len(src) == 0:
        return labels, "empty", 0.0
    keep = np.where(deg > 0)[0]
    remap = -np.ones(p, dtype=int)
    remap[keep] = np.arange(len(keep))
    e0, e1 = remap[src], remap[dst]
    ok = (e0 >= 0) & (e1 >= 0)
    algo, mod = "walktrap", 0.0
    try:
        import igraph as ig
        g = ig.Graph(n=len(keep), edges=list(zip(e0[ok].tolist(), e1[ok].tolist())), directed=False)
        g.es["weight"] = W[src[ok], dst[ok]].tolist()
        mem = g.community_walktrap(weights="weight", steps=4).as_clustering()
        labels[keep] = np.array(mem.membership)
        mod = float(g.modularity(labels[keep], weights=g.es["weight"]))
    except ImportError:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(len(keep)))
        G.add_weighted_edges_from(
            [(int(e0[i]), int(e1[i]), float(W[src[i], dst[i]])) for i in np.where(ok)[0]])
        com = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        for ci, c in enumerate(sorted(com, key=lambda s: -len(s))):
            for v in c:
                labels[keep[v]] = ci
        mod = float(nx.community.modularity(G, nx.community.louvain_communities(
            G, weight="weight", seed=SEED), weight="weight"))
        algo = "louvain"
    return labels, algo, mod


def vs_reference(labels, ref, mask=None):
    """社区 vs 12维参考标签: ARI + 卡方 + Cramér's V + 加权纯度。"""
    lab = np.asarray(labels).copy()
    ref = np.asarray(ref)
    if mask is not None:
        lab, ref = lab[mask], ref[mask]
    valid = lab >= 0
    lab, ref = lab[valid], ref[valid]
    if len(set(lab)) < 2:
        return {"ari": None, "note": "社区数<2无法对照"}
    ari = float(adjusted_rand_score(lab, ref))
    ct = pd.crosstab(pd.Series(lab, name="com"), pd.Series(ref, name="ref"))
    chi2, pval, dof, _ = chi2_contingency(ct)
    n = int(ct.values.sum())
    return {"ari": ari, "chi2": float(chi2), "p": float(pval), "dof": int(dof),
            "cramers_v": float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))),
            "weighted_purity": float(ct.max(axis=1).sum() / n)}


def bootstrap(X, base_labels, alpha, shrink_mode, n_boot):
    """被试bootstrap(重抽22列, 有放回) → fixed-λ网络 → 维度数分布/ARI/节点保留率。"""
    rng = np.random.default_rng(SEED)
    p, n = X.shape
    base_sets = {int(c): set(np.where(base_labels == c)[0])
                 for c in np.unique(base_labels[base_labels >= 0])}
    dims, aris = [], []
    retain = np.zeros(p)
    counts = np.zeros(p)
    n_fail = 0
    for _ in range(n_boot):
        Xb = X[:, rng.integers(0, n, n)]
        Rb, _ = shrink_corr(Xb, shrink_mode)
        try:
            _, Pb = graphical_lasso(Rb, alpha=alpha, max_iter=300)  # 返回(协方差估计, 精度矩阵)
        except Exception:
            n_fail += 1
            continue
        lab_b, _, _ = detect_communities(Pb)
        mb = lab_b >= 0
        dims.append(int(len(set(lab_b[mb]))) if mb.any() else 0)
        both = (base_labels >= 0) & mb
        if len(set(base_labels[both])) > 1 and len(set(lab_b[both])) > 1:
            aris.append(float(adjusted_rand_score(base_labels[both], lab_b[both])))
        for i in np.where(both)[0]:
            bm = set(np.where(lab_b == lab_b[i])[0])
            sm = base_sets[int(base_labels[i])]
            retain[i] += len(bm & sm) / len(sm)
            counts[i] += 1
    stable = counts > 0
    return {"n_boot": n_boot, "n_fail": n_fail,
            "dim_median": float(np.median(dims)) if dims else None,
            "dim_mode": int(pd.Series(dims).mode().iloc[0]) if dims else None,
            "dim_dist": {int(k): int(v) for k, v in pd.Series(dims).value_counts().items()},
            "ari_median": float(np.median(aris)) if aris else None,
            "ari_q1": float(np.percentile(aris, 25)) if aris else None,
            "ari_q3": float(np.percentile(aris, 75)) if aris else None,
            "node_retention_mean": float((retain[stable] / counts[stable]).mean()) if stable.any() else None,
            "_retain": np.where(stable, retain / np.maximum(counts, 1), np.nan)}


# ---------- 单层执行 ----------

def run_layer(name, X, node_names, node_ref, item_ref_labels, n_obs,
              configs, boots, primary_rule, do_boot=True, plot_title=None):
    """configs: [(mode, boot数)] — primary_rule(solutions)在全部配置求解后选主配置。"""
    print(f"\n===== [{name}] p={X.shape[0]} 节点, N={n_obs} 观测 =====", flush=True)
    out = {"n_nodes": int(X.shape[0]), "n_obs": int(n_obs), "configs": {}}
    solutions = {}
    for mode, n_boot in configs:
        R, shrink = shrink_corr(X, mode)
        best, curve = glasso_ebic(R, n_obs)
        if best is None:
            print(f"  [{mode}] glasso全路径失败", flush=True)
            continue
        labels, algo, mod = detect_communities(best["P"])
        n_iso = int((labels < 0).sum())
        n_lab = int(len(set(labels[labels >= 0])))
        vs12 = vs_reference(labels, node_ref)
        res = {"shrinkage": shrink, "alpha_star": best["alpha"], "n_edges": best["edges"],
               "n_isolated": n_iso, "n_communities_nonisolated": n_lab,
               "modularity": mod, "community_algo": algo,
               "density": round(2 * best["edges"] / (X.shape[0] * (X.shape[0] - 1)), 4),
               "vs_12dim": vs12, "ebic": best["ebic"]}
        # 社区规模
        sizes = pd.Series(labels[labels >= 0]).value_counts().sort_index()
        res["community_sizes"] = {int(k): int(v) for k, v in sizes.items()}
        solutions[mode] = (labels, res)
        print(f"  [{mode:9s}] 收缩={shrink if shrink is not None else '样本'} λ*={best['alpha']:.4f} "
              f"边={best['edges']} 孤立={n_iso} 社区={n_lab} 模块度={mod:.3f} "
              f"vs12维ARI={vs12.get('ari')}", flush=True)
    if not solutions:
        return out, {}, None
    primary_mode = primary_rule(solutions)
    if primary_mode not in solutions:
        primary_mode = list(solutions)[0]
    out["primary_mode"] = primary_mode
    # bootstrap: 全部配置(轻层)或仅主配置(L2)
    boot_modes = list(solutions) if X.shape[0] <= 60 else [primary_mode]
    for mode in boot_modes:
        labels, res = solutions[mode]
        n_boot = dict(configs)[mode] if X.shape[0] <= 60 else dict(configs)[primary_mode]
        if n_boot and n_boot > 0:
            t0 = time.time()
            bs = bootstrap(X, labels, solutions[mode][1]["alpha_star"], mode, n_boot)
            bs.pop("_retain")
            res["bootstrap"] = bs
            print(f"  [{mode}] bootstrap {bs['n_boot']}次({time.time()-t0:.0f}s): "
                  f"维度中位={bs['dim_median']} ARI中位={bs['ari_median']} "
                  f"节点保留率={bs['node_retention_mean']}", flush=True)
    for mode, (labels, res) in solutions.items():
        out["configs"][mode] = res
    # 主配置的社区构成(节点明细)
    if primary_mode:
        labels = solutions[primary_mode][0]
        out["primary_assignments"] = [
            {"node": node_names[i], "community": int(labels[i]),
             "ref_12dim": node_ref[i]} for i in range(len(node_names))]
    return out, solutions, primary_mode


# ---------- 可视化 ----------

def plot_network(name, X, node_names, labels, node_ref, P, out_png, title):
    """偏相关网络图: 左=按社区着色, 右=按12维着色。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    try:
        import networkx as nx
    except ImportError:
        return
    p = P.shape[0]
    d = np.sqrt(np.diag(P))
    W = np.abs(-P / np.outer(d, d))
    A = (P != 0)
    np.fill_diagonal(A, False)
    G = nx.Graph()
    G.add_nodes_from(range(p))
    src, dst = np.where(np.triu(A, 1))
    for i, j in zip(src, dst):
        G.add_edge(int(i), int(j), weight=float(W[i, j]))
    pos = nx.spring_layout(G, weight="weight", seed=SEED, k=1.5 / np.sqrt(max(p, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    com_colors = plt.cm.tab20(labels % 20)
    com_colors[labels < 0] = (0.8, 0.8, 0.8, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=axes[0])
    nx.draw_networkx_nodes(G, pos, node_color=com_colors, node_size=260, ax=axes[0])
    ref_uni = list(dict.fromkeys(node_ref))
    ref_cmap = {r: plt.cm.tab20(i % 20) for i, r in enumerate(ref_uni)}
    nx.draw_networkx_edges(G, pos, alpha=0.35, ax=axes[1])
    nx.draw_networkx_nodes(G, pos, node_color=[ref_cmap[r] for r in node_ref],
                           node_size=260, ax=axes[1])
    for ax, t in zip(axes, [f"{title}（按EGA社区着色）", f"{title}（按官方12维着色）"]):
        ax.set_title(t, fontsize=14)
        ax.axis("off")
    if p <= 60:
        for i, nname in enumerate(node_names):
            axes[0].annotate(str(nname)[:6], pos[i], fontsize=7, ha="center", va="center")
    fig.suptitle(f"{name}: |偏相关| 网络（glasso+EBIC）", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {out_png}", flush=True)


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="old")
    ap.add_argument("--layers", default="L0,L1,L2")
    ap.add_argument("--boots-l1", type=int, default=300)
    ap.add_argument("--boots-l2", type=int, default=0, help="0=按时长自适应")
    args = ap.parse_args()
    layers_wanted = set(args.layers.split(","))

    X, meta, colnames = load_matrices(args.source)
    res_dir = config.DATA_DIR / "results" / args.source
    n_obs = X.shape[1]
    result = {"meta": {"source": args.source, "n_items": int(X.shape[0]), "n_obs": n_obs,
                       "obs_columns": colnames, "gamma": GAMMA, "seed": SEED,
                       "note": "观测=22列(11模型×on/off); N<p层用收缩相关; bootstrap=重抽观测列"}}

    # ---- L2: 题目层(先跑路径计时, 决定bootstrap量) ----
    if "L2" in layers_wanted:
        item_ref = meta["core_value"].tolist()
        cfg = [("ridge:0.3", 0), ("ridge:0.5", 0), ("ridge:0.7", 0)]
        if args.boots_l2 > 0:
            cfg = [("ridge:0.3", 0), ("ridge:0.5", args.boots_l2), ("ridge:0.7", 0)]
        # 自适应: 主配置先试一次fit计时
        t0 = time.time()
        R, _ = shrink_corr(X, "ridge:0.5")
        best, _ = glasso_ebic(R, n_obs, alphas=np.logspace(-2.5, -0.3, 20))
        t_fit = (time.time() - t0) / 20
        if args.boots_l2 == 0 and best is not None:
            n_b = int(np.clip(1500 / max(t_fit, 1), 10, 60))
            cfg = [("ridge:0.3", 0), ("ridge:0.5", n_b), ("ridge:0.7", 0)]
            print(f"[L2] 单次glasso约{t_fit:.1f}s → 自适应bootstrap={n_b}次", flush=True)
        out, sols, pm = run_layer(
            "L2_items", X, meta.index.tolist(), item_ref, item_ref, n_obs,
            cfg, None, primary_rule=lambda s: "ridge:0.5")
        # 题目层对照用题目自身12维
        for mode, (labels, res) in sols.items():
            res["vs_12dim"] = vs_reference(labels, item_ref)
        result["L2_items"] = out
        # 保存题目归属CSV
        labels = sols[pm][0]
        pd.DataFrame({"rule_id": meta.index, "community": labels,
                      "core_value": item_ref}).to_csv(
            res_dir / "ega_L2_item_assignments.csv", index=False, encoding="utf-8-sig")

    # ---- L1: 子主题层(主标签分组) ----
    if "L1" in layers_wanted:
        keys = meta["derived_values"].str.split("|").str[0].tolist()
        Xa, names, refs, counts = aggregate(X, meta, keys, min_n=5)
        print(f"\n[L1] 主标签子主题层: {len(names)}节点(题数中位={int(np.median(list(counts.values())))})",
              flush=True)
        cfg = [("lw", args.boots_l1), ("ridge:0.3", args.boots_l1),
               ("ridge:0.5", args.boots_l1), ("ridge:0.7", args.boots_l1)]

        def primary_rule(sols):
            """LW收缩强度<0.85时LW为主配置, 否则退到ridge:0.5(防近单位阵伪空网)。"""
            if "lw" in sols:
                sh = sols["lw"][1]["shrinkage"]
                if sh is not None and sh < 0.85:
                    return "lw"
            return "ridge:0.5"
        out, sols, pm = run_layer(
            "L1_subthemes", Xa, names, refs, None, n_obs, cfg, None, primary_rule)
        result["L1_subthemes"] = out
        result["L1_subthemes"]["node_item_counts"] = counts
        # 保存CSV + 图
        labels = sols[pm][0]
        pd.DataFrame({"subtheme": names, "community": labels, "modal_12dim": refs,
                      "n_items": [counts[n] for n in names]}).to_csv(
            res_dir / "ega_L1_subtheme_assignments.csv", index=False, encoding="utf-8-sig")
        # 画图需要精度矩阵 → 重算主配置
        R, _ = shrink_corr(Xa, pm)
        best, _ = glasso_ebic(R, n_obs)
        plot_network("L1", Xa, names, labels, refs, best["P"], res_dir / "ega_network_L1.png",
                     f"L1 子主题层 ({len(names)}节点, 主配置={pm})")

    # ---- L0: 12维层 ----
    if "L0" in layers_wanted:
        keys = meta["core_value"].tolist()
        Xa, names, refs, counts = aggregate(X, meta, keys, min_n=3)
        cfg = [("sample", 500), ("ridge:0.3", 500)]
        out, sols, pm = run_layer(
            "L0_core12", Xa, names, refs, None, n_obs, cfg, None,
            primary_rule=lambda s: "sample")
        result["L0_core12"] = out
        result["L0_core12"]["node_item_counts"] = {k: int(v) for k, v in counts.items()}
        labels = sols[pm][0]
        R, _ = shrink_corr(Xa, pm)
        best, _ = glasso_ebic(R, n_obs)
        plot_network("L0", Xa, names, labels, refs, best["P"], res_dir / "ega_network_L0.png",
                     f"L0 十二核心价值层 (主配置={pm})")

    # ---- 保存 ----
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    out_path = res_dir / "ega_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_default)
    print(f"\n结果已保存: {out_path}", flush=True)


if __name__ == "__main__":
    main()
