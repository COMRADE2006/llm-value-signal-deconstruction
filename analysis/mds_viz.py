# -*- coding: utf-8 -*-
"""
武器一：MDS 构型可视化（50 子主题，总方案v2 §5 武器一）。

操作:
  - 50 子主题双重中心化行均值矩阵 → 1−Pearson 距离 → 非度量 MDS 二维
  - stress 报告（Kruskal 准则: <0.1 可接受；二维装不下本身是结论）
  - 12 维质心投影（该维度题目的 MDS 坐标均值，展示性参考，不做验证性解读）
  - 三结局预注册: 挤团/一维排序/环形扇形 —— 形态判读由人工看图完成，脚本不自动宣称

输入: data/results/{source}/centered_orig.csv + active_items.csv
输出: data/results/{source}/mds_subtopics.png, mds_subtopics_coords.csv,
      mds_result.json

用法: python analysis/mds_viz.py --source agree|old [--dim 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--dim", type=int, default=2)
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    cent = pd.read_csv(res_dir / "centered_orig.csv", index_col=0)
    active = pd.read_csv(res_dir / "active_items.csv")
    sub_map = active.set_index("rule_id")["derived_values"].fillna("").astype(str).str.split("|")

    # 子主题行均值（双重中心化后）
    rows, ids = [], []
    for rid in cent.index:
        for s in sub_map.get(rid, []):
            if s:
                ids.append(s)
                rows.append(cent.loc[rid].values.astype(float))
    sub_df = pd.DataFrame(rows, index=ids, columns=cent.columns)
    sub_mean = sub_df.groupby(level=0).mean()
    X = np.nan_to_num(sub_mean.values)

    # 1 − Pearson 距离
    xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(xc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    corr = (xc / norm) @ (xc / norm).T
    D = np.clip(1.0 - corr, 0.0, 2.0)

    mds = MDS(n_components=args.dim, dissimilarity="precomputed", random_state=config.SEED,
              n_init=10, max_iter=500)
    coords = mds.fit_transform(D)
    stress = float(mds.stress_)

    # stress 归一化（Kruskal stress-1: sqrt(stress_raw / sum(d²))，sklearn 已是 S-stress 口径，两者都报）
    norm_stress = float(np.sqrt(mds.stress_ / np.sum(D ** 2)))

    coords_df = pd.DataFrame(coords, index=sub_mean.index, columns=[f"MDS{i+1}" for i in range(args.dim)])
    coords_df.to_csv(res_dir / "mds_subtopics_coords.csv", encoding="utf-8-sig")

    # 12 维质心投影
    dim_pts = {}
    dim_map = active.set_index("rule_id")["core_value"]
    item_pos = {s: i for i, s in enumerate(sub_mean.index)}
    for v in dim_map.unique():
        pts = []
        for rid in cent.index:
            if dim_map.get(rid) == v:
                for s in sub_map.get(rid, []):
                    if s and s in item_pos:
                        pts.append(coords[item_pos[s]])
        if pts:
            dim_pts[v] = np.mean(pts, axis=0)

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c="#4C72B0", alpha=0.85, zorder=3)
    for i, s in enumerate(sub_mean.index):
        ax.annotate(s, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")
    if dim_pts:
        dx = np.array([p[0] for p in dim_pts.values()])
        dy = np.array([p[1] for p in dim_pts.values()])
        ax.scatter(dx, dy, s=340, marker="X", c="#C44E52", alpha=0.9, zorder=4,
                   edgecolors="white", linewidths=1.2)
        for v, p in dim_pts.items():
            ax.annotate(v, (p[0], p[1]), fontsize=11, fontweight="bold", color="#C44E52",
                        ha="center", va="center", zorder=5)
        ax.scatter([], [], marker="X", s=200, c="#C44E52", label="12维质心(展示性参考)")
        ax.scatter([], [], s=60, c="#4C72B0", label="50子主题")
        ax.legend(loc="best")
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_title(f"50 子主题 MDS 构型（orig 通道，stress={norm_stress:.3f}）\n"
                 f"形态判读（预注册三结局）: 挤团 / 一维排序 / 环形扇形 —— 人工看图认定，不宣称验证任何理论结构")
    fig.tight_layout()
    out_png = res_dir / "mds_subtopics.png"
    fig.savefig(out_png, dpi=150)

    verdict = ("可接受（<0.10）" if norm_stress < 0.10 else
               "偏高——二维构型装不下，'响应空间至少三维'本身是结论")
    result = {"source": args.source, "n_subtopics": int(len(sub_mean)),
              "stress_normalized": norm_stress, "stress_verdict": verdict}
    with open(res_dir / "mds_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"[source={args.source}] MDS 二维构型: {len(sub_mean)} 个子主题")
    print(f"归一化 stress = {norm_stress:.4f} → {verdict}")
    print(f"图: {out_png}")
    print(f"坐标: {res_dir / 'mds_subtopics_coords.csv'}")
    print("判读提醒: 三结局预注册（挤团/一维/环形扇形），只做探索性描述，不宣称验证")


if __name__ == "__main__":
    main()
