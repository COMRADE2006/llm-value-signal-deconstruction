# -*- coding: utf-8 -*-
"""
plot_items_distribution.py：画 12 维题目分布堆叠柱状图 + 输出统计 CSV。

输入：data/final_items_v2.jsonl
输出：
  - data/results/items_distribution.png  12 维堆叠柱状图（正面/负面）
  - data/results/items_distribution.csv  12 维 × 正面/负面 × 题数
  - data/results/subtopic_distribution.png  子主题横向堆叠条形图
  - data/results/subtopic_distribution.csv  50 个子主题 × 正面/负面 × 题数

用法：python run_models/plot_items_distribution.py
"""
import sys
import os
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# matplotlib 配置中文字体
import matplotlib
matplotlib.use("Agg")  # 无窗口模式
import matplotlib.pyplot as plt

# 尝试多个中文字体
for font in ["Microsoft YaHei", "SimHei", "Microsoft YaHei UI", "Source Han Sans CN"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def load_final_items(path: Path) -> list:
    """读最终量表 jsonl。"""
    if not path.exists():
        print(f"错误：{path} 不存在")
        print(f"请先运行: python run_models/apply_review_v2.py")
        sys.exit(1)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def plot_distribution():
    """画堆叠柱状图 + 输出统计 CSV。"""
    config.ensure_dirs()

    final_path = config.DATA_DIR / "final_items_v2.jsonl"
    items = load_final_items(final_path)
    print(f"读取最终量表: {len(items)} 条")

    # 统计：12 维 × 正面/负面
    stats = defaultdict(lambda: {"positive": 0, "negative": 0})
    for it in items:
        cv = it["core_value"]
        d = it["direction"]
        if d in ("positive", "negative"):
            stats[cv][d] += 1

    # 12 维固定顺序
    dims = config.CORE_VALUES
    pos_counts = [stats[d]["positive"] for d in dims]
    neg_counts = [stats[d]["negative"] for d in dims]
    total_counts = [p + n for p, n in zip(pos_counts, neg_counts)]

    # ============== 画图 ==============
    print(f"\n画堆叠柱状图...")
    fig, ax = plt.subplots(figsize=(14, 7))

    x = range(len(dims))
    # 负面在下（用深色），正面在上（用浅色）
    bar_neg = ax.bar(x, neg_counts, label="负面（反向）", color="#d62728", edgecolor="white")
    bar_pos = ax.bar(x, pos_counts, bottom=neg_counts, label="正面（正向）", color="#2ca02c", edgecolor="white")

    # 标注数字
    for i, (p, n, t) in enumerate(zip(pos_counts, neg_counts, total_counts)):
        # 负面段中间标数字
        if n > 0:
            ax.text(i, n / 2, str(n), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        # 正面段中间标数字
        if p > 0:
            ax.text(i, n + p / 2, str(p), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        # 柱顶标总数
        ax.text(i, t + 2, str(t), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=12)
    ax.set_ylabel("题数", fontsize=13)
    ax.set_title("C-VARC 12 维度最终量表题目分布（正面/负面）", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, max(total_counts) * 1.15)

    # 在右上角标注总题数
    total_all = sum(total_counts)
    ax.text(0.98, 0.98, f"总计 {total_all} 题", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_png = config.RESULTS_DIR / "items_distribution.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {out_png}")
    plt.close()

    # ============== 输出统计 CSV ==============
    out_csv = config.RESULTS_DIR / "items_distribution.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["维度", "正面题数", "负面题数", "总计"])
        for d, p, n, t in zip(dims, pos_counts, neg_counts, total_counts):
            writer.writerow([d, p, n, t])
        writer.writerow([])
        writer.writerow(["总计", sum(pos_counts), sum(neg_counts), total_all])
    print(f"  统计已保存: {out_csv}")

    # ============== 打印统计表 ==============
    print(f"\n=== 12 维题目分布统计 ===")
    print(f"{'维度':<6} {'正面':>6} {'负面':>6} {'总计':>6}")
    print("-" * 30)
    for d, p, n, t in zip(dims, pos_counts, neg_counts, total_counts):
        print(f"{d:<6} {p:>6} {n:>6} {t:>6}")
    print("-" * 30)
    print(f"{'总计':<6} {sum(pos_counts):>6} {sum(neg_counts):>6} {total_all:>6}")

    # ============== 子主题（derived_values）分布 ==============
    # 方案 A 全计入：一题挂多个子主题时，每个子主题都计一次
    sub_stats = defaultdict(lambda: {"positive": 0, "negative": 0})
    for it in items:
        d = it["direction"]
        if d not in ("positive", "negative"):
            continue
        for sv in it.get("derived_values", []):
            sub_stats[sv][d] += 1

    # 按总题数从大到小排序
    sub_sorted = sorted(
        sub_stats.items(),
        key=lambda kv: kv[1]["positive"] + kv[1]["negative"],
        reverse=True,
    )
    sub_names = [k for k, _ in sub_sorted]
    sub_pos = [v["positive"] for _, v in sub_sorted]
    sub_neg = [v["negative"] for _, v in sub_sorted]
    sub_tot = [p + n for p, n in zip(sub_pos, sub_neg)]

    print(f"\n画子主题分布横向堆叠条形图（{len(sub_names)} 个子主题）...")
    # 50 个标签用横向条形图，高度随数量自适应
    fig_h = max(10, len(sub_names) * 0.42)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    y = range(len(sub_names))
    # 正面（绿）在左，负面（红）接在右
    ax.barh(y, sub_pos, label="正面（正向）", color="#2ca02c", edgecolor="white")
    ax.barh(y, sub_neg, left=sub_pos, label="负面（反向）", color="#d62728", edgecolor="white")

    # 标注数字
    for i, (p, n, t) in enumerate(zip(sub_pos, sub_neg, sub_tot)):
        if p > 0:
            ax.text(p / 2, i, str(p), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        if n > 0:
            ax.text(p + n / 2, i, str(n), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        # 条右端标总数
        ax.text(t + max(sub_tot) * 0.01, i, str(t), ha="left", va="center",
                fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(sub_names, fontsize=10)
    ax.invert_yaxis()  # 题数最多的在最上方
    ax.set_xlabel("题数（一题多标签全计入）", fontsize=12)
    ax.set_title("最终量表子主题（derived values）题目分布（正面/负面）",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, max(sub_tot) * 1.12)

    plt.tight_layout()
    out_sub_png = config.RESULTS_DIR / "subtopic_distribution.png"
    plt.savefig(out_sub_png, dpi=150, bbox_inches="tight")
    print(f"  图已保存: {out_sub_png}")
    plt.close()

    # 子主题统计 CSV
    out_sub_csv = config.RESULTS_DIR / "subtopic_distribution.csv"
    with open(out_sub_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["子主题", "正面题数", "负面题数", "总计"])
        for name, p, n, t in zip(sub_names, sub_pos, sub_neg, sub_tot):
            writer.writerow([name, p, n, t])
    print(f"  统计已保存: {out_sub_csv}")

    # 打印子主题统计表
    print(f"\n=== 子主题题目分布统计（共 {len(sub_names)} 个，全计入口径）===")
    print(f"{'子主题':<10} {'正面':>5} {'负面':>5} {'总计':>5}")
    print("-" * 30)
    for name, p, n, t in zip(sub_names, sub_pos, sub_neg, sub_tot):
        print(f"{name:<10} {p:>5} {n:>5} {t:>5}")

    print(f"\n图和 CSV 已生成。下一步：人工检查两张图，然后讨论子主题下限")


if __name__ == "__main__":
    plot_distribution()
