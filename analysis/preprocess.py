# -*- coding: utf-8 -*-
"""
数据预处理（两步分析链的必要前置，总方案v2 §4 第二步配套手术）。

输入: data/results/{source}/response_matrix.csv（build_matrix.py 输出）
手术:
  1. 反向计分: 反向题 X* = 6 − X（方向对齐，高分=认同该行为背后的价值取向）
  2. 剔平指纹题: 计分后跨模型 SD=0（无区分度）或 有效列 < min-valid（覆盖不足）
  3. IR-Tree 三通道分解（Böckenholt 2012，计分后）:
       D 通道(方向):  X*>=4→+1, X*==3→0, X*<=2→−1
       I 通道(强度):  |X*−3|/2 ∈ {1, 0.5, 0}（丢方向，只有强度）
       P 通道(乘积):  D × I ∈ {+1,+0.5,0,−0.5,−1}（保方向保强度）
  4. 双重中心化: X_c = X − 行均值 − 列均值 + 总均值（扣题目支持度基线+模型作答风格，
     只留题目×模型交互结构；迭代 3 次处理 NaN 的 pairwise 偏差）

三通道聚类互为稳健性对照（2026-10-08 用户确认：同时保留强度和方向，分三次聚类）。

输出(全部在 data/results/{source}/ 下):
  scored_matrix.csv        计分后原始分（全题，行=rule_id）
  active_items.csv         剔除后活跃题目清单（含 direction/core_value/derived_values 标签）
  chan_{D,I,P}.csv         三通道矩阵（仅活跃题）
  centered_{orig,D,I,P}.csv 双重中心化矩阵（聚类主输入，仅活跃题）
  exclude_report.csv       剔除明细（平指纹 / 覆盖不足）

用法: python analysis/preprocess.py --source agree|old [--min-valid 11]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CHANNELS = {"orig": None, "D": None, "I": None, "P": None}


def reverse_score(mat, items):
    """反向计分 X*=6−X。mat: DataFrame(index=rule_id)，items: 题目清单。"""
    direction = {it["rule_id"]: it.get("direction", "positive") for it in items}
    neg_ids = [rid for rid in mat.index if direction.get(rid) == "negative"]
    scored = mat.copy()
    scored.loc[neg_ids] = 6.0 - scored.loc[neg_ids]
    return scored, neg_ids


def irtree_channels(scored):
    """IR-Tree 三通道分解（输入为计分后矩阵）。"""
    v = scored.values.astype(float)
    d = np.where(np.isnan(v), np.nan, np.where(v >= 4, 1.0, np.where(v == 3, 0.0, -1.0)))
    i = np.abs(v - 3.0) / 2.0
    p = d * i
    idx = scored.index
    return (pd.DataFrame(d, index=idx, columns=scored.columns),
            pd.DataFrame(i, index=idx, columns=scored.columns),
            pd.DataFrame(p, index=idx, columns=scored.columns))


def double_center(df, n_iter=3):
    """双重中心化（NaN 用 nanmean，迭代逼近 pairwise 一致性）。"""
    x = df.values.astype(float).copy()
    mask = ~np.isnan(x)
    x_filled = np.where(mask, x, 0.0)
    for _ in range(n_iter):
        # 行均值 / 列均值（仅对有效值）
        cnt = mask.sum(axis=1, keepdims=True)
        row_m = np.where(cnt > 0, x_filled.sum(axis=1, keepdims=True) / np.maximum(cnt, 1), 0.0)
        x_tmp = np.where(mask, x_filled - row_m, 0.0)
        cnt_c = mask.sum(axis=0, keepdims=True)
        col_m = np.where(cnt_c > 0, x_tmp.sum(axis=0, keepdims=True) / np.maximum(cnt_c, 1), 0.0)
        grand = x_tmp.sum() / max(mask.sum(), 1)
        x_filled = np.where(mask, x_filled - row_m - col_m + grand, 0.0)
    out = np.where(mask, x_filled, np.nan)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree")
    ap.add_argument("--min-valid", type=int, default=11,
                    help="题目最少有效列数（22列的一半），低于此值判覆盖不足剔除")
    args = ap.parse_args()

    res_dir = Path(config.DATA_DIR) / "results" / args.source
    mat = pd.read_csv(res_dir / "response_matrix.csv", index_col=0)

    # 题目标签
    items = []
    with open(config.FINAL_ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    item_meta = {it["rule_id"]: it for it in items}
    mat = mat.loc[[rid for rid in item_meta if rid in mat.index]]

    # 1) 反向计分
    scored, neg_ids = reverse_score(mat, items)

    # 2) 剔除：覆盖不足 / 平指纹
    valid_cnt = scored.notna().sum(axis=1)
    row_sd = scored.std(axis=1, skipna=True)
    excl_cov = scored.index[valid_cnt < args.min_valid]
    excl_flat = scored.index[(valid_cnt >= args.min_valid) & (row_sd < 1e-9)]
    keep = scored.index[(valid_cnt >= args.min_valid) & (row_sd >= 1e-9)]

    # 3) IR-Tree 三通道（仅活跃题）
    d_ch, i_ch, p_ch = irtree_channels(scored.loc[keep])

    # 4) 双重中心化（orig + 三通道）
    centered = {
        "orig": double_center(scored.loc[keep]),
        "D": double_center(d_ch),
        "I": double_center(i_ch),
        "P": double_center(p_ch),
    }

    # ---------- 落盘 ----------
    scored.to_csv(res_dir / "scored_matrix.csv", encoding="utf-8-sig")
    d_ch.to_csv(res_dir / "chan_D.csv", encoding="utf-8-sig")
    i_ch.to_csv(res_dir / "chan_I.csv", encoding="utf-8-sig")
    p_ch.to_csv(res_dir / "chan_P.csv", encoding="utf-8-sig")
    for name, df in centered.items():
        df.to_csv(res_dir / f"centered_{name}.csv", encoding="utf-8-sig")

    rows = []
    for rid in keep:
        it = item_meta[rid]
        rows.append({
            "rule_id": rid,
            "direction": it.get("direction", "positive"),
            "core_value": it.get("core_value", ""),
            "derived_values": "|".join(it.get("derived_values", [])),
            "n_valid": int(valid_cnt.loc[rid]),
        })
    pd.DataFrame(rows).to_csv(res_dir / "active_items.csv", index=False, encoding="utf-8-sig")

    excl_rows = (
        [{"rule_id": rid, "reason": "coverage", "n_valid": int(valid_cnt.loc[rid])} for rid in excl_cov]
        + [{"rule_id": rid, "reason": "flat_fingerprint", "n_valid": int(valid_cnt.loc[rid])} for rid in excl_flat]
    )
    pd.DataFrame(excl_rows).to_csv(res_dir / "exclude_report.csv", index=False, encoding="utf-8-sig")

    # ---------- 报告 ----------
    n = len(scored)
    print("=" * 70)
    print(f"[source={args.source}] 预处理完成")
    print(f"总题数 {n}（正向 {n - len(neg_ids)} / 反向 {len(neg_ids)}，反向已计分 X*=6−X）")
    print(f"剔除: 覆盖不足(<{args.min_valid}列) {len(excl_cov)} 题, 平指纹(SD=0) {len(excl_flat)} 题"
          f" → 共剔除 {n - len(keep)} 题 ({(n - len(keep)) / n * 100:.1f}%)")
    print(f"活跃题 {len(keep)} 题，三通道+双重中心化矩阵已输出")
    if len(keep) < 500:
        print("⚠ 活跃题过少，检查数据采集完整度")
    print(f"输出目录: {res_dir}")


if __name__ == "__main__":
    main()
