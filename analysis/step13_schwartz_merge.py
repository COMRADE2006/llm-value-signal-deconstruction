# -*- coding: utf-8 -*-
"""
step13 Schwartz标签合并与全管线重算（内部操作）：
将 gpt-4o 打标结果（含人工裁决修正）合并进题库，生成 v3 题库；
以 v3 的维度/方向重跑计分-剔除-IR-Tree-双重中心化全管线，输出到 results/schwartz/
（不覆盖 results/agree/，旧框架结果留作内部对照）。

输入:
  data/schwartz_labels.jsonl            gpt-4o 全量打标（rule_id/schwartz_main/valence/confidence）
  data/review_schwartz_lowconf.csv      低置信裁决CSV（可选，"修改为"列: 维度名 或 维度名|positive）
  data/review_schwartz_trial.csv        试标CSV（可选，同样格式，你已确认全通过）
  data/final_items_v2.jsonl             原题库
  data/results/agree/response_matrix.csv 原始响应（未计分）

输出:
  data/final_items_v3.jsonl             v3题库（direction=valence, core_value=schwartz维度,
                                         保留 old_core_value/old_direction 作内部对照）
  data/results/schwartz/ 全套           scored_matrix / chan_{D,I,P} / centered_{orig,D,I,P}
                                         active_items.csv（含schwartz标签列）/ exclude_report.csv

数学事实（预处理前已核实）：
  - 平指纹剔除与direction无关（SD(6-X)=SD(X)）→ keep集合必须与旧框架完全一致（断言）
  - I通道（强度|X-3|/2）与direction无关 → chan_I/centered_I 应与旧版逐元素一致（断言）

用法: python analysis/step13_schwartz_merge.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis.preprocess import irtree_channels, double_center

DATA = Path(config.DATA_DIR)
IN_SRC = DATA / "results" / "agree"      # 原始响应来源
OUT_DIR = DATA / "results" / "schwartz"  # 新框架输出
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_overrides():
    """读取裁决CSV的'修改为'列。格式: 维度名 或 维度名|positive/negative。"""
    ov = {}
    for csv_path in [DATA / "review_schwartz_lowconf.csv", DATA / "review_schwartz_trial.csv"]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")
        for _, row in df.iterrows():
            fix = row.get("修改为", "").strip() or row.get("修改为(留空=通过)", "").strip()
            if fix:
                parts = fix.split("|")
                dim = parts[0].strip()
                val = parts[1].strip().lower() if len(parts) > 1 else None
                ov[int(row["rule_id"])] = (dim, val)
    return ov


def main():
    # ---------- 1) 加载打标结果 + 裁决覆盖 ----------
    labels = {}
    with open(DATA / "schwartz_labels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels[r["rule_id"]] = r
    n_missing = 0
    items = []
    with open(DATA / "final_items_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    overrides = load_overrides()
    print(f"打标结果 {len(labels)} 条 | 题库 {len(items)} 条 | 人工裁决修正 {len(overrides)} 条")

    v3_items = []
    flip_cnt = 0
    for it in items:
        lab = labels.get(it["rule_id"])
        if lab is None:
            n_missing += 1
            continue
        dim, val = lab["schwartz_main"], lab["valence"]
        if it["rule_id"] in overrides:
            odim, oval = overrides[it["rule_id"]]
            dim = odim or dim
            val = oval or val
        if it["direction"] != val:
            flip_cnt += 1
        v3_items.append({
            "rule_id": it["rule_id"],
            "rule_text": it["rule_text"],
            "direction": val,                      # 新方向 = valence
            "core_value": dim,                     # 新维度
            "old_direction": it["direction"],      # 旧标签留档（内部对照）
            "old_core_value": it["core_value"],
            "derived_values": it.get("derived_values", []),
            "level": it.get("level", []),
            "all_core_values": it.get("all_core_values", []),
            "schwartz_confidence": lab["schwartz_confidence"],
            "review_status": it.get("review_status", ""),
        })
    assert n_missing == 0, f"有{n_missing}题未打标，先跑完 label_schwartz.py --full"
    with open(DATA / "final_items_v3.jsonl", "w", encoding="utf-8") as f:
        for it in v3_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(it["core_value"] for it in v3_items)
    dist_active_dims = [d for d, n in dist.items() if n >= 30]  # 题数>=30的维度
    print(f"v3题库 {len(v3_items)} 条已写入 final_items_v3.jsonl")
    print(f"维度分布: {dict(dist.most_common())}")
    print(f"方向翻转题数: {flip_cnt}/{len(v3_items)} ({flip_cnt/len(v3_items)*100:.1f}%)")
    print(f"题数>=30的维度（强设K候选）: {sorted(dist_active_dims)} → K={len(dist_active_dims)}")

    # ---------- 2) 重跑计分管线（direction=v3 valence） ----------
    mat = pd.read_csv(IN_SRC / "response_matrix.csv", index_col=0)
    item_meta = {it["rule_id"]: it for it in v3_items}
    mat = mat.loc[[rid for rid in item_meta if rid in mat.index]]

    direction = {it["rule_id"]: it["direction"] for it in v3_items}
    neg_ids = [rid for rid in mat.index if direction.get(rid) == "negative"]
    scored = mat.copy()
    scored.loc[neg_ids] = 6.0 - scored.loc[neg_ids]

    # 剔除逻辑（与preprocess.py一致）
    min_valid = 11
    valid_cnt = scored.notna().sum(axis=1)
    row_sd = scored.std(axis=1, skipna=True)
    excl_cov = scored.index[valid_cnt < min_valid]
    excl_flat = scored.index[(valid_cnt >= min_valid) & (row_sd < 1e-9)]
    keep = scored.index[(valid_cnt >= min_valid) & (row_sd >= 1e-9)]

    # 断言1：keep集合与旧框架完全一致（SD对翻转不变）
    old_active = pd.read_csv(IN_SRC / "active_items.csv", encoding="utf-8-sig")
    old_keep = set(old_active["rule_id"])
    assert set(keep) == old_keep, \
        f"keep集合不一致! 新{len(keep)} vs 旧{len(old_keep)}，差异{set(keep) ^ old_keep}"

    # IR-Tree三通道 + 双重中心化
    d_ch, i_ch, p_ch = irtree_channels(scored.loc[keep])
    centered = {
        "orig": double_center(scored.loc[keep]),
        "D": double_center(d_ch),
        "I": double_center(i_ch),
        "P": double_center(p_ch),
    }

    # 断言2：I通道与旧版逐元素一致（|X-3|/2对翻转不变）
    old_I = pd.read_csv(IN_SRC / "centered_I.csv", index_col=0)
    new_I = centered["I"].reindex(index=old_I.index, columns=old_I.columns)
    diff = np.nanmax(np.abs(new_I.values - old_I.values))
    assert diff < 1e-9, f"I通道不一致! max|diff|={diff}"

    # ---------- 3) 落盘 ----------
    scored.to_csv(OUT_DIR / "scored_matrix.csv", encoding="utf-8-sig")
    for name, df in [("D", d_ch), ("I", i_ch), ("P", p_ch)]:
        df.to_csv(OUT_DIR / f"chan_{name}.csv", encoding="utf-8-sig")
    for name, df in centered.items():
        df.to_csv(OUT_DIR / f"centered_{name}.csv", encoding="utf-8-sig")

    rows = []
    for rid in keep:
        it = item_meta[rid]
        rows.append({
            "rule_id": rid,
            "direction": it["direction"],
            "core_value": it["core_value"],
            "old_core_value": it["old_core_value"],
            "schwartz_confidence": it["schwartz_confidence"],
            "derived_values": "|".join(it.get("derived_values", [])),
            "n_valid": int(valid_cnt.loc[rid]),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "active_items.csv", index=False, encoding="utf-8-sig")
    excl_rows = (
        [{"rule_id": rid, "reason": "coverage", "n_valid": int(valid_cnt.loc[rid])} for rid in excl_cov]
        + [{"rule_id": rid, "reason": "flat_fingerprint", "n_valid": int(valid_cnt.loc[rid])} for rid in excl_flat]
    )
    pd.DataFrame(excl_rows).to_csv(OUT_DIR / "exclude_report.csv", index=False, encoding="utf-8-sig")

    # 活跃题维度分布
    act_dist = Counter(item_meta[rid]["core_value"] for rid in keep)
    print(f"\n[schwartz] 管线重算完成 → {OUT_DIR}")
    print(f"活跃题 {len(keep)}（与旧框架一致✓），剔除 覆盖{len(excl_cov)}+平指纹{len(excl_flat)}")
    print(f"I通道与旧版逐元素一致✓（max|diff|={diff:.2e}）")
    print(f"活跃题维度分布: {dict(act_dist.most_common())}")


if __name__ == "__main__":
    main()
