# -*- coding: utf-8 -*-
"""
构建响应矩阵 + 数据质量检查（聚类前置校验）。

输入 (--source agree, 默认):
  - data/final_items_v2.jsonl                正式量表 2388 题
  - data/raw_resp_agree/{alias}_agree-*.jsonl  盲评认同度新数据（论文主数据）
输入 (--source old):
  - data/raw_resp/{alias}_think-*.jsonl      旧"符合度"数据（仅管线空跑验证用）
输出:
  - data/results/{agree|old}/response_matrix.csv  行=rule_id, 列=模型_模式, 值=重复均值(NaN=无效)
  - stdout 质量检查报告

用法: python analysis/build_matrix.py [--source agree|old] [--include-appendix]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# 主分析 11 模型（hy4-preview / qwen3.5-plus / qwen3.5-flash 已弃用；step 仅 on 档放附录）
MAIN_MODELS = [
    "doubao-seed-2-1-pro", "ds-v4-flash", "ds-v4-pro",
    "glm-5-turbo", "glm-5.1", "glm-5.2",
    "hy3", "kimi-k2.6",
    "qwen3.7-max", "qwen3.8-max",
    "sensenova-flash-lite",
]
APPENDIX_MODELS = ["step-3.7-flash"]

SCORE_RE = re.compile(r"^[1-5]$")


def parse_score(raw):
    """非数字响应（空串/拒答话术/null）一律 NaN。"""
    if raw is None:
        return np.nan
    s = str(raw).strip()
    return float(s) if SCORE_RE.fullmatch(s) else np.nan


def load_items():
    items = []
    with open(config.FINAL_ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_model_column(alias, mode, qc, source):
    """读一个模型一个模式的 JSONL，返回 (Series: rule_id->重复均值, 明细dict)。"""
    if source == "agree":
        path = config.AGREE_RESP_DIR / f"{alias}_agree-{mode}.jsonl"
    else:
        path = config.RAW_RESP_DIR / f"{alias}_think-{mode}.jsonl"
    if not path.exists():
        qc.append(f"  ⚠ 文件不存在: {path.name}")
        return None, None
    # 按 (rule_id, repeat) 去重保留最后一条（防 sensenova 双进程重复写入类问题）
    latest = {}
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            r = json.loads(line)
            latest[(r["rule_id"], r["repeat"])] = parse_score(r.get("raw_response"))
    dup = n_lines - len(latest)
    if dup > 0:
        qc.append(f"  ⚠ {path.name}: 发现重复键 {dup} 条，已按最后写入去重")
    df = pd.DataFrame([(k[0], k[1], v) for k, v in latest.items()],
                      columns=["rule_id", "repeat", "score"])
    mean_by_item = df.groupby("rule_id")["score"].mean()
    # 重复间一致性：每题3次重复的标准差，再对全部题取均值
    sd_by_item = df.groupby("rule_id")["score"].std()
    rep_sd = float(np.nanmean(sd_by_item.values))
    return mean_by_item, {"n_lines": n_lines, "n_items": len(mean_by_item), "rep_sd": rep_sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["agree", "old"], default="agree",
                    help="agree=盲评认同度新数据(论文主数据)；old=旧符合度数据(仅空跑验证)")
    ap.add_argument("--include-appendix", action="store_true", help="把附录模型也放进矩阵（仅稳健性检验用）")
    args = ap.parse_args()

    items = load_items()
    all_ids = [it["rule_id"] for it in items]
    direction = {it["rule_id"]: it.get("direction", "positive") for it in items}
    models = MAIN_MODELS + (APPENDIX_MODELS if args.include_appendix else [])

    qc = []
    cols, meta = {}, {}
    for alias in models:
        for mode in ("on", "off"):
            col, m = load_model_column(alias, mode, qc, args.source)
            if col is None:
                continue
            cols[f"{alias}_{mode}"] = col
            meta[f"{alias}_{mode}"] = m

    mat = pd.DataFrame(index=all_ids, columns=list(cols.keys()))
    for c, s in cols.items():
        mat[c] = s.reindex(all_ids).values

    out_dir = Path(config.DATA_DIR) / "results" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "response_matrix.csv"
    mat.to_csv(out_csv, encoding="utf-8-sig")
    print(f"[source={args.source}]")

    # ---------- 质量报告 ----------
    n = len(all_ids)
    print("=" * 70)
    print(f"响应矩阵: {mat.shape[0]} 题 × {mat.shape[1]} 列  ->  {out_csv}")
    print("=" * 70)
    print(f"\n{'列':34s} {'有效题数':>8s} {'NaN题':>6s} {'重复sd':>7s} {'均值':>6s}  分布1..5")
    for c in mat.columns:
        v = mat[c].values.astype(float)
        valid = int(np.sum(~np.isnan(v)))
        nan = n - valid
        m = meta[c]
        dist = pd.Series(v[~np.isnan(v)]).value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
        print(f"{c:34s} {valid:>8d} {nan:>6d} {m['rep_sd']:>7.3f} {np.nanmean(v):>6.2f}  " +
              " ".join(f"{x:>5d}" for x in dist.values))

    # 全模型都 NaN 的题（内容拦截题）
    all_nan = mat.index[mat.isna().all(axis=1)]
    print(f"\n全列 NaN 题数: {len(all_nan)}" + (f"  rule_id={list(all_nan[:20])}" if len(all_nan) else ""))

    # 正反题均分 sanity check：反向题模型应普遍给低分（高分=误判）
    print("\n--- 正向题均分 / 反向题均分（反向应明显更低，否则该模型可能不理解反向计分）---")
    pos_mask = np.array([direction.get(rid, "positive") == "positive" for rid in mat.index])
    for c in mat.columns:
        v = mat[c].values.astype(float)
        p = np.nanmean(v[pos_mask])
        r = np.nanmean(v[~pos_mask])
        flag = "  ⚠ 反差不明显" if (r - p) > -0.3 else ""
        print(f"{c:34s} 正向={p:.2f}  反向={r:.2f}  差={r - p:+.2f}{flag}")

    # 聚类就绪判定
    print("\n" + "=" * 70)
    incomplete = [c for c in mat.columns if meta[c]["n_items"] < n]
    if incomplete:
        print("⚠ 未跑完的列（聚类前需补齐或剔除）:")
        for c in incomplete:
            print(f"   {c}: {meta[c]['n_items']}/{n}")
    else:
        print("✅ 所有列题目覆盖完整")
    for w in qc:
        print(w)


if __name__ == "__main__":
    main()
