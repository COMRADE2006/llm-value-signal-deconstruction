# -*- coding: utf-8 -*-
"""
apply_review_v2.py：应用人工审核结果，生成最终量表 final_items_v2.jsonl。

读取 review_items_v2.csv（GBK 编码，7 列）：
  列 0：序号（忽略）
  列 1：rule_id
  列 2：core_value（用户可能改过）
  列 3：原始规则（忽略）
  列 4：改写情景句（AI 版）
  列 5：方向（列名为空，值="正面"/"负面"）
  列 6：修改后的句子（留空=用 AI 版）

逻辑：
  1. rule_text = 列 6 有内容 → 用列 6；否则 → 用列 4
  2. direction = 列 5（"正面"→positive，"负面"→negative）
  3. core_value = 列 2（覆盖原始）
  4. derived_values / level / all_core_values 从 sampled_items_v2.jsonl 按 rule_id 查找
  5. config.EXCLUDE_RULE_IDS 中的兜底句（86913/144211）直接剔除

输出：data/final_items_v2.jsonl

用法：python run_models/apply_review_v2.py
"""
import sys
import os
import json
import csv
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============== 路径 ==============
CSV_PATH = config.DATA_DIR / "review_items_v2.csv"
ORIG_JSONL_PATH = config.DATA_DIR / "sampled_items_v2.jsonl"
OUT_PATH = config.DATA_DIR / "final_items_v2.jsonl"

def load_orig_jsonl(path: Path) -> dict:
    """读原始 jsonl，返回 {rule_id: item_dict}。"""
    mapping = {}
    if not path.exists():
        print(f"错误：{path} 不存在")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                mapping[d["rule_id"]] = d
    return mapping


def load_csv(path: Path) -> list:
    """读 CSV（GBK），返回行列表。"""
    if not path.exists():
        print(f"错误：{path} 不存在")
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="gbk") as f:
        reader = csv.reader(f)
        header = next(reader)  # 表头
        for row in reader:
            if len(row) < 7:
                print(f"警告：行长度不足 7，跳过：{row}")
                continue
            rows.append(row)
    return header, rows


def apply_review():
    """应用审核结果，生成最终量表。"""
    config.ensure_dirs()

    # 1. 读原始 jsonl
    print(f"[1/4] 读取原始 jsonl: {ORIG_JSONL_PATH.name}")
    orig_mapping = load_orig_jsonl(ORIG_JSONL_PATH)
    print(f"  共 {len(orig_mapping)} 条原始题")

    # 2. 读 CSV
    print(f"[2/4] 读取审核 CSV: {CSV_PATH.name}")
    header, rows = load_csv(CSV_PATH)
    print(f"  共 {len(rows)} 行")
    print(f"  表头: {header}")

    # 3. 生成最终量表
    print(f"[3/4] 生成最终量表...")
    final_items = []
    manual_modified = 0
    cv_changed = 0
    excluded = 0

    for row in rows:
        rid = int(row[1])

        # 剔除 C-VARC 兜底句（挂全部50子主题标签，无区分度）
        if rid in config.EXCLUDE_RULE_IDS:
            excluded += 1
            continue

        new_core_value = row[2].strip()
        ai_sentence = row[4].strip()
        direction_raw = row[5].strip()
        manual_sentence = row[6].strip()

        # 方向转换
        if direction_raw == "正面":
            direction = "positive"
        elif direction_raw == "负面":
            direction = "negative"
        else:
            print(f"警告：rule_id={rid} 方向值未知 '{direction_raw}'，跳过")
            continue

        # rule_text：优先手动修改
        if manual_sentence:
            rule_text = manual_sentence
            review_status = "manual_modified"
            manual_modified += 1
        else:
            rule_text = ai_sentence
            review_status = "ai_original"

        # core_value：跟原始对比，统计改动
        orig_item = orig_mapping.get(rid, {})
        orig_cv = orig_item.get("core_value", "")
        if new_core_value != orig_cv:
            cv_changed += 1

        # 合并字段
        item = {
            "rule_id": rid,
            "rule_text": rule_text,
            "direction": direction,
            "core_value": new_core_value,
            "derived_values": orig_item.get("derived_values", []),
            "level": orig_item.get("level", []),
            "all_core_values": orig_item.get("all_core_values", []),
            "review_status": review_status,
        }
        final_items.append(item)

    # 写出
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in final_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  写出: {OUT_PATH}")
    print(f"  总计: {len(final_items)} 条")

    # 4. 统计报告
    print(f"\n[4/4] 统计报告")
    dir_cnt = Counter(it["direction"] for it in final_items)
    status_cnt = Counter(it["review_status"] for it in final_items)
    cv_dist = Counter(it["core_value"] for it in final_items)

    print(f"  方向分布: {dict(dir_cnt)}")
    print(f"  审核状态: {dict(status_cnt)}")
    print(f"  手动修改句: {manual_modified} 条")
    print(f"  core_value 改动: {cv_changed} 条")
    print(f"  兜底句剔除: {excluded} 条 (rule_id {sorted(config.EXCLUDE_RULE_IDS)})")
    print(f"\n  最终 12 维分布:")
    for v in config.CORE_VALUES:
        print(f"    {v}: {cv_dist.get(v, 0)}")

    print(f"\n下一步: python run_models/plot_items_distribution.py")


if __name__ == "__main__":
    apply_review()
