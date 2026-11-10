# -*- coding: utf-8 -*-
"""
查看各模型 API 调用进度。
用法：python run_models/show_progress.py
"""
import sys
import os
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if __name__ == "__main__":
    config.ensure_dirs()
    items_path = config.FINAL_ITEMS_PATH
    total_items = 0
    if items_path.exists():
        with open(items_path, "r", encoding="utf-8") as f:
            total_items = sum(1 for line in f if line.strip())
    else:
        print("警告: 最终量表不存在（final_items_v2.jsonl）")
        print("请先运行: python run_models/apply_review_v2.py")
        sys.exit(0)

    repeat = config.REPEAT
    print("=" * 70)
    print(f"题目总数: {total_items}，每题重复 {repeat} 次（认同度盲评新Prompt，数据在 raw_resp_agree/）")
    print("=" * 70)
    print(f"{'模型':<25} {'thinking':<10} {'已完成':<8} {'目标':<8} {'进度':<8}")
    print("-" * 70)

    # 检查每个模型的进度
    for m in config.MODELS:
        alias = m["alias"]
        for mode in m["thinking_modes"]:
            fname = f"{alias}_agree-{mode}.jsonl"
            fpath = config.AGREE_RESP_DIR / fname
            done = 0
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    done = sum(1 for line in f if line.strip())
            target = total_items * repeat
            pct = f"{done/target*100:.1f}%" if target > 0 else "0%"
            print(f"{alias:<25} {mode:<10} {done:<8} {target:<8} {pct}")

    print("=" * 70)
    print("提示: 断点续跑——重复运行 run_*.py 会自动跳过已完成的题目。")
