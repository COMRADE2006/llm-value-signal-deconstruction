# -*- coding: utf-8 -*-
"""
T3 复测信度采集：从1227活跃题分层抽200题，3个模型（doubao/kimi/ds-v4-pro）
各跑 thinking-off 模式一遍，用于重测信度检验（step9_retest_reliability.py）。

- Prompt、模型参数与原全量采集完全一致（复用 runner._build_messages + client.chat）
- 输出 data/raw_resp_agree_retest/{alias}_agree-off.jsonl（独立目录，不污染原数据）
- 断点续跑：重复运行自动跳过已完成题目
- 先试跑：python run_models/retest_run.py --pilot   （每模型5题，验证通道）
- 全量：  python run_models/retest_run.py            （200题×3模型=600次调用）

依赖: llm_env 虚拟环境；API key 已在 .env。
"""
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from api_caller.client import chat, ContentFilterError, ReasoningExhaustedError
from api_caller.runner import _build_messages

logger = config.get_logger()

RETEST_DIR = config.DATA_DIR / "raw_resp_agree_retest"
RETEST_DIR.mkdir(parents=True, exist_ok=True)

RETEST_MODELS = ["doubao-seed-2-1-pro", "kimi-k2.6", "ds-v4-pro"]   # 轴两端+中位
N_SAMPLE = 200
SEED = 42


def sample_items():
    """从1227活跃题按12维分层抽200题（seed=42，可复现）。
    返回 [(rule_id, rule_text, core_value, direction), ...]"""
    active = pd.read_csv(config.DATA_DIR / "results" / "agree" / "active_items.csv")
    rng = np.random.default_rng(SEED)
    picked = []
    dims = active["core_value"].unique()
    # 按比例分配名额，余数给题量最多的维度
    quota = {d: int(round(N_SAMPLE * (active["core_value"] == d).mean())) for d in dims}
    while sum(quota.values()) < N_SAMPLE:
        dmax = max(dims, key=lambda d: (active["core_value"] == d).sum())
        quota[dmax] += 1
    for d in dims:
        pool = active[active["core_value"] == d]
        idx = rng.choice(len(pool), size=quota[d], replace=False)
        picked.append(pool.iloc[idx])
    picked = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)
    # 取题目文本
    texts = {}
    with open(config.FINAL_ITEMS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                texts[it["rule_id"]] = it
    out = []
    for _, r in picked.iterrows():
        it = texts[r["rule_id"]]
        out.append((r["rule_id"], it["rule_text"], it["core_value"],
                    it.get("direction", "positive")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="试跑：每模型只跑前5题")
    args = ap.parse_args()

    items = sample_items()
    if args.pilot:
        items = items[:5]
        print("【试跑模式】每模型仅前5题")

    models = [m for m in config.MODELS if m["alias"] in RETEST_MODELS]
    assert len(models) == 3, f"应找到3个模型，实际{len(models)}"

    print("=" * 62)
    print(f"T3 复测信度采集: {len(items)}题 × {len(models)}模型 × off模式")
    print(f"输出目录: {RETEST_DIR}")
    print("=" * 62)

    for m in models:
        alias = m["alias"]
        out_path = RETEST_DIR / f"{alias}_agree-off.jsonl"
        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["rule_id"])
        todo = [it for it in items if it[0] not in done]
        print(f"\n===== {alias} ===== 待跑 {len(todo)}/{len(items)} 题")

        ok = fail = 0
        for i, (rule_id, rule_text, core_value, direction) in enumerate(todo, 1):
            messages = _build_messages(rule_text, core_value)
            error_type = None
            try:
                raw = chat(m, messages, thinking_on=False)
            except ContentFilterError:
                raw, error_type = None, "content_filter"
            except ReasoningExhaustedError:
                raw, error_type = None, "reasoning_exhausted"

            if raw is None or str(raw).strip() == "" or error_type:
                fail += 1
                if error_type:  # 确定性失败落盘NaN，防止反复重试
                    rec = {"rule_id": rule_id, "repeat": 1, "alias": alias,
                           "thinking_mode": "off", "vendor": m["vendor"],
                           "model": m["model"], "direction": direction,
                           "core_value": core_value, "rule_text": rule_text,
                           "raw_response": None, "error_type": error_type}
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  [{i}/{len(todo)}] rule_id={rule_id} 失败({error_type or '空响应'})，跳过")
                continue
            ok += 1
            rec = {"rule_id": rule_id, "repeat": 1, "alias": alias,
                   "thinking_mode": "off", "vendor": m["vendor"],
                   "model": m["model"], "direction": direction,
                   "core_value": core_value, "rule_text": rule_text,
                   "raw_response": str(raw).strip(), "error_type": None}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(todo):
                print(f"  进度 {i}/{len(todo)}（成功{ok} 失败{fail}）")

    print("\n采集完成。下一步: python analysis/step9_retest_reliability.py")


if __name__ == "__main__":
    main()
