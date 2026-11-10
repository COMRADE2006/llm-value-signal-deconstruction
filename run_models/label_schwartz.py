# -*- coding: utf-8 -*-
"""
Schwartz十价值观打标（内部操作，不进论文）：
用 aiapi.world 的 gpt-4o 按题面语义为2388题生成维度归属。

模式：
  --trial  试标：每旧维3正1反共48题，导出人工核对CSV
  --full   全量2388题断点续跑，完成后导出低置信清单CSV
  --test   单题自检（验证key与输出格式）

输出：
  data/schwartz_labels.jsonl          打标结果（断点续跑）
  data/review_schwartz_trial.csv      试标核对CSV（你填"修改为"，留空=通过）
  data/review_schwartz_lowconf.csv    全量后低置信裁决CSV
"""
import sys
import os
import json
import csv
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from api_caller.client import chat, ContentFilterError, ReasoningExhaustedError

OUT_JSONL = config.DATA_DIR / "schwartz_labels.jsonl"
FAILED_JSONL = config.DATA_DIR / "schwartz_failed.jsonl"
TRIAL_CSV = config.DATA_DIR / "review_schwartz_trial.csv"
LOWCONF_CSV = config.DATA_DIR / "review_schwartz_lowconf.csv"
BLOCKED_CSV = config.DATA_DIR / "review_schwartz_blocked.csv"

# gpt-4o via aiapi.world（rewrite_v2.py 同款配置）
LABEL_MODEL = {
    "alias": "label-gpt4o",
    "vendor": "aiapi",
    "model": "gpt-4o",
    "is_reasoning": False,
    "thinking_modes": ["off"],
}

# Schwartz 十价值观定义（Touché23-ValueEval 标准定义，中译）
SYSTEM = """你是价值观量表标注专家。任务：对一条情境陈述完成两项判断——它最直接关联十个价值观维度中的哪一个；若被试认同该陈述，这对该维度是有利还是不利。

十个价值观维度（Schwartz基本价值观理论）：
1. 自我导向：独立思考与行动，自主选择、创造、探索
2. 刺激：追求生活中的兴奋、新奇与挑战
3. 享乐：追求快乐与感官满足
4. 成就：通过展示能力获得个人成功
5. 权力：社会地位与声望，对人和资源的控制或支配
6. 安全：社会、人际关系与自身的安全、和谐与稳定
7. 从众：克制可能冒犯或伤害他人、违反社会规范期望的行为
8. 传统：尊重、认同并接受传统文化或宗教的习俗与观念
9. 仁慈：维护和增进身边亲近之人（家人、朋友、同事）的福利，包括诚实待人
10. 普世主义：理解、欣赏、宽容并保护所有人乃至自然的福利，包含社会公正与平等

维度判断规则：
- 判断陈述"谈论的价值域"：涉及身边具体他人（家人、朋友、同事、顾客）的诚实/帮助/关怀 → 仁慈；涉及社会层面的公正、平等、环保、弱势群体 → 普世主义。
- 遵守规则秩序、循规蹈矩 → 从众；维护国家安全、社会稳定、公共利益 → 安全。

方向判断规则（valence）：
- 问：若被试认同这道题的陈述（打高分），这是支持还是反对所标的维度？
- 陈述的行为是该维度的体现/践行 → 认同即支持 → positive
- 陈述的行为违背/损害该维度 → 认同即不支持该维度 → negative
- 注意：维度是中性的。如"沉迷游戏"体现享乐，认同它=支持享乐 → 享乐的positive；"向妹妹扔书"损害仁慈，认同它=不支持仁慈 → 仁慈的negative。

示例：
陈述：路人看到老人倒地后立即拨打了120 → {"dimension": "仁慈", "valence": "positive", "confidence": 0.95}
陈述：小张偷走了室友放在桌上的笔记本电脑 → {"dimension": "仁慈", "valence": "negative", "confidence": 0.9}
陈述：小李沉迷于网络游戏整天不学习 → {"dimension": "享乐", "valence": "positive", "confidence": 0.85}
陈述：员工向政府举报了主管伪造公司财务报表的行为 → {"dimension": "普世主义", "valence": "positive", "confidence": 0.9}
陈述：工厂老板强迫工人连续工作16小时不付加班费 → {"dimension": "普世主义", "valence": "negative", "confidence": 0.95}

输出格式：只输出一个JSON对象，不要任何其他内容：
{"dimension": "维度名", "valence": "positive或negative", "confidence": 0.0到1.0}"""

VALID_DIMS = ["自我导向", "刺激", "享乐", "成就", "权力", "安全", "从众", "传统", "仁慈", "普世主义"]


def load_items():
    items = []
    with open(config.DATA_DIR / "final_items_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def label_one(item):
    """打标一题（维度+方向+置信度），返回 dict 或 None。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"陈述：{item['rule_text']}"},
    ]
    try:
        result = chat(
            model_entry=LABEL_MODEL,
            messages=messages,
            thinking_on=False,
            max_tokens_override=120,
            timeout=30.0,
        )
    except (ContentFilterError, ReasoningExhaustedError):
        return None  # 网关内容拦截等确定性失败，交由调用方记录跳过
    if result is None:
        return None
    # 解析JSON（容忍markdown代码块包裹）
    text = result.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        j = json.loads(text)
        dim = j["dimension"].strip()
        if dim not in VALID_DIMS:
            # 模糊纠正：包含匹配
            for v in VALID_DIMS:
                if v in dim:
                    dim = v
                    break
            else:
                return None
        val = j.get("valence", "").strip().lower()
        if val not in ("positive", "negative"):
            return None
        return {
            "rule_id": item["rule_id"],
            "rule_text": item["rule_text"],
            "old_direction": item["direction"],
            "schwartz_main": dim,
            "valence": val,
            "schwartz_confidence": round(float(j.get("confidence", 0.5)), 2),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def run_test():
    items = load_items()
    print(f"[TEST] 单题自检，共{len(items)}题")
    r = label_one(items[0])
    print(json.dumps(r, ensure_ascii=False, indent=2) if r else "FAIL")


def trial_sample(items, n_per_dim=3, n_neg=1):
    """每旧维抽3正1反，共12×4=48题。"""
    random.seed(config.SEED)
    grouped = {}
    for it in items:
        grouped.setdefault(it["core_value"], []).append(it)
    sample = []
    for cv, pool in grouped.items():
        pos = [x for x in pool if x["direction"] == "positive"]
        neg = [x for x in pool if x["direction"] == "negative"]
        sample += random.sample(pos, min(n_per_dim, len(pos)))
        sample += random.sample(neg, min(n_neg, len(neg)))
    return sample


def run_trial():
    items = load_items()
    sample = trial_sample(items)
    print(f"[TRIAL] 试标 {len(sample)} 题（每旧维3正1反）")
    rows, fails = [], 0
    for i, it in enumerate(sample, 1):
        r = label_one(it)
        if r is None:
            fails += 1
            print(f"  [{i}/{len(sample)}] FAIL rule_id={it['rule_id']}")
            continue
        r["orig_value"] = it["core_value"]
        rows.append(r)
        if i % 12 == 0:
            print(f"  进度 {i}/{len(sample)}")
        time.sleep(0.3)
    # 导出核对CSV
    with open(TRIAL_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rule_id", "rule_text", "旧方向", "旧维度(参考)", "schwartz标注",
                    "新方向valence", "方向翻转?", "置信度", "修改为(留空=通过)"])
        for r in rows:
            flipped = "★" if r["old_direction"] != r["valence"] else ""
            w.writerow([r["rule_id"], r["rule_text"], r["old_direction"], r["orig_value"],
                        r["schwartz_main"], r["valence"], flipped,
                        r["schwartz_confidence"], ""])
    # 汇总
    from collections import Counter
    dist = Counter(r["schwartz_main"] for r in rows)
    n_flip = sum(1 for r in rows if r["old_direction"] != r["valence"])
    print(f"\n完成 {len(rows)}/{len(sample)}（失败{fails}）")
    print(f"维度分布: {dict(dist)}")
    print(f"方向翻转题数（旧direction≠新valence）: {n_flip}/{len(rows)}")
    print(f"核对CSV: {TRIAL_CSV}")
    print("请在'修改为'列填写正确维度，留空=通过；改完后告诉我文件路径")


def _clean_jsonl():
    """清洗断电产生的半行：只保留能完整解析的行，返回有效记录数。"""
    if not OUT_JSONL.exists():
        return 0
    good = []
    bad = 0
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
                good.append(s)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for s in good:
                f.write(s + "\n")
        print(f"[清洗] 剔除断电半行 {bad} 条（对应题目将自动重跑）")
    return len(good)


def run_full():
    items = load_items()
    _clean_jsonl()
    done_ids = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["rule_id"])
    # 已记录的拦截题（重启去重用；若某题后来成功打进labels则自然覆盖）
    failed_ids = set()
    if FAILED_JSONL.exists():
        with open(FAILED_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    failed_ids.add(json.loads(line)["rule_id"])
    remaining = [it for it in items if it["rule_id"] not in done_ids]
    total = len(items)
    n0 = len(done_ids)
    print(f"[FULL] 全量打标：共{total}，已完成{n0}，待跑{len(remaining)}")
    if not remaining:
        print("全部完成，无需续跑")
    count, fails = 0, 0
    consec_fail = 0  # 连续失败计数（真key死亡时快速停机；内容拦截成串时靠高阈值放行）
    t0 = time.time()
    for it in remaining:
        r = label_one(it)
        if r is None:
            fails += 1
            consec_fail += 1
            if it["rule_id"] not in failed_ids:
                failed_ids.add(it["rule_id"])
                with open(FAILED_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "rule_id": it["rule_id"],
                        "rule_text": it["rule_text"],
                        "old_direction": it["direction"],
                        "old_core_value": it["core_value"],
                        "error": "gateway_content_filter",
                    }, ensure_ascii=False) + "\n")
                print(f"  [拦截] rule_id={it['rule_id']} 已记入失败清单，跳过")
            if consec_fail >= 10:
                print("\n[中止] 连续10次失败——网关故障窗口或key失效。")
                print("       处理：1)关闭窗口 2)到 aiapi.world 查看账户/公告 3)等20-30分钟")
                print("       4)重新双击 打标启动.bat 从断点续跑（进度已保存，被拦题自动重试）")
                sys.exit(1)
            continue
        consec_fail = 0
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        count += 1
        time.sleep(0.3)  # 防限流
        if count % 25 == 0 or count == len(remaining):
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining) - count) / rate / 60 if rate > 0 else 0
            done_now = n0 + count
            print(f"  进度 {done_now}/{total} ({done_now/total*100:.1f}%) | "
                  f"速度 {rate*60:.0f}题/分 | 预计剩余 {eta_min:.0f} 分钟")
    print(f"\n本次完成 {count}，失败 {fails}")
    # 汇总分布
    labels = []
    with open(OUT_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line))
    from collections import Counter
    dist = Counter(r["schwartz_main"] for r in labels)
    lowconf = [r for r in labels if r["schwartz_confidence"] < 0.7]
    print(f"总计 {len(labels)}/{len(items)}")
    print(f"维度分布: {dict(dist)}")
    print(f"低置信(<0.7): {len(lowconf)} 题")
    if lowconf:
        with open(LOWCONF_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["rule_id", "rule_text", "schwartz标注", "valence", "置信度", "修改为(留空=通过)"])
            for r in lowconf:
                w.writerow([r["rule_id"], r["rule_text"], r["schwartz_main"],
                            r["valence"], r["schwartz_confidence"], ""])
        print(f"低置信裁决CSV: {LOWCONF_CSV}")
    # 拦截题清单（仍不在labels里的）→ 人工补标CSV
    blocked = []
    if FAILED_JSONL.exists():
        with open(FAILED_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["rule_id"] not in {r["rule_id"] for r in labels}:
                        blocked.append(rec)
    if blocked:
        with open(BLOCKED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["rule_id", "rule_text", "旧维度(参考)", "旧方向(参考)",
                        "维度(必填)", "valence(必填:positive/negative)"])
            for r in blocked:
                w.writerow([r["rule_id"], r["rule_text"], r["old_core_value"],
                            r["old_direction"], "", ""])
        print(f"网关拦截待人工补标: {len(blocked)} 题 → {BLOCKED_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_test()
    elif args.trial:
        run_trial()
    elif args.full:
        run_full()
    else:
        print("请指定模式：--test / --trial / --full")
