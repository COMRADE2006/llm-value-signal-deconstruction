# -*- coding: utf-8 -*-
"""
全局配置 —— 多平台原生 API 模式。
每个模型用自己厂商的 base_url + API key，支持 thinking 开关控制。

重要：各厂商 temperature 策略不同，详见 VENDORS 配置。
"""
import os
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

def _load_dotenv(env_path: Path):
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if m:
                key, val = m.group(1), m.group(2).strip()
                if (val.startswith('"') and val.endswith('"')) or \
                   (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                os.environ.setdefault(key, val)

_load_dotenv(PROJECT_DIR / ".env")

# ============== 各厂商 API 凭证 ==============
# thinking_param 类型决定了怎么构造 extra_body：
#   "thinking_type" → {"thinking": {"type": "enabled/disabled"}} (DeepSeek, Zhipu, Kimi)
#   "enable_thinking" → {"enable_thinking": True/False} (Qwen)
#   None → 不需要或不支持
#
# temperature_policy:
#   "fixed" → 厂商强制固定 temperature，调用时不传（让 API 用默认值）
#   "free"  → 可以自由设置，我们用 TEMPERATURE=0
VENDORS = {
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "thinking_param": "thinking_type",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    "qwen": {
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "thinking_param": "enable_thinking",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    "zhipu": {
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "thinking_param": "thinking_type",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    "moonshot": {
        "base_url": os.getenv("MOONSHOT_BASE_URL", os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")),
        "api_key": os.getenv("MOONSHOT_API_KEY", os.getenv("KIMI_API_KEY", "")),
        "thinking_param": "thinking_type",
        "temperature_policy": "fixed",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 50,
    },
    "doubao": {
        "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "api_key": os.getenv("DOUBAO_API_KEY", ""),
        "thinking_param": "thinking_type",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "thinking_param": None,  # gpt-4o-mini 不是推理模型
        "temperature_policy": "free",
        "default_max_tokens": 10,
        "non_reasoning_max_tokens": 10,
    },
    "aiapi": {
        "base_url": os.getenv("AIAPI_BASE_URL", "https://aiapi.world/v1"),
        "api_key": os.getenv("AIAPI_API_KEY", ""),
        "thinking_param": None,
        "temperature_policy": "free",
        "default_max_tokens": 2000,
        "non_reasoning_max_tokens": 10,
    },
    # 腾讯云 TokenHub：统一 OpenAI 兼容，所有模型 thinking 统一为 thinking.type 格式
    # base_url 广州: https://tokenhub.tencentmaas.com/v1
    "tokenhub": {
        "base_url": os.getenv("TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com/v1"),
        "api_key": os.getenv("TOKENHUB_API_KEY", ""),
        "thinking_param": "thinking_type",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    # 阶跃星辰 Step 原生 API（按量计费标准端点）
    # 注意：不要用 /step_plan/v1，那是月度套餐端点，无套餐会报
    # "you have no active step plan subscription"（2026-10-05 实测排查）
    "stepfun": {
        "base_url": os.getenv("STEPFUN_BASE_URL", "https://api.stepfun.com/v1"),
        "api_key": os.getenv("STEPFUN_API_KEY", ""),
        # 实测（2026-10-05）：该模型只有 low/medium/high 三档推理强度，
        # enable_thinking / thinking.type=disabled 均被静默忽略，无法真正关闭思考。
        # 故模型条目中只保留 on 模式（=reasoning_effort=high），放附录报告。
        "thinking_param": "reasoning_effort",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
    # 商汤 SenseNova Token Plan（免费档）
    # thinking 用 reasoning_effort 顶层参数："high"=开, "none"=关
    "sensenova": {
        "base_url": os.getenv("SENSENOVA_BASE_URL", "https://token.sensenova.cn/v1"),
        "api_key": os.getenv("SENSENOVA_API_KEY", ""),
        "thinking_param": "reasoning_effort",
        "temperature_policy": "free",
        "default_max_tokens": 4000,
        "non_reasoning_max_tokens": 10,
    },
}

# ============== 被测模型清单 ==============
# thinking_modes: ["on", "off"] 表示跑两组；["off"] 表示只跑非推理
# 注意：只有 thinking_param != None 的厂商型号才支持开/关
MODELS = [
    {
        "alias": "ds-v4-pro",
        "vendor": "deepseek",
        "model": "deepseek-v4-pro",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "ds-v4-flash",
        "vendor": "deepseek",
        "model": "deepseek-v4-flash",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "qwen3.8-max",
        "vendor": "qwen",
        "model": "qwen3.8-max",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "qwen3.7-max",
        "vendor": "qwen",
        "model": "qwen3.7-max",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "glm-5.2",
        "vendor": "zhipu",
        "model": "glm-5.2",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "doubao-seed-2-1-pro",
        "vendor": "doubao",
        "model": "doubao-seed-2-1-pro-260915",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "kimi-k2.6",
        "vendor": "moonshot",
        "model": "kimi-k2.6",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # 注意：kimi-k2.7-code 强制思考不能关，必须用 kimi-k2.6
        # 注意：Kimi temperature 被强制固定（on=1.0, off=0.6），违反控制变量，主分析剔除，附录报告
    },
    # ===== 新增 8 个推理模型 =====
    # TokenHub 上的腾讯混元
    {
        "alias": "hy3",
        "vendor": "tokenhub",
        "model": "hy3",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
    },
    {
        "alias": "hy4-preview",
        "vendor": "tokenhub",
        "model": "hy4-preview",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # 实测：量表真实题目思考链远超 60s，默认超时导致 3 次重试全失败、
        # 每题空耗 3~9 分钟（2026-10-06 排查），单独放宽到 300s
        "timeout": 300,
        # 实测：单题思考链约 4600 tokens，厂商默认 4000 会截断导致 content 为空
        # （2026-10-06 诊断：mt=4000 返回''，mt=8000 正常返回分数），放宽到 8000
        "max_tokens": 8000,
    },
    # 阿里云百炼上的 Qwen3.5（从 TokenHub 迁移，阿里已充值且支持 enable_thinking）
    {
        "alias": "qwen3.5-flash",
        "vendor": "qwen",
        "model": "qwen3.5-flash",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # 实测：普通题约 20s 完成，个别难题思考链超 60s 触发超时重试，放宽到 180s
        "timeout": 180,
    },
    {
        "alias": "qwen3.5-plus",
        "vendor": "qwen",
        "model": "qwen3.5-plus",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        "timeout": 180,
    },
    # 智谱 GLM-5.1 / GLM-5-turbo
    # 【2026-10-10 改道】原走 TokenHub 网关（慢：on模式18s/次，且网关对 glm-5-turbo 的
    # thinking.type=disabled 一律报 400001 400错误 → off 模式完全跑不动）。
    # 实测智谱原生 API 两模型 on/off 四种组合全部成功且更快（4~8s），故切换到 zhipu 原生。
    # 模型暂未下线（原"10月8日下线"指 2027-10-08，实测 2026-10-10 仍可用）。
    {
        "alias": "glm-5.1",
        "vendor": "zhipu",
        "model": "glm-5.1",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # 实测：该模型 thinking=off 时仍产生约 120 token 非推理前缀，
        # 厂商默认上限 10 会截断答案，故单独放宽到 2000（2026-10-02 实测）
        "non_reasoning_max_tokens": 2000,
    },
    {
        "alias": "glm-5-turbo",
        "vendor": "zhipu",
        "model": "glm-5-turbo",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # GLM 系列共性：off 模式隐藏前缀会耗尽 max_tokens=10 导致空串（同 glm-5.1），
        # 放宽到 2000（智谱原生实测 off+mt2000 正常返回）
        "non_reasoning_max_tokens": 2000,
    },
    # 阶跃星辰 Step 3.7 Flash（附录模型，只跑 high 档）
    {
        "alias": "step-3.7-flash",
        "vendor": "stepfun",
        "model": "step-3.7-flash",
        "is_reasoning": True,
        # 无法关闭思考，只跑 reasoning_effort=high（详见 VENDORS.stepfun 注释）
        "thinking_modes": ["on"],
    },
    # 商汤 SenseNova 6.8 Flash Lite（免费档）
    {
        "alias": "sensenova-flash-lite",
        "vendor": "sensenova",
        "model": "sensenova-6.8-flash-lite",
        "is_reasoning": True,
        "thinking_modes": ["on", "off"],
        # 实测（2026-10-10）：发散题思考链 8000/16000 tokens 需 120~140s，
        # 默认 60s 超时会让"耗尽加倍重试"必超时，放宽到 300s
        "timeout": 300,
    },
]

# ============== 通用参数 ==============
# 只有 temperature_policy="free" 的厂商才会用这个值
TEMPERATURE = 0
REPEAT = int(os.getenv("REPEAT", "1"))

# ============== 数据路径 ==============
C_VARC_PATH = r"d:\competitionresearch\cognitivesafety\CVARC\C-VARC.jsonl"

DATA_DIR = PROJECT_DIR / "data"
RAW_RESP_DIR = DATA_DIR / "raw_resp"          # 旧"符合度"Prompt 数据（对照条件，封存不再写入）
AGREE_RESP_DIR = DATA_DIR / "raw_resp_agree"   # 新"盲评认同度"Prompt 数据（全量重跑主数据）
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

# 最终量表：人工审核（方向/维度/修改句）+ 剔除兜底句后的正式题目文件
FINAL_ITEMS_PATH = DATA_DIR / "final_items_v2.jsonl"

SEED = 42
MIN_RULE_LEN = 5

# 剔除的 rule_id：C-VARC 原文中的"兜底句"，被挂全部 12 维 + 全部 50 子主题标签，
# 无区分度（2026-10-02 核查 C-VARC 原始记录后确认，用户拍板舍弃）
EXCLUDE_RULE_IDS = {86913, 144211}

CORE_VALUES = [
    "富强", "民主", "文明", "和谐",
    "自由", "平等", "公正", "法治",
    "爱国", "敬业", "诚信", "友善",
]

def ensure_dirs():
    for d in [DATA_DIR, RAW_RESP_DIR, AGREE_RESP_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def get_logger(name: str = "cvarc"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger

import logging
