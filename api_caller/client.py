# -*- coding: utf-8 -*-
"""
多厂商原生 API 客户端。
- 每个模型用自己厂商的 base_url + api_key
- 根据 vendor 的 thinking_param 类型构造正确的 extra_body
- 根据 temperature_policy 决定是否传 temperature
- 统一重试逻辑
- 内容安全审查拦截（400/contentFilter/1301）立即判定，不重试
"""
import time
import random
from typing import Optional, List, Dict

import config

logger = config.get_logger()


class ContentFilterError(Exception):
    """内容安全审查拦截。此类错误不会随重试消失，应立即落盘为 NaN 并跳过。"""
    pass


class ReasoningExhaustedError(Exception):
    """推理链耗尽 max_tokens 导致无最终答案（finish_reason=length 且 content 为空）。
    实测确定性失败（sensenova 个别题思考链 8000+ token 不收敛），重试无意义，
    应立即落盘为 NaN 并跳过（2026-10-10 实测 rule_id=10346/17038 复现 5/5）。"""
    pass


def _is_content_filter_error(e) -> bool:
    """判断异常是否为内容安全审查拦截（不依赖具体厂商字段名）。"""
    try:
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
        body = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            body = resp.text or ""
        body_lower = body.lower()
        # 智谱: 400 + code=1301；通用: contentFilter / content_filter / 敏感内容 / 不安全
        if status == 400 and (
            "contentfilter" in body_lower
            or "content_filter" in body_lower
            or "1301" in body
            or "敏感内容" in body
            or "不安全" in body
            or "敏感" in body and "拦截" in body
        ):
            return True
        # 部分厂商（如豆包）不返回 400 而是在消息里提示内容安全
        if "content_filter" in body_lower and "sensitive" in body_lower:
            return True
    except Exception:
        pass
    return False


def _is_rate_limit_error(e) -> bool:
    """判断异常是否为速率限制（429/TPM限流）。此类错误等待足够长即可恢复。"""
    s = str(e)
    try:
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    except Exception:
        status = None
    if status == 429:
        return True
    s_lower = s.lower()
    return ("1302" in s or "速率限制" in s or "rate limit" in s_lower
            or "rate_limit" in s_lower or "too many requests" in s_lower)


def _build_thinking_extra(vendor_key: str, thinking_on: bool) -> Optional[dict]:
    """根据厂商参数类型，构造 extra_body 里的 thinking 控制字段。"""
    vendor = config.VENDORS.get(vendor_key, {})
    param_type = vendor.get("thinking_param")
    if param_type is None:
        return None

    if param_type == "thinking_type":
        # DeepSeek, Zhipu, Kimi 官方: {"thinking": {"type": "enabled/disabled"}}
        return {"thinking": {"type": "enabled" if thinking_on else "disabled"}}
    elif param_type == "enable_thinking":
        # Qwen DashScope: {"enable_thinking": True/False}
        return {"enable_thinking": thinking_on}
    return None


def make_client(vendor_key: str):
    """创建指定厂商的 OpenAI 兼容客户端。"""
    from openai import OpenAI
    vendor = config.VENDORS.get(vendor_key, {})
    api_key = vendor.get("api_key", "")
    base_url = vendor.get("base_url", "")
    if not api_key:
        raise RuntimeError(f"厂商 {vendor_key} 的 API Key 未配置（在 .env 中缺失对应的 {vendor_key.upper()}_API_KEY）")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    model_entry: dict,
    messages: List[Dict],
    thinking_on: bool = True,
    max_retries: int = 3,
    timeout: float = 60.0,
    max_tokens_override: int = None,
) -> Optional[str]:
    """
    调用指定模型。
    model_entry: config.MODELS 里的完整条目 dict
    thinking_on: 是否开启 thinking
    max_tokens_override: 若指定，则覆盖默认 max_tokens（用于改写等需要更长输出的场景）
    返回: content 文本字符串，失败返回 None
    """
    vendor_key = model_entry["vendor"]
    model_str = model_entry["model"]
    vendor = config.VENDORS.get(vendor_key, {})

    # 模型条目可单独指定 timeout（如 hy4-preview 思考链长，60s 不够用）
    timeout = model_entry.get("timeout", timeout)

    # max_tokens 优先级：
    #   1. 调用方显式 override
    #   2. 模型条目自带的 non_reasoning_max_tokens（个别模型关闭思考时仍需较大上限）
    #   3. 模型条目自带的 max_tokens（个别模型思考链超长，厂商默认上限会截断，如 hy4-preview）
    #   4. 推理开 → 厂商默认上限；推理关 → 厂商非推理默认值
    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    elif thinking_on and model_entry.get("max_tokens") is not None:
        max_tokens = model_entry["max_tokens"]
    elif thinking_on and vendor.get("thinking_param") is not None:
        max_tokens = vendor.get("default_max_tokens", 2000)
    elif model_entry.get("non_reasoning_max_tokens") is not None:
        max_tokens = model_entry["non_reasoning_max_tokens"]
    else:
        max_tokens = vendor.get("non_reasoning_max_tokens", 10)

    # temperature: 只有 "free" 策略才传 TEMPERATURE=0
    # "fixed" 策略（如 Kimi）不传，让 API 用默认值
    temp_policy = vendor.get("temperature_policy", "free")

    # 构造 thinking 控制参数
    param_type = vendor.get("thinking_param")
    extra_body = _build_thinking_extra(vendor_key, thinking_on)

    # 组装请求参数
    kwargs = dict(
        model=model_str,
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if temp_policy == "free":
        kwargs["temperature"] = config.TEMPERATURE
    # temperature_policy == "fixed" → 不传 temperature

    # 商汤 SenseNova: reasoning_effort 是顶层参数，不是 extra_body 里的嵌套对象
    if param_type == "reasoning_effort":
        kwargs["reasoning_effort"] = "high" if thinking_on else "none"
    elif extra_body:
        kwargs["extra_body"] = extra_body

    last_err = None
    exhausted_retried = False  # 是否已用过"加倍重试"机会
    for attempt in range(1, max_retries + 1):
        try:
            client = make_client(vendor_key)
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            # 推理链耗尽：finish=length 且 content 空
            finish = getattr(resp.choices[0], "finish_reason", None)
            if (content is None or str(content).strip() == "") and finish == "length":
                mode_str = "on" if thinking_on else "off"
                # 思考链可能只是"略长"被常规上限冤枉：自动加倍到16000重试一次
                # （实测sensenova发散题8000仍不收敛，16000可救回略长题；死循环题两次都耗尽后NaN）
                if not exhausted_retried and kwargs["max_tokens"] < 16000 and thinking_on:
                    exhausted_retried = True
                    old_mt = kwargs["max_tokens"]
                    kwargs["max_tokens"] = 16000
                    logger.warning(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] "
                                   f"推理链耗尽max_tokens={old_mt}，加倍至16000重试一次（给略长思考题机会）")
                    continue
                logger.warning(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] "
                               f"推理链耗尽max_tokens(finish=length, content空)，快速判定失败（不重试）")
                raise ReasoningExhaustedError(f"finish_reason=length, content empty")
            return content
        except (ContentFilterError, ReasoningExhaustedError):
            raise  # 确定性失败直接上抛，不重试
        except Exception as e:
            last_err = e
            # 内容安全审查拦截：不会随重试消失，立即判定失败
            if _is_content_filter_error(e):
                mode_str = "on" if thinking_on else "off"
                logger.warning(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] 内容安全拦截，快速判定失败（不重试）")
                raise ContentFilterError(str(e)) from e
            # aiapi.world 网关缺陷：内容拦截/密钥失效伪装成429限流（expired_secret_key），
            # 等待重试无意义，快速失败交由调用方处理
            if "expired_secret_key" in str(e) or "API key has expired" in str(e):
                mode_str = "on" if thinking_on else "off"
                logger.warning(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] 网关拦截或密钥失效(expired_secret_key)，快速判定失败（不重试）")
                raise ContentFilterError(str(e)) from e
            if _is_rate_limit_error(e):
                # 速率限制（429/TPM）：限流窗口按分钟计，短退避无效，等30~40s让窗口滑过
                wait = 30 + random.random() * 10
            else:
                wait = (2 ** (attempt - 1)) + random.random()
            mode_str = "on" if thinking_on else "off"
            logger.warning(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] 第 {attempt}/{max_retries} 次失败: {e}，等待 {wait:.1f}s")
            time.sleep(wait)
    mode_str = "on" if thinking_on else "off"
    logger.error(f"[{model_entry.get('alias', 'unknown')} thinking={mode_str}] 最终失败: {last_err}")
    return None
