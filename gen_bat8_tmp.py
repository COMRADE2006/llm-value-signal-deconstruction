# -*- coding: utf-8 -*-
"""临时脚本：生成 8-稳健性追加.bat（GBK+CRLF）。"""
from pathlib import Path

ROOT = r"d:\competitionresearch\cognitivesafety\实验2"
PY = r"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe"
OUT = Path(ROOT) / "分析脚本"

content = (
    "@echo off\r\n"
    "title 稳健性追加检验\r\n"
    "echo ============================================\r\n"
    "echo   [8/8] 稳健性追加检验\r\n"
    "echo   马蹄诊断 + PERMANOVA维度可区分性 + 质心椭圆\r\n"
    "echo   + 经验类标签武器三 + 纯度置换基线\r\n"
    "echo ============================================\r\n"
    f"cd /d \"{ROOT}\"\r\n"
    f"\"{PY}\" \"analysis\\robustness_extra.py\" --source agree\r\n"
    "echo.\r\n"
    "echo 运行结束。看四组判读：马蹄伪影是否成立、维度对可区分性、\r\n"
    "echo 经验类vs12维的正反互证对比、纯度是否超出置换基线。\r\n"
    "pause\r\n"
)
(OUT / "8-稳健性追加.bat").write_bytes(content.encode("gbk"))
print("生成: 8-稳健性追加.bat")
