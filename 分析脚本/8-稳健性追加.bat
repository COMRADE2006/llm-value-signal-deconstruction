@echo off
title 稳健性追加检验
echo ============================================
echo   [8/8] 稳健性追加检验
echo   马蹄诊断 + PERMANOVA维度可区分性 + 质心椭圆
echo   + 经验类标签武器三 + 纯度置换基线
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\robustness_extra.py" --source agree
echo.
echo 运行结束。看四组判读：马蹄伪影是否成立、维度对可区分性、
echo 经验类vs12维的正反互证对比、纯度是否超出置换基线。
pause
