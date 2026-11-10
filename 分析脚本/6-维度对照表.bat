@echo off
title 维度对照表
echo ============================================
echo   [6/7] 维度对照表
echo   第三步：经验类×12维交叉表 + 可合并维度组 + 跨类子主题清单
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\step3_mapping.py" --source agree
echo.
echo 运行结束。看卡方检验与维度纯度：纯度低的维度=LLM响应层面边界模糊
pause
