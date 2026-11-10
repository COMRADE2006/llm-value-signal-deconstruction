@echo off
title 数据预处理
echo ============================================
echo   [2/7] 数据预处理
echo   反向计分 + 剔平指纹题 + IR-Tree三通道 + 双重中心化
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\preprocess.py" --source agree
echo.
echo 运行结束。请检查剔除比例（平指纹题占比）与活跃题数量
pause
