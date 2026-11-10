@echo off
title 构建响应矩阵
echo ============================================
echo   [1/7] 构建响应矩阵
echo   读取盲评认同度新数据(raw_resp_agree/)，构建 2388×22 响应矩阵
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\build_matrix.py" --source agree
echo.
echo 运行结束。请检查质量报告：列覆盖是否完整、正反题均分差是否正常
pause
