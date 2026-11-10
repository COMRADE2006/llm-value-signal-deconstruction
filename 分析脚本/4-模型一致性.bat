@echo off
title 模型一致性检验
echo ============================================
echo   [4/7] 模型一致性检验
echo   第一步：22列 EFA + 平行分析 + Velicer MAP（KMO取样适切性）
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\step1_consistency.py" --source agree
echo.
echo 运行结束。看判读：[通过]一致性成立 → 聚类正当；[警告]不足 → 看载荷矩阵找模型分化结构
pause
