@echo off
title 正反题判别效度
echo ============================================
echo   [3/7] 正反题判别效度
echo   武器三先锋把关：同维 vs 跨维正反题分相关 + 置换检验
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\weapon3_posneg.py" --source agree
echo.
echo 运行结束。看判读：[通过]维度特异性存在 → 继续后续分析；[警告]不成立 → 后续聚类结果需谨慎解读
pause
