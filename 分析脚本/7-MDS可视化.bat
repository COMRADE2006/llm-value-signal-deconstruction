@echo off
title MDS构型可视化
echo ============================================
echo   [7/7] MDS构型可视化
echo   武器一：50子主题MDS二维图 + stress报告 + 12维质心投影
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\mds_viz.py" --source agree
echo.
echo 运行结束。看图判读三结局（挤团/一维/环形扇形），只做探索性描述
pause
