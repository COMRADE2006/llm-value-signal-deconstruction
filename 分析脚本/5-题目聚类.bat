@echo off
title 题目聚类主分析
echo ============================================
echo   [5/7] 题目聚类主分析
echo   第二步：三通道(orig/I/P) Ward聚类 + 轮廓扫K + 子抽样稳定性 + 子主题层双法互验
echo ============================================
cd /d "d:\competitionresearch\cognitivesafety\实验2"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" "analysis\step2_clustering.py" --source agree
echo.
echo 运行结束。耗时较长（子抽样稳定性约30-60分钟），请耐心等待。检查三通道ARI（通道稳健性）与难度因子警告
pause
