@echo off
chcp 936 >nul
cd /d "d:\competitionresearch\cognitivesafety\สตั้2"
echo ============================================================
echo  Values Rewrite v2 (Sampling + AI Rewrite + CSV)
echo  Resume supported, close anytime, double-click to continue
echo ============================================================
echo.
"d:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" run_models\rewrite_v2.py
echo.
echo ============================================================
echo  Done. Check data\review_items_v2.csv
echo ============================================================
pause