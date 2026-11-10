@echo off
title Schwartz Labeling (resume-able)
echo ==================================================
echo  Schwartz 10-value labeling  (2388 items)
echo.
echo  Resume-able: close this window anytime.
echo  Progress is saved after every single item.
echo  Double-click this bat again to continue.
echo ==================================================
echo.
cd /d "%~dp0"
"D:\competitionresearch\cognitivesafety\llm_env\Scripts\python.exe" -u "run_models\label_schwartz.py" --full
echo.
echo ==================================================
echo  Finished (or stopped). Progress is saved.
echo ==================================================
pause
