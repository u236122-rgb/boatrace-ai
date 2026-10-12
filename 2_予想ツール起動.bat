@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
title Kyotei AI Yosou

rem --- find real python (avoid Microsoft Store stub) ---
set "PY=C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=py"

echo Starting Kyotei AI prediction tool...
echo Python: %PY%
echo.

"%PY%" app.py

echo.
echo ====================================================
echo  Window will stay open so you can read any message.
echo  If the GUI did not appear, an error is shown above.
echo  (Run 1_model_train.bat first if model is missing)
echo ====================================================
pause
