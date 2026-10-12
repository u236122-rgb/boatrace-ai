@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
title Kyotei AI - Data Update

set "PY=C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=py"

echo ====================================================
echo  Kyotei AI - Update data to the latest
echo  (downloads official B/K files and updates CSV)
echo ====================================================
echo.

"%PY%" update_data.py
if errorlevel 1 (
  echo.
  echo [ERROR] Data update failed. See messages above.
  pause
  exit /b 1
)

echo.
echo ====================================================
echo  Training the model with the latest data (takes minutes)...
echo ====================================================
"%PY%" build_model.py

echo.
echo ====================================================
echo  All done. Press any key to close.
echo ====================================================
pause
