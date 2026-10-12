@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
title Kyotei Odds - Manual test (foreground)

set "PY=C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=py"

echo ====================================================
echo  Odds collector - run in this window (Ctrl+C to stop)
echo  Collects at 15/13/10/5 min before each race deadline.
echo  Files: data\odds\  Log: data\odds\_collector.log
echo ====================================================
echo.

"%PY%" odds_collector.py

echo.
pause
