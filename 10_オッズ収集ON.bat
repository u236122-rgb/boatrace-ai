@echo off
chcp 65001 > nul
rem --- self-elevate to admin (needed to register a scheduled task) ---
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting administrator permission... Please click "Yes".
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
setlocal
cd /d "%~dp0"
title Kyotei Odds - Enable auto collection

echo ====================================================
echo  Boatrace odds auto collection - ENABLE
echo  Registers a Windows scheduled task (runs at logon)
echo ====================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0register_odds_task.ps1"

echo.
echo Done. The collector runs in the background while the PC is on.
echo (Stop it anytime with the OFF batch.)
pause
