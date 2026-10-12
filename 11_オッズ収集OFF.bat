@echo off
chcp 65001 > nul
net session >nul 2>&1
if %errorlevel% neq 0 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
setlocal
title Kyotei Odds - Disable auto collection

echo Stopping and removing the scheduled task...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Stop-ScheduledTask -TaskName 'KyoteiOddsCollector' -ErrorAction SilentlyContinue; Unregister-ScheduledTask -TaskName 'KyoteiOddsCollector' -Confirm:$false -ErrorAction Stop; Write-Host 'Removed: KyoteiOddsCollector' } catch { Write-Host 'Task not found (already off).' }"
powershell -NoProfile -Command "Get-Process pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Python312*' } | Stop-Process -Force -ErrorAction SilentlyContinue"

echo.
echo Auto collection disabled.
pause
