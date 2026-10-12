@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"
title Kyotei AI - Public (Cloudflare Tunnel)

set "PY=C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=py"

echo ====================================================
echo  Kyotei AI - Public access (works on mobile data too)
echo  Wait for the https://....trycloudflare.com line,
echo  then open THAT url on your phone.
echo  Keep this window open while using.
echo ====================================================
echo.

"%PY%" tunnel.py

echo.
echo Stopped. Press any key to close.
pause
