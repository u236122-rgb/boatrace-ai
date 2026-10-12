# 競艇オッズ自動収集をWindowsタスクスケジューラに登録する
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw = "C:\Users\user\AppData\Local\Programs\Python\Python312\pythonw.exe"
if (-not (Test-Path $pyw)) { $pyw = "pythonw" }
$task = "KyoteiOddsCollector"

$action  = New-ScheduledTaskAction -Execute $pyw -Argument "odds_collector.py" -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 3)

Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Settings $settings `
    -Description "Boatrace 3rentan odds auto collector" -Force | Out-Null
Start-ScheduledTask -TaskName $task

Write-Host "===================================================="
Write-Host " Registered & started scheduled task: $task"
Write-Host " Runs at every logon, collects odds in background."
Write-Host " Logs: data\odds\_collector.log"
Write-Host "===================================================="
