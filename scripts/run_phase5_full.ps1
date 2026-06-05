$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
Set-Location $repoRoot

$pythonExe = (Get-Command python).Source
$runner = Join-Path $repoRoot "scripts\phase5_runner.py"
$modelPath = Join-Path $repoRoot "artifacts\bank_marketing_lr_full\model.json"
$calibrationPath = Join-Path $repoRoot "artifacts\bank_marketing_phase4_full\calibration\calibration.json"
$logDir = Join-Path $repoRoot "artifacts\bank_marketing_phase5_logs_full"

& $pythonExe $runner --mode full --model-path $modelPath --calibration-path $calibrationPath --log-dir $logDir
exit $LASTEXITCODE
