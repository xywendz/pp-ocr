param(
    [string]$ModelsDir = "./models"
)

$ErrorActionPreference = "Stop"
$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonExe = "python"
if (Test-Path ".venv/Scripts/python.exe") {
    $PythonExe = ".venv/Scripts/python.exe"
}

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[1/2] Installing model preparation dependencies..."
& $PythonExe -m pip install -i $IndexUrl -r requirements-build.txt
Assert-LastExitCode "Installing dependencies"

Write-Host "[2/2] Downloading models..."
& $PythonExe scripts/download_models.py --models-dir $ModelsDir
Assert-LastExitCode "Downloading models"

Write-Host ""
Write-Host "Models are ready at: $ModelsDir"
Write-Host "Now start the service:"
Write-Host '  .venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000'
