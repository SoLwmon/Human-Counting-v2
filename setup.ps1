$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.python'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot 'work\ultralytics'
New-Item -ItemType Directory -Path (Join-Path $env:YOLO_CONFIG_DIR 'Ultralytics') -Force | Out-Null
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required for setup: https://docs.astral.sh/uv/getting-started/installation/'
}
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Setup command failed: exit $LASTEXITCODE" } }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    uv python install 3.12 --no-bin
    Check-Exit
    uv venv --python 3.12 .venv
    Check-Exit
}
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
Check-Exit
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-setup.txt
Check-Exit
& .\.venv\Scripts\python.exe download_models.py
Check-Exit
uv pip check --python .venv\Scripts\python.exe
Check-Exit
Write-Host 'Setup complete. Example: .\run.cmd --source "C:\Videos\booth.mp4" --show'
