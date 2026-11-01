# Setup script for Industrial Energy Monitoring Application (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python installation..." -ForegroundColor Cyan
python --version

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "==> Virtual environment (.venv) already exists." -ForegroundColor Green
}

Write-Host "==> Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "==> Verifying Streamlit installation..." -ForegroundColor Cyan
streamlit --version

Write-Host "==> Setup complete! You can now start the web application by running:" -ForegroundColor Green
Write-Host "streamlit run ui/app.py" -ForegroundColor Yellow

