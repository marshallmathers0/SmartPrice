# Start the whole SmartPrice demo with one command (Windows).
#
#   .\scripts\start-demo.ps1        or double-click start-demo.bat
#
# 1. starts Docker Desktop if it is not running, and waits for it
# 2. docker compose up -d            PostgreSQL + Redis
# 3. alembic upgrade head            tables (does nothing if already up to date)
# 4. python -m app.database.seed     demo data (skips rows that already exist)
# 5. opens the API      (uvicorn,   port 8000) in its own window
# 6. opens the demo     (streamlit, port 8501) in its own window, then the browser
#
# No need to activate the virtual environment: the script uses .venv's Python directly.
# "Continue", not "Stop": docker and alembic print normal progress and warnings on stderr,
# which Windows PowerShell 5.1 would turn into fatal errors. Every step checks $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

if (-not (Test-Path $python)) {
    throw "No virtual environment at .venv - create it first: python -m venv .venv; .venv\Scripts\pip install -r requirements-dev.txt"
}

# ---- 1. Docker Desktop -------------------------------------------------------
Step "Checking Docker"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running - starting Docker Desktop..."
    Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker did not start within 3 minutes - open Docker Desktop manually and retry." }
}
Write-Host "Docker is running."

# ---- 2. PostgreSQL + Redis ---------------------------------------------------
Step "Starting PostgreSQL and Redis (docker compose up -d)"
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
foreach ($container in "smartprice-postgres", "smartprice-redis") {
    for ($i = 0; $i -lt 60; $i++) {
        if ((docker inspect -f "{{.State.Health.Status}}" $container) -eq "healthy") { break }
        Start-Sleep -Seconds 1
    }
    Write-Host "$container is healthy."
}

# ---- 3. Tables ---------------------------------------------------------------
Step "Applying database migrations (alembic upgrade head)"
& $python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "alembic upgrade failed" }

# ---- 4. Data -----------------------------------------------------------------
Step "Loading demo data (python -m app.database.seed)"
& $python -m app.database.seed
if ($LASTEXITCODE -ne 0) { throw "seed failed" }

# ---- 5 + 6. API and demo in their own windows --------------------------------
function Test-Port($port) { [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }

Step "Starting the API on http://localhost:8000"
if (Test-Port 8000) {
    Write-Host "Something is already listening on port 8000 - assuming the API is running."
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "Set-Location '$root'; `$host.UI.RawUI.WindowTitle = 'SmartPrice API'; & '$python' -m uvicorn app.main:app --reload"
}

Step "Starting the demo on http://localhost:8501"
if (Test-Port 8501) {
    Write-Host "Something is already listening on port 8501 - assuming the demo is running."
} else {
    Start-Process powershell -ArgumentList "-NoExit", "-Command",
        "Set-Location '$root'; `$host.UI.RawUI.WindowTitle = 'SmartPrice Demo'; & '$python' -m streamlit run demo/app.py --server.headless true"
}

# Wait for the demo to answer, then open it
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Port 8501) { break }
    Start-Sleep -Seconds 1
}
Start-Process "http://localhost:8501"

Write-Host "`nSmartPrice is running:" -ForegroundColor Green
Write-Host "  Demo      http://localhost:8501"
Write-Host "  API docs  http://localhost:8000/docs"
Write-Host "Stop everything with .\scripts\stop-demo.ps1 (or double-click stop-demo.bat)."
