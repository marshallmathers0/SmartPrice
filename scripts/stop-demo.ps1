# Stop the SmartPrice demo (Windows).
#
#   .\scripts\stop-demo.ps1         or double-click stop-demo.bat
#
# Stops the API (port 8000) and the demo (port 8501), then stops the
# PostgreSQL and Redis containers. Data is KEPT (docker compose stop, not down -v).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

foreach ($port in 8000, 8501) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($id in $owners) {
        # uvicorn --reload runs a child process; stop the whole tree so nothing keeps the port
        taskkill /PID $id /T /F *> $null
        Write-Host "Stopped the process on port $port (PID $id)."
    }
    if (-not $owners) { Write-Host "Nothing running on port $port." }
}

docker info *> $null
if ($LASTEXITCODE -eq 0) {
    docker compose stop
    Write-Host "PostgreSQL and Redis stopped (data kept)."
} else {
    Write-Host "Docker is not running - nothing to stop there."
}
