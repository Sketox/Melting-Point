# ============================================
# MeltingPoint Backend - Build Local + Push a Docker Hub
# ============================================
# Ejecutar desde: D:\devu\Kaggle\MeltingPoint\
#
# ANTES DE EJECUTAR:
#   1. Cambiar $DOCKER_USER por tu usuario de Docker Hub
#   2. docker login
# ============================================

$DOCKER_USER = "sketox"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MeltingPoint Backend - Build & Push" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# 1. BUILD BACKEND
# ============================================
Write-Host "[1/2] Building backend..." -ForegroundColor Yellow

docker build -t "${DOCKER_USER}/melting-point-backend:latest" -f Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Build OK" -ForegroundColor Green

# ============================================
# 2. PUSH
# ============================================
Write-Host ""
Write-Host "[2/2] Pushing a Docker Hub..." -ForegroundColor Yellow

docker push "${DOCKER_USER}/melting-point-backend:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Push failed! Hiciste docker login?" -ForegroundColor Red
    exit 1
}
Write-Host "Push OK" -ForegroundColor Green

# ============================================
# DONE
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Imagen pusheada a Docker Hub!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Imagen: ${DOCKER_USER}/melting-point-backend:latest" -ForegroundColor Cyan
Write-Host ""
Write-Host "En el server:" -ForegroundColor Cyan
Write-Host "  docker compose -f docker-compose.hub.yml pull"
Write-Host "  docker compose -f docker-compose.hub.yml up -d"
Write-Host ""
