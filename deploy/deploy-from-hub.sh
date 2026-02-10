#!/bin/bash
# ============================================
# MeltingPoint Backend - Deploy desde Docker Hub
# ============================================
# Ejecutar en el SERVER (EC2, VPS, etc.)
#
# Uso:
#   1. Subir: docker-compose.hub.yml + .env.production + este script
#   2. chmod +x deploy-from-hub.sh
#   3. ./deploy-from-hub.sh
# ============================================

set -e

echo "============================================"
echo " MeltingPoint Backend - Deploy desde Docker Hub"
echo "============================================"
echo ""

# ============================================
# 1. VERIFICAR DOCKER
# ============================================
echo "[1/5] Verificando Docker..."

if ! command -v docker &> /dev/null; then
    echo "Docker no esta instalado. Instalando..."

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS="unknown"
    fi

    if [ "$OS" = "amzn" ]; then
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ec2-user
    elif [ "$OS" = "ubuntu" ]; then
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-plugin
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ubuntu
    else
        echo "ERROR: Instala Docker manualmente"
        exit 1
    fi
fi

# Docker compose plugin
if ! docker compose version &> /dev/null 2>&1; then
    echo "Instalando Docker Compose plugin..."
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
    sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

echo "  Docker: $(docker --version)"
echo "  Compose: $(docker compose version)"

# ============================================
# 2. SWAP (si no existe)
# ============================================
echo ""
echo "[2/5] Verificando swap..."

if [ ! -f /swapfile ]; then
    echo "  Creando 2GB de swap..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  Swap activado"
else
    echo "  Swap ya existe"
fi

# ============================================
# 3. VERIFICAR .env.production
# ============================================
echo ""
echo "[3/5] Verificando .env.production..."

if [ ! -f .env.production ]; then
    echo "  ERROR: No se encontro .env.production"
    echo "  Crealo con:"
    echo "    MONGODB_URL=mongodb+srv://..."
    echo "    MONGODB_DB_NAME=melting_point_db"
    echo "    JWT_SECRET_KEY=..."
    exit 1
fi
echo "  .env.production OK"

# ============================================
# 4. PULL IMAGEN
# ============================================
echo ""
echo "[4/5] Descargando imagen de Docker Hub..."

sudo docker compose -f docker-compose.hub.yml pull

echo "  Imagen descargada"

# ============================================
# 5. LEVANTAR SERVICIO
# ============================================
echo ""
echo "[5/5] Levantando backend..."

sudo docker compose -f docker-compose.hub.yml down 2>/dev/null || true
sudo docker compose -f docker-compose.hub.yml up -d

echo ""
echo "  Esperando que inicie (30s)..."
sleep 30

# Health check
if curl -s http://localhost:8000/health | grep -q "ok"; then
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<TU-IP>')
    echo ""
    echo "============================================"
    echo " MeltingPoint API corriendo!"
    echo "============================================"
    echo ""
    echo " API:  http://${PUBLIC_IP}:8000"
    echo " Docs: http://${PUBLIC_IP}:8000/docs"
    echo ""
else
    echo ""
    echo "  Aun iniciando... Espera 1-2 min y prueba:"
    echo "    curl http://localhost:8000/health"
    echo ""
    echo "  Ver logs:"
    echo "    sudo docker compose -f docker-compose.hub.yml logs -f"
fi
