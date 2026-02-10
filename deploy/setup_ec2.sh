#!/bin/bash
# ============================================
# MeltingPoint Backend - EC2 Setup Script
# ============================================
# Ejecutar en una instancia EC2 t2.micro (Amazon Linux 2023 o Ubuntu 22.04)
#
# Uso:
#   1. Lanza una instancia EC2 t2.micro (free tier)
#      - AMI: Amazon Linux 2023 o Ubuntu 22.04
#      - Security Group: abrir puerto 22 (SSH) y 8000 (API)
#      - Storage: 30 GB (free tier)
#      - Key pair: crear o usar uno existente
#
#   2. Conectar por SSH:
#      ssh -i tu-key.pem ec2-user@<IP-PUBLICA>
#
#   3. Subir el proyecto (desde tu PC):
#      scp -i tu-key.pem -r MeltingPoint/ ec2-user@<IP-PUBLICA>:~/
#
#   4. Ejecutar este script:
#      cd ~/MeltingPoint
#      chmod +x deploy/setup_ec2.sh
#      ./deploy/setup_ec2.sh
#
# ============================================

set -e  # Salir si hay error

echo "============================================"
echo " MeltingPoint Backend - Setup EC2"
echo "============================================"
echo ""

# ============================================
# 1. DETECTAR OS
# ============================================
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

echo "[1/6] Sistema detectado: $OS"

# ============================================
# 2. CREAR SWAP (2GB)
# ============================================
echo ""
echo "[2/6] Configurando swap de 2GB..."

if [ -f /swapfile ]; then
    echo "  Swap ya existe, saltando..."
else
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    # Hacer permanente
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  Swap de 2GB activado"
fi

# Verificar
echo "  Memoria disponible:"
free -h | head -3

# ============================================
# 3. INSTALAR DOCKER
# ============================================
echo ""
echo "[3/6] Instalando Docker..."

if command -v docker &> /dev/null; then
    echo "  Docker ya está instalado"
else
    if [ "$OS" = "amzn" ]; then
        # Amazon Linux 2023
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ec2-user
    elif [ "$OS" = "ubuntu" ]; then
        # Ubuntu
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-plugin
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ubuntu
    else
        echo "ERROR: OS no soportado. Instala Docker manualmente."
        exit 1
    fi
    echo "  Docker instalado"
fi

# Instalar docker compose plugin si no existe
if ! docker compose version &> /dev/null 2>&1; then
    if [ "$OS" = "amzn" ]; then
        sudo mkdir -p /usr/local/lib/docker/cli-plugins
        COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
        sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" -o /usr/local/lib/docker/cli-plugins/docker-compose
        sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    fi
fi

echo "  Docker Compose: $(docker compose version 2>/dev/null || echo 'instalando...')"

# ============================================
# 4. CONFIGURAR VARIABLES DE ENTORNO
# ============================================
echo ""
echo "[4/6] Configurando variables de entorno..."

if [ ! -f .env.production ]; then
    echo "  ERROR: No se encontró .env.production"
    echo "  Copia el template y edítalo:"
    echo "    cp .env.production.example .env.production"
    echo "    nano .env.production"
    exit 1
fi

# Verificar que tiene valores reales (no los placeholder)
if grep -q "CAMBIA_ESTO" .env.production; then
    echo ""
    echo "  ⚠️  IMPORTANTE: Edita .env.production con tus valores reales:"
    echo "     nano .env.production"
    echo ""
    echo "  Necesitas configurar:"
    echo "    - MONGODB_URL (tu connection string de MongoDB Atlas)"
    echo "    - JWT_SECRET_KEY (genera uno con: python3 -c \"import secrets; print(secrets.token_urlsafe(64))\")"
    echo ""
    read -p "  ¿Ya editaste .env.production? (y/n): " EDITED
    if [ "$EDITED" != "y" ]; then
        echo "  Edita el archivo y ejecuta el script de nuevo."
        exit 0
    fi
fi

echo "  Variables de entorno OK"

# ============================================
# 5. BUILD DOCKER IMAGE
# ============================================
echo ""
echo "[5/6] Construyendo imagen Docker (esto toma 5-10 minutos)..."
echo "  (PyTorch CPU-only + RDKit + ChemProp + Ensemble)"
echo ""

# Usar docker compose o docker-compose
if docker compose version &> /dev/null 2>&1; then
    sudo docker compose build --no-cache
else
    sudo docker-compose build --no-cache
fi

echo ""
echo "  Imagen construida exitosamente"

# ============================================
# 6. INICIAR SERVICIO
# ============================================
echo ""
echo "[6/6] Iniciando MeltingPoint API..."

if docker compose version &> /dev/null 2>&1; then
    sudo docker compose up -d
else
    sudo docker-compose up -d
fi

# Esperar a que inicie
echo "  Esperando que el servicio inicie (30 segundos)..."
sleep 30

# Health check
echo ""
echo "  Verificando..."
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo ""
    echo "============================================"
    echo " ✅ MeltingPoint API corriendo exitosamente!"
    echo "============================================"
    echo ""
    echo " URL local:  http://localhost:8000"
    echo " URL publica: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<TU-IP-PUBLICA>'):8000"
    echo " Docs:       http://<TU-IP-PUBLICA>:8000/docs"
    echo ""
    echo " Para ver logs:"
    echo "   sudo docker compose logs -f"
    echo ""
    echo " Para parar:"
    echo "   sudo docker compose down"
    echo ""
else
    echo ""
    echo "  ⚠️  El servicio aún está iniciando (los modelos tardan en cargar)."
    echo "  Espera 1-2 minutos y prueba:"
    echo "    curl http://localhost:8000/health"
    echo ""
    echo "  Ver logs:"
    echo "    sudo docker compose logs -f"
fi
