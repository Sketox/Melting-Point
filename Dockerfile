# ============================================
# MeltingPoint Backend - Dockerfile
# Optimizado para EC2 t2.micro (1GB RAM + swap)
# PyTorch CPU-only para inferencia
# ============================================

FROM python:3.11-slim AS base

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para RDKit y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxrender1 \
    libxext6 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# ============================================
# Instalar dependencias Python
# ============================================

# Primero PyTorch CPU-only (mucho más liviano que con CUDA)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Requirements del proyecto (sin torch, sin catboost)
COPY backend/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Copiar código y datos
# ============================================

# Código del backend
COPY backend/app/ ./backend/app/
COPY backend/patch_chemprop_torch.py ./backend/

# Modelos necesarios (NO copiar chemprop_max - experimental)
COPY backend/models/model.joblib ./backend/models/model.joblib
COPY backend/models/ensemble_predictor.joblib ./backend/models/ensemble_predictor.joblib
COPY backend/models/best_params_paso6.json ./backend/models/best_params_paso6.json
COPY backend/models/model_chemprop/ ./backend/models/model_chemprop/

# Datos necesarios
COPY data/raw/train.csv ./data/raw/train.csv
COPY data/raw/test.csv ./data/raw/test.csv
COPY data/raw/train_with_names.csv ./data/raw/train_with_names.csv
COPY data/raw/test_with_names.csv ./data/raw/test_with_names.csv
COPY data/processed/test_processed.csv ./data/processed/test_processed.csv
COPY data/processed/test_chemprop_predictions.csv ./data/processed/test_chemprop_predictions.csv

# CSV de predicciones ChemProp
COPY data/processed/chemprop_predictions.csv ./data/processed/chemprop_predictions.csv

# ============================================
# Ejecutar patch de ChemProp (OBLIGATORIO)
# ============================================
RUN cd /app/backend && python patch_chemprop_torch.py || echo "Patch warning (puede que ya esté aplicado)"

# ============================================
# Configuración de runtime
# ============================================

# Puerto del API
EXPOSE 8000

# Variables de entorno por defecto
ENV MODEL_PATH=/app/backend/models/model.joblib
ENV CHEMPROP_MODEL_DIR=/app/backend/models/model_chemprop
ENV TRAIN_DATASET_PATH=/app/data/raw/train_with_names.csv
ENV TEST_DATASET_PATH=/app/data/raw/test_with_names.csv
ENV TEST_PROCESSED_PATH=/app/data/processed/test_processed.csv
ENV CHEMPROP_PREDICTIONS_PATH=/app/data/processed/test_chemprop_predictions.csv
ENV SMILES_CSV_PATH=""
ENV USER_COMPOUNDS_PATH=/app/backend/data/user_compounds.csv

# Crear directorio para datos de usuario
RUN mkdir -p /app/backend/data

# Comando de inicio
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
