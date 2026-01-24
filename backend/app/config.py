"""
config.py - Configuración de rutas y constantes para el backend.
"""

import os
from pathlib import Path

# Directorio base del backend
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de modelos
MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    str(BASE_DIR / "models" / "model.joblib")
)

CHEMPROP_MODEL_DIR = os.getenv(
    "CHEMPROP_MODEL_DIR",
    str(BASE_DIR / "models" / "model_chemprop")
)

# Rutas de datos
DATA_DIR = BASE_DIR.parent / "data"

TEST_PROCESSED_PATH = os.getenv(
    "TEST_PROCESSED_PATH",
    str(DATA_DIR / "processed" / "test_processed.csv")
)

# Archivo con SMILES mapeados a IDs (opcional)
SMILES_CSV_PATH = os.getenv(
    "SMILES_CSV_PATH",
    str(DATA_DIR / "processed" / "test_smiles.csv")
)

# Compuestos de usuarios
USER_COMPOUNDS_PATH = os.getenv(
    "USER_COMPOUNDS_PATH",
    str(BASE_DIR / "data" / "user_compounds.csv")
)

# Métricas del modelo (basado en entrenamiento)
MODEL_MAE = 28.85  # Mean Absolute Error en Kelvin
MODEL_MAE_STD = 3.16  # Desviación estándar del MAE
