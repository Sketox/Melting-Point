from pathlib import Path

# Este archivo se carga cuando levantas FastAPI desde la carpeta backend.
# BASE_DIR = carpeta backend
BASE_DIR = Path(__file__).resolve().parent.parent

# PROJECT_ROOT = carpeta raíz del proyecto (MeltingPoint)
PROJECT_ROOT = BASE_DIR.parent

# Ruta al modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

# Ruta al CSV de test procesado
TEST_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"
