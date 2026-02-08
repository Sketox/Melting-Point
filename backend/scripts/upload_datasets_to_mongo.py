"""
Script para subir los datasets train y test con nombres a MongoDB.
Ejecutar una sola vez para migrar los datos.

Uso:
    cd MeltingPoint/backend
    python scripts/upload_datasets_to_mongo.py
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Configuracion de MongoDB
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "melting_point_db")

# Rutas a los archivos CSV
DATA_DIR = backend_dir.parent / "data" / "raw"
TRAIN_FILE = DATA_DIR / "train_with_names.csv"
TEST_FILE = DATA_DIR / "test_with_names.csv"

# Ruta a predicciones pre-calculadas (para Tm de test)
# Usar submission_paso6_cp20.csv que tiene las predicciones del modelo hibrido (MAE 22.80 K)
PREDICTIONS_FILE = backend_dir.parent / "submissions" / "submission_paso6_cp20.csv"


def load_train_data():
    """Carga datos de entrenamiento con nombres."""
    print(f"Cargando train desde: {TRAIN_FILE}")
    # Solo cargar columnas necesarias (el archivo tiene 424+ columnas de grupos)
    df = pd.read_csv(TRAIN_FILE, usecols=['id', 'SMILES', 'Tm', 'name'])

    # Columnas necesarias: id, SMILES, Tm, name
    records = []
    for _, row in df.iterrows():
        record = {
            "compound_id": int(row["id"]),
            "smiles": row["SMILES"],
            "Tm": float(row["Tm"]),
            "Tm_celsius": round(float(row["Tm"]) - 273.15, 2),
            "name": row["name"] if pd.notna(row["name"]) else None,
            "source": "train",
            "created_at": datetime.utcnow(),
            "is_experimental": True  # Datos reales de entrenamiento
        }
        records.append(record)

    print(f"  - {len(records)} registros de train cargados")
    return records


def load_test_data():
    """Carga datos de test con nombres y predicciones."""
    print(f"Cargando test desde: {TEST_FILE}")
    # Solo cargar columnas necesarias (el archivo tiene 424+ columnas de grupos)
    df_test = pd.read_csv(TEST_FILE, usecols=['id', 'SMILES', 'name'])

    # Cargar predicciones si existen
    predictions = {}
    if PREDICTIONS_FILE.exists():
        print(f"Cargando predicciones desde: {PREDICTIONS_FILE}")
        df_preds = pd.read_csv(PREDICTIONS_FILE)
        for _, row in df_preds.iterrows():
            predictions[int(row["id"])] = float(row["Tm"])

    records = []
    for _, row in df_test.iterrows():
        compound_id = int(row["id"])

        # Obtener Tm predicho
        tm_pred = predictions.get(compound_id, None)

        record = {
            "compound_id": compound_id,
            "smiles": row["SMILES"],
            "Tm": tm_pred,
            "Tm_celsius": round(tm_pred - 273.15, 2) if tm_pred else None,
            "name": row["name"] if pd.notna(row["name"]) else None,
            "source": "test",
            "created_at": datetime.utcnow(),
            "is_experimental": False  # Datos predichos
        }
        records.append(record)

    print(f"  - {len(records)} registros de test cargados")
    return records


def upload_to_mongodb(records, collection_name="compounds"):
    """Sube los registros a MongoDB."""
    print(f"\nConectando a MongoDB: {MONGODB_URL}")
    client = MongoClient(MONGODB_URL)
    db = client[MONGODB_DB_NAME]
    collection = db[collection_name]

    # Limpiar coleccion existente (solo train y test, no user)
    result = collection.delete_many({"source": {"$in": ["train", "test"]}})
    print(f"Registros previos eliminados: {result.deleted_count}")

    # Insertar nuevos registros
    if records:
        result = collection.insert_many(records)
        print(f"Registros insertados: {len(result.inserted_ids)}")

    # Crear indices
    print("Creando indices...")
    collection.create_index("compound_id")
    collection.create_index("smiles")
    collection.create_index("source")
    collection.create_index("name")
    collection.create_index([("source", 1), ("compound_id", 1)])

    print("Indices creados correctamente")

    # Estadisticas
    total = collection.count_documents({})
    train_count = collection.count_documents({"source": "train"})
    test_count = collection.count_documents({"source": "test"})
    user_count = collection.count_documents({"source": "user"})

    print(f"\nEstadisticas de la coleccion '{collection_name}':")
    print(f"  - Total: {total}")
    print(f"  - Train: {train_count}")
    print(f"  - Test: {test_count}")
    print(f"  - User: {user_count}")

    client.close()
    return total


def main():
    print("=" * 60)
    print("MIGRACION DE DATASETS A MONGODB")
    print("=" * 60)

    # Verificar archivos
    if not TRAIN_FILE.exists():
        print(f"ERROR: No se encuentra {TRAIN_FILE}")
        return

    if not TEST_FILE.exists():
        print(f"ERROR: No se encuentra {TEST_FILE}")
        return

    # Cargar datos
    train_records = load_train_data()
    test_records = load_test_data()

    # Combinar
    all_records = train_records + test_records
    print(f"\nTotal de registros a subir: {len(all_records)}")

    # Subir a MongoDB
    total = upload_to_mongodb(all_records)

    print("\n" + "=" * 60)
    print(f"MIGRACION COMPLETADA: {total} registros en MongoDB")
    print("=" * 60)


if __name__ == "__main__":
    main()
