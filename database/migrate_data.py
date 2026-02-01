"""
Script de migración de datos CSV a Supabase
Carga los datos de train.csv y las predicciones a la base de datos
"""

import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL y SUPABASE_KEY deben estar configuradas en el archivo .env")

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'train.csv')
TEST_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'test.csv')
PREDICTIONS_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'chemprop_test_preds.csv')


def load_compounds_from_train():
    """Carga compuestos del dataset de entrenamiento"""
    logger.info("Cargando compuestos del dataset de entrenamiento...")
    
    df = pd.read_csv(TRAIN_CSV)
    
    compounds = []
    for _, row in df.iterrows():
        compound = {
            'compound_id': int(row['id']),
            'smiles': row['SMILES'],
            'tm_real': float(row['Tm']),
            'dataset_type': 'train'
        }
        compounds.append(compound)
    
    return compounds


def load_compounds_from_test():
    """Carga compuestos del dataset de test"""
    logger.info("Cargando compuestos del dataset de test...")
    
    df = pd.read_csv(TEST_CSV)
    
    compounds = []
    for idx, row in df.iterrows():
        compound = {
            'compound_id': 10000 + idx,  # ID único para test
            'smiles': row['smiles'],
            'tm_real': None,  # No hay valores reales en test
            'dataset_type': 'test'
        }
        compounds.append(compound)
    
    return compounds


def load_predictions():
    """Carga las predicciones del modelo"""
    logger.info("Cargando predicciones del modelo...")
    
    df = pd.read_csv(PREDICTIONS_CSV)
    
    predictions = []
    for idx, row in df.iterrows():
        prediction = {
            'smiles': row['smiles'],
            'tm_pred': float(row['target']),
            'model_version': 'chemprop_v1'
        }
        predictions.append(prediction)
    
    return predictions


def insert_in_batches(table: str, data: List[Dict], batch_size: int = 500):
    """Inserta datos en batches para mejor performance"""
    total = len(data)
    logger.info(f"Insertando {total} registros en la tabla '{table}'...")
    
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        try:
            response = supabase.table(table).insert(batch).execute()
            logger.info(f"Insertados {i + len(batch)}/{total} registros")
        except Exception as e:
            logger.error(f"Error al insertar batch {i}-{i+batch_size}: {str(e)}")
            # Intentar insertar uno por uno en caso de error
            for item in batch:
                try:
                    supabase.table(table).insert(item).execute()
                except Exception as e2:
                    logger.error(f"Error al insertar registro individual: {str(e2)}")
                    logger.error(f"Datos: {item}")
    
    logger.info(f"✓ Completada inserción en tabla '{table}'")


def link_predictions_to_compounds():
    """Vincula las predicciones con los compuestos mediante SMILES"""
    logger.info("Vinculando predicciones con compuestos...")
    
    # Obtener todas las predicciones sin compound_id
    predictions = supabase.table('predictions').select('id, smiles').is_('compound_id', 'null').execute()
    
    updated_count = 0
    for pred in predictions.data:
        # Buscar el compound_id correspondiente
        compound = supabase.table('compounds').select('id').eq('smiles', pred['smiles']).limit(1).execute()
        
        if compound.data:
            compound_id = compound.data[0]['id']
            supabase.table('predictions').update({'compound_id': compound_id}).eq('id', pred['id']).execute()
            updated_count += 1
    
    logger.info(f"✓ Vinculadas {updated_count} predicciones con compuestos")


def clear_all_tables():
    """Limpia todas las tablas (útil para re-migración)"""
    logger.warning("⚠️ LIMPIANDO TODAS LAS TABLAS...")
    
    tables = ['predictions', 'user_predictions', 'compounds', 'statistics_cache']
    
    for table in tables:
        try:
            # Supabase no tiene TRUNCATE directo, usamos delete
            supabase.table(table).delete().neq('id', 0).execute()
            logger.info(f"✓ Tabla '{table}' limpiada")
        except Exception as e:
            logger.error(f"Error al limpiar tabla '{table}': {str(e)}")


def main():
    """Función principal de migración"""
    print("=" * 60)
    print("MIGRACIÓN DE DATOS CSV A SUPABASE")
    print("=" * 60)
    
    # Preguntar si limpiar tablas
    clear = input("\n¿Deseas limpiar las tablas antes de migrar? (s/n): ")
    if clear.lower() == 's':
        clear_all_tables()
    
    print("\nIniciando migración...")
    
    try:
        # 1. Cargar e insertar compuestos de entrenamiento
        train_compounds = load_compounds_from_train()
        insert_in_batches('compounds', train_compounds)
        
        # 2. Cargar e insertar compuestos de test
        test_compounds = load_compounds_from_test()
        insert_in_batches('compounds', test_compounds)
        
        # 3. Cargar e insertar predicciones
        predictions = load_predictions()
        insert_in_batches('predictions', predictions)
        
        # 4. Vincular predicciones con compuestos
        link_predictions_to_compounds()
        
        print("\n" + "=" * 60)
        print("✅ MIGRACIÓN COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
        # Mostrar estadísticas
        compounds_count = supabase.table('compounds').select('id', count='exact').execute()
        predictions_count = supabase.table('predictions').select('id', count='exact').execute()
        
        print(f"\nEstadísticas:")
        print(f"  - Compuestos: {compounds_count.count}")
        print(f"  - Predicciones: {predictions_count.count}")
        
    except Exception as e:
        logger.error(f"❌ Error durante la migración: {str(e)}")
        raise


if __name__ == "__main__":
    main()
