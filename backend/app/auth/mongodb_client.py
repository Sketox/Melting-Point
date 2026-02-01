"""
Configuración de MongoDB
Cliente para conexión a MongoDB Atlas o local
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Configuración de MongoDB desde variables de entorno
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "melting_point_db")

# Cliente asíncrono (para FastAPI)
_async_client: Optional[AsyncIOMotorClient] = None
_async_db = None

# Cliente síncrono (para scripts)
_sync_client: Optional[MongoClient] = None
_sync_db = None


def get_async_database():
    """
    Obtiene la base de datos asíncrona (Motor)
    Usar en endpoints de FastAPI
    """
    global _async_client, _async_db
    
    if _async_client is None:
        logger.info(f"Conectando a MongoDB (async): {MONGODB_URL}")
        _async_client = AsyncIOMotorClient(MONGODB_URL)
        _async_db = _async_client[MONGODB_DB_NAME]
        logger.info(f"✓ Conectado a base de datos: {MONGODB_DB_NAME}")
    
    return _async_db


def get_sync_database():
    """
    Obtiene la base de datos síncrona (PyMongo)
    Usar en scripts de migración o testing
    """
    global _sync_client, _sync_db
    
    if _sync_client is None:
        logger.info(f"Conectando a MongoDB (sync): {MONGODB_URL}")
        _sync_client = MongoClient(MONGODB_URL)
        _sync_db = _sync_client[MONGODB_DB_NAME]
        logger.info(f"✓ Conectado a base de datos: {MONGODB_DB_NAME}")
    
    return _sync_db


async def close_mongodb_connection():
    """Cierra la conexión a MongoDB"""
    global _async_client, _sync_client
    
    if _async_client:
        _async_client.close()
        logger.info("✓ Conexión MongoDB async cerrada")
    
    if _sync_client:
        _sync_client.close()
        logger.info("✓ Conexión MongoDB sync cerrada")


async def test_mongodb_connection():
    """Prueba la conexión a MongoDB"""
    try:
        db = get_async_database()
        # Ping a la base de datos
        await db.command('ping')
        logger.info("✓ MongoDB connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        return False


# Colecciones de MongoDB
class Collections:
    """Nombres de las colecciones de MongoDB"""
    USERS = "users"
    USER_PREDICTIONS = "user_predictions"
    SESSIONS = "sessions"
    COMPOUNDS = "compounds"  # Opcional: si quieres migrar datos aquí también


# Índices para optimizar consultas
async def create_indexes():
    """Crea índices en las colecciones para mejor performance"""
    db = get_async_database()
    
    # Índices en la colección de usuarios
    await db[Collections.USERS].create_index("email", unique=True)
    await db[Collections.USERS].create_index("username", unique=True)
    
    # Índices en predicciones de usuario
    await db[Collections.USER_PREDICTIONS].create_index("user_id")
    await db[Collections.USER_PREDICTIONS].create_index("created_at")
    await db[Collections.USER_PREDICTIONS].create_index([("user_id", 1), ("created_at", -1)])
    
    # Índice en sesiones
    await db[Collections.SESSIONS].create_index("token", unique=True)
    await db[Collections.SESSIONS].create_index("expires_at")
    
    logger.info("✓ Índices de MongoDB creados")
