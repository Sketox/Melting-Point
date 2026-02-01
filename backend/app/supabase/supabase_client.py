"""
Configuración de Supabase para el backend
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Usar service_role para backend

# Cliente de Supabase (singleton)
_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Obtiene o crea el cliente de Supabase (singleton pattern)
    
    Raises:
        ValueError: Si las credenciales de Supabase no están configuradas
    """
    global _supabase_client
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL y SUPABASE_SERVICE_KEY deben estar configuradas en .env"
        )
    
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return _supabase_client


# Exportar cliente para uso directo (puede ser None si no está configurado)
try:
    supabase = get_supabase_client()
except ValueError:
    supabase = None
    import warnings
    warnings.warn(
        "Supabase no configurado. Los endpoints de Supabase no estarán disponibles. "
        "Configura SUPABASE_URL y SUPABASE_SERVICE_KEY en .env para habilitar."
    )
