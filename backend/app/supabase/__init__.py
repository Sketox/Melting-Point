"""
Módulo de integración con Supabase.

Incluye:
- Cliente de Supabase (supabase_client)
- Servicios de Supabase (supabase_service)
- Rutas de Supabase (supabase_routes)
"""

from .supabase_client import get_supabase_client
from .supabase_service import SupabaseService
from .supabase_routes import router as supabase_router

__all__ = [
    "get_supabase_client",
    "SupabaseService",
    "supabase_router",
]
