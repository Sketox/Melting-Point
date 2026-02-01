"""
Servicio de datos usando Supabase
Reemplaza la lógica de lectura de CSVs
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from .supabase_client import supabase
from ..schemas import (
    PredictResponse,
    StatsResponse,
    RangeResponse,
    CompoundResponse,
    DistributionResponse,
)

logger = logging.getLogger(__name__)


class SupabaseService:
    """Servicio para interactuar con Supabase"""
    
    @staticmethod
    async def get_all_predictions() -> List[Dict]:
        """Obtiene todas las predicciones del test set"""
        try:
            response = supabase.table('predictions_full')\
                .select('*')\
                .eq('dataset_type', 'test')\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error al obtener predicciones: {e}")
            raise
    
    @staticmethod
    async def get_prediction_by_id(compound_id: int) -> Optional[Dict]:
        """Obtiene una predicción específica por ID"""
        try:
            response = supabase.table('predictions_full')\
                .select('*')\
                .eq('original_id', compound_id)\
                .limit(1)\
                .execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error al obtener predicción {compound_id}: {e}")
            return None
    
    @staticmethod
    async def get_predictions_by_range(
        min_temp: float, 
        max_temp: float
    ) -> List[Dict]:
        """Obtiene predicciones en un rango de temperatura"""
        try:
            response = supabase.table('predictions')\
                .select('*')\
                .gte('tm_pred', min_temp)\
                .lte('tm_pred', max_temp)\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error al obtener predicciones por rango: {e}")
            raise
    
    @staticmethod
    async def get_statistics() -> Dict:
        """Obtiene estadísticas del modelo desde la vista"""
        try:
            # Primero intentar obtener del caché
            cache = supabase.table('statistics_cache')\
                .select('data')\
                .eq('stat_type', 'model_metrics')\
                .limit(1)\
                .execute()
            
            if cache.data:
                return cache.data[0]['data']
            
            # Si no hay caché, calcular desde la vista
            response = supabase.table('model_statistics')\
                .select('*')\
                .limit(1)\
                .execute()
            
            if response.data:
                stats = response.data[0]
                
                # Guardar en caché
                supabase.table('statistics_cache').upsert({
                    'stat_type': 'model_metrics',
                    'data': stats,
                    'updated_at': datetime.now().isoformat()
                }).execute()
                
                return stats
            
            return {}
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            raise
    
    @staticmethod
    async def get_distribution() -> List[Dict]:
        """Obtiene la distribución de temperaturas"""
        try:
            response = supabase.table('temperature_distribution')\
                .select('*')\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error al obtener distribución: {e}")
            raise
    
    @staticmethod
    async def save_user_prediction(
        smiles: str,
        tm_pred: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Guarda una predicción de usuario en Supabase"""
        try:
            data = {
                'smiles': smiles,
                'tm_pred': tm_pred,
                'user_id': user_id,
                'session_id': session_id,
                'metadata': metadata or {}
            }
            
            response = supabase.table('user_predictions')\
                .insert(data)\
                .execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error al guardar predicción de usuario: {e}")
            raise
    
    @staticmethod
    async def get_user_predictions(
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Obtiene predicciones de usuario"""
        try:
            query = supabase.table('user_predictions').select('*')
            
            if user_id:
                query = query.eq('user_id', user_id)
            elif session_id:
                query = query.eq('session_id', session_id)
            
            response = query.order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error al obtener predicciones de usuario: {e}")
            raise
    
    @staticmethod
    async def delete_user_prediction(prediction_id: int) -> bool:
        """Elimina una predicción de usuario"""
        try:
            response = supabase.table('user_predictions')\
                .delete()\
                .eq('id', prediction_id)\
                .execute()
            
            return True
        except Exception as e:
            logger.error(f"Error al eliminar predicción {prediction_id}: {e}")
            return False
    
    @staticmethod
    async def get_model_metadata() -> Dict:
        """Obtiene metadata del modelo activo"""
        try:
            response = supabase.table('model_metadata')\
                .select('*')\
                .eq('is_active', True)\
                .limit(1)\
                .execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error al obtener metadata del modelo: {e}")
            return {}
    
    @staticmethod
    async def search_compounds(
        smiles_pattern: Optional[str] = None,
        min_tm: Optional[float] = None,
        max_tm: Optional[float] = None,
        dataset_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Búsqueda avanzada de compuestos"""
        try:
            query = supabase.table('compounds').select('*')
            
            if smiles_pattern:
                query = query.ilike('smiles', f'%{smiles_pattern}%')
            
            if min_tm is not None:
                query = query.gte('tm_real', min_tm)
            
            if max_tm is not None:
                query = query.lte('tm_real', max_tm)
            
            if dataset_type:
                query = query.eq('dataset_type', dataset_type)
            
            response = query.limit(limit).execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error en búsqueda de compuestos: {e}")
            raise
    
    @staticmethod
    async def get_compound_by_id(compound_id: int) -> Optional[Dict]:
        """Obtiene un compuesto por su ID"""
        try:
            response = supabase.table('compounds')\
                .select('*')\
                .eq('compound_id', compound_id)\
                .limit(1)\
                .execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error al obtener compuesto {compound_id}: {e}")
            return None


# Instancia singleton del servicio
supabase_service = SupabaseService()
