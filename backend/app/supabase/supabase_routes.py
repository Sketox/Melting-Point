"""
Endpoints de API usando Supabase
Versión mejorada con base de datos en la nube
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import logging

from .supabase_service import supabase_service
from ..schemas import (
    PredictResponse,
    StatsResponse,
    DistributionResponse,
    CompoundResponse,
)

logger = logging.getLogger(__name__)

# Router para endpoints de Supabase
router = APIRouter(prefix="/api/v2", tags=["Supabase API"])


@router.get("/predictions", response_model=List[PredictResponse])
async def get_all_predictions_v2():
    """
    📊 Obtiene todas las predicciones desde Supabase
    
    Retorna todas las predicciones del test set almacenadas en la base de datos.
    """
    try:
        predictions = await supabase_service.get_all_predictions()
        
        result = []
        for pred in predictions:
            result.append(PredictResponse(
                id=pred.get('original_id'),
                smiles=pred.get('smiles'),
                Tm_pred=float(pred.get('tm_pred', 0))
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error al obtener predicciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{compound_id}", response_model=PredictResponse)
async def get_prediction_by_id_v2(compound_id: int):
    """
    🔍 Obtiene una predicción específica por ID desde Supabase
    
    - **compound_id**: ID del compuesto en el dataset
    """
    try:
        pred = await supabase_service.get_prediction_by_id(compound_id)
        
        if not pred:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontró predicción para compound_id={compound_id}"
            )
        
        return PredictResponse(
            id=pred.get('original_id'),
            smiles=pred.get('smiles'),
            Tm_pred=float(pred.get('tm_pred', 0))
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/range", response_model=List[PredictResponse])
async def get_predictions_by_range_v2(
    min_temp: float = Query(..., description="Temperatura mínima (K)"),
    max_temp: float = Query(..., description="Temperatura máxima (K)")
):
    """
    🌡️ Filtra predicciones por rango de temperatura desde Supabase
    
    - **min_temp**: Temperatura mínima en Kelvin
    - **max_temp**: Temperatura máxima en Kelvin
    """
    try:
        if min_temp > max_temp:
            raise HTTPException(
                status_code=400,
                detail="min_temp no puede ser mayor que max_temp"
            )
        
        predictions = await supabase_service.get_predictions_by_range(min_temp, max_temp)
        
        result = []
        for pred in predictions:
            result.append(PredictResponse(
                id=pred.get('id'),
                smiles=pred.get('smiles'),
                Tm_pred=float(pred.get('tm_pred', 0))
            ))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al filtrar predicciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=StatsResponse)
async def get_statistics_v2():
    """
    📈 Estadísticas del modelo desde Supabase
    
    Retorna métricas calculadas del modelo almacenadas en la base de datos.
    """
    try:
        stats = await supabase_service.get_statistics()
        
        return StatsResponse(
            total_predictions=int(stats.get('total_predictions', 0)),
            mae=float(stats.get('mae', 0)),
            rmse=float(stats.get('rmse', 0)),
            min_prediction=float(stats.get('min_prediction', 0)),
            max_prediction=float(stats.get('max_prediction', 0)),
            avg_prediction=float(stats.get('avg_prediction', 0))
        )
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution", response_model=List[DistributionResponse])
async def get_distribution_v2():
    """
    📊 Distribución de temperaturas desde Supabase
    
    Retorna la distribución de predicciones por rangos de temperatura.
    """
    try:
        distribution = await supabase_service.get_distribution()
        
        result = []
        for item in distribution:
            result.append(DistributionResponse(
                range=item.get('temperature_range'),
                count=int(item.get('count', 0)),
                percentage=float(item.get('percentage', 0))
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error al obtener distribución: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-metadata")
async def get_model_metadata_v2():
    """
    🧠 Metadata del modelo desde Supabase
    
    Retorna información detallada del modelo activo.
    """
    try:
        metadata = await supabase_service.get_model_metadata()
        return metadata
    except Exception as e:
        logger.error(f"Error al obtener metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/compounds")
async def search_compounds_v2(
    smiles_pattern: Optional[str] = Query(None, description="Patrón de búsqueda en SMILES"),
    min_tm: Optional[float] = Query(None, description="Temperatura mínima (K)"),
    max_tm: Optional[float] = Query(None, description="Temperatura máxima (K)"),
    dataset_type: Optional[str] = Query(None, description="Tipo de dataset: 'train' o 'test'"),
    limit: int = Query(100, le=1000, description="Límite de resultados")
):
    """
    🔍 Búsqueda avanzada de compuestos en Supabase
    
    Permite filtrar compuestos por múltiples criterios.
    """
    try:
        compounds = await supabase_service.search_compounds(
            smiles_pattern=smiles_pattern,
            min_tm=min_tm,
            max_tm=max_tm,
            dataset_type=dataset_type,
            limit=limit
        )
        
        return compounds
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        raise HTTPException(status_code=500, detail=str(e))
