"""
Endpoints para predicciones de usuario autenticado
Guardar, listar, actualizar, eliminar predicciones personales
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import logging

from .auth_service import get_current_user, get_current_user_optional
from .auth_schemas import UserResponse, UserPredictionInDB
from .mongodb_client import get_async_database, Collections
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/user-predictions",
    tags=["💾 User Predictions"]
)


# ============================================
# SCHEMAS
# ============================================

class SavePredictionRequest(BaseModel):
    """Request para guardar una predicción"""
    smiles: str
    tm_pred: float = Field(..., description="Temperatura predicha en Kelvin")
    compound_name: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=500)
    is_favorite: bool = False


class UpdatePredictionRequest(BaseModel):
    """Request para actualizar una predicción"""
    compound_name: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=500)
    is_favorite: Optional[bool] = None


class PredictionResponse(BaseModel):
    """Response de una predicción"""
    id: str
    user_id: str
    username: str
    smiles: str
    tm_pred: float
    tm_pred_celsius: float
    compound_name: Optional[str] = None
    notes: Optional[str] = None
    is_favorite: bool
    created_at: datetime
    
    class Config:
        json_encoders = {ObjectId: str}


# ============================================
# ENDPOINTS
# ============================================

@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def save_prediction(
    prediction_data: SavePredictionRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    💾 Guardar una predicción personal
    
    Requiere autenticación. Guarda una predicción de punto de fusión
    asociada al usuario actual.
    """
    db = get_async_database()
    
    try:
        # Calcular temperatura en Celsius
        tm_celsius = prediction_data.tm_pred - 273.15
        
        # Crear predicción
        prediction = UserPredictionInDB(
            user_id=current_user.id,
            username=current_user.username,
            smiles=prediction_data.smiles,
            tm_pred=prediction_data.tm_pred,
            tm_pred_celsius=tm_celsius,
            compound_name=prediction_data.compound_name,
            notes=prediction_data.notes,
            is_favorite=prediction_data.is_favorite,
            created_at=datetime.utcnow()
        )
        
        # Guardar en MongoDB
        prediction_dict = prediction.model_dump(by_alias=True, exclude={"id"})
        result = await db[Collections.USER_PREDICTIONS].insert_one(prediction_dict)
        
        # Incrementar contador de predicciones del usuario
        await db[Collections.USERS].update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"predictions_count": 1}}
        )
        
        prediction.id = result.inserted_id
        
        logger.info(f"✓ Predicción guardada: {current_user.username} - {prediction_data.smiles}")
        
        return PredictionResponse(
            id=str(prediction.id),
            user_id=prediction.user_id,
            username=prediction.username,
            smiles=prediction.smiles,
            tm_pred=prediction.tm_pred,
            tm_pred_celsius=prediction.tm_pred_celsius,
            compound_name=prediction.compound_name,
            notes=prediction.notes,
            is_favorite=prediction.is_favorite,
            created_at=prediction.created_at
        )
        
    except Exception as e:
        logger.error(f"Error al guardar predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al guardar la predicción"
        )


@router.get("/", response_model=List[PredictionResponse])
async def get_my_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=100),
    favorites_only: bool = Query(False),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    📋 Obtener mis predicciones
    
    Retorna las predicciones del usuario autenticado, ordenadas por fecha (más reciente primero).
    
    - **skip**: Número de predicciones a saltar (paginación)
    - **limit**: Máximo de predicciones a retornar (máx 100)
    - **favorites_only**: Solo retornar favoritas
    """
    db = get_async_database()
    
    try:
        # Construir query
        query = {"user_id": current_user.id}
        if favorites_only:
            query["is_favorite"] = True
        
        # Obtener predicciones
        cursor = db[Collections.USER_PREDICTIONS].find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)
        
        predictions = await cursor.to_list(length=limit)
        
        # Convertir a response
        result = []
        for pred in predictions:
            result.append(PredictionResponse(
                id=str(pred["_id"]),
                user_id=pred["user_id"],
                username=pred["username"],
                smiles=pred["smiles"],
                tm_pred=pred["tm_pred"],
                tm_pred_celsius=pred["tm_pred_celsius"],
                compound_name=pred.get("compound_name"),
                notes=pred.get("notes"),
                is_favorite=pred.get("is_favorite", False),
                created_at=pred["created_at"]
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error al obtener predicciones: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener predicciones"
        )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    🔍 Obtener una predicción específica
    
    Requiere que la predicción pertenezca al usuario autenticado.
    """
    db = get_async_database()
    
    try:
        prediction = await db[Collections.USER_PREDICTIONS].find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user.id
        })
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Predicción no encontrada"
            )
        
        return PredictionResponse(
            id=str(prediction["_id"]),
            user_id=prediction["user_id"],
            username=prediction["username"],
            smiles=prediction["smiles"],
            tm_pred=prediction["tm_pred"],
            tm_pred_celsius=prediction["tm_pred_celsius"],
            compound_name=prediction.get("compound_name"),
            notes=prediction.get("notes"),
            is_favorite=prediction.get("is_favorite", False),
            created_at=prediction["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener la predicción"
        )


@router.put("/{prediction_id}", response_model=PredictionResponse)
async def update_prediction(
    prediction_id: str,
    update_data: UpdatePredictionRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    ✏️ Actualizar una predicción
    
    Permite actualizar nombre del compuesto, notas y estado de favorito.
    """
    db = get_async_database()
    
    try:
        # Verificar que la predicción existe y pertenece al usuario
        prediction = await db[Collections.USER_PREDICTIONS].find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user.id
        })
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Predicción no encontrada"
            )
        
        # Preparar datos de actualización
        update_fields = {}
        if update_data.compound_name is not None:
            update_fields["compound_name"] = update_data.compound_name
        if update_data.notes is not None:
            update_fields["notes"] = update_data.notes
        if update_data.is_favorite is not None:
            update_fields["is_favorite"] = update_data.is_favorite
        
        # Actualizar en BD
        await db[Collections.USER_PREDICTIONS].update_one(
            {"_id": ObjectId(prediction_id)},
            {"$set": update_fields}
        )
        
        # Obtener predicción actualizada
        updated_prediction = await db[Collections.USER_PREDICTIONS].find_one({
            "_id": ObjectId(prediction_id)
        })
        
        logger.info(f"✓ Predicción actualizada: {current_user.username} - {prediction_id}")
        
        return PredictionResponse(
            id=str(updated_prediction["_id"]),
            user_id=updated_prediction["user_id"],
            username=updated_prediction["username"],
            smiles=updated_prediction["smiles"],
            tm_pred=updated_prediction["tm_pred"],
            tm_pred_celsius=updated_prediction["tm_pred_celsius"],
            compound_name=updated_prediction.get("compound_name"),
            notes=updated_prediction.get("notes"),
            is_favorite=updated_prediction.get("is_favorite", False),
            created_at=updated_prediction["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al actualizar predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al actualizar la predicción"
        )


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    🗑️ Eliminar una predicción
    
    Elimina permanentemente una predicción del usuario.
    """
    db = get_async_database()
    
    try:
        # Verificar que la predicción existe y pertenece al usuario
        prediction = await db[Collections.USER_PREDICTIONS].find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user.id
        })
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Predicción no encontrada"
            )
        
        # Eliminar predicción
        await db[Collections.USER_PREDICTIONS].delete_one({
            "_id": ObjectId(prediction_id)
        })
        
        # Decrementar contador de predicciones del usuario
        await db[Collections.USERS].update_one(
            {"_id": ObjectId(current_user.id)},
            {"$inc": {"predictions_count": -1}}
        )
        
        logger.info(f"✓ Predicción eliminada: {current_user.username} - {prediction_id}")
        
        return {
            "message": "Predicción eliminada exitosamente",
            "prediction_id": prediction_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al eliminar la predicción"
        )


@router.get("/search/by-smiles")
async def search_predictions_by_smiles(
    smiles: str = Query(..., min_length=1),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    🔎 Buscar predicciones por SMILES
    
    Busca predicciones del usuario que contengan el SMILES especificado.
    """
    db = get_async_database()
    
    try:
        cursor = db[Collections.USER_PREDICTIONS].find({
            "user_id": current_user.id,
            "smiles": {"$regex": smiles, "$options": "i"}
        }).sort("created_at", -1)
        
        predictions = await cursor.to_list(length=50)
        
        result = []
        for pred in predictions:
            result.append(PredictionResponse(
                id=str(pred["_id"]),
                user_id=pred["user_id"],
                username=pred["username"],
                smiles=pred["smiles"],
                tm_pred=pred["tm_pred"],
                tm_pred_celsius=pred["tm_pred_celsius"],
                compound_name=pred.get("compound_name"),
                notes=pred.get("notes"),
                is_favorite=pred.get("is_favorite", False),
                created_at=pred["created_at"]
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error en la búsqueda"
        )
