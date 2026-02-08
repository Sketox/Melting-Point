"""
schemas.py - Esquemas Pydantic para la API.

ACTUALIZADO:
- ValidateSmilesRequest/Response para validación de SMILES
- ModelInfoResponse para información del modelo
- CompoundResponse con campo de incertidumbre
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


# ============================================
# REQUEST MODELS
# ============================================

class PredictByIdRequest(BaseModel):
    """Request para predicción por ID."""
    id: int = Field(..., description="ID de la molécula (1-667)", ge=1)


class CompoundCreateRequest(BaseModel):
    """Request para crear un compuesto de usuario."""
    smiles: str = Field(..., description="Estructura SMILES de la molécula", min_length=1)
    name: str = Field(..., description="Nombre del compuesto", min_length=1)


class ValidateSmilesRequest(BaseModel):
    """Request para validar un SMILES."""
    smiles: str = Field(..., description="String SMILES a validar")


# ============================================
# RESPONSE MODELS - Basic
# ============================================

class RootResponse(BaseModel):
    """Respuesta del endpoint raíz."""
    message: str
    status: str
    version: str
    docs: str
    endpoints_count: int


class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str
    model_loaded: bool
    dataset_size: int


class ModelInfoResponse(BaseModel):
    """Información del modelo ML."""
    name: str = Field(..., description="Nombre del modelo")
    type: str = Field(..., description="Tipo de modelo")
    mae: float = Field(..., description="Mean Absolute Error en Kelvin")
    mae_std: float = Field(..., description="Desviación estándar del MAE")
    folds: int = Field(..., description="Número de folds en cross-validation")
    epochs: int = Field(..., description="Épocas de entrenamiento")
    hidden_size: int = Field(..., description="Tamaño de capas ocultas")
    depth: int = Field(..., description="Profundidad del modelo")
    uncertainty_interval: str = Field(..., description="Intervalo de incertidumbre")


class ValidateSmilesResponse(BaseModel):
    """Respuesta de validación de SMILES."""
    valid: bool = Field(..., description="Si el SMILES es válido")
    canonical_smiles: Optional[str] = Field(None, description="SMILES canónico")
    num_atoms: Optional[int] = Field(None, description="Número de átomos")
    molecular_weight: Optional[float] = Field(None, description="Peso molecular")
    error: Optional[str] = Field(None, description="Mensaje de error si es inválido")
    warning: Optional[str] = Field(None, description="Advertencia opcional")


class CompoundNameResponse(BaseModel):
    """Respuesta con el nombre del compuesto desde PubChem."""
    smiles: str = Field(..., description="SMILES consultado")
    name: str = Field(..., description="Nombre del compuesto")
    source: str = Field(..., description="Fuente del nombre (pubchem/cache/unknown)")


# ============================================
# RESPONSE MODELS - Predictions
# ============================================

class PredictResponse(BaseModel):
    """Respuesta de predicción."""
    id: int
    Tm_pred: float = Field(..., description="Punto de fusión predicho en Kelvin")
    smiles: Optional[str] = Field(None, description="Estructura SMILES (si disponible)")


class DataItemResponse(BaseModel):
    """Respuesta de un item de datos (train/test/user)."""
    id: Any = Field(..., description="ID del compuesto (int para dataset, str para user)")
    smiles: str = Field(..., description="Estructura SMILES")
    Tm_pred: float = Field(..., description="Punto de fusión (K) - real para train, predicho para test/user")
    source: str = Field(..., description="Fuente: train (real), test (predicción), user")
    name: Optional[str] = Field(None, description="Nombre del compuesto (de PubChem o definido por usuario)")


class StatsResponse(BaseModel):
    """Estadísticas del dataset."""
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    variance: float
    range: float


class RangeFilter(BaseModel):
    """Filtro de rango aplicado."""
    min_tm: float
    max_tm: float


class RangeResponse(BaseModel):
    """Respuesta de predicciones filtradas por rango."""
    filter: RangeFilter
    count: int
    percentage: float
    predictions: List[PredictResponse]


# ============================================
# RESPONSE MODELS - User Compounds
# ============================================

class CompoundResponse(BaseModel):
    """Respuesta de compuesto de usuario."""
    id: str = Field(..., description="ID del compuesto (ej: USR_001)")
    smiles: str = Field(..., description="Estructura SMILES")
    name: str = Field(..., description="Nombre del compuesto")
    Tm_pred: float = Field(..., description="Punto de fusión predicho (K)")
    Tm_celsius: float = Field(..., description="Punto de fusión en Celsius")
    uncertainty: str = Field("±29 K", description="Intervalo de incertidumbre")
    created_at: str = Field(..., description="Fecha de creación ISO")
    source: str = Field(..., description="Fuente del compuesto")


class CompoundsListResponse(BaseModel):
    """Lista de compuestos de usuario."""
    total: int
    compounds: List[CompoundResponse]


class DeleteResponse(BaseModel):
    """Respuesta de eliminación."""
    message: str
    deleted_id: str


# ============================================
# RESPONSE MODELS - Analytics
# ============================================

class FunctionalGroupStats(BaseModel):
    """Estadísticas de un grupo funcional."""
    name: str
    pattern: str
    count: int
    avg_tm: float
    min_tm: float
    max_tm: float


class FunctionalGroupsResponse(BaseModel):
    """Respuesta de análisis por grupos funcionales."""
    total_molecules: int
    groups: List[FunctionalGroupStats]
    note: Optional[str] = Field(None, description="Nota sobre la fuente de datos")


class DistributionCategory(BaseModel):
    """Categoría de distribución de temperatura."""
    name: str
    description: str
    range_min: float
    range_max: float
    count: int
    percentage: float


class DistributionResponse(BaseModel):
    """Respuesta de distribución por categorías."""
    total: int
    categories: List[DistributionCategory]


class MoleculeSizeGroup(BaseModel):
    """Grupo por tamaño molecular."""
    name: str
    smiles_length_min: int
    smiles_length_max: int
    count: int
    avg_tm: float
    min_tm: float
    max_tm: float


class MoleculeSizeResponse(BaseModel):
    """Respuesta de análisis por tamaño molecular."""
    total_molecules: int
    size_groups: List[MoleculeSizeGroup]
    note: Optional[str] = Field(None, description="Nota sobre la fuente de datos")
