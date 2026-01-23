from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ============================================
# REQUEST SCHEMAS
# ============================================

class PredictByIdRequest(BaseModel):
    """Request para predicción por ID."""
    id: int = Field(..., description="ID de la molécula en el dataset", ge=1)


class RangeFilterRequest(BaseModel):
    """Request para filtrar por rango de temperatura."""
    min_tm: float = Field(..., description="Temperatura mínima en Kelvin")
    max_tm: float = Field(..., description="Temperatura máxima en Kelvin")


class CompoundCreateRequest(BaseModel):
    """Request para crear un nuevo compuesto de usuario."""
    smiles: str = Field(..., description="Estructura SMILES de la molécula", min_length=1)
    name: str = Field(..., description="Nombre del compuesto", min_length=1)


# ============================================
# RESPONSE SCHEMAS
# ============================================

class PredictResponse(BaseModel):
    """Response básica de predicción."""
    id: int
    Tm_pred: float


class PredictResponseWithCelsius(BaseModel):
    """Response de predicción con temperatura en Celsius."""
    id: int
    Tm_pred: float = Field(..., description="Temperatura en Kelvin")
    Tm_celsius: float = Field(..., description="Temperatura en Celsius")


# --- Stats ---
class StatsResponse(BaseModel):
    """Estadísticas del dataset."""
    count: int = Field(..., description="Número total de muestras")
    mean: float = Field(..., description="Media de Tm")
    std: float = Field(..., description="Desviación estándar")
    min: float = Field(..., description="Valor mínimo")
    max: float = Field(..., description="Valor máximo")
    median: float = Field(..., description="Mediana")
    q25: float = Field(..., description="Percentil 25")
    q75: float = Field(..., description="Percentil 75")
    variance: float = Field(..., description="Varianza")
    range: float = Field(..., description="Rango (max - min)")


# --- Range Filter ---
class RangeFilter(BaseModel):
    """Filtro aplicado."""
    min_tm: float
    max_tm: float


class RangeResponse(BaseModel):
    """Response de filtro por rango."""
    filter: RangeFilter
    count: int = Field(..., description="Cantidad de moléculas en el rango")
    percentage: float = Field(..., description="Porcentaje del total")
    predictions: List[PredictResponse]


# --- User Compounds ---
class CompoundResponse(BaseModel):
    """Response de un compuesto de usuario."""
    id: str = Field(..., description="ID único del compuesto (USR_XXX)")
    smiles: str = Field(..., description="Estructura SMILES")
    name: str = Field(..., description="Nombre del compuesto")
    Tm_pred: float = Field(..., description="Temperatura predicha en Kelvin")
    Tm_celsius: float = Field(..., description="Temperatura predicha en Celsius")
    created_at: str = Field(..., description="Fecha de creación")
    source: str = Field(default="user_submitted", description="Origen del dato")


class CompoundsListResponse(BaseModel):
    """Response lista de compuestos de usuarios."""
    total: int
    compounds: List[CompoundResponse]


class DeleteResponse(BaseModel):
    """Response de eliminación."""
    message: str
    deleted_id: str


# --- Functional Groups ---
class FunctionalGroupStats(BaseModel):
    """Estadísticas de un grupo funcional."""
    name: str = Field(..., description="Nombre del grupo funcional")
    pattern: str = Field(..., description="Patrón SMILES del grupo")
    count: int = Field(..., description="Cantidad de moléculas")
    avg_tm: float = Field(..., description="Tm promedio")
    min_tm: float = Field(..., description="Tm mínimo")
    max_tm: float = Field(..., description="Tm máximo")


class FunctionalGroupsResponse(BaseModel):
    """Response de análisis por grupos funcionales."""
    total_molecules: int
    groups: List[FunctionalGroupStats]


# --- Distribution Categories ---
class DistributionCategory(BaseModel):
    """Categoría de distribución de temperatura."""
    name: str = Field(..., description="Nombre de la categoría")
    description: str = Field(..., description="Descripción")
    range_min: float = Field(..., description="Límite inferior del rango")
    range_max: float = Field(..., description="Límite superior del rango")
    count: int = Field(..., description="Cantidad de moléculas")
    percentage: float = Field(..., description="Porcentaje del total")


class DistributionResponse(BaseModel):
    """Response de distribución por categorías."""
    total: int
    categories: List[DistributionCategory]


# --- Molecule Size Analysis ---
class MoleculeSizeGroup(BaseModel):
    """Grupo por tamaño de molécula."""
    name: str = Field(..., description="Nombre del grupo")
    smiles_length_min: int = Field(..., description="Longitud mínima de SMILES")
    smiles_length_max: int = Field(..., description="Longitud máxima de SMILES")
    count: int = Field(..., description="Cantidad de moléculas")
    avg_tm: float = Field(..., description="Tm promedio")
    min_tm: float = Field(..., description="Tm mínimo")
    max_tm: float = Field(..., description="Tm máximo")


class MoleculeSizeResponse(BaseModel):
    """Response de análisis por tamaño molecular."""
    total_molecules: int
    size_groups: List[MoleculeSizeGroup]


# --- Root/Health ---
class RootResponse(BaseModel):
    """Response del endpoint raíz."""
    message: str
    status: str
    version: str
    docs: str
    endpoints_count: int


class HealthResponse(BaseModel):
    """Response del health check."""
    status: str
    model_loaded: bool
    dataset_size: int