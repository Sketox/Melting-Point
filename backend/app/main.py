from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ml_service import MLService
from .schemas import (
    # Request
    PredictByIdRequest,
    CompoundCreateRequest,
    # Response
    PredictResponse,
    StatsResponse,
    RangeResponse,
    CompoundResponse,
    CompoundsListResponse,
    DeleteResponse,
    FunctionalGroupsResponse,
    DistributionResponse,
    MoleculeSizeResponse,
    RootResponse,
    HealthResponse,
)

app = FastAPI(
    title="Melting Point API",
    description="""
    🧪 API para predecir el punto de fusión (Tm) de compuestos orgánicos.
    
    ## Características
    - Predicciones de punto de fusión en Kelvin
    - Estadísticas del dataset
    - Filtrado por rango de temperatura
    - Análisis por grupos funcionales
    - Distribución por categorías de temperatura
    - Gestión de compuestos de usuarios
    
    ## Competencia
    [Kaggle - Thermophysical Property: Melting Point](https://www.kaggle.com/competitions/melting-point)
    """,
    version="1.0.0",
    contact={
        "name": "Melting Point Team",
        "url": "https://www.kaggle.com/competitions/melting-point",
    },
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev server
        "http://127.0.0.1:3000",      # Alternativa
        "http://localhost:5173",       # Vite
        "*",                           # Permitir todo (solo para desarrollo)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service: MLService | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Carga el modelo y el CSV procesado al iniciar la aplicación."""
    global ml_service
    ml_service = MLService()


# ============================================
# 1. ROOT - Info del API
# ============================================
@app.get("/", response_model=RootResponse, tags=["Info"])
def root():
    """
    🏠 Endpoint raíz con información del API.
    
    Devuelve información general sobre el API y cantidad de endpoints disponibles.
    """
    return RootResponse(
        message="Melting Point API",
        status="running",
        version="1.0.0",
        docs="/docs",
        endpoints_count=12
    )


# ============================================
# 2. HEALTH - Health Check
# ============================================
@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """
    💚 Health check del servidor.
    
    Verifica que el servidor está corriendo y el modelo está cargado.
    """
    return HealthResponse(
        status="ok",
        model_loaded=ml_service is not None,
        dataset_size=ml_service.get_dataset_size() if ml_service else 0
    )


# ============================================
# 3. PREDICT BY ID - Predicción individual
# ============================================
@app.post("/predict-by-id", response_model=PredictResponse, tags=["Predictions"])
def predict_by_id(request: PredictByIdRequest):
    """
    🔮 Predicción por ID.
    
    Dado un ID presente en el dataset de test, devuelve la predicción de Tm.
    
    - **id**: ID de la molécula (1-667)
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    try:
        pred = ml_service.predict_by_id(request.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(id=request.id, Tm_pred=round(pred, 2))


# ============================================
# 4. PREDICT ALL - Todas las predicciones
# ============================================
@app.get("/predict-all", response_model=List[PredictResponse], tags=["Predictions"])
def predict_all():
    """
    📊 Todas las predicciones.
    
    Devuelve las predicciones de Tm para TODOS los IDs del dataset de test.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    results = ml_service.predict_all()

    return [
        PredictResponse(id=sample_id, Tm_pred=round(pred, 2)) 
        for sample_id, pred in results
    ]


# ============================================
# 5. STATS - Estadísticas del dataset
# ============================================
@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
def get_stats():
    """
    📈 Estadísticas del dataset.
    
    Devuelve estadísticas calculadas de todas las predicciones:
    - Count, Mean, Std, Min, Max
    - Median, Q25, Q75
    - Variance, Range
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    stats = ml_service.get_stats()
    
    return StatsResponse(
        count=stats["count"],
        mean=round(stats["mean"], 2),
        std=round(stats["std"], 2),
        min=round(stats["min"], 2),
        max=round(stats["max"], 2),
        median=round(stats["median"], 2),
        q25=round(stats["q25"], 2),
        q75=round(stats["q75"], 2),
        variance=round(stats["variance"], 2),
        range=round(stats["range"], 2)
    )


# ============================================
# 6. PREDICTIONS RANGE - Filtrar por rango
# ============================================
@app.get("/predictions/range", response_model=RangeResponse, tags=["Analytics"])
def get_predictions_range(
    min_tm: float = Query(..., description="Temperatura mínima en Kelvin", ge=0),
    max_tm: float = Query(..., description="Temperatura máxima en Kelvin", le=1000)
):
    """
    🎚️ Filtrar predicciones por rango de temperatura.
    
    Devuelve todas las moléculas cuyo Tm predicho está dentro del rango especificado.
    
    - **min_tm**: Límite inferior del rango (Kelvin)
    - **max_tm**: Límite superior del rango (Kelvin)
    
    Ejemplo: `/predictions/range?min_tm=200&max_tm=350`
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    if min_tm > max_tm:
        raise HTTPException(
            status_code=400, 
            detail="min_tm debe ser menor o igual que max_tm"
        )

    result = ml_service.get_predictions_in_range(min_tm, max_tm)
    
    return RangeResponse(
        filter=result["filter"],
        count=result["count"],
        percentage=result["percentage"],
        predictions=[
            PredictResponse(id=p["id"], Tm_pred=round(p["Tm_pred"], 2))
            for p in result["predictions"]
        ]
    )


# ============================================
# 7. POST COMPOUNDS - Agregar compuesto
# ============================================
@app.post("/compounds", response_model=CompoundResponse, tags=["User Compounds"])
def create_compound(request: CompoundCreateRequest):
    """
    ➕ Agregar un nuevo compuesto.
    
    Permite a los usuarios agregar sus propios compuestos para obtener una predicción.
    Los compuestos se guardan en un CSV separado del dataset oficial.
    
    - **smiles**: Estructura SMILES de la molécula
    - **name**: Nombre del compuesto
    
    Ejemplo:
    ```json
    {
        "smiles": "CCO",
        "name": "Ethanol"
    }
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    compound = ml_service.add_user_compound(request.smiles, request.name)
    
    return CompoundResponse(
        id=compound["id"],
        smiles=compound["smiles"],
        name=compound["name"],
        Tm_pred=compound["Tm_pred"],
        Tm_celsius=compound["Tm_celsius"],
        created_at=compound["created_at"],
        source=compound["source"]
    )


# ============================================
# 8. GET COMPOUNDS - Listar compuestos
# ============================================
@app.get("/compounds", response_model=CompoundsListResponse, tags=["User Compounds"])
def get_compounds():
    """
    📋 Listar compuestos de usuarios.
    
    Devuelve todos los compuestos agregados por usuarios.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_user_compounds()
    
    return CompoundsListResponse(
        total=result["total"],
        compounds=[
            CompoundResponse(
                id=c["id"],
                smiles=c["smiles"],
                name=c["name"],
                Tm_pred=c["Tm_pred"],
                Tm_celsius=c["Tm_celsius"],
                created_at=c["created_at"],
                source=c["source"]
            )
            for c in result["compounds"]
        ]
    )


# ============================================
# 9. DELETE COMPOUNDS - Eliminar compuesto
# ============================================
@app.delete("/compounds/{compound_id}", response_model=DeleteResponse, tags=["User Compounds"])
def delete_compound(compound_id: str):
    """
    🗑️ Eliminar un compuesto de usuario.
    
    Elimina un compuesto de la lista de compuestos de usuarios.
    
    - **compound_id**: ID del compuesto (ej: USR_001)
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    success = ml_service.delete_user_compound(compound_id)
    
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Compuesto {compound_id} no encontrado"
        )
    
    return DeleteResponse(
        message="Compuesto eliminado exitosamente",
        deleted_id=compound_id
    )


# ============================================
# 10. FUNCTIONAL GROUPS - Análisis por grupos
# ============================================
@app.get("/predictions/by-functional-group", response_model=FunctionalGroupsResponse, tags=["Analytics"])
def get_by_functional_group():
    """
    🧬 Análisis por grupos funcionales.
    
    Agrupa las moléculas por tipo de grupo funcional detectado y muestra
    estadísticas de Tm para cada grupo.
    
    Grupos incluidos:
    - Alcohols (OH)
    - Carboxylic Acids (COOH)
    - Amines (NH2)
    - Halogenated (F, Cl, Br, I)
    - Aromatic Rings
    - Hydrocarbons
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_functional_group()
    
    return FunctionalGroupsResponse(
        total_molecules=result["total_molecules"],
        groups=result["groups"]
    )


# ============================================
# 11. DISTRIBUTION - Distribución por categorías
# ============================================
@app.get("/predictions/distribution", response_model=DistributionResponse, tags=["Analytics"])
def get_distribution():
    """
    🥧 Distribución por categorías de temperatura.
    
    Clasifica las moléculas en categorías según su punto de fusión:
    
    - **Muy bajo (<150K)**: Gases a temperatura ambiente
    - **Bajo (150-250K)**: Líquidos volátiles
    - **Medio (250-350K)**: Líquidos/Sólidos a temp. ambiente
    - **Alto (350-450K)**: Sólidos estables
    - **Muy alto (>450K)**: Sólidos de alto punto de fusión
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_distribution()
    
    return DistributionResponse(
        total=result["total"],
        categories=result["categories"]
    )


# ============================================
# 12. MOLECULE SIZE - Análisis por tamaño
# ============================================
@app.get("/predictions/by-molecule-size", response_model=MoleculeSizeResponse, tags=["Analytics"])
def get_by_molecule_size():
    """
    📏 Análisis por tamaño molecular.
    
    Agrupa las moléculas por tamaño (estimado por longitud del SMILES):
    
    - **Pequeñas (1-10 átomos)**
    - **Medianas (11-25 átomos)**
    - **Grandes (26-50 átomos)**
    - **Muy grandes (>50 átomos)**
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_molecule_size()
    
    return MoleculeSizeResponse(
        total_molecules=result["total_molecules"],
        size_groups=result["size_groups"]
    )