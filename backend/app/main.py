"""
main.py - API FastAPI para predicción de puntos de fusión.

ACTUALIZADO:
- Endpoint de validación de SMILES
- Endpoint de información del modelo
- Manejo mejorado de errores para SMILES inválidos
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .ml_service import MLService, SMILESValidationError
from .schemas import (
    # Request
    PredictByIdRequest,
    CompoundCreateRequest,
    ValidateSmilesRequest,
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
    ValidateSmilesResponse,
    ModelInfoResponse,
)

# Importar rutas de Supabase
from .supabase import supabase_router

# Importar rutas de autenticación y MongoDB
from .auth import (
    auth_router,
    user_predictions_router,
    get_async_database,
    create_indexes,
    test_mongodb_connection,
    close_mongodb_connection
)

app = FastAPI(
    title="Melting Point API",
    description="""
    API para predecir el punto de fusión (Tm) de compuestos orgánicos.
    
    ## Características
    - Predicciones de punto de fusión en Kelvin
    - **Validación de SMILES** con RDKit
    - Estadísticas del dataset
    - Filtrado por rango de temperatura
    - Análisis por grupos funcionales
    - Distribución por categorías de temperatura
    - Gestión de compuestos de usuarios
    - **Información de incertidumbre del modelo** (MAE ±29 K)
    
    ## Competencia
    [Kaggle - Thermophysical Property: Melting Point](https://www.kaggle.com/competitions/melting-point)
    """,
    version="2.0.0",
    contact={
        "name": "Melting Point Team",
        "url": "https://www.kaggle.com/competitions/melting-point",
    },
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service: MLService | None = None

# Incluir rutas de Supabase
app.include_router(supabase_router)

# Incluir rutas de autenticación y predicciones de usuario
app.include_router(auth_router)
app.include_router(user_predictions_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Carga el modelo, CSV y conecta a MongoDB al iniciar la aplicación."""
    global ml_service
    ml_service = MLService()
    
    # Conectar a MongoDB y crear índices
    try:
        db = get_async_database()
        await create_indexes()
        connection_ok = await test_mongodb_connection()
        if connection_ok:
            print("✓ MongoDB conectado y listo")
        else:
            print("⚠️ MongoDB no disponible - funcionalidades de usuario deshabilitadas")
    except Exception as e:
        print(f"⚠️ Error al conectar MongoDB: {e}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cierra las conexiones al apagar la aplicación."""
    await close_mongodb_connection()


# ============================================
# 1. ROOT - Info del API
# ============================================
@app.get("/", response_model=RootResponse, tags=["Info"])
def root():
    """
    🏠 Endpoint raíz con información del API.
    """
    return RootResponse(
        message="Melting Point API - Now with Supabase & Authentication!",
        status="running",
        version="2.1.0",
        docs="/docs",
        endpoints_count=21
    )


# ============================================
# 2. HEALTH - Health Check
# ============================================
@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """
     Health check del servidor.
    """
    return HealthResponse(
        status="ok",
        model_loaded=ml_service is not None,
        dataset_size=ml_service.get_dataset_size() if ml_service else 0
    )


# ============================================
# 3. MODEL INFO - Información del modelo
# ============================================
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
def get_model_info():
    """
    🧠 Información del modelo ML.
    
    Devuelve detalles del modelo incluyendo métricas de rendimiento
    y el intervalo de incertidumbre de las predicciones.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")
    
    info = ml_service.get_model_info()
    return ModelInfoResponse(**info)


# ============================================
# 4. VALIDATE SMILES - Validar estructura SMILES
# ============================================
@app.post("/validate-smiles", response_model=ValidateSmilesResponse, tags=["Validation"])
def validate_smiles(request: ValidateSmilesRequest):
    """
    ✅ Valida una estructura SMILES.
    
    Verifica que el SMILES sea válido usando RDKit y devuelve
    información sobre la molécula.
    
    - **smiles**: String SMILES a validar
    
    Ejemplo:
    ```json
    {"smiles": "CCO"}  // Etanol
    ```
    
    Respuesta exitosa:
    ```json
    {
        "valid": true,
        "canonical_smiles": "CCO",
        "num_atoms": 3,
        "molecular_weight": 46.07,
        "error": null
    }
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")
    
    result = ml_service.validate_smiles(request.smiles)
    return ValidateSmilesResponse(**result)


# ============================================
# 5. PREDICT BY ID - Predicción individual
# ============================================
@app.post("/predict-by-id", response_model=PredictResponse, tags=["Predictions"])
def predict_by_id(request: PredictByIdRequest):
    """
    🔮 Predicción por ID.
    
    Dado un ID presente en el dataset de test, devuelve la predicción de Tm.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    try:
        pred = ml_service.predict_by_id(request.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(id=request.id, Tm_pred=round(pred, 2))


# ============================================
# 6. PREDICT ALL - Todas las predicciones
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
# 7. STATS - Estadísticas del dataset
# ============================================
@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
def get_stats():
    """
    📈 Estadísticas del dataset.
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
# 8. PREDICTIONS RANGE - Filtrar por rango
# ============================================
@app.get("/predictions/range", response_model=RangeResponse, tags=["Analytics"])
def get_predictions_range(
    min_tm: float = Query(..., description="Temperatura mínima en Kelvin", ge=0),
    max_tm: float = Query(..., description="Temperatura máxima en Kelvin", le=1000)
):
    """
    🎚️ Filtrar predicciones por rango de temperatura.
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
# 9. POST COMPOUNDS - Agregar compuesto
# ============================================
@app.post("/compounds", response_model=CompoundResponse, tags=["User Compounds"])
def create_compound(request: CompoundCreateRequest):
    """
    ➕ Agregar un nuevo compuesto.
    
    Valida el SMILES antes de agregar. Si el SMILES es inválido,
    devuelve un error 400 con detalles.
    
    - **smiles**: Estructura SMILES de la molécula (debe ser válido)
    - **name**: Nombre del compuesto
    
    Ejemplo válido:
    ```json
    {"smiles": "CCO", "name": "Ethanol"}
    ```
    
    Ejemplo inválido:
    ```json
    {"smiles": "xyz123", "name": "Invalid"}  // Error 400
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    try:
        compound = ml_service.add_user_compound(request.smiles, request.name)
    except SMILESValidationError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"SMILES inválido: {str(e)}"
        )
    except Exception as e:
        # Log del error para debugging
        import traceback
        print(f"Error creating compound: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno al crear compuesto: {str(e)}"
        )
    
    return CompoundResponse(
        id=compound["id"],
        smiles=compound["smiles"],
        name=compound["name"],
        Tm_pred=compound["Tm_pred"],
        Tm_celsius=compound["Tm_celsius"],
        uncertainty=compound.get("uncertainty", "±29 K"),
        created_at=compound["created_at"],
        source=compound["source"]
    )


# ============================================
# 10. GET COMPOUNDS - Listar compuestos
# ============================================
@app.get("/compounds", response_model=CompoundsListResponse, tags=["User Compounds"])
def get_compounds():
    """
    📋 Listar compuestos de usuarios.
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
                uncertainty=c.get("uncertainty", "±29 K"),
                created_at=c["created_at"],
                source=c["source"]
            )
            for c in result["compounds"]
        ]
    )


# ============================================
# 11. DELETE COMPOUNDS - Eliminar compuesto
# ============================================
@app.delete("/compounds/{compound_id}", response_model=DeleteResponse, tags=["User Compounds"])
def delete_compound(compound_id: str):
    """
    🗑️ Eliminar un compuesto de usuario.
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
# 12. FUNCTIONAL GROUPS - Análisis por grupos
# ============================================
@app.get("/predictions/by-functional-group", response_model=FunctionalGroupsResponse, tags=["Analytics"])
def get_by_functional_group():
    """
    🧬 Análisis por grupos funcionales.
    
    Agrupa las moléculas por tipo de grupo funcional detectado usando
    patrones SMARTS. Si los SMILES no están disponibles, usa estimación.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_functional_group()
    
    return FunctionalGroupsResponse(
        total_molecules=result["total_molecules"],
        groups=result["groups"]
    )


# ============================================
# 13. DISTRIBUTION - Distribución por categorías
# ============================================
@app.get("/predictions/distribution", response_model=DistributionResponse, tags=["Analytics"])
def get_distribution():
    """
    📊 Distribución por categorías de temperatura.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_distribution()
    
    return DistributionResponse(
        total=result["total"],
        categories=result["categories"]
    )


# ============================================
# 14. MOLECULE SIZE - Análisis por tamaño
# ============================================
@app.get("/predictions/by-molecule-size", response_model=MoleculeSizeResponse, tags=["Analytics"])
def get_by_molecule_size():
    """
    📏 Análisis por tamaño molecular.
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_molecule_size()
    
    return MoleculeSizeResponse(
        total_molecules=result["total_molecules"],
        size_groups=result["size_groups"]
    )