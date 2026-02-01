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

# Metadata para tags de la documentación
tags_metadata = [
    {
        "name": "🏠 System",
        "description": "Información del sistema, salud y estado general del API",
    },
    {
        "name": "🔐 Authentication",
        "description": "Autenticación de usuarios con JWT, registro, login y gestión de perfil",
    },
    {
        "name": "💾 User Predictions",
        "description": "Gestión de predicciones guardadas por usuario autenticado",
    },
    {
        "name": "🔬 Predictions",
        "description": "Predicciones de punto de fusión usando modelos ML",
    },
    {
        "name": "✅ Validation",
        "description": "Validación de estructuras químicas SMILES",
    },
    {
        "name": "📊 Analytics",
        "description": "Estadísticas, distribuciones y análisis del dataset",
    },
    {
        "name": "🧪 Compounds",
        "description": "Gestión de compuestos del dataset",
    },
    {
        "name": "🗄️ Supabase",
        "description": "Endpoints opcionales de Supabase (requiere configuración)",
    },
]

app = FastAPI(
    title="🔥 Melting Point Prediction API",
    description="""
    ## 🎯 Descripción
    API completa para predecir el punto de fusión (Tm) de compuestos orgánicos usando Machine Learning.
    
    ## ✨ Características Principales
    
    ### 🤖 Machine Learning
    - **Modelo**: ChemProp Ensemble (5 checkpoints)
    - **Precisión**: MAE ±29 K
    - **Dataset**: 666 compuestos pre-calculados
    - **Validación**: RDKit para estructuras SMILES
    
    ### 🔐 Autenticación
    - Sistema completo de usuarios con MongoDB
    - JWT tokens seguros
    - Gestión de predicciones por usuario
    
    ### 📊 Analytics
    - Estadísticas del dataset
    - Filtrado por rango de temperatura
    - Análisis de grupos funcionales
    - Distribución por categorías
    
    ### 🗄️ Bases de Datos
    - **MongoDB Atlas**: Autenticación y datos de usuario
    - **Supabase** (opcional): Datos adicionales
    
    ## 🚀 Inicio Rápido
    
    1. **Health Check**: `GET /health`
    2. **Registrarse**: `POST /auth/register`
    3. **Login**: `POST /auth/login`
    4. **Predecir**: `POST /predict-by-id?id=123`
    
    ## 📖 Documentación
    
    - **Swagger UI**: `/docs` (esta página)
    - **ReDoc**: `/redoc`
    - **OpenAPI Schema**: `/openapi.json`
    
    ## 🏆 Competencia
    [Kaggle - Thermophysical Property: Melting Point](https://www.kaggle.com/competitions/playground-series-s5e6)
    
    ## 👥 Equipo
    Desarrollado para Kaggle Playground Series S5E6
    """,
    version="2.1.0",
    contact={
        "name": "Melting Point Team",
        "url": "https://github.com/Sketox/Melting-Point",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=tags_metadata,
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
# 1. SYSTEM - Info & Health
# ============================================
@app.get(
    "/",
    response_model=RootResponse,
    tags=["🏠 System"],
    summary="🏠 Información del API",
    description="Endpoint raíz que proporciona información general sobre el API y sus capacidades."
)
def root():
    """
    Retorna información básica del API.
    
    **Returns:**
    - Mensaje de bienvenida
    - Estado del servicio
    - Versión actual
    - Link a documentación
    - Número total de endpoints
    """
    return RootResponse(
        message="Melting Point API - Predicciones ML con Autenticación MongoDB",
        status="running",
        version="2.1.0",
        docs="/docs",
        endpoints_count=25
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["🏠 System"],
    summary="💊 Health Check",
    description="Verifica el estado de salud del servidor y sus componentes."
)
def health():
    """
    Health check del servidor.
    
    **Verifica:**
    - Estado general del API
    - Carga del modelo ML
    - Tamaño del dataset disponible
    
    **Returns:**
    - `status`: Estado del servidor (ok/error)
    - `model_loaded`: Si el modelo ML está cargado
    - `dataset_size`: Número de compuestos disponibles
    """
    return HealthResponse(
        status="ok",
        model_loaded=ml_service is not None,
        dataset_size=ml_service.get_dataset_size() if ml_service else 0
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["🏠 System"],
    summary="🧠 Información del Modelo",
    description="Obtiene detalles técnicos del modelo ML y sus métricas de rendimiento."
)
def get_model_info():
    """
    Información detallada del modelo de Machine Learning.
    
    **Incluye:**
    - Tipo de modelo (ChemProp Ensemble)
    - Métricas de precisión (MAE, RMSE)
    - Número de checkpoints
    - Intervalo de confianza
    - Tamaño del dataset de entrenamiento
    
    **Returns:**
    - `model_type`: Tipo de modelo usado
    - `mae`: Error absoluto medio
    - `uncertainty_interval`: Rango de incertidumbre (±K)
    - `num_checkpoints`: Número de modelos en el ensemble
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")
    
    info = ml_service.get_model_info()
    return ModelInfoResponse(**info)


# ============================================
# 2. VALIDATION - Validación de Estructuras
# ============================================
@app.post(
    "/validate-smiles",
    response_model=ValidateSmilesResponse,
    tags=["✅ Validation"],
    summary="✅ Validar SMILES",
    description="Valida una estructura química en formato SMILES usando RDKit."
)
def validate_smiles(request: ValidateSmilesRequest):
    """
    Valida una estructura SMILES y retorna información de la molécula.
    
    **Verificaciones:**
    - Sintaxis correcta del SMILES
    - Estructura químicamente válida
    - Conversión a SMILES canónico
    - Cálculo de propiedades básicas
    
    **Ejemplo de request:**
    ```json
    {
        "smiles": "CCO"
    }
    ```
    
    **Ejemplo de respuesta exitosa:**
    ```json
    {
        "valid": true,
        "canonical_smiles": "CCO",
        "num_atoms": 9,
        "molecular_weight": 46.07,
        "error": null
    }
    ```
    
    **Ejemplo de respuesta con error:**
    ```json
    {
        "valid": false,
        "canonical_smiles": null,
        "num_atoms": 0,
        "molecular_weight": 0.0,
        "error": "Invalid SMILES string"
    }
    ```
    
    **Moléculas de ejemplo:**
    - Etanol: `CCO`
    - Benceno: `c1ccccc1`
    - Aspirina: `CC(=O)Oc1ccccc1C(=O)O`
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")
    
    result = ml_service.validate_smiles(request.smiles)
    return ValidateSmilesResponse(**result)


# ============================================
# 3. PREDICTIONS - Predicciones de Punto de Fusión
# ============================================
@app.post(
    "/predict-by-id",
    response_model=PredictResponse,
    tags=["🔬 Predictions"],
    summary="🔮 Predicción por ID",
    description="Predice el punto de fusión de un compuesto usando su ID del dataset."
)
def predict_by_id(request: PredictByIdRequest):
    """
    Predice el punto de fusión (Tm) usando el ID del compuesto.
    
    **Cómo funciona:**
    1. Busca el compuesto en el dataset por ID
    2. Obtiene la predicción pre-calculada del modelo ChemProp
    3. Retorna el valor en Kelvin con 2 decimales
    
    **Parámetros:**
    - `id`: ID del compuesto (int, ejemplo: 123)
    
    **Returns:**
    - `id`: ID del compuesto consultado
    - `Tm_pred`: Temperatura de fusión predicha (K)
    
    **Ejemplo de request:**
    ```json
    {
        "id": 123
    }
    ```
    
    **Ejemplo de respuesta:**
    ```json
    {
        "id": 123,
        "Tm_pred": 350.25
    }
    ```
    
    **Nota:** El modelo tiene una incertidumbre de ±29 K (MAE).
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    try:
        pred = ml_service.predict_by_id(request.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(id=request.id, Tm_pred=round(pred, 2))


@app.get(
    "/predict-all",
    response_model=List[PredictResponse],
    tags=["🔬 Predictions"],
    summary="📊 Todas las Predicciones",
    description="Obtiene las predicciones de todos los compuestos del dataset."
)
def predict_all():
    """
    Retorna todas las predicciones de Tm del dataset.
    
    **Cómo funciona:**
    - Retorna las 666 predicciones pre-calculadas
    - Cada predicción incluye ID y Tm predicho
    - Ordenadas por ID ascendente
    
    **Returns:**
    Lista de objetos con:
    - `id`: ID del compuesto
    - `Tm_pred`: Temperatura de fusión predicha (K)
    
    **Ejemplo de respuesta:**
    ```json
    [
        {"id": 0, "Tm_pred": 298.15},
        {"id": 1, "Tm_pred": 350.42},
        ...
    ]
    ```
    
    **Total de predicciones:** 666 compuestos
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    results = ml_service.predict_all()

    return [
        PredictResponse(id=sample_id, Tm_pred=round(pred, 2)) 
        for sample_id, pred in results
    ]


# ============================================
# 4. ANALYTICS - Estadísticas y Análisis
# ============================================
@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["📊 Analytics"],
    summary="📈 Estadísticas del Dataset",
    description="Obtiene estadísticas descriptivas completas del dataset de predicciones."
)
def get_stats():
    """
    Estadísticas descriptivas de todas las predicciones.
    
    **Métricas incluidas:**
    - `count`: Número total de predicciones
    - `mean`: Media de temperaturas (K)
    - `std`: Desviación estándar
    - `min`: Temperatura mínima
    - `max`: Temperatura máxima
    - `median`: Mediana
    - `q25`: Primer cuartil (25%)
    - `q75`: Tercer cuartil (75%)
    - `variance`: Varianza
    - `range`: Rango (max - min)
    
    **Ejemplo de respuesta:**
    ```json
    {
        "count": 666,
        "mean": 350.25,
        "std": 45.32,
        "min": 250.00,
        "max": 450.00,
        "median": 345.50,
        "q25": 320.00,
        "q75": 380.00,
        "variance": 2053.90,
        "range": 200.00
    }
    ```
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
# 5. RANGE FILTER - Filtrado por Rango
# ============================================
@app.get(
    "/predictions/range",
    response_model=RangeResponse,
    tags=["📊 Analytics"],
    summary="🎚️ Filtrar por Rango de Temperatura",
    description="Filtra predicciones dentro de un rango específico de temperaturas."
)
def get_predictions_range(
    min_tm: float = Query(..., description="Temperatura mínima en Kelvin", ge=0, example=300),
    max_tm: float = Query(..., description="Temperatura máxima en Kelvin", le=1000, example=400)
):
    """
    Filtra predicciones por rango de temperatura.
    
    **Parámetros:**
    - `min_tm`: Temperatura mínima (K) - debe ser ≥ 0
    - `max_tm`: Temperatura máxima (K) - debe ser ≤ 1000
    
    **Returns:**
    - `filter`: Descripción del rango aplicado
    - `count`: Número de predicciones en el rango
    - `percentage`: Porcentaje del total
    - `predictions`: Lista de predicciones filtradas
    
    **Ejemplo de uso:**
    ```
    GET /predictions/range?min_tm=300&max_tm=400
    ```
    
    **Ejemplo de respuesta:**
    ```json
    {
        "filter": "300.00 K - 400.00 K",
        "count": 150,
        "percentage": 22.52,
        "predictions": [
            {"id": 10, "Tm_pred": 305.23},
            {"id": 15, "Tm_pred": 398.76}
        ]
    }
    ```
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
# 6. COMPOUNDS - Gestión de Compuestos
# ============================================
@app.post(
    "/compounds",
    response_model=CompoundResponse,
    tags=["🧪 Compounds"],
    summary="➕ Agregar Compuesto",
    description="Agrega un nuevo compuesto al dataset y predice su punto de fusión.",
    status_code=201
)
def create_compound(request: CompoundCreateRequest):
    """
    Agrega un nuevo compuesto validando su estructura SMILES.
    
    **Proceso:**
    1. Valida el SMILES con RDKit
    2. Genera predicción de Tm usando ChemProp
    3. Guarda el compuesto en CSV local
    4. Retorna información completa
    
    **Parámetros:**
    - `smiles`: Estructura SMILES válida (string)
    - `name`: Nombre del compuesto (string, opcional)
    
    **Ejemplos válidos:**
    ```json
    {"smiles": "CCO", "name": "Etanol"}
    {"smiles": "c1ccccc1", "name": "Benceno"}
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirina"}
    ```
    
    **Error 400 (SMILES inválido):**
    ```json
    {
        "detail": "SMILES inválido: Invalid SMILES syntax"
    }
    ```
    
    **Respuesta exitosa (201):**
    ```json
    {
        "id": 667,
        "smiles": "CCO",
        "name": "Etanol",
        "Tm_pred": 159.05,
        "Tm_celsius": -114.10,
        "uncertainty": "±29 K",
        "created_at": "2026-02-01T10:30:00",
        "source": "user"
    }
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


@app.get(
    "/compounds",
    response_model=CompoundsListResponse,
    tags=["🧪 Compounds"],
    summary="📋 Listar Compuestos",
    description="Obtiene la lista completa de compuestos agregados por usuarios."
)
def get_compounds():
    """
    Lista todos los compuestos agregados por usuarios.
    
    **Returns:**
    - `total`: Número total de compuestos
    - `compounds`: Lista de compuestos con sus predicciones
    
    **Cada compuesto incluye:**
    - ID único
    - SMILES canónico
    - Nombre
    - Predicción de Tm (K)
    - Temperatura en Celsius
    - Intervalo de incertidumbre
    - Fecha de creación
    - Fuente (user/dataset)
    
    **Ejemplo de respuesta:**
    ```json
    {
        "total": 10,
        "compounds": [
            {
                "id": 667,
                "smiles": "CCO",
                "name": "Etanol",
                "Tm_pred": 159.05,
                "Tm_celsius": -114.10,
                "uncertainty": "±29 K",
                "created_at": "2026-02-01T10:30:00",
                "source": "user"
            }
        ]
    }
    ```
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


@app.delete(
    "/compounds/{compound_id}",
    response_model=DeleteResponse,
    tags=["🧪 Compounds"],
    summary="🗑️ Eliminar Compuesto",
    description="Elimina un compuesto agregado por el usuario."
)
def delete_compound(compound_id: str):
    """
    Elimina un compuesto del dataset local.
    
    **Parámetros:**
    - `compound_id`: ID del compuesto a eliminar
    
    **Response (200 OK):**
    ```json
    {
        "message": "Compuesto eliminado exitosamente",
        "deleted_id": "667"
    }
    ```
    
    **Error 404:**
    ```json
    {
        "detail": "Compuesto 999 no encontrado"
    }
    ```
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
# 7. FUNCTIONAL GROUPS - Análisis Químico
# ============================================
@app.get(
    "/predictions/by-functional-group",
    response_model=FunctionalGroupsResponse,
    tags=["📊 Analytics"],
    summary="🧬 Análisis por Grupos Funcionales",
    description="Agrupa moléculas según sus grupos funcionales químicos."
)
def get_by_functional_group():
    """
    Análisis químico por grupos funcionales.
    
    **Detecta grupos como:**
    - Alcoholes (OH)
    - Cetonas (C=O)
    - Ácidos carboxílicos (COOH)
    - Aminas (NH2)
    - Aromáticos (benceno)
    - Etc.
    
    **Método:**
    Usa patrones SMARTS para identificar subestructuras químicas.
    
    **Response:**
    ```json
    {
        "total_molecules": 666,
        "groups": {
            "alcohols": {"count": 45, "avg_Tm": 320.5},
            "ketones": {"count": 32, "avg_Tm": 305.2},
            "aromatics": {"count": 150, "avg_Tm": 350.8}
        }
    }
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_functional_group()
    
    return FunctionalGroupsResponse(
        total_molecules=result["total_molecules"],
        groups=result["groups"]
    )


@app.get(
    "/predictions/distribution",
    response_model=DistributionResponse,
    tags=["📊 Analytics"],
    summary="📊 Distribución por Categorías",
    description="Distribuye predicciones en categorías de temperatura."
)
def get_distribution():
    """
    Distribución de predicciones por rangos de temperatura.
    
    **Categorías:**
    - **Muy baja** (< 200 K): Sólidos muy fríos
    - **Baja** (200-273 K): Por debajo de 0°C
    - **Media** (273-373 K): Temperatura ambiente
    - **Alta** (373-500 K): Temperaturas elevadas
    - **Muy alta** (> 500 K): Sólidos muy estables
    
    **Response:**
    ```json
    {
        "total": 666,
        "categories": [
            {
                "name": "Muy baja (< 200 K)",
                "count": 25,
                "percentage": 3.75,
                "range": "< 200 K"
            },
            {
                "name": "Media (273-373 K)",
                "count": 200,
                "percentage": 30.03,
                "range": "273-373 K"
            }
        ]
    }
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_distribution()
    
    return DistributionResponse(
        total=result["total"],
        categories=result["categories"]
    )


@app.get(
    "/predictions/by-molecule-size",
    response_model=MoleculeSizeResponse,
    tags=["📊 Analytics"],
    summary="📏 Análisis por Tamaño Molecular",
    description="Agrupa moléculas según su número de átomos."
)
def get_by_molecule_size():
    """
    Análisis estadístico por tamaño molecular.
    
    **Categorías de tamaño:**
    - **Pequeña**: 1-10 átomos
    - **Mediana**: 11-25 átomos
    - **Grande**: 26-50 átomos
    - **Muy grande**: > 50 átomos
    
    **Incluye:**
    - Número de moléculas por categoría
    - Temperatura promedio de fusión
    - Temperatura mínima y máxima
    - Ejemplos de moléculas
    
    **Response:**
    ```json
    {
        "total_molecules": 666,
        "size_groups": [
            {
                "size_category": "Pequeña (1-10 átomos)",
                "count": 120,
                "avg_Tm": 280.5,
                "min_Tm": 200.0,
                "max_Tm": 350.0
            }
        ]
    }
    ```
    """
    if ml_service is None:
        raise HTTPException(status_code=500, detail="MLService no está inicializado.")

    result = ml_service.get_predictions_by_molecule_size()
    
    return MoleculeSizeResponse(
        total_molecules=result["total_molecules"],
        size_groups=result["size_groups"]
    )