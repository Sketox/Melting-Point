from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 

from .ml_service import MLService
from .schemas import PredictByIdRequest, PredictResponse

app = FastAPI(
    title="Melting Point API",
    description="API para predecir el punto de fusión (Tm) usando RandomForestRegressor.",
    version="0.1.0",
)

#CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev server
        "http://127.0.0.1:3000",      # Alternativa
        "http://localhost:5173",       # Vite (por si acaso)
        "*",                           # Permitir todo (solo para desarrollo)
    ],
    allow_credentials=True,
    allow_methods=["*"],               # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],               # Permitir todos los headers
)

ml_service: MLService | None = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Carga el modelo y el CSV procesado al iniciar la aplicación.
    """
    global ml_service
    ml_service = MLService()


@app.get("/")
def root():
    """Endpoint raíz para verificar que el servidor está corriendo."""
    return {"message": "Melting Point API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-by-id", response_model=PredictResponse)
def predict_by_id(request: PredictByIdRequest):
    """
    Dado un ID presente en test_processed.csv, devuelve la predicción de Tm.
    """
    if ml_service is None:
        raise HTTPException(
            status_code=500,
            detail="MLService no está inicializado.",
        )

    try:
        pred = ml_service.predict_by_id(request.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(id=request.id, Tm_pred=pred)


@app.get("/predict-all", response_model=List[PredictResponse])
def predict_all():
    """
    Devuelve las predicciones de Tm para TODOS los IDs presentes en test_processed.csv.
    """
    if ml_service is None:
        raise HTTPException(
            status_code=500,
            detail="MLService no está inicializado.",
        )

    results = ml_service.predict_all()

    responses: List[PredictResponse] = [
        PredictResponse(id=sample_id, Tm_pred=pred) for sample_id, pred in results
    ]

    return responses