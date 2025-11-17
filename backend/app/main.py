from typing import List

from fastapi import FastAPI, HTTPException

from .ml_service import MLService
from .schemas import PredictByIdRequest, PredictResponse

app = FastAPI(
    title="Melting Point API",
    description="API para predecir el punto de fusión (Tm) usando RandomForestRegressor.",
    version="0.1.0",
)

ml_service: MLService | None = None


@app.on_event("startup")
def startup_event() -> None:
    """
    Carga el modelo y el CSV procesado al iniciar la aplicación.
    """
    global ml_service
    ml_service = MLService()


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
        # id no encontrado
        raise HTTPException(status_code=404, detail=str(e))

    return PredictResponse(id=request.id, Tm_pred=pred)


@app.get("/predict-all", response_model=List[PredictResponse])
def predict_all():
    """
    Devuelve las predicciones de Tm para TODOS los IDs presentes en test_processed.csv.
    Formato: [{ "id": xxx, "Tm_pred": yyy }, ...]
    """
    if ml_service is None:
        raise HTTPException(
            status_code=500,
            detail="MLService no está inicializado.",
        )

    results = ml_service.predict_all()

    # Convertimos la lista de (id, pred) a lista de PredictResponse
    responses: List[PredictResponse] = [
        PredictResponse(id=sample_id, Tm_pred=pred) for sample_id, pred in results
    ]

    return responses
