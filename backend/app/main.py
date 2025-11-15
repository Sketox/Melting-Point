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
