from pydantic import BaseModel


class PredictByIdRequest(BaseModel):
    id: int


class PredictResponse(BaseModel):
    id: int
    Tm_pred: float
