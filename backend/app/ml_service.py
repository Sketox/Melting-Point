from typing import List

import joblib
import pandas as pd

from .config import MODEL_PATH, TEST_PROCESSED_PATH


class MLService:
    def __init__(self) -> None:
        # Cargar modelo
        self.model = joblib.load(MODEL_PATH)

        # Cargar test procesado
        self.test_df = pd.read_csv(TEST_PROCESSED_PATH)

        # Columnas de features = todas menos 'id'
        self.feature_cols: List[str] = [
            col for col in self.test_df.columns if col != "id"
        ]

        if self.test_df.empty:
            raise RuntimeError("El archivo test_processed.csv está vacío.")

    def predict_by_id(self, sample_id: int) -> float:
        # Buscar la fila con ese id
        row = self.test_df[self.test_df["id"] == sample_id]

        if row.empty:
            raise ValueError(f"id {sample_id} no encontrado en test_processed.csv")

        X = row[self.feature_cols]
        pred = self.model.predict(X)[0]

        # Asegurarnos de devolver un float nativo de Python
        return float(pred)
