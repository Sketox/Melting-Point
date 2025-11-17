from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH, TEST_PROCESSED_PATH


class MLService:
    """
    Servicio de ML que:
    - Carga el modelo entrenado (joblib).
    - Carga el CSV test_processed.csv.
    - Permite predecir Tm por id o para todos los ids.
    """

    def __init__(self) -> None:
        model_path = Path(MODEL_PATH)
        csv_path = Path(TEST_PROCESSED_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV procesado no encontrado en: {csv_path}")

        # Carga modelo
        self.model = joblib.load(model_path)

        # Carga test procesado
        self.test_df = pd.read_csv(csv_path)

        if "id" not in self.test_df.columns:
            raise ValueError("test_processed.csv debe contener una columna 'id'.")

        # Todas las columnas de features son todas excepto 'id'
        self.feature_cols = [c for c in self.test_df.columns if c != "id"]

        if not self.feature_cols:
            raise ValueError(
                "No se encontraron columnas de features en test_processed.csv (solo 'id')."
            )

    def predict_by_id(self, sample_id: int) -> float:
        """
        Devuelve la predicción de Tm para un id concreto del test.
        """
        row = self.test_df[self.test_df["id"] == sample_id]

        if row.empty:
            raise ValueError(
                f"ID {sample_id} no encontrado en test_processed.csv."
            )

        X = row[self.feature_cols]
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_all(self) -> list[tuple[int, float]]:
        """
        Devuelve una lista de pares (id, predicción) para TODOS los registros del test.
        """
        X = self.test_df[self.feature_cols]
        ids = self.test_df["id"].tolist()

        preds = self.model.predict(X)

        # Empaquetar como lista de (id, pred)
        results: list[tuple[int, float]] = [
            (int(sample_id), float(pred)) for sample_id, pred in zip(ids, preds)
        ]
        return results
