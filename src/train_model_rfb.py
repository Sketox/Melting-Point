import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    # Detectar la raíz del proyecto (carpeta MeltingPoint)
    project_root = Path(__file__).resolve().parents[1]

    # Rutas de entrada
    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"

    # Rutas de salida
    model_path = project_root / "backend" / "models" / "model.joblib"
    processed_dir = project_root / "data" / "processed"
    test_processed_path = processed_dir / "test_processed.csv"

    # Crear carpetas necesarias
    model_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Cargando train.csv desde: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"📂 Cargando test.csv desde: {test_path}")
    test_df = pd.read_csv(test_path)

    # Definir columnas de features (todas excepto id, SMILES, Tm)
    drop_cols = ["id", "SMILES", "Tm"]
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    print(f"✅ Número de features: {len(feature_cols)}")
    print(f"🔹 Primeras columnas de features: {feature_cols[:10]}")

    # Separar X, y
    X = train_df[feature_cols].copy()
    y = train_df["Tm"].copy()

    # Manejo simple de NaNs: usar la mediana de train para cada columna
    medians = X.median(numeric_only=True)

    # Rellenar NaNs en train y test con las medianas calculadas en train
    X = X.fillna(medians)
    test_features = test_df[feature_cols].copy().fillna(medians)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("🚀 Entrenando RandomForestRegressor (baseline)...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluación con MAE y MAPE
    train_pred = model.predict(X_train)
    val_pred  = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae   = mean_absolute_error(y_val, val_pred)

    # Cálculo manual del MAPE para evitar divisiones por cero
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        eps = 1e-8  # evitar división por cero
        return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

    train_mape = mape(y_train, train_pred)
    val_mape   = mape(y_val, val_pred)

    print("\n📊 Métricas del modelo:")
    print(f"   • MAE (train):      {train_mae:.4f}")
    print(f"   • MAE (validation): {val_mae:.4f}")
    print(f"   • MAPE (train):     {train_mape:.2f}%")
    print(f"   • MAPE (validation):{val_mape:.2f}%\n")

    # Guardar modelo
    print(f"💾 Guardando modelo en: {model_path}")
    joblib.dump(model, model_path)

    # Guardar test procesado (id + mismas columnas de features)
    print(f"💾 Guardando test procesado en: {test_processed_path}")
    test_processed = pd.DataFrame()
    test_processed["id"] = test_df["id"]
    for col in feature_cols:
        test_processed[col] = test_features[col]

    test_processed.to_csv(test_processed_path, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
