from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def mape(y_true, y_pred):
    """
    Cálculo manual de MAPE para evitar divisiones por cero.
    Devuelve el porcentaje de error medio absoluto.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


def main() -> None:
    # Raíz del proyecto (carpeta MeltingPoint)
    project_root = Path(__file__).resolve().parents[1]

    # Rutas de entrada (ya con RDKit + Groups)
    train_path = project_root / "data" / "processed" / "train_rdkit.csv"
    test_path = project_root / "data" / "processed" / "test_rdkit.csv"

    # Rutas de salida
    model_path = project_root / "backend" / "models" / "model.joblib"
    processed_dir = project_root / "data" / "processed"
    test_processed_path = processed_dir / "test_processed.csv"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Cargando train_rdkit.csv desde: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"📂 Cargando test_rdkit.csv desde:  {test_path}")
    test_df = pd.read_csv(test_path)

    # Features = todo menos id / SMILES / Tm
    drop_cols = ["id", "SMILES", "Tm"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    print(f"✅ Número de features: {len(feature_cols)}")
    print(f"🔹 Primeras columnas de features: {feature_cols[:10]}")

    X = train_df[feature_cols].copy()
    y = train_df["Tm"].copy()

    # Relleno de NaNs con la mediana del train
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    test_features = test_df[feature_cols].copy().fillna(medians)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("🚀 Entrenando XGBRegressor (RDKit + Groups, configuración TUNED)...")

    # Versión "tuneada" pero compatible con tu xgboost (sin early_stopping_rounds)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,      # más árboles
        max_depth=8,           # árboles más profundos que el básico
        learning_rate=0.03,    # learning rate más bajo
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,        # L2
        reg_alpha=0.0,         # L1
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
    )

    # IMPORTANTE: no pasar early_stopping_rounds ni eval_set para evitar el error
    model.fit(X_train, y_train)

    # Predicciones
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Métricas
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    train_mape = mape(y_train, train_pred)
    val_mape = mape(y_val, val_pred)

    print("\n📊 Métricas del modelo (XGBoost + RDKit TUNED):")
    print(f"   • MAE (train):      {train_mae:.4f}")
    print(f"   • MAE (validation): {val_mae:.4f}")
    print(f"   • MAPE (train):     {train_mape:.2f}%")
    print(f"   • MAPE (validation):{val_mape:.2f}%\n")

    # Guardar modelo
    print(f"💾 Guardando modelo XGBoost en: {model_path}")
    joblib.dump(model, model_path)

    # Guardar test_processed con id + features combinadas
    print("💾 Generando test_processed.csv con id + RDKit/Group-features...")
    test_processed = pd.DataFrame()
    test_processed["id"] = test_df["id"]
    for col in feature_cols:
        test_processed[col] = test_features[col]

    test_processed.to_csv(test_processed_path, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
