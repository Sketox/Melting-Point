from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def mape(y_true, y_pred) -> float:
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

    # Rutas de entrada (ya con RDKit + Groups generados)
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

    print(f"📂 Cargando test_rdkit.csv desde: {test_path}")
    test_df = pd.read_csv(test_path)

    # Definir columnas de features (todas excepto id, SMILES, Tm si existen)
    drop_cols = {"id", "SMILES", "Tm"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    print(f"✅ Número de features: {len(feature_cols)}")
    print(f"🔹 Primeras columnas de features: {feature_cols[:10]}")

    # Separar features y target
    X = train_df[feature_cols].copy()
    y = train_df["Tm"].copy()

    # Relleno sencillo de NaN con medianas de train
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    test_features = test_df[feature_cols].copy().fillna(medians)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("🚀 Entrenando XGBRegressor (RDKit + Groups, configuración básica)...")

    model = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",  # CPU-friendly
    )

    model.fit(X_train, y_train)

    # Predicciones
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Métricas
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_mape = mape(y_train, train_pred)
    val_mape = mape(y_val, val_pred)

    print("\n📊 Métricas del modelo (XGBoost + RDKit básico):")
    print(f"   • MAE (train):      {train_mae:.4f}")
    print(f"   • MAE (validation): {val_mae:.4f}")
    print(f"   • MAPE (train):     {train_mape:.2f}%")
    print(f"   • MAPE (validation):{val_mape:.2f}%\n")

    # Guardar modelo
    print(f"💾 Guardando modelo XGBoost en: {model_path}")
    joblib.dump(model, model_path)

    # Guardar test_processed.csv (id + todas las features)
    print(f"💾 Generando test_processed.csv con id + RDKit/Group-features...")
    test_processed = pd.concat(
        [
            test_df[["id"]].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_processed.to_csv(test_processed_path, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
