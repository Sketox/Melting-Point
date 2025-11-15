import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "backend" / "models"

TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"

MODEL_OUTPUT_PATH = MODEL_DIR / "model.joblib"
TEST_PROCESSED_OUTPUT_PATH = DATA_PROCESSED_DIR / "test_processed.csv"


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error en %."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8  # evitar división por cero
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def main() -> None:
    print(f"📂 Cargando train.csv desde: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)

    print(f"📂 Cargando test.csv desde:  {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)

    # ----------------------------------------------------------------
    # 1) Seleccionar solo columnas Group como features
    # ----------------------------------------------------------------
    group_cols = [c for c in train_df.columns if c.startswith("Group")]
    print(f"🔢 Número de columnas Group: {len(group_cols)}")
    if not group_cols:
        raise ValueError(
            "No se encontraron columnas que empiecen con 'Group'. "
            "Revisa que el CSV tenga columnas Group 1, Group 2, etc."
        )

    X_all = train_df[group_cols].copy()
    y_all = train_df["Tm"].values

    test_features = test_df[group_cols].copy()

    # Imputar NaN por la mediana (simple y estable)
    medians = X_all.median()
    X_all = X_all.fillna(medians)
    test_features = test_features.fillna(medians)

    print("📊 Ejemplo de columnas de features:", group_cols[:10])

    # ----------------------------------------------------------------
    # 2) Train / Validation split
    # ----------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
    )

    print(f"📏 Tamaño train: {X_train.shape}, Tamaño val: {X_val.shape}")

    # ----------------------------------------------------------------
    # 3) Modelo LightGBM (solo Groups)
    # ----------------------------------------------------------------
    print("🌱 Entrenando LGBMRegressor (solo Group-features)...")

    model = LGBMRegressor(
        n_estimators=600,          # menos árboles para reducir overfitting
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,              # -1 = sin límite, controlado por num_leaves
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="regression",
    )

    # En tu versión de LightGBM no podemos usar early_stopping_rounds
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
    )

    # Si existe best_iteration_ la usamos, si no, usamos todos los árboles
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None:
        print(f"🏁 Mejor iteración reportada por el modelo: {best_iter}")
        pred_kwargs = {"num_iteration": best_iter}
    else:
        print("ℹ️ El modelo no tiene atributo best_iteration_, se usan todas las iteraciones.")
        pred_kwargs = {}

    # ----------------------------------------------------------------
    # 4) Métricas MAE / MAPE
    # ----------------------------------------------------------------
    train_pred = model.predict(X_train, **pred_kwargs)
    val_pred = model.predict(X_val, **pred_kwargs)

    mae_train = mean_absolute_error(y_train, train_pred)
    mae_val = mean_absolute_error(y_val, val_pred)
    mape_train = mape(y_train, train_pred)
    mape_val = mape(y_val, val_pred)

    print("\n📊 Métricas del modelo (LightGBM + Groups):")
    print(f"   • MAE (train):       {mae_train:.4f}")
    print(f"   • MAE (validation):  {mae_val:.4f}")
    print(f"   • MAPE (train):      {mape_train:.2f}%")
    print(f"   • MAPE (validation): {mape_val:.2f}%")

    # ----------------------------------------------------------------
    # 5) Guardar modelo y test_processed.csv
    # ----------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Guardando modelo LightGBM en: {MODEL_OUTPUT_PATH}")
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("💾 Generando test_processed.csv con id + Group-features...")
    test_processed = pd.concat(
        [
            test_df[["id"]].reset_index(drop=True),
            test_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_processed.to_csv(TEST_PROCESSED_OUTPUT_PATH, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
