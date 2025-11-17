from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


def mape(y_true, y_pred):
    """MAPE en porcentaje, con protección contra división por cero."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


def main() -> None:
    # Raíz del proyecto (MeltingPoint)
    project_root = Path(__file__).resolve().parents[1]

    # Data con RDKit + Groups
    train_path = project_root / "data" / "processed" / "train_rdkit.csv"
    test_path = project_root / "data" / "processed" / "test_rdkit.csv"

    # Salidas
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

    # Rellenar NaNs con la mediana del train
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    test_features = test_df[feature_cols].copy().fillna(medians)

    # ---------------------------
    # 1) BÚSQUEDA DE HIPERPARÁMETROS
    # ---------------------------
    print("🎯 Iniciando RandomizedSearchCV para XGBRegressor...")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    # Espacio de búsqueda (puedes ajustar rangos y listas)
    param_distributions = {
        "n_estimators": [400, 600, 800, 1000, 1200],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=25,  # puedes subir/bajar esto
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # viene negativo

    print("\n🏆 Mejores hiperparámetros encontrados:")
    for k, v in best_params.items():
        print(f"   • {k}: {v}")
    print(f"   → MAE CV (5-fold): {best_score:.4f}")

    # ---------------------------
    # 2) ENTRENAR MODELO FINAL + MÉTRICAS HOLD-OUT
    # ---------------------------

    # Split 80/20 para tener métricas comparables a las que ya usas
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\n🚀 Entrenando modelo final con los mejores hiperparámetros...")

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )

    final_model.fit(X_train, y_train)

    # Predicciones
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)

    # Métricas
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    train_mape = mape(y_train, train_pred)
    val_mape = mape(y_val, val_pred)

    print("\n📊 Métricas del modelo (XGBoost + RDKit AUTOTUNE):")
    print(f"   • MAE (train):      {train_mae:.4f}")
    print(f"   • MAE (validation): {val_mae:.4f}")
    print(f"   • MAPE (train):     {train_mape:.2f}%")
    print(f"   • MAPE (validation):{val_mape:.2f}%\n")

    # ---------------------------
    # 3) GUARDAR MODELO + TEST_PROCESSED
    # ---------------------------
    print(f"💾 Guardando modelo XGBoost en: {model_path}")
    joblib.dump(final_model, model_path)

    print("💾 Generando test_processed.csv con id + RDKit/Group-features...")
    test_processed = pd.DataFrame()
    test_processed["id"] = test_df["id"]
    for col in feature_cols:
        test_processed[col] = test_features[col]
    test_processed.to_csv(test_processed_path, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
