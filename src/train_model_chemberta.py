import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from xgboost import XGBRegressor


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

CHEMBERT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


# --------------------------------------------------------------------
# Utilidad: cálculo de embeddings ChemBERTa para una lista de SMILES
# --------------------------------------------------------------------
def compute_chemberta_embeddings(smiles: List[str],
                                 batch_size: int = 32,
                                 device: str = None) -> np.ndarray:
    """
    Calcula embeddings ChemBERTa (mean pooling) para una lista de SMILES.
    Devuelve un array numpy de shape [n_moléculas, dim_embedding].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🧪 Usando dispositivo para ChemBERTa: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(CHEMBERT_MODEL_NAME)
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        n = len(smiles)
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_smiles = smiles[start:end]

            # Tokenización
            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Forward
            outputs = model(**encoded)
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

            # Mean pooling sobre la dimensión de secuencia
            emb = last_hidden_state.mean(dim=1)  # [batch, hidden]
            all_embeddings.append(emb.cpu().numpy())

            print(f"   -> Procesadas {min(end, n)}/{n} moléculas", end="\r")

    print()  # salto de línea después del progreso
    embeddings = np.vstack(all_embeddings)
    return embeddings


# --------------------------------------------------------------------
# Entrenamiento principal
# --------------------------------------------------------------------
def main():
    print("📂 Cargando train.csv desde:", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    print("📂 Cargando test.csv desde:", TEST_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --------------------------------------------------------------
    # 1) Features numéricos: columnas Group 1..N (ignoramos id, SMILES, Tm)
    # --------------------------------------------------------------
    group_cols = [c for c in train_df.columns if c.startswith("Group")]
    print(f"🔢 Número de columnas Group: {len(group_cols)}")

    X_groups_train = train_df[group_cols].copy()
    X_groups_test = test_df[group_cols].copy()

    # Imputación simple por mediana en las Group
    medians = X_groups_train.median()
    X_groups_train = X_groups_train.fillna(medians)
    X_groups_test = X_groups_test.fillna(medians)

    # --------------------------------------------------------------
    # 2) Embeddings ChemBERTa de SMILES
    # --------------------------------------------------------------
    print("🧬 Calculando embeddings ChemBERTa para train SMILES...")
    train_smiles = train_df["SMILES"].astype(str).tolist()
    train_chem_embeddings = compute_chemberta_embeddings(train_smiles, batch_size=32)

    print("🧬 Calculando embeddings ChemBERTa para test SMILES...")
    test_smiles = test_df["SMILES"].astype(str).tolist()
    test_chem_embeddings = compute_chemberta_embeddings(test_smiles, batch_size=32)

    chem_dim = train_chem_embeddings.shape[1]
    chem_cols = [f"chem_{i}" for i in range(chem_dim)]

    train_chem_df = pd.DataFrame(train_chem_embeddings, columns=chem_cols)
    test_chem_df = pd.DataFrame(test_chem_embeddings, columns=chem_cols)

    print(f"✅ Dimensión de embedding ChemBERTa: {chem_dim}")

    # --------------------------------------------------------------
    # 3) Combinar Group-features + ChemBERTa embeddings
    # --------------------------------------------------------------
    X_all_train = pd.concat(
        [X_groups_train.reset_index(drop=True), train_chem_df.reset_index(drop=True)],
        axis=1,
    )
    X_all_test = pd.concat(
        [X_groups_test.reset_index(drop=True), test_chem_df.reset_index(drop=True)],
        axis=1,
    )

    feature_cols = X_all_train.columns.tolist()
    print(f"🔗 Total de features combinados: {len(feature_cols)}")

    y = train_df["Tm"].values

    # --------------------------------------------------------------
    # 4) Train/validation split
    # --------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"📊 Tamaño train: {X_train.shape}, Tamaño val: {X_val.shape}")

    # --------------------------------------------------------------
    # 5) Modelo XGBoostRegressor (fuerte)
    # --------------------------------------------------------------
    print("🌲 Entrenando XGBoostRegressor (ChemBERTa + Groups)...")
    model = XGBRegressor(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        reg_lambda=1.0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # --------------------------------------------------------------
    # 6) Métricas MAE / MAPE
    # --------------------------------------------------------------
    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    mae_train = mean_absolute_error(y_train, train_pred)
    mae_val = mean_absolute_error(y_val, val_pred)
    mape_train = mape(y_train, train_pred)
    mape_val = mape(y_val, val_pred)

    print("\n📏 Métricas del modelo (XGBoost + ChemBERTa + Groups):")
    print(f"   • MAE (train):       {mae_train:.4f}")
    print(f"   • MAE (validation):  {mae_val:.4f}")
    print(f"   • MAPE (train):      {mape_train:.2f}%")
    print(f"   • MAPE (validation): {mape_val:.2f}%")

    # --------------------------------------------------------------
    # 7) Guardar modelo y test_processed.csv
    #     (MISMAS RUTAS que el baseline anterior)
    # --------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n💾 Guardando modelo mejorado en:", MODEL_OUTPUT_PATH)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print("💾 Generando test_processed.csv con todas las features combinadas...")
    test_processed = pd.concat(
        [
            test_df[["id"]].reset_index(drop=True),
            X_all_test[feature_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    test_processed.to_csv(TEST_PROCESSED_OUTPUT_PATH, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")


if __name__ == "__main__":
    main()
