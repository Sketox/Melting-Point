from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from xgboost import XGBRegressor


def morgan_fp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Genera un fingerprint de Morgan (ECFP) binario a partir de un SMILES.
    Si el SMILES es inválido, devuelve un vector de ceros.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_morgan_df(smiles_series: pd.Series, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """
    Aplica morgan_fp_from_smiles a toda una columna de SMILES
    y devuelve un DataFrame con columnas morgan_0 ... morgan_{n_bits-1}.
    """
    fps = smiles_series.apply(lambda s: morgan_fp_from_smiles(s, radius=radius, n_bits=n_bits))
    fp_matrix = np.stack(fps.values)

    fp_cols = [f"morgan_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=smiles_series.index)
    return fp_df


def mape(y_true, y_pred) -> float:
    """
    Cálculo manual del MAPE para evitar divisiones por cero.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0


def main():
    # 1) Rutas
    project_root = Path(__file__).resolve().parents[1]

    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"

    model_path = project_root / "backend" / "models" / "model.joblib"
    processed_dir = project_root / "data" / "processed"
    test_processed_path = processed_dir / "test_processed.csv"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2) Cargar datos
    print(f"📂 Cargando train.csv desde: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"📂 Cargando test.csv desde: {test_path}")
    test_df = pd.read_csv(test_path)

    # 3) Fingerprints Morgan a partir de SMILES
    print("🧪 Generando Morgan fingerprints (ECFP-2048) para train...")
    train_morgan = compute_morgan_df(train_df["SMILES"], radius=2, n_bits=2048)

    print("🧪 Generando Morgan fingerprints (ECFP-2048) para test...")
    test_morgan = compute_morgan_df(test_df["SMILES"], radius=2, n_bits=2048)

    # 4) Features base (todo excepto id, SMILES, Tm)
    drop_cols = ["id", "SMILES", "Tm"]
    base_feature_cols = [col for col in train_df.columns if col not in drop_cols]

    print(f"🔢 Número de columnas base (Groups / otros): {len(base_feature_cols)}")
    print(f"🔹 Ejemplo columnas base: {base_feature_cols[:10]}")

    X_base_train = train_df[base_feature_cols].copy()
    X_base_test = test_df[base_feature_cols].copy()

    # 5) Combinar base + Morgan
    X_train_full = pd.concat(
        [X_base_train.reset_index(drop=True), train_morgan.reset_index(drop=True)],
        axis=1,
    )
    X_test_full = pd.concat(
        [X_base_test.reset_index(drop=True), test_morgan.reset_index(drop=True)],
        axis=1,
    )

    # Rellenar NaNs con mediana de cada columna (solo en X_train_full)
    medians = X_train_full.median(numeric_only=True)
    X_train_full = X_train_full.fillna(medians)
    X_test_full = X_test_full.fillna(medians)

    print(f"✅ Número total de features (base + Morgan): {X_train_full.shape[1]}")

    # 6) Target
    y = train_df["Tm"].copy()

    # 7) Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 8) Definir modelo XGBoost (configuración razonable para empezar)
    print("🚀 Entrenando XGBRegressor (Groups + Morgan, configuración básica)...")
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # suele ir bien en CPU
    )

    model.fit(X_tr, y_tr)

    # 9) Evaluación
    y_tr_pred = model.predict(X_tr)
    y_val_pred = model.predict(X_val)

    mae_tr = mean_absolute_error(y_tr, y_tr_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)

    mape_tr = mape(y_tr, y_tr_pred)
    mape_val = mape(y_val, y_val_pred)

    # 10) Guardar modelo
    print(f"💾 Guardando modelo XGBoost en: {model_path}")
    joblib.dump(model, model_path)

    # 11) Guardar test_processed.csv con id + mismas features que X_train_full
    print(f"💾 Generando test_processed.csv con id + Morgan/base-features...")
    test_processed = pd.DataFrame()
    test_processed["id"] = test_df["id"]
    # aseguramos mismo orden de columnas que en entrenamiento
    for col in X_train_full.columns:
        test_processed[col] = X_test_full[col].values

    test_processed.to_csv(test_processed_path, index=False)

    print("✅ Entrenamiento completado y archivos guardados correctamente.")

    print("\n📊 Métricas del modelo (XGBoost + Morgan + Groups):")
    print(f"   • MAE (train):      {mae_tr:.4f}")
    print(f"   • MAE (validation): {mae_val:.4f}")
    print(f"   • MAPE (train):     {mape_tr:.2f}%")
    print(f"   • MAPE (validation):{mape_val:.2f}%\n")


if __name__ == "__main__":
    main()
