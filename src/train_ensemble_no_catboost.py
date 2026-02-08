#!/usr/bin/env python3
"""
===============================================================================
ENTRENAR ENSEMBLE SIN CATBOOST (XGBoost + LightGBM)
===============================================================================

Ejecutar desde la carpeta src/:
    cd MeltingPoint/src
    python train_ensemble_no_catboost.py

Guarda los modelos en:
    backend/models/ensemble_predictor.joblib

Este ensemble se combina con ChemProp para lograr predicciones en tiempo real.
Sin CatBoost el MAE es ligeramente mayor, pero funciona sin Visual Studio 2022.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')

# ===============================================================================
# CONFIG
# ===============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent  # MeltingPoint/

DATA_RAW = PROJECT_ROOT / "data" / "raw"
BACKEND_MODELS = PROJECT_ROOT / "backend" / "models"

N_FOLDS = 5
RANDOM_STATE = 42

print(f"[+] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[+] DATA_RAW: {DATA_RAW}")
print(f"[+] BACKEND_MODELS: {BACKEND_MODELS}")


# ===============================================================================
# FEATURES
# ===============================================================================

def extract_features(smiles: str) -> np.ndarray:
    """Extrae features moleculares de un SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = []

    # Morgan FP (2048)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features.extend(list(fp))

    # MACCS (167)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    features.extend(list(maccs))

    # RDKit Descriptors
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    try:
        rdkit_desc = list(calc.CalcDescriptors(mol))
    except:
        rdkit_desc = [0.0] * len(desc_names)
    features.extend(rdkit_desc)

    # SMILES features
    smi = str(smiles)
    features.extend([
        len(smi), sum(c.isdigit() for c in smi), smi.count("("),
        smi.count("="), smi.count("#"), sum(c.islower() for c in smi),
        smi.count("N") + smi.count("n"), smi.count("O") + smi.count("o"),
        smi.count("F"), smi.count("Cl"), smi.count("Br"),
        smi.count("S") + smi.count("s"), smi.count("P"),
    ])

    X = np.array(features, dtype=np.float32)
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


def create_feature_matrix(smiles_list):
    """Crea matriz de features."""
    print("  Extrayendo features...")
    features_list = []
    n_features = None

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(smiles_list)}...")
        feat = extract_features(smi)
        if feat is not None:
            if n_features is None:
                n_features = len(feat)
            features_list.append(feat)
        else:
            # Use zeros if extraction fails
            if n_features is None:
                n_features = 2428  # Default
            features_list.append(np.zeros(n_features, dtype=np.float32))

    return np.array(features_list)


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("""
================================================================================
  ENTRENANDO ENSEMBLE SIN CATBOOST (XGBoost + LightGBM)
  Objetivo: Predicciones en tiempo real para nuevos compuestos
================================================================================
    """)

    # Verificar archivo
    train_path = DATA_RAW / "train.csv"
    if not train_path.exists():
        print(f"[ERROR] No se encontro: {train_path}")
        sys.exit(1)

    # Cargar datos
    print("[*] Cargando datos...")
    train_df = pd.read_csv(train_path)
    y = train_df['Tm'].values
    smiles_list = train_df['SMILES'].tolist()
    print(f"   Muestras: {len(train_df)}, Target: {y.min():.1f} - {y.max():.1f} K")

    # Features
    print("\n[*] Extrayendo features...")
    X = create_feature_matrix(smiles_list)
    print(f"   Shape: {X.shape}")

    # Parametros optimizados
    xgb_params = {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_lambda': 3.0,
        'reg_alpha': 0.2,
        'min_child_weight': 4,
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    lgb_params = {
        'n_estimators': 2000,
        'max_depth': 10,
        'num_leaves': 90,
        'learning_rate': 0.02,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_lambda': 3.0,
        'reg_alpha': 0.2,
        'min_child_samples': 12,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }

    # Pesos: sin CatBoost, ajustamos 55% XGB + 45% LGB
    weights = {'XGBoost': 0.55, 'LightGBM': 0.45}

    # Entrenar
    print(f"\n[*] Entrenando ({N_FOLDS} folds)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {'XGBoost': [], 'LightGBM': []}
    oof_preds = {'XGBoost': np.zeros(len(y)), 'LightGBM': np.zeros(len(y))}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n   Fold {fold + 1}/{N_FOLDS}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # XGBoost
        print("      XGBoost...", end=" ", flush=True)
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models['XGBoost'].append(m_xgb)
        oof_preds['XGBoost'][val_idx] = m_xgb.predict(X_val)
        xgb_mae = mean_absolute_error(y_val, oof_preds['XGBoost'][val_idx])
        print(f"MAE: {xgb_mae:.2f}", end=" | ")

        # LightGBM
        print("LightGBM...", end=" ", flush=True)
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        models['LightGBM'].append(m_lgb)
        oof_preds['LightGBM'][val_idx] = m_lgb.predict(X_val)
        lgb_mae = mean_absolute_error(y_val, oof_preds['LightGBM'][val_idx])
        print(f"MAE: {lgb_mae:.2f}")

    # Resultados finales
    print("\n" + "="*70)
    print("[*] RESULTADOS OOF:")
    mae_results = {}
    for name, preds in oof_preds.items():
        mae_results[name] = mean_absolute_error(y, preds)
        print(f"   {name}: {mae_results[name]:.2f} K")

    # Ensemble weighted average
    ensemble_oof = sum(weights.get(n, 0.5) * oof_preds[n] for n in oof_preds)
    ensemble_mae = mean_absolute_error(y, ensemble_oof)
    mae_results['Ensemble'] = ensemble_mae
    print(f"   Ensemble (XGB+LGB): {ensemble_mae:.2f} K [BEST]")
    print("="*70)

    # Guardar
    print("\n[*] Guardando modelos...")
    BACKEND_MODELS.mkdir(parents=True, exist_ok=True)
    save_path = BACKEND_MODELS / "ensemble_predictor.joblib"

    # Backup del anterior si existe
    if save_path.exists():
        backup_path = save_path.with_suffix('.joblib.bak')
        save_path.rename(backup_path)
        print(f"   Backup anterior: {backup_path.name}")

    joblib.dump({
        'models': models,
        'weights': weights,
        'oof_mae': mae_results,
        'n_features': X.shape[1],
        'chemprop_weight': 0.20,  # Para combinacion con ChemProp
        'trained_at': datetime.now().isoformat(),
        'note': 'Ensemble sin CatBoost (XGBoost + LightGBM)'
    }, save_path)

    file_size = save_path.stat().st_size / 1024 / 1024
    print(f"   [OK] Guardado: {save_path} ({file_size:.1f} MB)")

    # Estimacion con ChemProp
    # Asumiendo ChemProp MAE ~28.85 y combinacion optima 20%/80%
    chemprop_mae = 28.85
    estimated_combined = 0.20 * chemprop_mae + 0.80 * ensemble_mae

    print(f"""
================================================================================
  ENTRENAMIENTO COMPLETADO
--------------------------------------------------------------------------------
  Ensemble OOF (XGB+LGB):  {ensemble_mae:>6.2f} K
  ChemProp estimado:       {chemprop_mae:>6.2f} K
  Combinado estimado:      {estimated_combined:>6.2f} K (20% CP + 80% Ensemble)
--------------------------------------------------------------------------------
  Reinicia el backend para usar el nuevo modelo:
     cd backend && uvicorn app.main:app --reload --port 8000
================================================================================
    """)


if __name__ == "__main__":
    main()
