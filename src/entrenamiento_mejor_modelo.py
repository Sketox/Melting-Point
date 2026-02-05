#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
💾 ENTRENAR ENSEMBLE PARA PRODUCCIÓN
═══════════════════════════════════════════════════════════════════════════════

Ejecutar desde la carpeta src/:
    cd src
    python train_ensemble_production.py

Guarda los modelos en:
    backend/models/ensemble_predictor.joblib

El ensemble se combina con ChemProp para lograr MAE ~22.80 K (Kaggle).

Autor: Sketo
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
from catboost import CatBoostRegressor

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent  # MeltingPoint/

DATA_RAW = PROJECT_ROOT / "data" / "raw"
BACKEND_MODELS = PROJECT_ROOT / "backend" / "models"

N_FOLDS = 5
RANDOM_STATE = 42

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
print(f"📁 DATA_RAW: {DATA_RAW}")
print(f"📁 BACKEND_MODELS: {BACKEND_MODELS}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

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
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(smiles_list)}...")
        feat = extract_features(smi)
        features_list.append(feat if feat is not None else np.zeros(2428, dtype=np.float32))
    return np.array(features_list)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  💾 ENTRENANDO ENSEMBLE PARA PRODUCCIÓN                                     ║
║  Objetivo: MAE ~26.64 → Con ChemProp → MAE ~22.80 (Kaggle)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar archivo
    train_path = DATA_RAW / "train.csv"
    if not train_path.exists():
        print(f"❌ No se encontró: {train_path}")
        sys.exit(1)
    
    # Cargar datos
    print("📥 Cargando datos...")
    train_df = pd.read_csv(train_path)
    y = train_df['Tm'].values
    smiles_list = train_df['SMILES'].tolist()
    print(f"   Muestras: {len(train_df)}, Target: {y.min():.1f} - {y.max():.1f} K")
    
    # Features
    print("\n🔧 Extrayendo features...")
    X = create_feature_matrix(smiles_list)
    print(f"   Shape: {X.shape}")
    
    # Parámetros
    params_file = BACKEND_MODELS / "best_params_paso6.json"
    if params_file.exists():
        print(f"\n📂 Cargando parámetros de {params_file.name}...")
        with open(params_file) as f:
            saved = json.load(f)
        xgb_params = saved.get('xgboost', {})
        lgb_params = saved.get('lightgbm', {})
        cat_params = saved.get('catboost', {})
        weights = saved.get('weights', {'XGBoost': 0.35, 'LightGBM': 0.30, 'CatBoost': 0.35})
    else:
        print("\n⚠️ Usando parámetros default...")
        xgb_params = {'n_estimators': 1800, 'max_depth': 8, 'learning_rate': 0.025,
                      'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 2.5,
                      'reg_alpha': 0.15, 'min_child_weight': 4, 'tree_method': 'hist'}
        lgb_params = {'n_estimators': 1800, 'max_depth': 10, 'num_leaves': 80,
                      'learning_rate': 0.025, 'subsample': 0.85, 'colsample_bytree': 0.85,
                      'reg_lambda': 2.5, 'reg_alpha': 0.15, 'min_child_samples': 15}
        cat_params = {'iterations': 1800, 'depth': 8, 'learning_rate': 0.025, 'l2_leaf_reg': 4}
        weights = {'XGBoost': 0.35, 'LightGBM': 0.30, 'CatBoost': 0.35}
    
    # Ajustar params
    xgb_params.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    lgb_params.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
    cat_params.update({'random_seed': RANDOM_STATE, 'verbose': False})
    
    # Entrenar
    print(f"\n🏋️ Entrenando ({N_FOLDS} folds)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    models = {'XGBoost': [], 'LightGBM': [], 'CatBoost': []}
    oof_preds = {'XGBoost': np.zeros(len(y)), 'LightGBM': np.zeros(len(y)), 'CatBoost': np.zeros(len(y))}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n   Fold {fold + 1}/{N_FOLDS}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # XGBoost
        print("      XGB...", end=" ", flush=True)
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models['XGBoost'].append(m)
        oof_preds['XGBoost'][val_idx] = m.predict(X_val)
        print(f"{mean_absolute_error(y_val, oof_preds['XGBoost'][val_idx]):.2f}", end=" | ")
        
        # LightGBM
        print("LGB...", end=" ", flush=True)
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        models['LightGBM'].append(m)
        oof_preds['LightGBM'][val_idx] = m.predict(X_val)
        print(f"{mean_absolute_error(y_val, oof_preds['LightGBM'][val_idx]):.2f}", end=" | ")
        
        # CatBoost
        print("CAT...", end=" ", flush=True)
        m = CatBoostRegressor(**cat_params)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        models['CatBoost'].append(m)
        oof_preds['CatBoost'][val_idx] = m.predict(X_val)
        print(f"{mean_absolute_error(y_val, oof_preds['CatBoost'][val_idx]):.2f}")
    
    # Resultados
    print("\n" + "="*60)
    mae_results = {}
    for name, preds in oof_preds.items():
        mae_results[name] = mean_absolute_error(y, preds)
        print(f"   {name}: {mae_results[name]:.2f}")
    
    ensemble_oof = sum(weights.get(n, 0.33) * oof_preds[n] for n in oof_preds)
    ensemble_mae = mean_absolute_error(y, ensemble_oof)
    mae_results['Ensemble'] = ensemble_mae
    print(f"   Ensemble: {ensemble_mae:.2f} ⭐")
    
    # Guardar
    print("\n💾 Guardando...")
    BACKEND_MODELS.mkdir(parents=True, exist_ok=True)
    save_path = BACKEND_MODELS / "ensemble_predictor.joblib"
    
    joblib.dump({
        'models': models, 'weights': weights, 'oof_mae': mae_results,
        'n_features': X.shape[1], 'chemprop_weight': 0.20,
        'trained_at': datetime.now().isoformat()
    }, save_path)
    
    print(f"   ✅ {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅ COMPLETADO                                                               ║
║                                                                              ║
║  Ensemble OOF: {ensemble_mae:.2f} K                                                    ║
║  + ChemProp 20%: ~22.80 K (Kaggle)                                          ║
║                                                                              ║
║  🚀 Reinicia el backend: uvicorn app.main:app --reload                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()