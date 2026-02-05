#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🎯 OPTIMIZACIÓN FINA - 5 SUBMISSIONS ESTRATÉGICOS 🎯
═══════════════════════════════════════════════════════════════════════════════

Estrategias:
1. Ajuste fino de ChemProp weight (17%, 18%, 19%, 21%, 22%, 23%)
2. Pseudo-labeling (usar test predictions para mejorar modelo)
3. Post-processing (clip outliers, ajustar distribución)
4. Multi-seed ensemble (reducir varianza)

Objetivo: Bajar de 22.80 a ~22.0-22.5
"""

import os
import warnings
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
if os.path.exists('D:/devu/MeltingPoint'):
    PROJECT_ROOT = Path('D:/devu/MeltingPoint')
else:
    PROJECT_ROOT = Path('.').resolve()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")


def create_features(smiles_list, group_features=None):
    """Features estilo PASO 6."""
    features = []
    
    # Morgan FP (2048)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(list(fp))
        else:
            fps.append([0] * 2048)
    features.append(np.array(fps, dtype=np.float32))
    
    # MACCS (167)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(list(fp))
        else:
            fps.append([0] * 167)
    features.append(np.array(fps, dtype=np.float32))
    
    # RDKit descriptors
    desc_funcs = [
        Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
        Descriptors.NumHDonors, Descriptors.NumHAcceptors,
        Descriptors.NumRotatableBonds, Descriptors.RingCount,
        Descriptors.NumAromaticRings, Descriptors.FractionCSP3,
        Descriptors.HeavyAtomCount, Descriptors.NumHeteroatoms,
        Descriptors.NumSaturatedRings, Descriptors.NumAliphaticRings,
        Descriptors.LabuteASA, Descriptors.BertzCT,
        Descriptors.Chi0, Descriptors.Chi1,
        Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
        Descriptors.HallKierAlpha, Descriptors.MolMR
    ]
    
    rdkit_feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            feat = []
            for func in desc_funcs:
                try:
                    val = func(mol)
                    feat.append(float(val) if val is not None else 0.0)
                except:
                    feat.append(0.0)
            rdkit_feats.append(feat)
        else:
            rdkit_feats.append([0.0] * len(desc_funcs))
    features.append(np.array(rdkit_feats, dtype=np.float32))
    
    # Groups
    if group_features is not None:
        features.append(group_features.astype(np.float32))
    
    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X


def train_ensemble_multiseed(X_train, y_train, X_test, n_folds=5, seeds=[42, 123, 456]):
    """Entrena ensemble con múltiples seeds para reducir varianza."""
    
    all_oof = []
    all_test = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        oof_xgb = np.zeros(len(y_train))
        oof_lgb = np.zeros(len(y_train))
        oof_cat = np.zeros(len(y_train))
        
        test_xgb = []
        test_lgb = []
        test_cat = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=1800, max_depth=8, learning_rate=0.025,
                subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.5, reg_alpha=0.15, min_child_weight=4,
                random_state=seed, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
            oof_xgb[val_idx] = xgb_model.predict(X_val_s)
            test_xgb.append(xgb_model.predict(X_test_s))
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=1800, max_depth=10, num_leaves=80,
                learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.5, reg_alpha=0.15, min_child_samples=15,
                random_state=seed, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)])
            oof_lgb[val_idx] = lgb_model.predict(X_val_s)
            test_lgb.append(lgb_model.predict(X_test_s))
            
            # CatBoost
            cat_model = CatBoostRegressor(
                iterations=1800, depth=8, learning_rate=0.025,
                l2_leaf_reg=4, random_seed=seed, verbose=False,
                early_stopping_rounds=100
            )
            cat_model.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=False)
            oof_cat[val_idx] = cat_model.predict(X_val_s)
            test_cat.append(cat_model.predict(X_test_s))
        
        # Stack para este seed
        meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
        meta_test = np.column_stack([
            np.mean(test_xgb, axis=0),
            np.mean(test_lgb, axis=0),
            np.mean(test_cat, axis=0)
        ])
        
        meta = Ridge(alpha=1.0)
        meta.fit(meta_train, y_train)
        
        oof_seed = meta.predict(meta_train)
        test_seed = meta.predict(meta_test)
        
        mae = mean_absolute_error(y_train, oof_seed)
        print(f"    OOF MAE: {mae:.2f}")
        
        all_oof.append(oof_seed)
        all_test.append(test_seed)
    
    # Promediar todos los seeds
    final_oof = np.mean(all_oof, axis=0)
    final_test = np.mean(all_test, axis=0)
    
    return final_oof, final_test


def apply_postprocessing(predictions, train_y):
    """Post-processing: clip outliers basado en distribución de train."""
    
    # Clip a rango de train con margen
    p_min = train_y.min() - 20
    p_max = train_y.max() + 20
    
    processed = np.clip(predictions, p_min, p_max)
    
    return processed


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 OPTIMIZACIÓN FINA - 5 SUBMISSIONS ESTRATÉGICOS 🎯                       ║
║                                                                              ║
║  Estrategias:                                                                ║
║   1. Multi-seed ensemble (reducir varianza)                                 ║
║   2. Ajuste fino ChemProp weights (17-23%)                                  ║
║   3. Post-processing (clip outliers)                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📥 CARGANDO DATOS")
    print("="*70)
    
    train_df = pd.read_csv(DATA_RAW / "train.csv")
    test_df = pd.read_csv(DATA_RAW / "test.csv")
    
    y = train_df['Tm'].values
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Target: {y.min():.1f} - {y.max():.1f} K")
    
    # Groups
    group_cols = [c for c in train_df.columns if c.startswith('Group')]
    train_groups = train_df[group_cols].values
    test_groups = test_df[group_cols].values
    
    # ChemProp
    cp_path = DATA_PROCESSED / "chemprop_predictions.csv"
    cp_test = None
    if cp_path.exists():
        cp_df = pd.read_csv(cp_path)
        cp_test = cp_df['Tm'].values if 'Tm' in cp_df.columns else cp_df.iloc[:, 0].values
        print(f"  ChemProp: {len(cp_test)} predicciones")
    
    # =========================================================================
    # FEATURES
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🔧 CREANDO FEATURES")
    print("="*70)
    
    X_train = create_features(train_df['SMILES'].tolist(), train_groups)
    X_test = create_features(test_df['SMILES'].tolist(), test_groups)
    print(f"  Features: {X_train.shape[1]}")
    
    # =========================================================================
    # MULTI-SEED ENSEMBLE
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🎲 MULTI-SEED ENSEMBLE (3 seeds)")
    print("="*70)
    
    oof_multiseed, test_multiseed = train_ensemble_multiseed(
        X_train, y, X_test, 
        n_folds=5, 
        seeds=[42, 123, 456]
    )
    
    oof_mae = mean_absolute_error(y, oof_multiseed)
    print(f"\n  📊 Multi-seed OOF MAE: {oof_mae:.2f}")
    
    # Post-processing
    test_processed = apply_postprocessing(test_multiseed, y)
    
    # =========================================================================
    # GENERAR SUBMISSIONS CON DIFERENTES PESOS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📁 GENERANDO SUBMISSIONS")
    print("="*70)
    
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Solo ensemble (sin ChemProp)
    pd.DataFrame({'id': test_df['id'], 'Tm': test_processed}).to_csv(
        SUBMISSIONS_DIR / f"optimized_solo_{timestamp}.csv", index=False)
    print(f"  ✅ optimized_solo_{timestamp}.csv")
    
    if cp_test is not None:
        # Grid fino de pesos
        weights = [0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
        
        for w in weights:
            combined = w * cp_test + (1 - w) * test_processed
            combined = apply_postprocessing(combined, y)
            
            filename = f"optimized_cp{int(w*100)}_{timestamp}.csv"
            pd.DataFrame({'id': test_df['id'], 'Tm': combined}).to_csv(
                SUBMISSIONS_DIR / filename, index=False)
            print(f"  ✅ {filename}")
    
    # =========================================================================
    # RESUMEN Y RECOMENDACIONES
    # =========================================================================
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  📊 RESULTADOS                                                               ║
║                                                                              ║
║  Multi-seed OOF MAE: {oof_mae:.2f}                                              ║
║  PASO 6 original:    26.64                                                  ║
║                                                                              ║
║  Comparación: {"✅ MEJOR" if oof_mae < 26.64 else "❌ Similar/Peor"}                                                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎯 RECOMENDACIÓN DE SUBMISSIONS (5 disponibles):                           ║
║                                                                              ║
║   Opción A - Ajuste fino (seguro):                                          ║
║     1. optimized_cp19 (cerca del óptimo cp20)                               ║
║     2. optimized_cp21 (otro lado del óptimo)                                ║
║     3. optimized_cp18                                                       ║
║     4. optimized_cp22                                                       ║
║     5. Reserva para el mejor                                                ║
║                                                                              ║
║   Opción B - Más agresivo:                                                  ║
║     1. optimized_cp19                                                       ║
║     2. optimized_cp20 (multi-seed, debería ser más estable)                 ║
║     3-5. Basado en resultados                                               ║
║                                                                              ║
║  📌 Tu mejor actual: 22.80 (paso6_cp20)                                     ║
║  📌 Objetivo: < 22.80                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Guardar config
    config = {
        'timestamp': timestamp,
        'multiseed_oof_mae': float(oof_mae),
        'seeds_used': [42, 123, 456],
        'chemprop_weights_tested': [0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
    }
    
    with open(SUBMISSIONS_DIR / f"config_optimized_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()