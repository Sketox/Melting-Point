#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🎯 PASO 6 + FEATURE SELECTION 🎯
═══════════════════════════════════════════════════════════════════════════════

Hipótesis: El PASO 6 tiene 2757 features, pero muchas son redundantes.
Si seleccionamos solo las más importantes, podríamos mejorar.

Técnica: Mutual Information Regression para seleccionar top K features.

Paper reference: "Feature Selection for High-Dimensional Data" 
"""

import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
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


def create_paso6_features(smiles_list: List[str], 
                          group_features=None) -> Tuple[np.ndarray, List[str]]:
    """Features estilo PASO 6."""
    
    all_features = []
    all_names = []
    
    # Morgan FP (2048)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(list(fp))
        else:
            fps.append([0] * 2048)
    morgan = np.array(fps, dtype=np.float32)
    all_features.append(morgan)
    all_names.extend([f'morgan_{i}' for i in range(2048)])
    
    # MACCS (167)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(list(fp))
        else:
            fps.append([0] * 167)
    maccs = np.array(fps, dtype=np.float32)
    all_features.append(maccs)
    all_names.extend([f'maccs_{i}' for i in range(167)])
    
    # RDKit 2D
    desc_funcs = [
        ('MolWt', Descriptors.MolWt), ('LogP', Descriptors.MolLogP),
        ('TPSA', Descriptors.TPSA), ('NumHDonors', Descriptors.NumHDonors),
        ('NumHAcceptors', Descriptors.NumHAcceptors),
        ('NumRotatableBonds', Descriptors.NumRotatableBonds),
        ('RingCount', Descriptors.RingCount), ('NumAromaticRings', Descriptors.NumAromaticRings),
        ('NumSaturatedRings', Descriptors.NumSaturatedRings),
        ('NumAliphaticRings', Descriptors.NumAliphaticRings),
        ('NumHeteroatoms', Descriptors.NumHeteroatoms),
        ('FractionCSP3', Descriptors.FractionCSP3),
        ('HeavyAtomCount', Descriptors.HeavyAtomCount),
        ('LabuteASA', Descriptors.LabuteASA), ('BertzCT', Descriptors.BertzCT),
        ('Chi0', Descriptors.Chi0), ('Chi1', Descriptors.Chi1),
        ('Kappa1', Descriptors.Kappa1), ('Kappa2', Descriptors.Kappa2),
        ('Kappa3', Descriptors.Kappa3), ('HallKierAlpha', Descriptors.HallKierAlpha),
        ('MolMR', Descriptors.MolMR),
    ]
    
    rdkit_feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            feat = []
            for name, func in desc_funcs:
                try:
                    val = func(mol)
                    feat.append(float(val) if val is not None else 0.0)
                except:
                    feat.append(0.0)
            rdkit_feats.append(feat)
        else:
            rdkit_feats.append([0.0] * len(desc_funcs))
    
    rdkit_arr = np.array(rdkit_feats, dtype=np.float32)
    all_features.append(rdkit_arr)
    all_names.extend([f'rdkit_{name}' for name, _ in desc_funcs])
    
    # Groups
    if group_features is not None:
        all_features.append(group_features.astype(np.float32))
        all_names.extend([f'group_{i}' for i in range(group_features.shape[1])])
    
    X = np.hstack(all_features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, all_names


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎯 PASO 6 + FEATURE SELECTION                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Cargar datos
    train_df = pd.read_csv(DATA_RAW / "train.csv")
    test_df = pd.read_csv(DATA_RAW / "test.csv")
    
    y = train_df['Tm'].values
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    group_cols = [c for c in train_df.columns if c.startswith('Group')]
    train_groups = train_df[group_cols].values
    test_groups = test_df[group_cols].values
    
    # Features
    print("\n📊 Creando features PASO 6...")
    X_train_full, names = create_paso6_features(train_df['SMILES'].tolist(), train_groups)
    X_test_full, _ = create_paso6_features(test_df['SMILES'].tolist(), test_groups)
    print(f"   Total features: {X_train_full.shape[1]}")
    
    # Feature Selection con diferentes K
    K_VALUES = [500, 1000, 1500, 2000, 2500, 2757]  # 2757 = todas (baseline)
    
    results = {}
    
    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  🔍 K = {K} features")
        print(f"{'='*60}")
        
        # Seleccionar features
        if K < X_train_full.shape[1]:
            print("  Calculando mutual information...")
            selector = SelectKBest(score_func=f_regression, k=K)
            X_train = selector.fit_transform(X_train_full, y)
            X_test = selector.transform(X_test_full)
            selected_idx = selector.get_support(indices=True)
            print(f"  Features seleccionadas: {len(selected_idx)}")
        else:
            X_train = X_train_full
            X_test = X_test_full
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        oof_xgb = np.zeros(len(train_df))
        oof_lgb = np.zeros(len(train_df))
        oof_cat = np.zeros(len(train_df))
        
        test_preds_xgb = []
        test_preds_lgb = []
        test_preds_cat = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # XGB
            xgb_model = xgb.XGBRegressor(
                n_estimators=1800, max_depth=8, learning_rate=0.025,
                subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.5, reg_alpha=0.15,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
            oof_xgb[val_idx] = xgb_model.predict(X_val_s)
            test_preds_xgb.append(xgb_model.predict(X_test_s))
            
            # LGB
            lgb_model = lgb.LGBMRegressor(
                n_estimators=1800, max_depth=10, num_leaves=80,
                learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.5, reg_alpha=0.15,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)])
            oof_lgb[val_idx] = lgb_model.predict(X_val_s)
            test_preds_lgb.append(lgb_model.predict(X_test_s))
            
            # CAT
            cat_model = CatBoostRegressor(
                iterations=1800, depth=8, learning_rate=0.025,
                l2_leaf_reg=4, random_seed=RANDOM_STATE, verbose=False
            )
            cat_model.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=False)
            oof_cat[val_idx] = cat_model.predict(X_val_s)
            test_preds_cat.append(cat_model.predict(X_test_s))
        
        # Stacking
        meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
        meta_test = np.column_stack([
            np.mean(test_preds_xgb, axis=0),
            np.mean(test_preds_lgb, axis=0),
            np.mean(test_preds_cat, axis=0)
        ])
        
        meta = Ridge(alpha=1.0)
        meta.fit(meta_train, y)
        
        stacked_oof = meta.predict(meta_train)
        stacked_test = meta.predict(meta_test)
        
        stacked_mae = mean_absolute_error(y, stacked_oof)
        
        print(f"  XGB OOF:     {mean_absolute_error(y, oof_xgb):.2f}")
        print(f"  LGB OOF:     {mean_absolute_error(y, oof_lgb):.2f}")
        print(f"  CAT OOF:     {mean_absolute_error(y, oof_cat):.2f}")
        print(f"  Stacked OOF: {stacked_mae:.2f}")
        
        results[K] = {
            'stacked_mae': stacked_mae,
            'stacked_test': stacked_test
        }
    
    # Encontrar mejor K
    best_k = min(results.keys(), key=lambda k: results[k]['stacked_mae'])
    best_mae = results[best_k]['stacked_mae']
    
    print(f"\n{'='*60}")
    print(f"  📊 RESUMEN")
    print(f"{'='*60}")
    
    for k in sorted(results.keys()):
        marker = " ⭐" if k == best_k else ""
        print(f"  K={k:4d}: OOF MAE = {results[k]['stacked_mae']:.2f}{marker}")
    
    print(f"\n  Mejor K: {best_k} con MAE {best_mae:.2f}")
    print(f"  PASO 6 original: 26.64")
    
    # Guardar submissions para el mejor K
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Cargar ChemProp
    cp_path = DATA_PROCESSED / "chemprop_predictions.csv"
    cp_test = None
    if cp_path.exists():
        cp_df = pd.read_csv(cp_path)
        cp_test = cp_df['Tm'].values if 'Tm' in cp_df.columns else cp_df.iloc[:, 0].values
    
    # Guardar submissions para el mejor K
    best_test = results[best_k]['stacked_test']
    
    pd.DataFrame({'id': test_df['id'], 'Tm': best_test}).to_csv(
        SUBMISSIONS_DIR / f"fs_k{best_k}_stacked_{timestamp}.csv", index=False)
    
    if cp_test is not None:
        for w in [0.15, 0.20, 0.25, 0.30]:
            combined = w * cp_test + (1 - w) * best_test
            pd.DataFrame({'id': test_df['id'], 'Tm': combined}).to_csv(
                SUBMISSIONS_DIR / f"fs_k{best_k}_cp{int(w*100)}_{timestamp}.csv", index=False)
    
    print(f"\n✅ Submissions guardados en {SUBMISSIONS_DIR}")
    
    # Comparación final
    comparison = "✅ MEJOR" if best_mae < 26.64 else "❌ PEOR"
    print(f"\nComparación vs PASO 6: {comparison}")
    print(f"  PASO 6:      26.64")
    print(f"  Feature Sel: {best_mae:.2f}")


if __name__ == "__main__":
    main()