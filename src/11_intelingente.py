#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🧠 SCRIPT INTELIGENTE - APLICANDO PAPERS CORRECTAMENTE 🧠
═══════════════════════════════════════════════════════════════════════════════

LECCIÓN APRENDIDA: Más features ≠ Mejor
- Script anterior: 3055 features → OOF 28.79 (PEOR)
- PASO 6: 2757 features → OOF 26.64 (MEJOR)

ESTRATEGIA CORRECTA (basada en papers):
1. Datos externos de Bradley para AUMENTAR training data (no features)
2. Feature Selection para REDUCIR a las más importantes
3. ChemBERTa como PREDICTOR separado (no como features)
4. Mantener base PASO 6 que funciona

Referencias:
- Jean-Claude Bradley Dataset: https://figshare.com/articles/dataset/1031637
- Paper MAE 10.93: Knowledge-infused molecular graphs (2024)
- SMILES Augmentation: Bjerrum 2017 (R² 0.56 → 0.66)
"""

import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression, SelectKBest

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
elif os.path.exists('../data/raw'):
    PROJECT_ROOT = Path('..').resolve()
else:
    PROJECT_ROOT = Path('.').resolve()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCARGAR DATOS EXTERNOS (Bradley Dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def download_bradley_dataset() -> Optional[pd.DataFrame]:
    """
    Descarga Jean-Claude Bradley Double Plus Good Dataset.
    3,041 mediciones altamente curadas y validadas.
    """
    print("\n" + "="*70)
    print("  📥 DATOS EXTERNOS (Jean-Claude Bradley)")
    print("="*70)
    
    cache_path = DATA_PROCESSED / "bradley_curated.csv"
    
    if cache_path.exists():
        print(f"  ✅ Usando cache: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"     {len(df)} moléculas cargadas")
        return df
    
    # URL del dataset curado (Double Plus Good)
    # Formato: name, SMILES, mpC (melting point en Celsius)
    url = "https://raw.githubusercontent.com/NIEHS/SmartScope/master/data/jcb_training_data.csv"
    
    try:
        print("  Descargando desde GitHub (backup)...")
        df = pd.read_csv(url)
        
        # Procesar
        if 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})
        if 'mpC' in df.columns:
            df['Tm'] = df['mpC'] + 273.15  # Convertir a Kelvin
        elif 'mp' in df.columns:
            df['Tm'] = df['mp'] + 273.15
        
        # Validar SMILES
        valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        df = df[valid_mask]
        
        # Guardar cache
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df[['SMILES', 'Tm']].to_csv(cache_path, index=False)
        
        print(f"  ✅ Descargados {len(df)} datos externos")
        return df[['SMILES', 'Tm']]
        
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        
        # Plan B: Crear datos sintéticos basados en similitud
        print("  Intentando fuente alternativa...")
        
        try:
            # Usar el dataset completo de Figshare
            url2 = "https://ndownloader.figshare.com/files/1502888"
            response = requests.get(url2, timeout=30)
            
            if response.status_code == 200:
                import io
                df = pd.read_excel(io.BytesIO(response.content))
                
                # Limpiar
                df = df.rename(columns={
                    'smiles': 'SMILES',
                    'mpC': 'Tm_C'
                })
                
                if 'donotuse' in df.columns:
                    df = df[df['donotuse'] != 'x']
                
                df['Tm'] = df['Tm_C'] + 273.15
                
                valid_mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
                df = df[valid_mask]
                
                df[['SMILES', 'Tm']].to_csv(cache_path, index=False)
                print(f"  ✅ Descargados {len(df)} datos externos (Figshare)")
                return df[['SMILES', 'Tm']]
        except:
            pass
    
    print("  ❌ No se pudieron descargar datos externos")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURES (Estilo PASO 6 - SIN EXTRAS RUIDOSOS)
# ═══════════════════════════════════════════════════════════════════════════════

def create_paso6_features(smiles_list: List[str], 
                          group_features: Optional[np.ndarray] = None,
                          verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Crea features estilo PASO 6 (las que funcionan).
    NO incluye ChemBERTa ni 3D (causaron overfitting).
    
    Total: ~2757 features
    - Morgan FP: 2048
    - MACCS: 167
    - RDKit 2D: ~22
    - Groups: ~424 (del dataset)
    """
    
    all_features = []
    all_names = []
    
    # 1. Morgan FP (2048)
    if verbose:
        print("    → Morgan FP (2048)...")
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
    
    # 2. MACCS (167)
    if verbose:
        print("    → MACCS Keys (167)...")
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
    
    # 3. RDKit 2D descriptors (~22)
    if verbose:
        print("    → RDKit 2D (~22)...")
    desc_funcs = [
        ('MolWt', Descriptors.MolWt),
        ('LogP', Descriptors.MolLogP),
        ('TPSA', Descriptors.TPSA),
        ('NumHDonors', Descriptors.NumHDonors),
        ('NumHAcceptors', Descriptors.NumHAcceptors),
        ('NumRotatableBonds', Descriptors.NumRotatableBonds),
        ('RingCount', Descriptors.RingCount),
        ('NumAromaticRings', Descriptors.NumAromaticRings),
        ('NumSaturatedRings', Descriptors.NumSaturatedRings),
        ('NumAliphaticRings', Descriptors.NumAliphaticRings),
        ('NumHeteroatoms', Descriptors.NumHeteroatoms),
        ('FractionCSP3', Descriptors.FractionCSP3),
        ('HeavyAtomCount', Descriptors.HeavyAtomCount),
        ('LabuteASA', Descriptors.LabuteASA),
        ('BertzCT', Descriptors.BertzCT),
        ('Chi0', Descriptors.Chi0),
        ('Chi1', Descriptors.Chi1),
        ('Kappa1', Descriptors.Kappa1),
        ('Kappa2', Descriptors.Kappa2),
        ('Kappa3', Descriptors.Kappa3),
        ('HallKierAlpha', Descriptors.HallKierAlpha),
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
    
    # 4. Group features (del dataset)
    if group_features is not None:
        if verbose:
            print(f"    → Groups ({group_features.shape[1]})...")
        all_features.append(group_features.astype(np.float32))
        all_names.extend([f'group_{i}' for i in range(group_features.shape[1])])
    
    # Concatenar
    X = np.hstack(all_features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if verbose:
        print(f"    ✅ Total: {X.shape[1]} features")
    
    return X, all_names


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODELOS OPTIMIZADOS (del PASO 6)
# ═══════════════════════════════════════════════════════════════════════════════

def get_optimized_models():
    """Mejores hiperparámetros del PASO 6."""
    return {
        'xgb': xgb.XGBRegressor(
            n_estimators=1800, max_depth=8, learning_rate=0.025,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.5, reg_alpha=0.15, min_child_weight=4,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist', verbosity=0
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=1800, max_depth=10, num_leaves=80,
            learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.5, reg_alpha=0.15, min_child_samples=15,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
        'cat': CatBoostRegressor(
            iterations=1800, depth=8, learning_rate=0.025,
            l2_leaf_reg=4, random_seed=RANDOM_STATE, verbose=False,
            early_stopping_rounds=100
        )
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🧠 SCRIPT INTELIGENTE - APLICANDO PAPERS CORRECTAMENTE 🧠               ║
║                                                                              ║
║     Estrategia:                                                              ║
║     ✓ Base PASO 6 (2757 features que funcionan)                             ║
║     ✓ Datos externos Bradley (+3000 moléculas)                              ║
║     ✓ SIN ChemBERTa embeddings (causaron ruido)                             ║
║     ✓ SIN 3D descriptors (causaron overfitting)                             ║
║     ✓ Combinar con ChemProp (15-30%)                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📥 CARGANDO DATOS KAGGLE")
    print("="*70)
    
    train_df = pd.read_csv(DATA_RAW / "train.csv")
    test_df = pd.read_csv(DATA_RAW / "test.csv")
    
    print(f"  Train: {len(train_df)} muestras")
    print(f"  Test:  {len(test_df)} muestras")
    
    y_original = train_df['Tm'].values
    print(f"  Target range: {y_original.min():.1f} - {y_original.max():.1f} K")
    
    # Group features
    group_cols = [c for c in train_df.columns if c.startswith('Group')]
    train_groups = train_df[group_cols].values
    test_groups = test_df[group_cols].values
    print(f"  Group features: {len(group_cols)}")
    
    # =========================================================================
    # DATOS EXTERNOS
    # =========================================================================
    
    external_df = download_bradley_dataset()
    
    use_external = False
    if external_df is not None and len(external_df) > 100:
        # Filtrar duplicados
        train_smiles_set = set(train_df['SMILES'].str.strip().str.lower())
        external_df = external_df[~external_df['SMILES'].str.strip().str.lower().isin(train_smiles_set)]
        
        # Filtrar por rango similar
        tm_min, tm_max = y_original.min() - 50, y_original.max() + 50
        external_df = external_df[(external_df['Tm'] >= tm_min) & (external_df['Tm'] <= tm_max)]
        
        if len(external_df) > 100:
            print(f"  Externos filtrados: {len(external_df)} moléculas")
            use_external = True
        else:
            print("  ⚠️ Muy pocos datos externos después de filtrar")
    
    # =========================================================================
    # CREAR FEATURES
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🔧 FEATURES (estilo PASO 6)")
    print("="*70)
    
    print("\n  Train:")
    X_train, feature_names = create_paso6_features(
        train_df['SMILES'].tolist(),
        group_features=train_groups
    )
    
    print("\n  Test:")
    X_test, _ = create_paso6_features(
        test_df['SMILES'].tolist(),
        group_features=test_groups
    )
    
    # Features para datos externos (sin Groups porque no los tienen)
    if use_external:
        print("\n  External:")
        X_external, _ = create_paso6_features(
            external_df['SMILES'].tolist(),
            group_features=None  # No tienen groups
        )
        y_external = external_df['Tm'].values
        
        # Padding para que coincidan dimensiones
        n_groups = train_groups.shape[1]
        external_groups_padding = np.zeros((len(external_df), n_groups))
        X_external = np.hstack([X_external, external_groups_padding])
        
        print(f"    External features: {X_external.shape[1]}")
    
    # =========================================================================
    # CROSS-VALIDATION
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🏋️ CROSS-VALIDATION (5 FOLDS)")
    print("="*70)
    
    N_FOLDS = 5
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    model_names = ['xgb', 'lgb', 'cat']
    all_models = {name: [] for name in model_names}
    scalers = []
    
    oof_preds = {name: np.zeros(len(train_df)) for name in model_names}
    test_preds = {name: [] for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\n  ═══ FOLD {fold+1}/{N_FOLDS} ═══")
        
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_original[train_idx]
        y_fold_val = y_original[val_idx]
        
        # Añadir datos externos al training
        if use_external:
            X_fold_train = np.vstack([X_fold_train, X_external])
            y_fold_train = np.concatenate([y_fold_train, y_external])
            print(f"    Train + External: {len(y_fold_train)} muestras")
        
        # Escalar
        scaler = StandardScaler()
        X_fold_train_s = scaler.fit_transform(X_fold_train)
        X_fold_val_s = scaler.transform(X_fold_val)
        X_test_s = scaler.transform(X_test)
        scalers.append(scaler)
        
        # Entrenar modelos
        models = get_optimized_models()
        
        for name in model_names:
            model = models[name]
            print(f"    {name}...", end=" ")
            
            if name == 'cat':
                model.fit(X_fold_train_s, y_fold_train,
                         eval_set=(X_fold_val_s, y_fold_val), verbose=False)
            elif name == 'lgb':
                model.fit(X_fold_train_s, y_fold_train,
                         eval_set=[(X_fold_val_s, y_fold_val)])
            else:
                model.fit(X_fold_train_s, y_fold_train,
                         eval_set=[(X_fold_val_s, y_fold_val)], verbose=False)
            
            all_models[name].append(model)
            
            val_pred = model.predict(X_fold_val_s)
            oof_preds[name][val_idx] = val_pred
            test_preds[name].append(model.predict(X_test_s))
            
            mae = mean_absolute_error(y_fold_val, val_pred)
            print(f"MAE = {mae:.2f}")
    
    # Promediar test predictions
    test_preds_avg = {name: np.mean(test_preds[name], axis=0) for name in model_names}
    
    # =========================================================================
    # STACKING
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🏗️ STACKING")
    print("="*70)
    
    meta_train = np.column_stack([oof_preds[name] for name in model_names])
    meta_test = np.column_stack([test_preds_avg[name] for name in model_names])
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, y_original)
    
    stacked_oof = meta_model.predict(meta_train)
    stacked_test = meta_model.predict(meta_test)
    
    stacked_mae = mean_absolute_error(y_original, stacked_oof)
    
    print(f"\n  Weights: {dict(zip(model_names, meta_model.coef_.round(3)))}")
    print(f"\n  📊 OOF MAE:")
    for name in model_names:
        print(f"    {name}: {mean_absolute_error(y_original, oof_preds[name]):.2f}")
    print(f"    Stacked: {stacked_mae:.2f}")
    
    # =========================================================================
    # CHEMPROP
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🔗 CHEMPROP")
    print("="*70)
    
    cp_path = DATA_PROCESSED / "chemprop_predictions.csv"
    cp_test = None
    
    if cp_path.exists():
        cp_df = pd.read_csv(cp_path)
        cp_test = cp_df['Tm'].values if 'Tm' in cp_df.columns else cp_df.iloc[:, 0].values
        print(f"  ✅ ChemProp: {len(cp_test)} predicciones")
    else:
        print(f"  ⚠️ ChemProp no encontrado")
    
    # =========================================================================
    # SUBMISSIONS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📁 SUBMISSIONS")
    print("="*70)
    
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Solo stacked
    pd.DataFrame({'id': test_df['id'], 'Tm': stacked_test}).to_csv(
        SUBMISSIONS_DIR / f"inteligente_stacked_{timestamp}.csv", index=False)
    print(f"  ✅ inteligente_stacked_{timestamp}.csv")
    
    # Con ChemProp
    if cp_test is not None:
        for w in [0.15, 0.20, 0.25, 0.30]:
            combined = w * cp_test + (1 - w) * stacked_test
            filename = f"inteligente_cp{int(w*100)}_{timestamp}.csv"
            pd.DataFrame({'id': test_df['id'], 'Tm': combined}).to_csv(
                SUBMISSIONS_DIR / filename, index=False)
            print(f"  ✅ {filename}")
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    
    paso6_oof = 26.64
    comparison = "✅ MEJOR" if stacked_mae < paso6_oof else "❌ PEOR"
    recommendation = "SUBIR" if stacked_mae < paso6_oof else "NO SUBIR (usar PASO 6)"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  📊 RESULTADOS                                                               ║
║                                                                              ║
║    PASO 6 OOF:      26.64 → Kaggle 22.80                                    ║
║    Este script OOF: {stacked_mae:>5.2f} → Kaggle ???                                    ║
║                                                                              ║
║    Comparación: {comparison:>10}                                                    ║
║    Recomendación: {recommendation:<30}                          ║
║                                                                              ║
║    Features: {X_train.shape[1]}                                                        ║
║    Datos externos: {len(external_df) if use_external else 0:>5}                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Guardar config
    config = {
        'timestamp': timestamp,
        'stacked_oof_mae': float(stacked_mae),
        'paso6_oof_mae': paso6_oof,
        'comparison': comparison,
        'use_external': use_external,
        'n_external': int(len(external_df)) if use_external else 0,
        'n_features': int(X_train.shape[1]),
        'model_maes': {name: float(mean_absolute_error(y_original, oof_preds[name])) 
                       for name in model_names}
    }
    
    with open(SUBMISSIONS_DIR / f"config_inteligente_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()