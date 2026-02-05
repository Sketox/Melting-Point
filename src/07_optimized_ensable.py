"""
═══════════════════════════════════════════════════════════════════════════════
                PASO 5: Ensemble Optimizado + Optuna Tuning
                MeltingPoint Kaggle Competition
═══════════════════════════════════════════════════════════════════════════════

MEJORAS RESPECTO AL PASO 4:
1. ❌ Quitar Neural Network (empeoraba: MAE 29.72)
2. ✅ Agregar más fingerprints (AtomPair, TopologicalTorsion, MorganCounts)
3. ✅ Hyperparameter tuning con Optuna
4. ✅ Usar TODOS los Groups del dataset (424 columnas)

FEATURES USADAS:
- SMILES text features (17) - del string SMILES
- RDKit Descriptors (~200) - calculados desde SMILES
- Morgan Fingerprints (2048) - calculados desde SMILES
- Morgan Counts (1024) - con conteos, no solo bits
- MACCS Keys (167) - calculados desde SMILES
- Atom Pair FP (1024) - pares de átomos
- TopologicalTorsion FP (1024) - secuencias de 4 átomos
- Group Features (~337) - ★ DEL DATASET ORIGINAL ★

OBJETIVO: Bajar de MAE 22.94 a ~20-21

Autor: Sketo
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Suprimir warnings de RDKit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

N_FOLDS = 5
RANDOM_STATE = 42
USE_OPTUNA = True  # Activar/desactivar tuning
OPTUNA_TRIALS = 30  # Número de trials para Optuna


# ═══════════════════════════════════════════════════════════════════════════════
# FINGERPRINTS Y FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Morgan Fingerprints (ECFP4) - subestructuras circulares."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    
    print(f"  📊 Morgan FP (radius={radius}, bits={n_bits})...", end=" ", flush=True)
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(list(fp))
        else:
            fps.append([0] * n_bits)
    
    df = pd.DataFrame(fps, columns=[f"mfp_{i}" for i in range(n_bits)])
    print(f"✓ {n_bits}")
    return df


def get_morgan_counts(smiles_list, radius=2, n_bits=1024):
    """Morgan Fingerprints con CONTEOS (no solo presencia/ausencia)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    
    print(f"  📊 Morgan Counts (radius={radius}, bits={n_bits})...", end=" ", flush=True)
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits)
            for idx, count in fp.GetNonzeroElements().items():
                arr[idx] = count
            fps.append(arr)
        else:
            fps.append([0] * n_bits)
    
    df = pd.DataFrame(fps, columns=[f"mfp_count_{i}" for i in range(n_bits)])
    print(f"✓ {n_bits}")
    return df


def get_maccs_keys(smiles_list):
    """MACCS Keys - 167 bits predefinidos."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    
    print(f"  📊 MACCS Keys...", end=" ", flush=True)
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(list(fp))
        else:
            fps.append([0] * 167)
    
    df = pd.DataFrame(fps, columns=[f"maccs_{i}" for i in range(167)])
    print(f"✓ 167")
    return df


def get_atom_pair_fingerprints(smiles_list, n_bits=1024):
    """Atom Pair Fingerprints - pares de átomos y distancia entre ellos."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    
    print(f"  📊 Atom Pair FP (bits={n_bits})...", end=" ", flush=True)
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            fps.append(list(fp))
        else:
            fps.append([0] * n_bits)
    
    df = pd.DataFrame(fps, columns=[f"ap_{i}" for i in range(n_bits)])
    print(f"✓ {n_bits}")
    return df


def get_topological_torsion_fingerprints(smiles_list, n_bits=1024):
    """Topological Torsion FP - secuencias de 4 átomos."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    
    print(f"  📊 Topological Torsion FP (bits={n_bits})...", end=" ", flush=True)
    
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
            fps.append(list(fp))
        else:
            fps.append([0] * n_bits)
    
    df = pd.DataFrame(fps, columns=[f"tt_{i}" for i in range(n_bits)])
    print(f"✓ {n_bits}")
    return df


def get_rdkit_descriptors(smiles_list):
    """Descriptores físico-químicos de RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    
    print(f"  📊 RDKit Descriptors...", end=" ", flush=True)
    
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                desc = calculator.CalcDescriptors(mol)
                features.append(desc)
            except:
                features.append([np.nan] * len(descriptor_names))
        else:
            features.append([np.nan] * len(descriptor_names))
    
    df = pd.DataFrame(features, columns=[f"rdkit_{name}" for name in descriptor_names])
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.std() > 0]
    df = df.fillna(df.median())
    
    print(f"✓ {df.shape[1]}")
    return df


def get_smiles_features(smiles_list):
    """Features básicas extraídas del string SMILES."""
    print(f"  📊 SMILES Features...", end=" ", flush=True)
    
    features = []
    for smi in smiles_list:
        smi = str(smi)
        feat = {
            "smi_len": len(smi),
            "smi_rings": sum(c.isdigit() for c in smi),
            "smi_branches": smi.count("("),
            "smi_double": smi.count("="),
            "smi_triple": smi.count("#"),
            "smi_aromatic": sum(c.islower() for c in smi),
            "smi_N": smi.count("N") + smi.count("n"),
            "smi_O": smi.count("O") + smi.count("o"),
            "smi_F": smi.count("F"),
            "smi_Cl": smi.count("Cl"),
            "smi_Br": smi.count("Br"),
            "smi_S": smi.count("S") + smi.count("s"),
            "smi_P": smi.count("P"),
            "smi_I": smi.count("I"),
            "smi_ring_density": sum(c.isdigit() for c in smi) / max(len(smi), 1),
            "smi_branch_density": smi.count("(") / max(len(smi), 1),
            "smi_hetero_ratio": (smi.count("N") + smi.count("O") + smi.count("S") + 
                                 smi.count("F") + smi.count("Cl")) / max(len(smi), 1),
        }
        features.append(feat)
    
    df = pd.DataFrame(features)
    print(f"✓ {df.shape[1]}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAR FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_all_features(train_df, test_df):
    """Prepara todas las features combinadas."""
    
    print("\n" + "="*70)
    print("  EXTRAYENDO FEATURES")
    print("="*70 + "\n")
    
    train_smiles = train_df["SMILES"].tolist()
    test_smiles = test_df["SMILES"].tolist()
    
    feature_dfs_train = []
    feature_dfs_test = []
    
    # 1. SMILES features básicas
    feature_dfs_train.append(get_smiles_features(train_smiles))
    feature_dfs_test.append(get_smiles_features(test_smiles))
    
    # 2. RDKit Descriptors
    feature_dfs_train.append(get_rdkit_descriptors(train_smiles))
    feature_dfs_test.append(get_rdkit_descriptors(test_smiles))
    
    # 3. Morgan Fingerprints (bits)
    feature_dfs_train.append(get_morgan_fingerprints(train_smiles, radius=2, n_bits=2048))
    feature_dfs_test.append(get_morgan_fingerprints(test_smiles, radius=2, n_bits=2048))
    
    # 4. Morgan Counts (con conteos)
    feature_dfs_train.append(get_morgan_counts(train_smiles, radius=2, n_bits=1024))
    feature_dfs_test.append(get_morgan_counts(test_smiles, radius=2, n_bits=1024))
    
    # 5. MACCS Keys
    feature_dfs_train.append(get_maccs_keys(train_smiles))
    feature_dfs_test.append(get_maccs_keys(test_smiles))
    
    # 6. Atom Pair Fingerprints
    feature_dfs_train.append(get_atom_pair_fingerprints(train_smiles, n_bits=1024))
    feature_dfs_test.append(get_atom_pair_fingerprints(test_smiles, n_bits=1024))
    
    # 7. Topological Torsion
    feature_dfs_train.append(get_topological_torsion_fingerprints(train_smiles, n_bits=1024))
    feature_dfs_test.append(get_topological_torsion_fingerprints(test_smiles, n_bits=1024))
    
    # ══════════════════════════════════════════════════════════════════════════
    # 8. ★★★ GROUP FEATURES DEL DATASET ORIGINAL ★★★
    # ══════════════════════════════════════════════════════════════════════════
    group_cols = [c for c in train_df.columns if c.startswith("Group")]
    train_group = train_df[group_cols].copy()
    test_group = test_df[group_cols].copy()
    
    # Eliminar columnas con varianza 0 (no aportan información)
    nonzero_cols = train_group.columns[train_group.var() > 0]
    train_group = train_group[nonzero_cols]
    test_group = test_group[nonzero_cols]
    
    print(f"  📊 ★ GROUP FEATURES (DATASET) ★... ✓ {train_group.shape[1]} de {len(group_cols)}")
    
    feature_dfs_train.append(train_group.reset_index(drop=True))
    feature_dfs_test.append(test_group.reset_index(drop=True))
    
    # Combinar todas las features
    X_train = pd.concat([df.reset_index(drop=True) for df in feature_dfs_train], axis=1)
    X_test = pd.concat([df.reset_index(drop=True) for df in feature_dfs_test], axis=1)
    
    # Asegurar mismas columnas
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Limpiar datos
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  ✅ TOTAL FEATURES: {X_train.shape[1]}")
    print(f"  ═══════════════════════════════════════")
    
    return X_train, X_test


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_xgboost(X, y, n_trials=30):
    """Optimiza hiperparámetros de XGBoost con Optuna."""
    import optuna
    from xgboost import XGBRegressor
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'tree_method': 'hist',
        }
        
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params


def optimize_lightgbm(X, y, n_trials=30):
    """Optimiza hiperparámetros de LightGBM con Optuna."""
    import optuna
    from lightgbm import LGBMRegressor
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'num_leaves': trial.suggest_int('num_leaves', 30, 120),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params


def optimize_catboost(X, y, n_trials=30):
    """Optimiza hiperparámetros de CatBoost con Optuna (CV manual)."""
    import optuna
    from catboost import CatBoostRegressor
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Convertir a numpy si es DataFrame
    X_np = X.values if hasattr(X, 'values') else X
    y_np = y if isinstance(y, np.ndarray) else np.array(y)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 800, 2000),
            'depth': trial.suggest_int('depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 8.0),
            'random_seed': RANDOM_STATE,
            'verbose': False,
        }
        
        # CV manual para evitar incompatibilidad con sklearn
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        maes = []
        
        for train_idx, val_idx in kf.split(X_np):
            X_tr, X_val = X_np[train_idx], X_np[val_idx]
            y_tr, y_val = y_np[train_idx], y_np[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
            
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            maes.append(mae)
        
        return np.mean(maes)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(X_train, y_train, X_test, xgb_params=None, lgbm_params=None, cat_params=None):
    """Entrena los 3 modelos con los parámetros dados."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    
    # Parámetros por defecto mejorados
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.025,
            'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_lambda': 2.0,
            'reg_alpha': 0.1, 'min_child_weight': 4, 'random_state': RANDOM_STATE,
            'n_jobs': -1, 'tree_method': 'hist'
        }
    
    if lgbm_params is None:
        lgbm_params = {
            'n_estimators': 1500, 'max_depth': 10, 'num_leaves': 64,
            'learning_rate': 0.025, 'subsample': 0.85, 'colsample_bytree': 0.85,
            'reg_lambda': 2.0, 'reg_alpha': 0.1, 'min_child_samples': 15,
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
        }
    
    if cat_params is None:
        cat_params = {
            'iterations': 1500, 'depth': 8, 'learning_rate': 0.025,
            'l2_leaf_reg': 4, 'random_seed': RANDOM_STATE, 'verbose': False
        }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    for name, Model, params in [
        ('XGBoost', XGBRegressor, xgb_params),
        ('LightGBM', LGBMRegressor, lgbm_params),
        ('CatBoost', CatBoostRegressor, cat_params),
    ]:
        print(f"\n  🔧 Entrenando {name}...")
        
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        maes = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = Model(**params)
            
            if name == 'CatBoost':
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=100)
            elif name == 'LightGBM':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            test_preds += model.predict(X_test) / N_FOLDS
            
            mae = mean_absolute_error(y_val, val_pred)
            maes.append(mae)
            print(f"      Fold {fold+1}: MAE = {mae:.2f}")
        
        oof_mae = mean_absolute_error(y_train, oof_preds)
        print(f"      📊 {name} OOF MAE: {oof_mae:.2f} (±{np.std(maes):.2f})")
        
        results[name] = {
            'oof': oof_preds,
            'test': test_preds,
            'mae': oof_mae
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZAR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_ensemble_weights(y_true, predictions_dict):
    """Encuentra los pesos óptimos usando scipy."""
    from scipy.optimize import minimize
    
    names = list(predictions_dict.keys())
    preds = np.array([predictions_dict[name]['oof'] for name in names])
    
    def objective(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))
        ensemble = np.sum(weights.reshape(-1, 1) * preds, axis=0)
        return mean_absolute_error(y_true, ensemble)
    
    # Probar varias inicializaciones
    best_mae = float('inf')
    best_weights = None
    
    for _ in range(20):
        initial = np.random.random(len(names))
        initial = initial / initial.sum()
        
        result = minimize(objective, initial, method='Nelder-Mead', 
                         options={'maxiter': 2000})
        
        if result.fun < best_mae:
            best_mae = result.fun
            best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    
    return dict(zip(names, best_weights)), best_mae


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🚀 PASO 5: Ensemble Optimizado + Optuna Tuning                       ║
║        MeltingPoint Kaggle Competition                                       ║
║                                                                              ║
║        Mejor actual: MAE 22.94                                              ║
║        Objetivo: MAE < 22                                                   ║
║                                                                              ║
║        Mejoras:                                                              ║
║        • Sin Neural Network (empeoraba)                                     ║
║        • Más fingerprints (AtomPair, TopologicalTorsion, MorganCounts)      ║
║        • Hyperparameter tuning con Optuna                                   ║
║        • Usando TODOS los Group features del dataset                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Cargar datos
    print("\n" + "="*70)
    print("  CARGANDO DATOS")
    print("="*70)
    
    train_df = pd.read_csv(DATA_RAW / "train.csv")
    test_df = pd.read_csv(DATA_RAW / "test.csv")
    y_train = train_df["Tm"].values
    
    group_cols_count = len([c for c in train_df.columns if c.startswith('Group')])
    
    print(f"\n  Train: {len(train_df)} muestras")
    print(f"  Test: {len(test_df)} muestras")
    print(f"  Group columns en dataset: {group_cols_count}")
    
    # Preparar features
    X_train, X_test = prepare_all_features(train_df, test_df)
    
    # Optuna tuning
    xgb_params = None
    lgbm_params = None
    cat_params = None
    
    if USE_OPTUNA:
        print("\n" + "="*70)
        print(f"  OPTUNA HYPERPARAMETER TUNING ({OPTUNA_TRIALS} trials cada uno)")
        print("="*70)
        
        print("\n  🔍 Optimizando XGBoost...")
        xgb_params = optimize_xgboost(X_train, y_train, n_trials=OPTUNA_TRIALS)
        xgb_params['random_state'] = RANDOM_STATE
        xgb_params['n_jobs'] = -1
        xgb_params['tree_method'] = 'hist'
        print(f"      ✓ Mejor configuración encontrada")
        
        print("\n  🔍 Optimizando LightGBM...")
        lgbm_params = optimize_lightgbm(X_train, y_train, n_trials=OPTUNA_TRIALS)
        lgbm_params['random_state'] = RANDOM_STATE
        lgbm_params['n_jobs'] = -1
        lgbm_params['verbose'] = -1
        print(f"      ✓ Mejor configuración encontrada")
        
        print("\n  🔍 Optimizando CatBoost...")
        cat_params = optimize_catboost(X_train, y_train, n_trials=OPTUNA_TRIALS)
        cat_params['random_seed'] = RANDOM_STATE
        cat_params['verbose'] = False
        print(f"      ✓ Mejor configuración encontrada")
    
    # Entrenar modelos
    print("\n" + "="*70)
    print("  ENTRENANDO MODELOS (5-Fold CV)")
    print("="*70)
    
    results = train_models(X_train, y_train, X_test, xgb_params, lgbm_params, cat_params)
    
    # Optimizar ensemble
    print("\n" + "="*70)
    print("  OPTIMIZANDO ENSEMBLE")
    print("="*70)
    
    weights, ensemble_mae = optimize_ensemble_weights(y_train, results)
    
    print(f"\n  Pesos óptimos:")
    for name, weight in weights.items():
        print(f"      {name}: {weight:.3f} ({weight*100:.1f}%)")
    print(f"\n  📊 Ensemble OOF MAE: {ensemble_mae:.2f}")
    
    # Crear predicciones del ensemble
    ensemble_test = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_test += weight * results[name]['test']
    
    # Cargar predicciones de ChemProp
    print("\n" + "="*70)
    print("  COMBINANDO CON CHEMPROP")
    print("="*70)
    
    cp_path = DATA_PROCESSED / "chemprop_predictions.csv"
    cp_test = None
    if cp_path.exists():
        cp_preds = pd.read_csv(cp_path)
        cp_test = cp_preds["Tm"].values if "Tm" in cp_preds.columns else cp_preds.iloc[:, 0].values
        print(f"\n  ✓ ChemProp cargado: {len(cp_test)} predicciones")
    else:
        print(f"\n  ⚠️ No se encontró ChemProp en {cp_path}")
    
    # Guardar submissions
    print("\n" + "="*70)
    print("  GUARDANDO SUBMISSIONS")
    print("="*70)
    
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Ensemble solo
    pd.DataFrame({
        "id": test_df["id"],
        "Tm": ensemble_test
    }).to_csv(SUBMISSION_DIR / "submission_optimized_ensemble.csv", index=False)
    print(f"\n  ✓ submission_optimized_ensemble.csv")
    
    # 2. Modelos individuales
    for name in results:
        filename = f"submission_{name.lower()}_v3.csv"
        pd.DataFrame({
            "id": test_df["id"],
            "Tm": results[name]['test']
        }).to_csv(SUBMISSION_DIR / filename, index=False)
        print(f"  ✓ {filename}")
    
    # 3. Combinaciones con ChemProp
    if cp_test is not None:
        for w_cp in [0.0, 0.10, 0.15, 0.20]:
            combined = w_cp * cp_test + (1 - w_cp) * ensemble_test
            filename = f"submission_opt_cp{int(w_cp*100)}.csv"
            pd.DataFrame({
                "id": test_df["id"],
                "Tm": combined
            }).to_csv(SUBMISSION_DIR / filename, index=False)
            print(f"  ✓ {filename}")
    
    # Resumen final
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ✅ PASO 5 COMPLETADO                                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 RESULTADOS OOF:                                                         ║
║      • XGBoost:   {results['XGBoost']['mae']:>6.2f}                                              ║
║      • LightGBM:  {results['LightGBM']['mae']:>6.2f}                                              ║
║      • CatBoost:  {results['CatBoost']['mae']:>6.2f}                                              ║
║      • Ensemble:  {ensemble_mae:>6.2f}                                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📁 SUBMISSIONS:                                                             ║
║      • submission_optimized_ensemble.csv (PRINCIPAL)                        ║
║      • submission_opt_cp0/10/15/20.csv (con ChemProp)                       ║
║                                                                              ║
║  🎯 Sube todos a Kaggle y compara!                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Guardar parámetros
    if USE_OPTUNA:
        import json
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "best_params_v2.json", 'w') as f:
            json.dump({
                'xgboost': xgb_params,
                'lightgbm': lgbm_params,
                'catboost': cat_params,
                'weights': {k: float(v) for k, v in weights.items()}
            }, f, indent=2)
        print(f"\n  💾 Parámetros guardados en: {MODELS_DIR / 'best_params_v2.json'}")


if __name__ == "__main__":
    main()