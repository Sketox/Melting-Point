"""
═══════════════════════════════════════════════════════════════════════════════
                PASO 4: Modelos Avanzados
                MeltingPoint Kaggle Competition
═══════════════════════════════════════════════════════════════════════════════

OBJETIVO: Bajar de MAE 23.4 a ~18-20

NUEVAS TÉCNICAS:
1. Morgan Fingerprints (ECFP4) - 2048 bits
2. MACCS Keys - 167 bits  
3. CatBoost - Gradient boosting adicional
4. Neural Network - MLP con todas las features
5. Ensemble optimizado con todos los modelos

Autor: Sketo
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

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


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES: MORGAN FINGERPRINTS
# ═══════════════════════════════════════════════════════════════════════════════

def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Genera Morgan Fingerprints (ECFP4).
    
    Morgan FPs capturan subestructuras circulares alrededor de cada átomo.
    Son muy efectivos para propiedades moleculares.
    
    - radius=2 → ECFP4 (diámetro 4)
    - n_bits=2048 → Vector de 2048 dimensiones
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    print(f"\n  📊 Generando Morgan Fingerprints (radius={radius}, bits={n_bits})...")
    
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(list(fp))
        else:
            fingerprints.append([0] * n_bits)
    
    columns = [f"morgan_{i}" for i in range(n_bits)]
    df = pd.DataFrame(fingerprints, columns=columns)
    
    print(f"      ✓ {df.shape[1]} features Morgan FP")
    return df


def get_maccs_keys(smiles_list):
    """
    Genera MACCS Keys (166 bits).
    
    MACCS son fingerprints predefinidos basados en subestructuras
    químicas conocidas. Complementan a Morgan FPs.
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    
    print(f"\n  📊 Generando MACCS Keys...")
    
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.append(list(fp))
        else:
            fingerprints.append([0] * 167)
    
    columns = [f"maccs_{i}" for i in range(167)]
    df = pd.DataFrame(fingerprints, columns=columns)
    
    print(f"      ✓ {df.shape[1]} features MACCS")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES: RDKIT DESCRIPTORS (del paso anterior)
# ═══════════════════════════════════════════════════════════════════════════════

def get_rdkit_descriptors(smiles_list):
    """Extrae ~200 descriptores RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    
    print(f"\n  📊 Extrayendo descriptores RDKit...")
    
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
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
    
    print(f"      ✓ {df.shape[1]} descriptores RDKit")
    return df


def get_smiles_features(smiles_list):
    """Features básicas del string SMILES."""
    print(f"\n  📊 Extrayendo features SMILES...")
    
    features = []
    for smi in smiles_list:
        smi = str(smi)
        feat = {
            "smiles_len": len(smi),
            "n_rings": sum(c.isdigit() for c in smi),
            "n_branches": smi.count("("),
            "n_double_bonds": smi.count("="),
            "n_triple_bonds": smi.count("#"),
            "n_aromatic": sum(c.islower() for c in smi),
            "count_N": smi.count("N") + smi.count("n"),
            "count_O": smi.count("O") + smi.count("o"),
            "count_F": smi.count("F"),
            "count_Cl": smi.count("Cl"),
            "count_S": smi.count("S") + smi.count("s"),
            "count_Br": smi.count("Br"),
            "heteroatom_ratio": (smi.count("N") + smi.count("O") + smi.count("S") + 
                                 smi.count("F") + smi.count("Cl")) / max(len(smi), 1),
        }
        features.append(feat)
    
    df = pd.DataFrame(features)
    print(f"      ✓ {df.shape[1]} features SMILES")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PREPARAR TODAS LAS FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_all_features(train_df, test_df):
    """Prepara todas las features combinadas."""
    
    print("\n" + "="*70)
    print("  PREPARANDO FEATURES")
    print("="*70)
    
    train_smiles = train_df["SMILES"].tolist()
    test_smiles = test_df["SMILES"].tolist()
    
    # 1. SMILES features
    train_smi_feat = get_smiles_features(train_smiles)
    test_smi_feat = get_smiles_features(test_smiles)
    
    # 2. RDKit descriptors
    train_rdkit = get_rdkit_descriptors(train_smiles)
    test_rdkit = get_rdkit_descriptors(test_smiles)
    
    # 3. Morgan Fingerprints
    train_morgan = get_morgan_fingerprints(train_smiles)
    test_morgan = get_morgan_fingerprints(test_smiles)
    
    # 4. MACCS Keys
    train_maccs = get_maccs_keys(train_smiles)
    test_maccs = get_maccs_keys(test_smiles)
    
    # 5. Group features (originales)
    group_cols = [c for c in train_df.columns if c.startswith("Group")]
    train_group = train_df[group_cols]
    test_group = test_df[group_cols]
    
    # Eliminar columnas con varianza 0
    nonzero_cols = train_group.columns[train_group.var() > 0]
    train_group = train_group[nonzero_cols]
    test_group = test_group[nonzero_cols]
    
    print(f"\n  📊 Features Group: {train_group.shape[1]}")
    
    # Combinar todas
    X_train = pd.concat([
        train_smi_feat.reset_index(drop=True),
        train_rdkit.reset_index(drop=True),
        train_morgan.reset_index(drop=True),
        train_maccs.reset_index(drop=True),
        train_group.reset_index(drop=True),
    ], axis=1)
    
    X_test = pd.concat([
        test_smi_feat.reset_index(drop=True),
        test_rdkit.reset_index(drop=True),
        test_morgan.reset_index(drop=True),
        test_maccs.reset_index(drop=True),
        test_group.reset_index(drop=True),
    ], axis=1)
    
    # Asegurar mismas columnas
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Rellenar NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Reemplazar infinitos
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    print(f"\n  ✅ TOTAL FEATURES: {X_train.shape[1]}")
    
    return X_train, X_test


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO: CATBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_catboost(X_train, y_train, X_test):
    """Entrena CatBoost con 5-Fold CV."""
    from catboost import CatBoostRegressor
    
    print("\n" + "="*70)
    print("  ENTRENANDO CATBOOST")
    print("="*70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = CatBoostRegressor(
            iterations=2000,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=False,
            early_stopping_rounds=100,
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / N_FOLDS
        
        mae = mean_absolute_error(y_val, val_pred)
        maes.append(mae)
        print(f"      Fold {fold+1}: MAE = {mae:.2f}")
    
    oof_mae = mean_absolute_error(y_train, oof_preds)
    print(f"\n  📊 CatBoost OOF MAE: {oof_mae:.2f} (±{np.std(maes):.2f})")
    
    return oof_preds, test_preds


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO: XGBOOST (mejorado)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test):
    """Entrena XGBoost con 5-Fold CV."""
    from xgboost import XGBRegressor
    
    print("\n" + "="*70)
    print("  ENTRENANDO XGBOOST")
    print("="*70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = XGBRegressor(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            min_child_weight=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=100,
        )
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / N_FOLDS
        
        mae = mean_absolute_error(y_val, val_pred)
        maes.append(mae)
        print(f"      Fold {fold+1}: MAE = {mae:.2f}")
    
    oof_mae = mean_absolute_error(y_train, oof_preds)
    print(f"\n  📊 XGBoost OOF MAE: {oof_mae:.2f} (±{np.std(maes):.2f})")
    
    return oof_preds, test_preds


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO: LIGHTGBM (mejorado)
# ═══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train, X_test):
    """Entrena LightGBM con 5-Fold CV."""
    from lightgbm import LGBMRegressor
    
    print("\n" + "="*70)
    print("  ENTRENANDO LIGHTGBM")
    print("="*70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = LGBMRegressor(
            n_estimators=2000,
            max_depth=10,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            min_child_samples=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / N_FOLDS
        
        mae = mean_absolute_error(y_val, val_pred)
        maes.append(mae)
        print(f"      Fold {fold+1}: MAE = {mae:.2f}")
    
    oof_mae = mean_absolute_error(y_train, oof_preds)
    print(f"\n  📊 LightGBM OOF MAE: {oof_mae:.2f} (±{np.std(maes):.2f})")
    
    return oof_preds, test_preds


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO: NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

def train_neural_network(X_train, y_train, X_test):
    """Entrena una Red Neuronal MLP con 5-Fold CV."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\n" + "="*70)
    print("  ENTRENANDO NEURAL NETWORK")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Dispositivo: {device}")
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    maes = []
    
    input_dim = X_train.shape[1]
    
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(256, 64),
                nn.ReLU(),
                
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.net(x).squeeze()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = torch.FloatTensor(X_train_scaled[train_idx]).to(device)
        y_tr = torch.FloatTensor(y_train[train_idx]).to(device)
        X_val = torch.FloatTensor(X_train_scaled[val_idx]).to(device)
        y_val_np = y_train[val_idx]
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        model = MLP(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.L1Loss()  # MAE loss
        
        best_val_mae = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(200):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validación
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).cpu().numpy()
                val_mae = mean_absolute_error(y_val_np, val_pred)
            
            scheduler.step(val_mae)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                break
        
        # Cargar mejor modelo
        model.load_state_dict(best_model_state)
        model.eval()
        
        with torch.no_grad():
            oof_preds[val_idx] = model(X_val).cpu().numpy()
            test_preds += model(X_test_t).cpu().numpy() / N_FOLDS
        
        maes.append(best_val_mae)
        print(f"      Fold {fold+1}: MAE = {best_val_mae:.2f}")
    
    oof_mae = mean_absolute_error(y_train, oof_preds)
    print(f"\n  📊 Neural Network OOF MAE: {oof_mae:.2f} (±{np.std(maes):.2f})")
    
    return oof_preds, test_preds


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZAR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_ensemble(y_true, predictions_dict):
    """Encuentra los pesos óptimos para el ensemble."""
    from scipy.optimize import minimize
    
    print("\n" + "="*70)
    print("  OPTIMIZANDO ENSEMBLE")
    print("="*70)
    
    names = list(predictions_dict.keys())
    preds = np.array([predictions_dict[name] for name in names])
    
    def objective(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalizar
        ensemble = np.sum(weights.reshape(-1, 1) * preds, axis=0)
        return mean_absolute_error(y_true, ensemble)
    
    # Inicial: pesos iguales
    initial_weights = np.ones(len(names)) / len(names)
    
    # Optimizar
    result = minimize(objective, initial_weights, method='Nelder-Mead')
    
    # Normalizar pesos finales
    best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    best_mae = result.fun
    
    print(f"\n  Pesos óptimos:")
    for name, weight in zip(names, best_weights):
        print(f"      {name}: {weight:.3f} ({weight*100:.1f}%)")
    
    print(f"\n  📊 Ensemble OOF MAE: {best_mae:.2f}")
    
    return dict(zip(names, best_weights)), best_mae


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🚀 PASO 4: Modelos Avanzados                                         ║
║        MeltingPoint Kaggle Competition                                       ║
║                                                                              ║
║        OBJETIVO: Bajar de MAE 23.4 a ~18-20                                 ║
║                                                                              ║
║        Nuevas técnicas:                                                      ║
║        • Morgan Fingerprints (2048 bits)                                    ║
║        • MACCS Keys (167 bits)                                              ║
║        • CatBoost                                                           ║
║        • Neural Network (MLP)                                               ║
║        • Ensemble optimizado                                                ║
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
    
    print(f"\n  Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Preparar features
    X_train, X_test = prepare_all_features(train_df, test_df)
    
    # Entrenar modelos
    xgb_oof, xgb_test = train_xgboost(X_train, y_train, X_test)
    lgbm_oof, lgbm_test = train_lightgbm(X_train, y_train, X_test)
    cat_oof, cat_test = train_catboost(X_train, y_train, X_test)
    nn_oof, nn_test = train_neural_network(X_train, y_train, X_test)
    
    # Cargar predicciones de ChemProp
    print("\n" + "="*70)
    print("  CARGANDO CHEMPROP")
    print("="*70)
    
    cp_path = DATA_PROCESSED / "chemprop_predictions.csv"
    if cp_path.exists():
        cp_preds = pd.read_csv(cp_path)
        cp_test = cp_preds["Tm"].values if "Tm" in cp_preds.columns else cp_preds.iloc[:, 0].values
        print(f"\n  ✓ ChemProp test: {len(cp_test)} predicciones")
    else:
        cp_test = None
        print("\n  ⚠️ No hay predicciones de ChemProp")
    
    # Optimizar ensemble (sin ChemProp primero, porque no tenemos OOF)
    oof_predictions = {
        "XGBoost": xgb_oof,
        "LightGBM": lgbm_oof,
        "CatBoost": cat_oof,
        "NeuralNet": nn_oof,
    }
    
    test_predictions = {
        "XGBoost": xgb_test,
        "LightGBM": lgbm_test,
        "CatBoost": cat_test,
        "NeuralNet": nn_test,
    }
    
    weights, oof_mae = optimize_ensemble(y_train, oof_predictions)
    
    # Crear ensemble de test
    ensemble_test = np.zeros(len(X_test))
    for name, weight in weights.items():
        ensemble_test += weight * test_predictions[name]
    
    # Guardar submissions
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("  GUARDANDO SUBMISSIONS")
    print("="*70)
    
    # Submission: Ensemble sin ChemProp
    sub_ensemble = pd.DataFrame({"id": test_df["id"], "Tm": ensemble_test})
    sub_ensemble.to_csv(SUBMISSION_DIR / "submission_advanced_ensemble.csv", index=False)
    print(f"\n  ✓ submission_advanced_ensemble.csv")
    
    # Submissions individuales
    for name, preds in test_predictions.items():
        filename = f"submission_{name.lower()}_v2.csv"
        pd.DataFrame({"id": test_df["id"], "Tm": preds}).to_csv(SUBMISSION_DIR / filename, index=False)
        print(f"  ✓ {filename}")
    
    # Combinar con ChemProp si existe
    if cp_test is not None:
        for w_cp in [0.20, 0.25, 0.30, 0.35]:
            w_ensemble = 1 - w_cp
            final = w_cp * cp_test + w_ensemble * ensemble_test
            filename = f"submission_advanced_cp{int(w_cp*100)}.csv"
            pd.DataFrame({"id": test_df["id"], "Tm": final}).to_csv(SUBMISSION_DIR / filename, index=False)
            print(f"  ✓ {filename}")
    
    # Resumen
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ✅ PASO 4 COMPLETADO                                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 RESULTADOS OOF:                                                         ║
║      • XGBoost:    {mean_absolute_error(y_train, xgb_oof):>6.2f}                                              ║
║      • LightGBM:   {mean_absolute_error(y_train, lgbm_oof):>6.2f}                                              ║
║      • CatBoost:   {mean_absolute_error(y_train, cat_oof):>6.2f}                                              ║
║      • NeuralNet:  {mean_absolute_error(y_train, nn_oof):>6.2f}                                              ║
║      • Ensemble:   {oof_mae:>6.2f}                                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📁 SUBMISSIONS:                                                             ║
║      • submission_advanced_ensemble.csv (sin ChemProp)                      ║
║      • submission_advanced_cp20/25/30/35.csv (con ChemProp)                 ║
║      • submission_xgboost_v2.csv, etc.                                      ║
║                                                                              ║
║  🎯 Sube todos a Kaggle y compara!                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()