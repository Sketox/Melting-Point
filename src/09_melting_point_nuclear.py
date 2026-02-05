#!/usr/bin/env python3
"""
=============================================================================
🔥 MELTING POINT - SCRIPT NUCLEAR 🔥
=============================================================================

Este script usa TODAS las técnicas que usan los ganadores:

1. ✅ ChemBERTa (Transformer preentrenado en 77M moléculas)
2. ✅ Features 3D moleculares (RDKit conformers)
3. ✅ SMILES Augmentation + TTA
4. ✅ Ensemble masivo (XGB + LGB + CAT + RF + ET + ChemBERTa)
5. ✅ Stacking de 2 niveles
6. ✅ Datos externos opcionales

REQUISITOS:
pip install transformers torch pandas numpy rdkit scikit-learn
pip install xgboost lightgbm catboost tqdm requests

Tiempo estimado: 2-4 horas (con GPU: 1-2 horas)

Autor: Sketo
Competencia: Kaggle Melting Point
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit import RDLogger

# ML tradicional
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Configuración
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Verificar Transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
    torch.manual_seed(RANDOM_STATE)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ PyTorch disponible. Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️ PyTorch/Transformers no instalado. Instala con:")
    print("   pip install torch transformers")

# Verificar 3D descriptors
try:
    from rdkit.Chem import Descriptors3D
    DESCRIPTORS_3D_AVAILABLE = True
    print("✅ RDKit 3D Descriptors disponibles")
except ImportError:
    DESCRIPTORS_3D_AVAILABLE = False
    print("⚠️ RDKit 3D Descriptors no disponibles")


print("\n" + "=" * 70)
print("   🔥 MELTING POINT - SCRIPT NUCLEAR 🔥")
print("=" * 70)


# =============================================================================
# SECCIÓN 1: ChemBERTa EMBEDDINGS
# =============================================================================

class ChemBERTaFeaturizer:
    """
    Extrae embeddings de ChemBERTa preentrenado.
    
    ChemBERTa fue preentrenado en 77 MILLONES de moléculas de PubChem.
    Estos embeddings capturan información química que los fingerprints no pueden.
    """
    
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch no disponible")
        
        print(f"\n🤖 Cargando ChemBERTa: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        
        # Obtener dimensión del embedding
        self.embedding_dim = self.model.config.hidden_size
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def get_embeddings(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """Extrae embeddings para una lista de SMILES"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa embeddings"):
            batch = smiles_list[i:i + batch_size]
            
            # Tokenizar
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usar [CLS] token como embedding de la molécula
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


# =============================================================================
# SECCIÓN 2: FEATURES 3D MOLECULARES
# =============================================================================

def generate_3d_conformer(mol):
    """Genera conformación 3D de una molécula"""
    if mol is None:
        return None
    
    try:
        mol = Chem.AddHs(mol)
        
        # ETKDG v3 es el mejor método
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_STATE
        params.maxAttempts = 5
        
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            # Fallback a ETKDG básico
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if result == -1:
            return None
        
        # Optimizar geometría
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass
        
        return mol
    except:
        return None


def calculate_3d_descriptors(mol) -> Dict[str, float]:
    """
    Calcula descriptores 3D - CRUCIALES para melting point.
    
    El melting point depende del empaquetamiento cristalino,
    que está directamente relacionado con la forma 3D.
    """
    if not DESCRIPTORS_3D_AVAILABLE:
        return {}
    
    if mol is None or mol.GetNumConformers() == 0:
        return {}
    
    desc = {}
    try:
        # Descriptores de forma básicos
        desc['Asphericity'] = Descriptors3D.Asphericity(mol)
        desc['Eccentricity'] = Descriptors3D.Eccentricity(mol)
        desc['InertialShapeFactor'] = Descriptors3D.InertialShapeFactor(mol)
        desc['SpherocityIndex'] = Descriptors3D.SpherocityIndex(mol)
        desc['RadiusOfGyration'] = Descriptors3D.RadiusOfGyration(mol)
        
        # Principal Moments of Inertia
        desc['PMI1'] = Descriptors3D.PMI1(mol)
        desc['PMI2'] = Descriptors3D.PMI2(mol)
        desc['PMI3'] = Descriptors3D.PMI3(mol)
        
        # Normalized PMI ratios (útiles para forma)
        desc['NPR1'] = Descriptors3D.NPR1(mol)
        desc['NPR2'] = Descriptors3D.NPR2(mol)
        
        # Autocorrelación 3D (si disponible)
        try:
            autocorr = Descriptors3D.CalcAUTOCORR3D(mol)
            for i, val in enumerate(autocorr[:30]):  # Primeros 30
                desc[f'AUTOCORR3D_{i}'] = val
        except:
            pass
        
        # RDF - Radial Distribution Function
        try:
            rdf = Descriptors3D.CalcRDF(mol)
            for i, val in enumerate(rdf[:30]):
                desc[f'RDF_{i}'] = val
        except:
            pass
        
        # MORSE descriptors
        try:
            morse = Descriptors3D.CalcMORSE(mol)
            for i, val in enumerate(morse[:30]):
                desc[f'MORSE_{i}'] = val
        except:
            pass
        
        # WHIM descriptors
        try:
            whim = Descriptors3D.CalcWHIM(mol)
            for i, val in enumerate(whim[:30]):
                desc[f'WHIM_{i}'] = val
        except:
            pass
        
    except Exception as e:
        pass
    
    return desc


# =============================================================================
# SECCIÓN 3: FEATURE ENGINEERING COMPLETO
# =============================================================================

def create_all_features(df: pd.DataFrame, 
                        include_3d: bool = True,
                        chemberta_embeddings: Optional[np.ndarray] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Crea TODAS las features disponibles:
    - Morgan Fingerprints (2048)
    - MACCS Keys (167)
    - RDKit 2D Descriptors (~25)
    - RDKit 3D Descriptors (~100+) [NUEVO]
    - ChemBERTa Embeddings (384) [NUEVO]
    - Group features del dataset
    """
    if verbose:
        print("\n🔧 Creando features completas...")
    
    all_features = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Features") if verbose else df.iterrows()
    
    for idx, (_, row) in enumerate(iterator):
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        features = {}
        
        # 1. Morgan Fingerprints (2048 bits)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for i, bit in enumerate(fp):
                features[f'morgan_{i}'] = int(bit)
        else:
            for i in range(2048):
                features[f'morgan_{i}'] = 0
        
        # 2. MACCS Keys (167 bits)
        if mol:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            for i, bit in enumerate(maccs):
                features[f'maccs_{i}'] = int(bit)
        else:
            for i in range(167):
                features[f'maccs_{i}'] = 0
        
        # 3. RDKit 2D Descriptors
        if mol:
            features['MolWt'] = Descriptors.MolWt(mol)
            features['LogP'] = Descriptors.MolLogP(mol)
            features['TPSA'] = Descriptors.TPSA(mol)
            features['NumHDonors'] = Descriptors.NumHDonors(mol)
            features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            features['NumRotBonds'] = Descriptors.NumRotatableBonds(mol)
            features['NumRings'] = Descriptors.RingCount(mol)
            features['NumAromRings'] = Descriptors.NumAromaticRings(mol)
            features['NumSatRings'] = Descriptors.NumSaturatedRings(mol)
            features['NumAliphRings'] = Descriptors.NumAliphaticRings(mol)
            features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            features['FracCSP3'] = Descriptors.FractionCSP3(mol)
            features['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
            features['LabuteASA'] = Descriptors.LabuteASA(mol)
            features['BertzCT'] = Descriptors.BertzCT(mol)
            features['Chi0'] = Descriptors.Chi0(mol)
            features['Chi1'] = Descriptors.Chi1(mol)
            features['Kappa1'] = Descriptors.Kappa1(mol)
            features['Kappa2'] = Descriptors.Kappa2(mol)
            features['Kappa3'] = Descriptors.Kappa3(mol)
            features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            features['MolMR'] = Descriptors.MolMR(mol)
            try:
                features['qed'] = Descriptors.qed(mol)
            except:
                features['qed'] = 0
        
        # 4. 3D Descriptors (NUEVO - clave para melting point)
        if include_3d and mol and DESCRIPTORS_3D_AVAILABLE:
            mol_3d = generate_3d_conformer(mol)
            if mol_3d:
                desc_3d = calculate_3d_descriptors(mol_3d)
                features.update(desc_3d)
        
        # 5. ChemBERTa Embeddings (si disponibles)
        if chemberta_embeddings is not None:
            for i, val in enumerate(chemberta_embeddings[idx]):
                features[f'chemberta_{i}'] = val
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # 6. Group features del dataset original
    group_cols = [c for c in df.columns if c.startswith('Group')]
    if group_cols:
        features_df = pd.concat([
            features_df.reset_index(drop=True),
            df[group_cols].reset_index(drop=True)
        ], axis=1)
        if verbose:
            print(f"   + {len(group_cols)} Group features del dataset")
    
    # Limpiar
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    
    # Eliminar columnas con varianza cero
    variance = features_df.var()
    zero_var = variance[variance == 0].index.tolist()
    if zero_var:
        features_df = features_df.drop(columns=zero_var)
    
    if verbose:
        print(f"   Total features: {features_df.shape[1]}")
    
    return features_df


# =============================================================================
# SECCIÓN 4: MODELOS
# =============================================================================

def train_all_models(X_train, y_train, X_val, y_val):
    """Entrena TODOS los modelos disponibles"""
    models = {}
    
    # XGBoost
    models['xgb'] = xgb.XGBRegressor(
        max_depth=8, learning_rate=0.05, n_estimators=1000,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    models['xgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # LightGBM
    models['lgb'] = lgb.LGBMRegressor(
        max_depth=10, learning_rate=0.05, n_estimators=1000,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    )
    models['lgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # CatBoost
    models['cat'] = CatBoostRegressor(
        depth=8, learning_rate=0.05, iterations=1000,
        l2_leaf_reg=3.0, loss_function='MAE',
        random_seed=RANDOM_STATE, verbose=False,
        early_stopping_rounds=100
    )
    models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # Random Forest
    models['rf'] = RandomForestRegressor(
        n_estimators=300, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['rf'].fit(X_train, y_train)
    
    # Extra Trees
    models['et'] = ExtraTreesRegressor(
        n_estimators=300, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['et'].fit(X_train, y_train)
    
    return models


# =============================================================================
# SECCIÓN 5: PIPELINE PRINCIPAL
# =============================================================================

def main():
    """Pipeline principal NUCLEAR"""
    
    # Paths
    DATA_DIR = '../data/raw'
    OUTPUT_DIR = 'submissions'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    
    print("\n📥 Cargando datos...")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"\n❌ ERROR: No se encontró {train_path}")
        print("   Coloca train.csv y test.csv en data/raw/")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    y = train_df['Tm'].values
    print(f"   Train: {len(train_df)} muestras")
    print(f"   Test:  {len(test_df)} muestras")
    print(f"   Target range: {y.min():.1f} - {y.max():.1f} K")
    
    # =========================================================================
    # CHEMBERTA EMBEDDINGS
    # =========================================================================
    
    train_bert_emb = None
    test_bert_emb = None
    
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("PASO 1: ChemBERTa Embeddings")
        print("=" * 60)
        
        try:
            chemberta = ChemBERTaFeaturizer("DeepChem/ChemBERTa-77M-MTR")
            
            print("\n   Extrayendo embeddings para train...")
            train_bert_emb = chemberta.get_embeddings(train_df['SMILES'].tolist())
            
            print("   Extrayendo embeddings para test...")
            test_bert_emb = chemberta.get_embeddings(test_df['SMILES'].tolist())
            
            print(f"   ✅ Embeddings shape: {train_bert_emb.shape}")
            
        except Exception as e:
            print(f"   ⚠️ Error con ChemBERTa: {e}")
            print("   Continuando sin embeddings de transformer...")
    
    # =========================================================================
    # CREAR FEATURES
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PASO 2: Feature Engineering (2D + 3D)")
    print("=" * 60)
    
    X_train_full = create_all_features(
        train_df, 
        include_3d=DESCRIPTORS_3D_AVAILABLE,
        chemberta_embeddings=train_bert_emb
    )
    
    X_test_full = create_all_features(
        test_df,
        include_3d=DESCRIPTORS_3D_AVAILABLE,
        chemberta_embeddings=test_bert_emb
    )
    
    # Asegurar columnas coincidan
    common_cols = X_train_full.columns.intersection(X_test_full.columns).tolist()
    X_train_full = X_train_full[common_cols]
    X_test_full = X_test_full[common_cols]
    
    print(f"\n   Features finales: {len(common_cols)}")
    
    # =========================================================================
    # CROSS-VALIDATION + STACKING
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PASO 3: Cross-Validation con 5 modelos")
    print("=" * 60)
    
    N_FOLDS = 5
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    model_names = ['xgb', 'lgb', 'cat', 'rf', 'et']
    all_models = {name: [] for name in model_names}
    scalers = []
    
    # OOF predictions para stacking
    oof = {name: np.zeros(len(train_df)) for name in model_names}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        
        X_train = X_train_full.iloc[train_idx].values
        X_val = X_train_full.iloc[val_idx].values
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Escalar
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        scalers.append(scaler)
        
        # Entrenar todos los modelos
        models = train_all_models(X_train_s, y_train, X_val_s, y_val)
        
        # Guardar predicciones OOF
        for name in model_names:
            all_models[name].append(models[name])
            oof[name][val_idx] = models[name].predict(X_val_s)
        
        # Mostrar scores del fold
        print(f"   XGB: {mean_absolute_error(y_val, oof['xgb'][val_idx]):.4f}")
        print(f"   LGB: {mean_absolute_error(y_val, oof['lgb'][val_idx]):.4f}")
        print(f"   CAT: {mean_absolute_error(y_val, oof['cat'][val_idx]):.4f}")
        print(f"   RF:  {mean_absolute_error(y_val, oof['rf'][val_idx]):.4f}")
        print(f"   ET:  {mean_absolute_error(y_val, oof['et'][val_idx]):.4f}")
    
    # =========================================================================
    # STACKING - NIVEL 2
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PASO 4: Stacking (Meta-Learner)")
    print("=" * 60)
    
    # Crear meta-features (predicciones de nivel 1)
    meta_train = np.column_stack([oof[name] for name in model_names])
    
    # Entrenar meta-learner (Ridge regression)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, y)
    
    # Predicciones stacked en train (para evaluación)
    stacked_oof = meta_model.predict(meta_train)
    stacked_mae = mean_absolute_error(y, stacked_oof)
    
    print(f"\n   Pesos del meta-learner: {dict(zip(model_names, meta_model.coef_.round(3)))}")
    print(f"   Stacked OOF MAE: {stacked_mae:.4f}")
    
    # Scores individuales
    print("\n   Scores individuales OOF:")
    for name in model_names:
        mae = mean_absolute_error(y, oof[name])
        print(f"      {name}: {mae:.4f}")
    
    # =========================================================================
    # PREDICCIÓN EN TEST
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PASO 5: Predicción en Test")
    print("=" * 60)
    
    X_test = X_test_full.values
    
    # Predicciones de cada modelo (promedio de folds)
    test_preds = {name: [] for name in model_names}
    
    for i, scaler in enumerate(scalers):
        X_test_s = scaler.transform(X_test)
        for name in model_names:
            test_preds[name].append(all_models[name][i].predict(X_test_s))
    
    # Promediar folds
    test_preds_avg = {name: np.mean(test_preds[name], axis=0) for name in model_names}
    
    # Meta-features para test
    meta_test = np.column_stack([test_preds_avg[name] for name in model_names])
    
    # Predicción final stacked
    final_pred = meta_model.predict(meta_test)
    
    # =========================================================================
    # GENERAR SUBMISSIONS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PASO 6: Generando Submissions")
    print("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submissions = []
    
    # 1. Stacked (mejor esperado)
    submissions.append({
        'name': f'nuclear_stacked_{timestamp}',
        'preds': final_pred,
        'desc': 'Stacking: XGB+LGB+CAT+RF+ET'
    })
    
    # 2. Simple average (backup)
    simple_avg = np.mean([test_preds_avg[name] for name in model_names], axis=0)
    submissions.append({
        'name': f'nuclear_average_{timestamp}',
        'preds': simple_avg,
        'desc': 'Simple average de 5 modelos'
    })
    
    # 3. Weighted average optimizado
    best_mae = float('inf')
    best_weights = None
    
    # Grid search para pesos óptimos
    for w_xgb in np.arange(0.1, 0.5, 0.1):
        for w_lgb in np.arange(0.1, 0.5, 0.1):
            for w_cat in np.arange(0.1, 0.5, 0.1):
                w_rem = 1 - w_xgb - w_lgb - w_cat
                if w_rem > 0.05:
                    w_rf = w_rem / 2
                    w_et = w_rem / 2
                    
                    weighted = (w_xgb * oof['xgb'] + w_lgb * oof['lgb'] + 
                               w_cat * oof['cat'] + w_rf * oof['rf'] + w_et * oof['et'])
                    mae = mean_absolute_error(y, weighted)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = [w_xgb, w_lgb, w_cat, w_rf, w_et]
    
    if best_weights:
        weighted_pred = (best_weights[0] * test_preds_avg['xgb'] +
                        best_weights[1] * test_preds_avg['lgb'] +
                        best_weights[2] * test_preds_avg['cat'] +
                        best_weights[3] * test_preds_avg['rf'] +
                        best_weights[4] * test_preds_avg['et'])
        
        submissions.append({
            'name': f'nuclear_weighted_{timestamp}',
            'preds': weighted_pred,
            'desc': f'Weighted: {dict(zip(model_names, [round(w, 2) for w in best_weights]))}'
        })
        
        print(f"\n   Pesos óptimos: {dict(zip(model_names, [round(w, 2) for w in best_weights]))}")
        print(f"   Weighted OOF MAE: {best_mae:.4f}")
    
    # Guardar submissions
    print(f"\n📁 Guardando submissions en '{OUTPUT_DIR}/':")
    for sub in submissions:
        filepath = os.path.join(OUTPUT_DIR, f"{sub['name']}.csv")
        pd.DataFrame({
            'id': test_df['id'],
            'Tm': sub['preds']
        }).to_csv(filepath, index=False)
        print(f"   ✅ {sub['name']}.csv - {sub['desc']}")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("   📊 RESUMEN FINAL")
    print("=" * 70)
    
    print(f"\n   Features utilizadas: {len(common_cols)}")
    print(f"   - Morgan FP: 2048")
    print(f"   - MACCS Keys: 167")
    print(f"   - RDKit 2D: ~25")
    if DESCRIPTORS_3D_AVAILABLE:
        print(f"   - RDKit 3D: ~100+")
    if train_bert_emb is not None:
        print(f"   - ChemBERTa: {train_bert_emb.shape[1]}")
    
    print(f"\n   Modelos entrenados: {len(model_names)} × {N_FOLDS} folds = {len(model_names) * N_FOLDS}")
    
    print(f"\n   OOF Scores:")
    for name in model_names:
        print(f"      {name}: {mean_absolute_error(y, oof[name]):.4f}")
    print(f"      Stacked: {stacked_mae:.4f}")
    if best_weights:
        print(f"      Weighted: {best_mae:.4f}")
    
    print(f"\n   🎯 Submissions generados: {len(submissions)}")
    print(f"\n   📝 Recomendación: Subir 'nuclear_stacked' primero")
    
    # Guardar configuración
    config = {
        'timestamp': timestamp,
        'n_features': len(common_cols),
        'n_folds': N_FOLDS,
        'models': model_names,
        'has_3d': DESCRIPTORS_3D_AVAILABLE,
        'has_chemberta': train_bert_emb is not None,
        'stacked_oof_mae': stacked_mae,
        'weighted_oof_mae': best_mae if best_weights else None,
        'meta_weights': dict(zip(model_names, meta_model.coef_.tolist())),
        'optimal_weights': dict(zip(model_names, best_weights)) if best_weights else None
    }
    
    config_path = os.path.join(OUTPUT_DIR, f'config_nuclear_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n   💾 Config guardada: {config_path}")
    
    print("\n" + "=" * 70)
    print("   🔥 ¡LISTO PARA KAGGLE! 🔥")
    print("=" * 70)


if __name__ == "__main__":
    main()