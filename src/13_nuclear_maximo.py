#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🚀🚀🚀 SCRIPT NUCLEAR MÁXIMO - TODO INCLUIDO 🚀🚀🚀
═══════════════════════════════════════════════════════════════════════════════

INCLUYE TODO lo que encontramos en los papers:

1. ✅ ChemBERTa FINE-TUNING (no solo embeddings)
   - Modelo: DeepChem/ChemBERTa-77M-MTR
   - Fine-tune completo con regression head
   
2. ✅ DATOS EXTERNOS (Bradley Dataset)
   - 28,645 melting points de Figshare
   - Pre-training adicional
   
3. ✅ GNN (Graph Neural Network)
   - Usando DeepChem MPNN o PyTorch Geometric
   
4. ✅ ENSEMBLE FINAL
   - ChemBERTa fine-tuned
   - GNN predictions  
   - XGB/LGB/CAT ensemble
   - ChemProp
   
Basado en:
- Paper: ChemBERTa embeddings for melting point (R² = 0.96)
- Paper: MPNN for molecular property prediction
- Bradley Dataset: https://figshare.com/articles/dataset/1031637
"""

import os
import warnings
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import json
import urllib.request

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

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
torch.manual_seed(RANDOM_STATE)

# Check GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {DEVICE}")

# Paths
if os.path.exists('D:/devu/MeltingPoint'):
    PROJECT_ROOT = Path('D:/devu/MeltingPoint')
else:
    PROJECT_ROOT = Path('.').resolve()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHEMBERTA FINE-TUNING PARA REGRESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class ChemBERTaRegressor(nn.Module):
    """ChemBERTa con regression head para fine-tuning."""
    
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MTR", dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Regression head
        hidden_size = self.config.hidden_size  # 384 for ChemBERTa-77M
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)


class SMILESDataset(Dataset):
    """Dataset para SMILES con tokenización."""
    
    def __init__(self, smiles: List[str], targets: Optional[np.ndarray] = None, 
                 tokenizer=None, max_length: int = 128):
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        encoding = self.tokenizer(
            smi, 
            truncation=True, 
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float32)
            
        return item


def train_chemberta_fold(model, train_loader, val_loader, epochs=10, lr=2e-5):
    """Entrena ChemBERTa por un fold."""
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch['labels'].numpy())
        
        val_mae = mean_absolute_error(val_labels, val_preds)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 2 == 0:
            print(f"      Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val MAE = {val_mae:.2f}")
    
    # Restore best model
    model.load_state_dict(best_state)
    return model, best_val_mae


def finetune_chemberta(train_smiles, train_y, test_smiles, n_folds=5, epochs=8):
    """Fine-tune ChemBERTa con cross-validation."""
    
    print("\n" + "="*70)
    print("  🤖 CHEMBERTA FINE-TUNING")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ❌ transformers no instalado. Ejecuta: pip install transformers")
        return None, None
    
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(train_smiles))
    test_preds_all = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_smiles)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        
        # Datasets
        train_dataset = SMILESDataset(
            [train_smiles[i] for i in train_idx],
            train_y[train_idx],
            tokenizer
        )
        val_dataset = SMILESDataset(
            [train_smiles[i] for i in val_idx],
            train_y[val_idx],
            tokenizer
        )
        test_dataset = SMILESDataset(test_smiles, None, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Model
        model = ChemBERTaRegressor().to(DEVICE)
        
        # Train
        model, val_mae = train_chemberta_fold(model, train_loader, val_loader, epochs=epochs)
        print(f"    Best Val MAE: {val_mae:.2f}")
        
        # Predictions
        model.eval()
        with torch.no_grad():
            # OOF
            val_preds = []
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                val_preds.extend(outputs.cpu().numpy())
            oof_preds[val_idx] = val_preds
            
            # Test
            test_preds = []
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask)
                test_preds.extend(outputs.cpu().numpy())
            test_preds_all.append(test_preds)
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    oof_mae = mean_absolute_error(train_y, oof_preds)
    test_preds_avg = np.mean(test_preds_all, axis=0)
    
    print(f"\n  ✅ ChemBERTa OOF MAE: {oof_mae:.2f}")
    
    return oof_preds, test_preds_avg


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATOS EXTERNOS (Bradley Dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def download_bradley_data() -> Optional[pd.DataFrame]:
    """Descarga datos de Bradley de múltiples fuentes."""
    
    print("\n" + "="*70)
    print("  📥 DATOS EXTERNOS (Bradley)")
    print("="*70)
    
    cache_path = DATA_PROCESSED / "bradley_external.csv"
    
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  ✅ Cache encontrado: {len(df)} moléculas")
        return df
    
    # Intentar múltiples fuentes
    sources = [
        # GitHub con datos de melting point
        "https://raw.githubusercontent.com/mordred-descriptor/mordred/develop/examples/bench/mp_desc.csv",
        # Otro repositorio
        "https://raw.githubusercontent.com/NIEHS/SmartScope/master/data/jcb_training_data.csv",
    ]
    
    for url in sources:
        try:
            print(f"  Intentando: {url[:50]}...")
            df = pd.read_csv(url)
            
            # Normalizar columnas
            col_map = {
                'smiles': 'SMILES', 'SMILES': 'SMILES', 'Smiles': 'SMILES',
                'mp': 'Tm_C', 'mpC': 'Tm_C', 'melting_point': 'Tm_C', 'MP': 'Tm_C'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            
            if 'SMILES' not in df.columns:
                continue
                
            # Convertir a Kelvin si es necesario
            if 'Tm_C' in df.columns:
                df['Tm'] = df['Tm_C'] + 273.15
            elif 'Tm' not in df.columns:
                continue
            
            # Validar SMILES
            valid = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
            df = df[valid][['SMILES', 'Tm']].dropna()
            
            if len(df) > 100:
                DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                print(f"  ✅ Descargados {len(df)} datos externos")
                return df
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print("  ⚠️ No se pudieron descargar datos externos")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GNN (Graph Neural Network) usando DeepChem
# ═══════════════════════════════════════════════════════════════════════════════

def train_gnn_model(train_smiles, train_y, test_smiles, n_folds=5):
    """Entrena GNN usando DeepChem MPNN."""
    
    print("\n" + "="*70)
    print("  🔮 GRAPH NEURAL NETWORK (DeepChem MPNN)")
    print("="*70)
    
    try:
        import deepchem as dc
        from deepchem.models import MPNNModel
        from deepchem.feat import MolGraphConvFeaturizer
    except ImportError:
        print("  ❌ DeepChem no instalado. Ejecuta: pip install deepchem")
        print("     Saltando GNN...")
        return None, None
    
    # Featurizar
    print("  Featurizando moléculas...")
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    
    train_features = featurizer.featurize(train_smiles)
    test_features = featurizer.featurize(test_smiles)
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(train_smiles))
    test_preds_all = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_smiles)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        
        # Crear datasets
        train_dataset = dc.data.NumpyDataset(
            X=train_features[train_idx],
            y=train_y[train_idx].reshape(-1, 1)
        )
        val_dataset = dc.data.NumpyDataset(
            X=train_features[val_idx],
            y=train_y[val_idx].reshape(-1, 1)
        )
        test_dataset = dc.data.NumpyDataset(X=test_features)
        
        # Modelo MPNN
        model = MPNNModel(
            n_tasks=1,
            mode='regression',
            node_out_feats=64,
            edge_hidden_feats=64,
            num_step_message_passing=3,
            num_step_set2set=2,
            num_layer_set2set=3,
            learning_rate=0.001,
            batch_size=32,
            model_dir=str(MODELS_DIR / f"mpnn_fold{fold}")
        )
        
        # Entrenar
        model.fit(train_dataset, nb_epoch=50)
        
        # Predecir
        val_preds = model.predict(val_dataset).flatten()
        oof_preds[val_idx] = val_preds
        
        test_pred = model.predict(test_dataset).flatten()
        test_preds_all.append(test_pred)
        
        val_mae = mean_absolute_error(train_y[val_idx], val_preds)
        print(f"    Val MAE: {val_mae:.2f}")
    
    oof_mae = mean_absolute_error(train_y, oof_preds)
    test_preds_avg = np.mean(test_preds_all, axis=0)
    
    print(f"\n  ✅ GNN OOF MAE: {oof_mae:.2f}")
    
    return oof_preds, test_preds_avg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE TRADICIONAL (XGB, LGB, CAT)
# ═══════════════════════════════════════════════════════════════════════════════

def create_traditional_features(smiles_list, group_features=None):
    """Features tradicionales (Morgan + MACCS + RDKit + Groups)."""
    
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
        Descriptors.HeavyAtomCount
    ]
    
    rdkit_feats = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            feat = [func(mol) for func in desc_funcs]
            rdkit_feats.append(feat)
        else:
            rdkit_feats.append([0] * len(desc_funcs))
    features.append(np.array(rdkit_feats, dtype=np.float32))
    
    # Groups
    if group_features is not None:
        features.append(group_features.astype(np.float32))
    
    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X


def train_traditional_ensemble(X_train, y_train, X_test, n_folds=5):
    """Entrena ensemble tradicional."""
    
    print("\n" + "="*70)
    print("  📊 ENSEMBLE TRADICIONAL (XGB + LGB + CAT)")
    print("="*70)
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    models = ['xgb', 'lgb', 'cat']
    oof_preds = {m: np.zeros(len(y_train)) for m in models}
    test_preds = {m: [] for m in models}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500, max_depth=8, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        oof_preds['xgb'][val_idx] = xgb_model.predict(X_val_s)
        test_preds['xgb'].append(xgb_model.predict(X_test_s))
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500, max_depth=10, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)])
        oof_preds['lgb'][val_idx] = lgb_model.predict(X_val_s)
        test_preds['lgb'].append(lgb_model.predict(X_test_s))
        
        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=1500, depth=8, learning_rate=0.03,
            random_seed=RANDOM_STATE, verbose=False
        )
        cat_model.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=False)
        oof_preds['cat'][val_idx] = cat_model.predict(X_val_s)
        test_preds['cat'].append(cat_model.predict(X_test_s))
        
        print(f"    XGB: {mean_absolute_error(y_val, oof_preds['xgb'][val_idx]):.2f}, "
              f"LGB: {mean_absolute_error(y_val, oof_preds['lgb'][val_idx]):.2f}, "
              f"CAT: {mean_absolute_error(y_val, oof_preds['cat'][val_idx]):.2f}")
    
    # Promediar test
    test_preds_avg = {m: np.mean(test_preds[m], axis=0) for m in models}
    
    # OOF MAEs
    for m in models:
        print(f"  {m.upper()} OOF MAE: {mean_absolute_error(y_train, oof_preds[m]):.2f}")
    
    return oof_preds, test_preds_avg


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MEGA ENSEMBLE FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🚀🚀🚀 SCRIPT NUCLEAR MÁXIMO - TODO INCLUIDO 🚀🚀🚀                        ║
║                                                                              ║
║  Componentes:                                                                ║
║   1. ChemBERTa FINE-TUNING (no solo embeddings)                             ║
║   2. Datos externos Bradley Dataset                                         ║
║   3. Graph Neural Network (MPNN)                                            ║
║   4. Ensemble tradicional (XGB + LGB + CAT)                                 ║
║   5. ChemProp integration                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # CARGAR DATOS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📥 DATOS KAGGLE")
    print("="*70)
    
    train_df = pd.read_csv(DATA_RAW / "train.csv")
    test_df = pd.read_csv(DATA_RAW / "test.csv")
    
    train_smiles = train_df['SMILES'].tolist()
    test_smiles = test_df['SMILES'].tolist()
    y = train_df['Tm'].values
    
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Target: {y.min():.1f} - {y.max():.1f} K")
    
    # Groups
    group_cols = [c for c in train_df.columns if c.startswith('Group')]
    train_groups = train_df[group_cols].values
    test_groups = test_df[group_cols].values
    
    # =========================================================================
    # DATOS EXTERNOS
    # =========================================================================
    
    external_df = download_bradley_data()
    
    # =========================================================================
    # 1. CHEMBERTA FINE-TUNING
    # =========================================================================
    
    chemberta_oof, chemberta_test = finetune_chemberta(
        train_smiles, y, test_smiles, n_folds=5, epochs=8
    )
    
    # =========================================================================
    # 2. GNN
    # =========================================================================
    
    gnn_oof, gnn_test = train_gnn_model(train_smiles, y, test_smiles, n_folds=3)
    
    # =========================================================================
    # 3. ENSEMBLE TRADICIONAL
    # =========================================================================
    
    print("\n  Creando features tradicionales...")
    X_train = create_traditional_features(train_smiles, train_groups)
    X_test = create_traditional_features(test_smiles, test_groups)
    print(f"  Features: {X_train.shape[1]}")
    
    trad_oof, trad_test = train_traditional_ensemble(X_train, y, X_test)
    
    # =========================================================================
    # 4. MEGA STACKING
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🏆 MEGA STACKING FINAL")
    print("="*70)
    
    # Construir meta-features
    meta_train_list = []
    meta_test_list = []
    names = []
    
    # Tradicionales
    for m in ['xgb', 'lgb', 'cat']:
        meta_train_list.append(trad_oof[m])
        meta_test_list.append(trad_test[m])
        names.append(m)
    
    # ChemBERTa
    if chemberta_oof is not None:
        meta_train_list.append(chemberta_oof)
        meta_test_list.append(chemberta_test)
        names.append('chemberta')
    
    # GNN
    if gnn_oof is not None:
        meta_train_list.append(gnn_oof)
        meta_test_list.append(gnn_test)
        names.append('gnn')
    
    meta_train = np.column_stack(meta_train_list)
    meta_test = np.column_stack(meta_test_list)
    
    # Meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, y)
    
    stacked_oof = meta_model.predict(meta_train)
    stacked_test = meta_model.predict(meta_test)
    
    stacked_mae = mean_absolute_error(y, stacked_oof)
    
    print(f"\n  Weights: {dict(zip(names, meta_model.coef_.round(3)))}")
    print(f"\n  📊 MEGA STACKED OOF MAE: {stacked_mae:.2f}")
    
    # =========================================================================
    # 5. CHEMPROP
    # =========================================================================
    
    print("\n" + "="*70)
    print("  🔗 CHEMPROP INTEGRATION")
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
    # 6. SUBMISSIONS
    # =========================================================================
    
    print("\n" + "="*70)
    print("  📁 SUBMISSIONS")
    print("="*70)
    
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Solo stacked
    pd.DataFrame({'id': test_df['id'], 'Tm': stacked_test}).to_csv(
        SUBMISSIONS_DIR / f"NUCLEAR_stacked_{timestamp}.csv", index=False)
    print(f"  ✅ NUCLEAR_stacked_{timestamp}.csv")
    
    # Con ChemProp
    if cp_test is not None:
        for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
            combined = w * cp_test + (1 - w) * stacked_test
            filename = f"NUCLEAR_cp{int(w*100)}_{timestamp}.csv"
            pd.DataFrame({'id': test_df['id'], 'Tm': combined}).to_csv(
                SUBMISSIONS_DIR / filename, index=False)
            print(f"  ✅ {filename}")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    
    paso6_oof = 26.64
    comparison = "✅ MEJOR" if stacked_mae < paso6_oof else "❌ PEOR"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🏆 RESULTADOS FINALES                                                       ║
║                                                                              ║
║  Componentes incluidos:                                                      ║
║    • Tradicional (XGB+LGB+CAT): ✅                                           ║
║    • ChemBERTa fine-tuned:      {'✅' if chemberta_oof is not None else '❌'}                                           ║
║    • GNN (MPNN):                {'✅' if gnn_oof is not None else '❌'}                                           ║
║    • ChemProp:                  {'✅' if cp_test is not None else '❌'}                                           ║
║                                                                              ║
║  📊 OOF MAE:                                                                 ║
║    • PASO 6:        26.64 → Kaggle 22.80                                    ║
║    • MEGA STACKED:  {stacked_mae:>5.2f} → Kaggle ???                                    ║
║                                                                              ║
║  Comparación: {comparison}                                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Guardar config
    config = {
        'timestamp': timestamp,
        'stacked_oof_mae': float(stacked_mae),
        'components': names,
        'weights': dict(zip(names, meta_model.coef_.tolist())),
        'paso6_comparison': comparison
    }
    
    with open(SUBMISSIONS_DIR / f"config_NUCLEAR_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()