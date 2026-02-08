"""
ml_service.py - Servicio de Machine Learning para predicción de puntos de fusión.

ACTUALIZADO v3.0:
- Integración de Ensemble (XGB + LGB + CAT) con ChemProp
- MAE mejorado: ~28.85 K (ChemProp solo) → ~22.80 K (combinado)
- Validación de SMILES con RDKit
- Detección real de grupos funcionales con SMARTS
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import re
import os

import pandas as pd
import numpy as np
import joblib

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    from rdkit.Chem import Descriptors as RDKitDescriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit no está instalado. La validación de SMILES estará limitada.")

# ChemProp
CHEMPROP_AVAILABLE = False
CHEMPROP_VERSION = None

try:
    import chemprop
    CHEMPROP_VERSION = getattr(chemprop, '__version__', '1.x')
    CHEMPROP_AVAILABLE = True
    print(f"INFO: ChemProp {CHEMPROP_VERSION} detectado correctamente.")
except ImportError:
    print("WARNING: ChemProp no está instalado. Se usará solo ensemble.")

from .config import (
    MODEL_PATH, TEST_PROCESSED_PATH, USER_COMPOUNDS_PATH,
    CHEMPROP_MODEL_DIR, SMILES_CSV_PATH, CHEMPROP_PREDICTIONS_PATH,
    TRAIN_DATASET_PATH, TEST_DATASET_PATH
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

# MAE esperado según configuración
MODEL_MAE_CHEMPROP_ONLY = 28.85  # Solo ChemProp
MODEL_MAE_ENSEMBLE_ONLY = 26.64  # Solo Ensemble
MODEL_MAE_COMBINED = 22.80      # Ensemble + ChemProp (mejor)
MODEL_STD = 3.16

# Peso óptimo de ChemProp en la combinación
CHEMPROP_WEIGHT = 0.20  # 20% ChemProp + 80% Ensemble


class SMILESValidationError(Exception):
    """Error de validación de SMILES."""
    pass


class MLService:
    """
    Servicio de ML que combina:
    - Ensemble (XGBoost + LightGBM + CatBoost): MAE ~26.64 K
    - ChemProp D-MPNN: MAE ~28.85 K
    - Combinación óptima: MAE ~22.80 K (Kaggle)
    """

    def __init__(self) -> None:
        model_path = Path(MODEL_PATH)
        csv_path = Path(TEST_PROCESSED_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV procesado no encontrado en: {csv_path}")

        # Carga modelo sklearn (fallback)
        self.model = joblib.load(model_path)

        # =====================================================================
        # CARGAR ENSEMBLE (XGB + LGB + CAT)
        # =====================================================================
        self.ensemble_models = None
        self.ensemble_weights = None
        self.use_ensemble = False
        
        ensemble_path = Path(MODEL_PATH).parent / "ensemble_predictor.joblib"
        if ensemble_path.exists():
            try:
                ensemble_data = joblib.load(ensemble_path)
                self.ensemble_models = ensemble_data.get('models', {})
                self.ensemble_weights = ensemble_data.get('weights', {
                    'XGBoost': 0.35, 'LightGBM': 0.30, 'CatBoost': 0.35
                })
                self.use_ensemble = bool(self.ensemble_models)
                if self.use_ensemble:
                    n_models = sum(len(v) for v in self.ensemble_models.values())
                    print(f"INFO: Ensemble cargado con {n_models} modelos.")
            except Exception as e:
                print(f"WARNING: Error cargando ensemble: {e}")
        else:
            print(f"INFO: Ensemble no encontrado en {ensemble_path}")
            print("      Ejecuta: cd src && python train_ensemble_production.py")

        # =====================================================================
        # CARGAR CHEMPROP
        # =====================================================================
        self.chemprop_model_dir = Path(CHEMPROP_MODEL_DIR) if CHEMPROP_MODEL_DIR else None
        self.use_chemprop = False
        self.chemprop_checkpoints = []
        
        if CHEMPROP_AVAILABLE and self.chemprop_model_dir:
            if self.chemprop_model_dir.exists():
                fold_dirs = list(self.chemprop_model_dir.glob("fold_*"))
                if fold_dirs:
                    for fold_dir in sorted(fold_dirs):
                        model_file = fold_dir / "model_0" / "model.pt"
                        if not model_file.exists():
                            model_file = fold_dir / "model.pt"
                        if model_file.exists():
                            self.chemprop_checkpoints.append(str(model_file))
                    
                    if self.chemprop_checkpoints:
                        self.use_chemprop = True
                        print(f"INFO: ChemProp habilitado con {len(self.chemprop_checkpoints)} checkpoints.")

        # Determinar MAE a reportar según modelos disponibles
        if self.use_ensemble and self.use_chemprop:
            self.effective_mae = MODEL_MAE_COMBINED
            print(f"INFO: Modo COMBINADO activo (MAE ~{MODEL_MAE_COMBINED} K)")
        elif self.use_ensemble:
            self.effective_mae = MODEL_MAE_ENSEMBLE_ONLY
            print(f"INFO: Modo ENSEMBLE activo (MAE ~{MODEL_MAE_ENSEMBLE_ONLY} K)")
        elif self.use_chemprop:
            self.effective_mae = MODEL_MAE_CHEMPROP_ONLY
            print(f"INFO: Modo CHEMPROP activo (MAE ~{MODEL_MAE_CHEMPROP_ONLY} K)")
        else:
            self.effective_mae = 30.0  # Fallback sklearn
            print("WARNING: Sin ensemble ni ChemProp, usando modelo fallback.")

        # Carga test procesado
        self.test_df = pd.read_csv(csv_path)
        if "id" not in self.test_df.columns:
            raise ValueError("test_processed.csv debe contener una columna 'id'.")

        # Cargar SMILES si existe
        self.smiles_df = None
        smiles_path = Path(SMILES_CSV_PATH) if SMILES_CSV_PATH else None
        if smiles_path and smiles_path.exists():
            self.smiles_df = pd.read_csv(smiles_path)
            if "smiles" not in self.smiles_df.columns:
                self.smiles_df = None

        # Cargar predicciones pre-calculadas si existen
        self.predictions_df = None
        predictions_path = Path(CHEMPROP_PREDICTIONS_PATH) if CHEMPROP_PREDICTIONS_PATH else None
        if predictions_path and predictions_path.exists():
            try:
                self.predictions_df = pd.read_csv(predictions_path)
                print(f"INFO: Predicciones pre-calculadas cargadas ({len(self.predictions_df)} registros)")
            except Exception as e:
                print(f"WARNING: Error cargando predicciones: {e}")

        self.feature_cols = [c for c in self.test_df.columns if c != "id"]

        # =====================================================================
        # CARGAR DATASETS PROCESADOS (train + test)
        # =====================================================================
        self.train_dataset: Optional[pd.DataFrame] = None
        self.test_dataset: Optional[pd.DataFrame] = None

        train_path = Path(TRAIN_DATASET_PATH) if TRAIN_DATASET_PATH else None
        test_path = Path(TEST_DATASET_PATH) if TEST_DATASET_PATH else None

        if train_path and train_path.exists():
            try:
                # Cargar solo columnas necesarias
                df = pd.read_csv(train_path, usecols=['id', 'SMILES', 'Tm', 'name'])
                df = df.rename(columns={'SMILES': 'smiles'})
                self.train_dataset = df
                print(f"INFO: Dataset train cargado ({len(self.train_dataset)} registros)")
            except Exception as e:
                print(f"WARNING: Error cargando train dataset: {e}")

        if test_path and test_path.exists():
            try:
                # Cargar solo columnas necesarias (test no tiene Tm)
                df = pd.read_csv(test_path, usecols=['id', 'SMILES', 'name'])
                df = df.rename(columns={'SMILES': 'smiles'})
                self.test_dataset = df
                print(f"INFO: Dataset test cargado ({len(self.test_dataset)} registros)")
            except Exception as e:
                print(f"WARNING: Error cargando test dataset: {e}")

        # Calcular predicciones
        self._predictions_cache: List[Tuple[int, float]] = []
        self._predictions_with_smiles: List[Dict[str, Any]] = []
        self._calculate_all_predictions()

        # Compuestos de usuarios
        self.user_compounds_path = Path(USER_COMPOUNDS_PATH)
        self._load_user_compounds()

        # Grupos funcionales
        self._init_functional_group_patterns()

    # =========================================================================
    # EXTRACCIÓN DE FEATURES PARA ENSEMBLE
    # =========================================================================
    
    def _extract_ensemble_features(self, smiles: str) -> Optional[np.ndarray]:
        """Extrae features moleculares para el ensemble."""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
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
            desc_names = [d[0] for d in RDKitDescriptors._descList]
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
            
        except Exception as e:
            print(f"WARNING: Error extrayendo features: {e}")
            return None

    # =========================================================================
    # PREDICCIÓN CON ENSEMBLE
    # =========================================================================
    
    def _predict_with_ensemble(self, smiles: str) -> Optional[float]:
        """Predice usando el ensemble de XGB + LGB + CAT."""
        if not self.use_ensemble or not self.ensemble_models:
            return None
        
        X = self._extract_ensemble_features(smiles)
        if X is None:
            return None
        
        X = X.reshape(1, -1)
        
        predictions = []
        weights = []
        
        for name, model_list in self.ensemble_models.items():
            if not model_list:
                continue
            
            weight = self.ensemble_weights.get(name, 1.0 / len(self.ensemble_models))
            
            # Promedio de los folds
            fold_preds = []
            for model in model_list:
                try:
                    pred = model.predict(X)[0]
                    fold_preds.append(pred)
                except Exception as e:
                    print(f"WARNING: Error prediciendo con {name}: {e}")
            
            if fold_preds:
                predictions.append(np.mean(fold_preds))
                weights.append(weight)
        
        if not predictions:
            return None
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return float(np.sum(np.array(predictions) * weights))

    # =========================================================================
    # PREDICCIÓN CON CHEMPROP
    # =========================================================================

    def _predict_with_chemprop(self, smiles: str) -> Optional[float]:
        """Predice usando ChemProp D-MPNN."""
        if not self.use_chemprop or not self.chemprop_checkpoints:
            return None
        
        import tempfile
        import csv
        
        temp_input = None
        temp_output = None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("smiles\n")
                f.write(f"{smiles}\n")
                temp_input = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_output = f.name
            
            try:
                from chemprop.train import make_predictions
                from chemprop.args import PredictArgs
                
                args = PredictArgs().parse_args([
                    '--test_path', temp_input,
                    '--preds_path', temp_output,
                    '--checkpoint_dir', str(self.chemprop_model_dir),
                    '--no_cuda',
                    '--num_workers', '0'
                ])
                
                preds = make_predictions(args=args)
                
                if preds and len(preds) > 0:
                    if isinstance(preds[0], (list, tuple)):
                        return float(preds[0][0])
                    return float(preds[0])
                        
            except SystemExit:
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    with open(temp_output, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            for key in ['target', 'Tm', 'pred', 'prediction']:
                                if key in row and row[key]:
                                    return float(row[key])
                        
            except Exception as e:
                print(f"WARNING: Error ChemProp: {e}")
                
        finally:
            if temp_input and os.path.exists(temp_input):
                os.unlink(temp_input)
            if temp_output and os.path.exists(temp_output):
                os.unlink(temp_output)
        
        return None

    # =========================================================================
    # PREDICCIÓN COMBINADA (PRINCIPAL)
    # =========================================================================

    def predict_melting_point(self, smiles: str) -> Dict[str, Any]:
        """
        Predice el punto de fusión combinando ensemble y ChemProp.
        
        Combinación óptima: 20% ChemProp + 80% Ensemble = MAE ~22.80 K
        
        Returns:
            Dict con Tm_pred, Tm_celsius, uncertainty, method, predictions
        """
        # Validar SMILES primero
        validation = self.validate_smiles(smiles)
        if not validation['valid']:
            raise SMILESValidationError(validation['error'])
        
        ensemble_pred = None
        chemprop_pred = None
        
        # Obtener predicciones
        if self.use_ensemble:
            ensemble_pred = self._predict_with_ensemble(smiles)
        
        if self.use_chemprop:
            chemprop_pred = self._predict_with_chemprop(smiles)
        
        # Combinar predicciones
        if ensemble_pred is not None and chemprop_pred is not None:
            # Combinación óptima: 20% ChemProp + 80% Ensemble
            final_pred = CHEMPROP_WEIGHT * chemprop_pred + (1 - CHEMPROP_WEIGHT) * ensemble_pred
            method = f"combined (cp={CHEMPROP_WEIGHT:.0%})"
            uncertainty = f"±{MODEL_MAE_COMBINED:.0f} K"
        elif ensemble_pred is not None:
            final_pred = ensemble_pred
            method = "ensemble_only"
            uncertainty = f"±{MODEL_MAE_ENSEMBLE_ONLY:.0f} K"
        elif chemprop_pred is not None:
            final_pred = chemprop_pred
            method = "chemprop_only"
            uncertainty = f"±{MODEL_MAE_CHEMPROP_ONLY:.0f} K"
        else:
            # Fallback al modelo sklearn original
            X = self._extract_ensemble_features(smiles)
            if X is not None:
                final_pred = float(self.model.predict(X.reshape(1, -1))[0])
            else:
                final_pred = 300.0  # Default
            method = "fallback"
            uncertainty = "±30 K"
        
        return {
            'Tm_pred': round(final_pred, 2),
            'Tm_celsius': round(final_pred - 273.15, 2),
            'uncertainty': uncertainty,
            'method': method,
            'predictions': {
                'ensemble': round(ensemble_pred, 2) if ensemble_pred else None,
                'chemprop': round(chemprop_pred, 2) if chemprop_pred else None,
                'final': round(final_pred, 2)
            }
        }

    # =========================================================================
    # GESTIÓN DE COMPUESTOS DE USUARIO
    # =========================================================================

    def add_user_compound(self, smiles: str, name: str) -> Dict[str, Any]:
        """Añade un compuesto de usuario con predicción."""
        # Validar SMILES
        validation = self.validate_smiles(smiles)
        if not validation['valid']:
            raise SMILESValidationError(validation['error'])
        
        # Usar SMILES canónico
        canonical_smiles = validation['canonical_smiles'] or smiles
        
        # Obtener predicción
        prediction = self.predict_melting_point(canonical_smiles)
        
        # Crear registro
        new_id = self._get_next_user_id()
        compound = {
            'id': new_id,
            'smiles': canonical_smiles,
            'name': name,
            'Tm_pred': prediction['Tm_pred'],
            'Tm_celsius': prediction['Tm_celsius'],
            'uncertainty': prediction['uncertainty'],
            'created_at': datetime.now().isoformat(),
            'source': 'user_input',
            'method': prediction['method']
        }
        
        # Añadir al DataFrame
        self.user_compounds_df = pd.concat([
            self.user_compounds_df,
            pd.DataFrame([compound])
        ], ignore_index=True)
        
        self._save_user_compounds()
        
        return compound

    # =========================================================================
    # INFORMACIÓN DEL MODELO
    # =========================================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Devuelve información del modelo."""
        return {
            "name": "ChemProp D-MPNN + Ensemble (XGB+LGB+CAT)",
            "type": "Hybrid GNN + GBDT",
            "mae": self.effective_mae,
            "mae_std": MODEL_STD,
            "folds": len(self.chemprop_checkpoints) if self.use_chemprop else 5,
            "epochs": 50,
            "hidden_size": 300,
            "depth": 6,
            "ensemble_enabled": self.use_ensemble,
            "chemprop_enabled": self.use_chemprop,
            "combination": f"{CHEMPROP_WEIGHT:.0%} ChemProp + {1-CHEMPROP_WEIGHT:.0%} Ensemble",
            "uncertainty_interval": f"±{self.effective_mae:.0f} K"
        }

    # =========================================================================
    # RESTO DE MÉTODOS (sin cambios significativos)
    # =========================================================================

    def _init_functional_group_patterns(self) -> None:
        """Inicializa patrones SMARTS para grupos funcionales."""
        self.functional_groups = [
            {"name": "Alcohols (OH)", "pattern": "[OX2H]", "smarts": "[OX2H]"},
            {"name": "Carboxylic Acids (COOH)", "pattern": "[CX3](=O)[OX2H1]", "smarts": "[CX3](=O)[OX2H1]"},
            {"name": "Amines (NH2)", "pattern": "[NX3;H2,H1;!$(NC=O)]", "smarts": "[NX3;H2,H1;!$(NC=O)]"},
            {"name": "Halogenated (F, Cl, Br, I)", "pattern": "[F,Cl,Br,I]", "smarts": "[F,Cl,Br,I]"},
            {"name": "Aromatic Rings", "pattern": "c1ccccc1", "smarts": "c1ccccc1"},
            {"name": "Hydrocarbons", "pattern": "[CX4]", "smarts": "[CX4]"}
        ]

    def _calculate_all_predictions(self) -> None:
        """
        Calcula todas las predicciones y las cachea.
        PRIORIDAD:
        1. Predicciones pre-calculadas de CSV (test_chemprop_predictions.csv)
        2. Modelo XGBoost (si está disponible)
        3. Valores placeholder
        """
        # ========================================
        # Método 1: Predicciones pre-calculadas
        # ========================================
        if self.predictions_df is not None:
            # Determinar nombre de columnas
            id_col = "id" if "id" in self.predictions_df.columns else self.predictions_df.columns[0]
            pred_col = None
            for col in ["Tm_pred", "target", "prediction", "pred"]:
                if col in self.predictions_df.columns:
                    pred_col = col
                    break
            if pred_col is None:
                pred_col = self.predictions_df.columns[1]
            
            self._predictions_cache = [
                (int(row[id_col]), float(row[pred_col])) 
                for _, row in self.predictions_df.iterrows()
            ]
            print(f"INFO: {len(self._predictions_cache)} predicciones cargadas de archivo pre-calculado.")
        
        # ========================================
        # Método 2: XGBoost (fallback)
        # ========================================
        elif self.model is not None and self.test_df is not None:
            try:
                X = self.test_df[self.feature_cols]
                ids = self.test_df["id"].tolist()
                
                # XGBoost 3.x requiere DataFrame con nombres de columnas
                if hasattr(self.model, 'feature_names_in_'):
                    X = pd.DataFrame(X.values, columns=self.model.feature_names_in_)
                
                preds = self.model.predict(X)
                
                self._predictions_cache = [
                    (int(sample_id), float(pred)) for sample_id, pred in zip(ids, preds)
                ]
                print(f"INFO: {len(self._predictions_cache)} predicciones calculadas con XGBoost.")
            except Exception as e:
                print(f"WARNING: Error con XGBoost: {e}")
                self._predictions_cache = []
        
        if self.smiles_df is not None:
            for sample_id, pred in self._predictions_cache:
                smiles_row = self.smiles_df[self.smiles_df["id"] == sample_id]
                smiles = smiles_row["smiles"].iloc[0] if not smiles_row.empty else None
                self._predictions_with_smiles.append({
                    "id": sample_id, "Tm_pred": pred, "smiles": smiles
                })
        else:
            for sample_id, pred in self._predictions_cache:
                self._predictions_with_smiles.append({
                    "id": sample_id, "Tm_pred": pred, "smiles": None
                })

    def _load_user_compounds(self) -> None:
        """Carga o crea DataFrame de compuestos de usuarios."""
        try:
            self.user_compounds_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.user_compounds_path.exists():
                self.user_compounds_df = pd.read_csv(self.user_compounds_path)
                required_cols = ['id', 'smiles', 'name', 'Tm_pred', 'Tm_celsius', 
                                'uncertainty', 'created_at', 'source']
                for col in required_cols:
                    if col not in self.user_compounds_df.columns:
                        self.user_compounds_df[col] = None
            else:
                self.user_compounds_df = pd.DataFrame(columns=[
                    'id', 'smiles', 'name', 'Tm_pred', 'Tm_celsius', 
                    'uncertainty', 'created_at', 'source'
                ])
                self.user_compounds_df.to_csv(self.user_compounds_path, index=False)
        except Exception as e:
            print(f"WARNING: Error cargando user_compounds: {e}")
            self.user_compounds_df = pd.DataFrame(columns=[
                'id', 'smiles', 'name', 'Tm_pred', 'Tm_celsius', 
                'uncertainty', 'created_at', 'source'
            ])

    def _save_user_compounds(self) -> None:
        """Guarda compuestos de usuarios."""
        try:
            self.user_compounds_path.parent.mkdir(parents=True, exist_ok=True)
            self.user_compounds_df.to_csv(self.user_compounds_path, index=False)
        except Exception as e:
            print(f"WARNING: Error guardando user_compounds: {e}")

    def _get_next_user_id(self) -> str:
        """Genera siguiente ID de usuario."""
        if self.user_compounds_df.empty:
            return "USR_001"
        
        existing_ids = self.user_compounds_df['id'].tolist()
        max_num = 0
        for uid in existing_ids:
            match = re.search(r'USR_(\d+)', str(uid))
            if match:
                max_num = max(max_num, int(match.group(1)))
        
        return f"USR_{max_num + 1:03d}"

    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """Valida un SMILES."""
        if not smiles or not smiles.strip():
            return {"valid": False, "error": "SMILES vacío", 
                    "canonical_smiles": None, "num_atoms": None, "molecular_weight": None}
        
        smiles = smiles.strip()
        
        if not RDKIT_AVAILABLE:
            valid_chars = set("CNOSPFClBrI[]()=#+-@/\\%0123456789cnospfclbri")
            if all(c in valid_chars for c in smiles):
                return {"valid": True, "canonical_smiles": smiles,
                        "num_atoms": len(re.findall(r'[A-Z][a-z]?', smiles)),
                        "molecular_weight": None, "error": None,
                        "warning": "RDKit no disponible"}
            return {"valid": False, "error": "Caracteres inválidos",
                    "canonical_smiles": None, "num_atoms": None, "molecular_weight": None}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"valid": False, "error": "SMILES inválido",
                        "canonical_smiles": None, "num_atoms": None, "molecular_weight": None}
            
            return {
                "valid": True,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "num_atoms": mol.GetNumAtoms(),
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "error": None
            }
        except Exception as e:
            return {"valid": False, "error": str(e),
                    "canonical_smiles": None, "num_atoms": None, "molecular_weight": None}

    def get_dataset_size(self) -> int:
        return len(self.test_df)

    def predict_by_id(self, sample_id: int) -> float:
        for sid, pred in self._predictions_cache:
            if sid == sample_id:
                return pred
        raise ValueError(f"ID {sample_id} no encontrado")

    def predict_all(self) -> List[Tuple[int, float]]:
        return self._predictions_cache

    def get_all_data(self) -> List[Dict[str, Any]]:
        """
        Retorna todos los datos: train (real), test (prediccion), user.

        Returns:
            Lista de diccionarios con id, smiles, Tm, source, name
        """
        all_data = []

        # Crear diccionario de predicciones para test
        predictions_dict = {sid: pred for sid, pred in self._predictions_cache}

        # 1. Datos de train (Tm REAL)
        if self.train_dataset is not None:
            for _, row in self.train_dataset.iterrows():
                name = row.get("name", None)
                if pd.isna(name):
                    name = None
                all_data.append({
                    "id": int(row["id"]),
                    "smiles": row.get("smiles", ""),
                    "Tm_pred": float(row["Tm"]),
                    "source": "train",
                    "name": name
                })

        # 2. Datos de test (Tm PREDICHO desde cache)
        if self.test_dataset is not None:
            for _, row in self.test_dataset.iterrows():
                sample_id = int(row["id"])
                tm_pred = predictions_dict.get(sample_id, None)
                if tm_pred is None:
                    continue  # Skip si no hay prediccion

                name = row.get("name", None)
                if pd.isna(name):
                    name = None
                all_data.append({
                    "id": sample_id,
                    "smiles": row.get("smiles", ""),
                    "Tm_pred": float(tm_pred),
                    "source": "test",
                    "name": name
                })

        # 3. Compuestos de usuario
        if not self.user_compounds_df.empty:
            for _, row in self.user_compounds_df.iterrows():
                name = row.get("name", None)
                if pd.isna(name):
                    name = None
                all_data.append({
                    "id": str(row["id"]),
                    "smiles": row.get("smiles", ""),
                    "Tm_pred": float(row["Tm_pred"]),
                    "source": "user",
                    "name": name
                })

        return all_data

    def get_stats(self) -> Dict[str, float]:
        preds = [p for _, p in self._predictions_cache]
        return {
            "count": len(preds), "mean": np.mean(preds), "std": np.std(preds),
            "min": np.min(preds), "max": np.max(preds), "median": np.median(preds),
            "q25": np.percentile(preds, 25), "q75": np.percentile(preds, 75),
            "variance": np.var(preds), "range": np.max(preds) - np.min(preds)
        }

    def get_predictions_in_range(self, min_tm: float, max_tm: float) -> Dict[str, Any]:
        filtered = [(id_, pred) for id_, pred in self._predictions_cache if min_tm <= pred <= max_tm]
        total = len(self._predictions_cache)
        return {
            "filter": {"min_tm": min_tm, "max_tm": max_tm},
            "count": len(filtered),
            "percentage": round((len(filtered) / total) * 100, 2) if total > 0 else 0,
            "predictions": [{"id": id_, "Tm_pred": pred} for id_, pred in filtered]
        }

    def get_user_compounds(self) -> Dict[str, Any]:
        compounds = self.user_compounds_df.to_dict('records')
        return {"total": len(compounds), "compounds": compounds}

    def delete_user_compound(self, compound_id: str) -> bool:
        initial_len = len(self.user_compounds_df)
        self.user_compounds_df = self.user_compounds_df[
            self.user_compounds_df['id'] != compound_id
        ]
        if len(self.user_compounds_df) < initial_len:
            self._save_user_compounds()
            return True
        return False

    def get_predictions_by_functional_group(self) -> Dict[str, Any]:
        """Análisis por grupos funcionales usando SMARTS patterns."""
        if not RDKIT_AVAILABLE:
            return {
                "total_molecules": 0,
                "groups": [],
                "note": "RDKit no disponible"
            }

        # Definir patrones SMARTS para grupos funcionales
        functional_group_patterns = [
            {"name": "Alcohols (OH)", "smarts": "[OX2H]"},
            {"name": "Carboxylic Acids", "smarts": "[CX3](=O)[OX2H1]"},
            {"name": "Amines (NH2/NHR)", "smarts": "[NX3;H2,H1;!$(NC=O)]"},
            {"name": "Amides", "smarts": "[NX3][CX3](=[OX1])"},
            {"name": "Esters", "smarts": "[CX3](=O)[OX2][#6]"},
            {"name": "Ketones", "smarts": "[CX3](=[OX1])([#6])[#6]"},
            {"name": "Aldehydes", "smarts": "[CX3H1](=O)[#6]"},
            {"name": "Ethers", "smarts": "[OD2]([#6])[#6]"},
            {"name": "Halogenated (F,Cl,Br,I)", "smarts": "[F,Cl,Br,I]"},
            {"name": "Aromatic Rings", "smarts": "c1ccccc1"},
            {"name": "Nitro Groups", "smarts": "[NX3](=O)=O"},
            {"name": "Sulfides/Thiols", "smarts": "[#16X2]"},
        ]

        # Obtener todos los datos con SMILES
        all_data = self.get_all_data()

        groups_data = []

        for fg in functional_group_patterns:
            try:
                pattern = Chem.MolFromSmarts(fg["smarts"])
                if pattern is None:
                    continue

                matching_tms = []

                for item in all_data:
                    smiles = item.get("smiles", "")
                    if not smiles:
                        continue

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    if mol.HasSubstructMatch(pattern):
                        matching_tms.append(item["Tm_pred"])

                if len(matching_tms) > 0:
                    groups_data.append({
                        "name": fg["name"],
                        "pattern": fg["smarts"],
                        "count": len(matching_tms),
                        "avg_tm": round(np.mean(matching_tms), 2),
                        "min_tm": round(min(matching_tms), 2),
                        "max_tm": round(max(matching_tms), 2),
                        "std_tm": round(np.std(matching_tms), 2) if len(matching_tms) > 1 else 0
                    })
            except Exception as e:
                print(f"WARNING: Error procesando grupo {fg['name']}: {e}")
                continue

        # Ordenar por cantidad descendente
        groups_data.sort(key=lambda x: x["count"], reverse=True)

        return {
            "total_molecules": len(all_data),
            "groups": groups_data,
            "note": "Análisis basado en patrones SMARTS"
        }

    def get_distribution(self) -> Dict[str, Any]:
        """Distribución por categorías de temperatura."""
        categories_config = [
            {"name": "Muy bajo (<150K)", "description": "Gases", "range": (0, 150)},
            {"name": "Bajo (150-250K)", "description": "Líquidos volátiles", "range": (150, 250)},
            {"name": "Medio (250-350K)", "description": "Líquidos/Sólidos", "range": (250, 350)},
            {"name": "Alto (350-450K)", "description": "Sólidos estables", "range": (350, 450)},
            {"name": "Muy alto (>450K)", "description": "Alto punto de fusión", "range": (450, 1000)}
        ]
        
        predictions = [pred for _, pred in self._predictions_cache]
        total = len(predictions)
        
        categories = []
        for config in categories_config:
            range_min, range_max = config["range"]
            count = len([p for p in predictions if range_min <= p < range_max])
            categories.append({
                "name": config["name"], "description": config["description"],
                "range_min": range_min, "range_max": range_max,
                "count": count, "percentage": round((count / total) * 100, 2) if total > 0 else 0
            })
        
        return {"total": total, "categories": categories}

    def get_predictions_by_molecule_size(self) -> Dict[str, Any]:
        """Análisis por tamaño molecular basado en longitud de SMILES."""
        # Definir rangos de tamaño (longitud de SMILES)
        size_bins = [
            {"name": "Muy pequeno (1-10)", "min": 1, "max": 10},
            {"name": "Pequeno (11-20)", "min": 11, "max": 20},
            {"name": "Mediano (21-35)", "min": 21, "max": 35},
            {"name": "Grande (36-50)", "min": 36, "max": 50},
            {"name": "Muy grande (51-75)", "min": 51, "max": 75},
            {"name": "Enorme (>75)", "min": 76, "max": 999},
        ]

        # Obtener todos los datos con SMILES
        all_data = self.get_all_data()

        size_groups = []

        for size_bin in size_bins:
            matching_tms = []

            for item in all_data:
                smiles = item.get("smiles", "")
                if not smiles:
                    continue

                smiles_length = len(smiles)
                if size_bin["min"] <= smiles_length <= size_bin["max"]:
                    matching_tms.append(item["Tm_pred"])

            if len(matching_tms) > 0:
                size_groups.append({
                    "name": size_bin["name"],
                    "smiles_length_min": size_bin["min"],
                    "smiles_length_max": size_bin["max"],
                    "count": len(matching_tms),
                    "avg_tm": round(np.mean(matching_tms), 2),
                    "min_tm": round(min(matching_tms), 2),
                    "max_tm": round(max(matching_tms), 2),
                    "std_tm": round(np.std(matching_tms), 2) if len(matching_tms) > 1 else 0
                })

        return {
            "total_molecules": len(all_data),
            "size_groups": size_groups,
            "note": "Tamaño basado en longitud de SMILES"
        }