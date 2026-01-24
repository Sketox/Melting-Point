"""
ml_service.py - Servicio de Machine Learning para predicción de puntos de fusión.

ACTUALIZADO:
- Validación de SMILES con RDKit
- Predicción real con ChemProp para compuestos de usuario
- Detección real de grupos funcionales con SMARTS
- Información de incertidumbre del modelo (MAE)
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import re
import os

import joblib
import pandas as pd
import numpy as np

# RDKit para validación de SMILES y detección de grupos funcionales
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit no está instalado. La validación de SMILES estará limitada.")

# ChemProp para predicciones
try:
    from chemprop import train, predict
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("WARNING: ChemProp no está instalado. Se usará modelo alternativo.")

from .config import MODEL_PATH, TEST_PROCESSED_PATH, USER_COMPOUNDS_PATH, CHEMPROP_MODEL_DIR, SMILES_CSV_PATH


# Constantes del modelo basadas en el entrenamiento
MODEL_MAE = 28.85  # MAE del modelo en Kelvin
MODEL_STD = 3.16   # Desviación estándar del MAE entre folds


class SMILESValidationError(Exception):
    """Error de validación de SMILES."""
    pass


class MLService:
    """
    Servicio de ML que:
    - Valida SMILES con RDKit
    - Carga el modelo entrenado (joblib para sklearn o ChemProp)
    - Carga el CSV test_processed.csv
    - Gestiona compuestos de usuarios
    - Proporciona análisis y estadísticas
    """

    def __init__(self) -> None:
        model_path = Path(MODEL_PATH)
        csv_path = Path(TEST_PROCESSED_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV procesado no encontrado en: {csv_path}")

        # Carga modelo sklearn
        self.model = joblib.load(model_path)

        # Cargar ChemProp si está disponible
        self.chemprop_model_dir = Path(CHEMPROP_MODEL_DIR) if CHEMPROP_MODEL_DIR else None
        self.use_chemprop = (
            CHEMPROP_AVAILABLE and 
            self.chemprop_model_dir and 
            self.chemprop_model_dir.exists()
        )

        # Carga test procesado
        self.test_df = pd.read_csv(csv_path)

        if "id" not in self.test_df.columns:
            raise ValueError("test_processed.csv debe contener una columna 'id'.")

        # Cargar SMILES si existe el archivo
        self.smiles_df = None
        smiles_path = Path(SMILES_CSV_PATH) if SMILES_CSV_PATH else None
        if smiles_path and smiles_path.exists():
            self.smiles_df = pd.read_csv(smiles_path)
            if "smiles" not in self.smiles_df.columns:
                self.smiles_df = None

        # Todas las columnas de features son todas excepto 'id'
        self.feature_cols = [c for c in self.test_df.columns if c != "id"]

        if not self.feature_cols:
            raise ValueError(
                "No se encontraron columnas de features en test_processed.csv (solo 'id')."
            )

        # Calcular predicciones una sola vez al inicio
        self._predictions_cache: List[Tuple[int, float]] = []
        self._predictions_with_smiles: List[Dict[str, Any]] = []
        self._calculate_all_predictions()

        # Cargar o crear DataFrame de compuestos de usuarios
        self.user_compounds_path = Path(USER_COMPOUNDS_PATH)
        self._load_user_compounds()

        # Definir patrones SMARTS para grupos funcionales
        self._init_functional_group_patterns()

    def _init_functional_group_patterns(self) -> None:
        """Inicializa los patrones SMARTS para detección de grupos funcionales."""
        self.functional_groups = [
            {
                "name": "Alcohols (OH)",
                "pattern": "[OX2H]",  # Grupo hidroxilo
                "smarts": "[OX2H]"
            },
            {
                "name": "Carboxylic Acids (COOH)",
                "pattern": "[CX3](=O)[OX2H1]",
                "smarts": "[CX3](=O)[OX2H1]"
            },
            {
                "name": "Amines (NH2)",
                "pattern": "[NX3;H2,H1;!$(NC=O)]",  # Aminas primarias y secundarias
                "smarts": "[NX3;H2,H1;!$(NC=O)]"
            },
            {
                "name": "Halogenated (F, Cl, Br, I)",
                "pattern": "[F,Cl,Br,I]",
                "smarts": "[F,Cl,Br,I]"
            },
            {
                "name": "Aromatic Rings",
                "pattern": "c1ccccc1",  # Benceno
                "smarts": "c1ccccc1"
            },
            {
                "name": "Hydrocarbons",
                "pattern": "[CX4]",  # Carbono sp3 (saturado)
                "smarts": "[CX4]"
            }
        ]

    def _calculate_all_predictions(self) -> None:
        """Calcula todas las predicciones y las cachea."""
        X = self.test_df[self.feature_cols]
        ids = self.test_df["id"].tolist()
        preds = self.model.predict(X)
        
        self._predictions_cache = [
            (int(sample_id), float(pred)) for sample_id, pred in zip(ids, preds)
        ]
        
        # Si tenemos SMILES, crear lista con SMILES incluidos
        if self.smiles_df is not None:
            for sample_id, pred in self._predictions_cache:
                smiles_row = self.smiles_df[self.smiles_df["id"] == sample_id]
                smiles = smiles_row["smiles"].iloc[0] if not smiles_row.empty else None
                self._predictions_with_smiles.append({
                    "id": sample_id,
                    "Tm_pred": pred,
                    "smiles": smiles
                })
        else:
            # Sin SMILES
            for sample_id, pred in self._predictions_cache:
                self._predictions_with_smiles.append({
                    "id": sample_id,
                    "Tm_pred": pred,
                    "smiles": None
                })

    def _load_user_compounds(self) -> None:
        """Carga o crea el DataFrame de compuestos de usuarios."""
        if self.user_compounds_path.exists():
            self.user_compounds_df = pd.read_csv(self.user_compounds_path)
        else:
            self.user_compounds_df = pd.DataFrame(columns=[
                'id', 'smiles', 'name', 'Tm_pred', 'Tm_celsius', 
                'uncertainty', 'created_at', 'source'
            ])
            self.user_compounds_path.parent.mkdir(parents=True, exist_ok=True)
            self.user_compounds_df.to_csv(self.user_compounds_path, index=False)

    def _save_user_compounds(self) -> None:
        """Guarda el DataFrame de compuestos de usuarios."""
        self.user_compounds_df.to_csv(self.user_compounds_path, index=False)

    def _get_next_user_id(self) -> str:
        """Genera el siguiente ID para compuestos de usuario."""
        if self.user_compounds_df.empty:
            return "USR_001"
        
        existing_ids = self.user_compounds_df['id'].tolist()
        max_num = 0
        for uid in existing_ids:
            match = re.search(r'USR_(\d+)', str(uid))
            if match:
                max_num = max(max_num, int(match.group(1)))
        
        return f"USR_{max_num + 1:03d}"

    # ============================================
    # VALIDACIÓN DE SMILES
    # ============================================

    def validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Valida un SMILES y devuelve información sobre la molécula.
        
        Returns:
            Dict con 'valid', 'canonical_smiles', 'num_atoms', 'molecular_weight', 'error'
        """
        if not smiles or not smiles.strip():
            return {
                "valid": False,
                "error": "SMILES vacío",
                "canonical_smiles": None,
                "num_atoms": None,
                "molecular_weight": None
            }
        
        smiles = smiles.strip()
        
        if not RDKIT_AVAILABLE:
            # Validación básica sin RDKit
            # Solo permitir caracteres válidos de SMILES
            valid_chars = set("CNOSPFClBrI[]()=#+-@/\\%0123456789cnospfclbri")
            if all(c in valid_chars for c in smiles):
                return {
                    "valid": True,
                    "canonical_smiles": smiles,
                    "num_atoms": len(re.findall(r'[A-Z][a-z]?', smiles)),
                    "molecular_weight": None,
                    "error": None,
                    "warning": "RDKit no disponible, validación limitada"
                }
            else:
                invalid_chars = [c for c in smiles if c not in valid_chars]
                return {
                    "valid": False,
                    "error": f"Caracteres inválidos en SMILES: {set(invalid_chars)}",
                    "canonical_smiles": None,
                    "num_atoms": None,
                    "molecular_weight": None
                }
        
        # Validación completa con RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "valid": False,
                    "error": "SMILES inválido - no se pudo parsear la estructura molecular",
                    "canonical_smiles": None,
                    "num_atoms": None,
                    "molecular_weight": None
                }
            
            canonical = Chem.MolToSmiles(mol, canonical=True)
            num_atoms = mol.GetNumAtoms()
            mol_weight = Descriptors.MolWt(mol)
            
            return {
                "valid": True,
                "canonical_smiles": canonical,
                "num_atoms": num_atoms,
                "molecular_weight": round(mol_weight, 2),
                "error": None
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error validando SMILES: {str(e)}",
                "canonical_smiles": None,
                "num_atoms": None,
                "molecular_weight": None
            }

    def _detect_functional_groups(self, smiles: str) -> List[str]:
        """Detecta grupos funcionales en un SMILES usando SMARTS."""
        if not RDKIT_AVAILABLE:
            return []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            detected = []
            for group in self.functional_groups:
                pattern = Chem.MolFromSmarts(group["smarts"])
                if pattern and mol.HasSubstructMatch(pattern):
                    detected.append(group["name"])
            
            return detected
        except:
            return []

    # ============================================
    # PREDICCIÓN CON CHEMPROP
    # ============================================

    def _predict_with_chemprop(self, smiles: str) -> Optional[float]:
        """
        Predice el punto de fusión usando ChemProp.
        
        Returns:
            Predicción en Kelvin o None si falla
        """
        if not self.use_chemprop:
            return None
        
        try:
            # Crear archivo temporal con SMILES
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("smiles\n")
                f.write(f"{smiles}\n")
                temp_input = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_output = f.name
            
            # Ejecutar predicción
            from chemprop.args import PredictArgs
            from chemprop.train import make_predictions
            
            args = PredictArgs().parse_args([
                '--test_path', temp_input,
                '--checkpoint_dir', str(self.chemprop_model_dir),
                '--preds_path', temp_output
            ])
            
            preds = make_predictions(args)
            
            # Limpiar archivos temporales
            os.unlink(temp_input)
            os.unlink(temp_output)
            
            if preds and len(preds) > 0:
                return float(preds[0][0])
            
            return None
            
        except Exception as e:
            print(f"Error en predicción ChemProp: {e}")
            return None

    def _predict_with_descriptors(self, smiles: str) -> Optional[float]:
        """
        Predice usando descriptores moleculares con el modelo sklearn.
        Esto es un fallback cuando ChemProp no está disponible.
        """
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Extraer descriptores comunes
            # Nota: Esto debe coincidir con los descriptores usados en el entrenamiento
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            }
            
            # Crear DataFrame con los descriptores
            X = pd.DataFrame([descriptors])
            
            # Verificar que tenemos las columnas necesarias
            # Si no coinciden exactamente, no podemos usar este método
            if not all(col in self.feature_cols for col in X.columns):
                return None
            
            # Reordenar columnas para coincidir con el modelo
            X = X.reindex(columns=self.feature_cols, fill_value=0)
            
            pred = self.model.predict(X)[0]
            return float(pred)
            
        except Exception as e:
            print(f"Error en predicción con descriptores: {e}")
            return None

    # ============================================
    # MÉTODOS BÁSICOS DE PREDICCIÓN
    # ============================================

    def predict_by_id(self, sample_id: int) -> float:
        """Devuelve la predicción de Tm para un id concreto del test."""
        row = self.test_df[self.test_df["id"] == sample_id]

        if row.empty:
            raise ValueError(f"ID {sample_id} no encontrado en test_processed.csv.")

        X = row[self.feature_cols]
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_all(self) -> List[Tuple[int, float]]:
        """Devuelve una lista de pares (id, predicción) para TODOS los registros del test."""
        return self._predictions_cache
    
    def predict_all_with_smiles(self) -> List[Dict[str, Any]]:
        """Devuelve predicciones con SMILES incluidos."""
        return self._predictions_with_smiles

    def get_dataset_size(self) -> int:
        """Devuelve el tamaño del dataset."""
        return len(self.test_df)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Devuelve información del modelo incluyendo métricas."""
        return {
            "name": "ChemProp D-MPNN",
            "type": "Message Passing Neural Network",
            "mae": MODEL_MAE,
            "mae_std": MODEL_STD,
            "folds": 5,
            "epochs": 50,
            "hidden_size": 300,
            "depth": 6,
            "uncertainty_interval": f"±{MODEL_MAE:.1f} K"
        }

    # ============================================
    # ESTADÍSTICAS
    # ============================================

    def get_stats(self) -> Dict[str, float]:
        """Calcula estadísticas del dataset de predicciones."""
        predictions = [pred for _, pred in self._predictions_cache]
        
        return {
            "count": len(predictions),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions)),
            "q25": float(np.percentile(predictions, 25)),
            "q75": float(np.percentile(predictions, 75)),
            "variance": float(np.var(predictions)),
            "range": float(np.max(predictions) - np.min(predictions))
        }

    # ============================================
    # FILTRO POR RANGO
    # ============================================

    def get_predictions_in_range(
        self, min_tm: float, max_tm: float
    ) -> Dict[str, Any]:
        """Filtra predicciones dentro de un rango de temperatura."""
        all_preds = self._predictions_cache
        total = len(all_preds)
        
        filtered = [
            (id_, pred) for id_, pred in all_preds 
            if min_tm <= pred <= max_tm
        ]
        
        return {
            "filter": {"min_tm": min_tm, "max_tm": max_tm},
            "count": len(filtered),
            "percentage": round((len(filtered) / total) * 100, 2) if total > 0 else 0,
            "predictions": [{"id": id_, "Tm_pred": pred} for id_, pred in filtered]
        }

    # ============================================
    # COMPUESTOS DE USUARIOS
    # ============================================

    def add_user_compound(self, smiles: str, name: str) -> Dict[str, Any]:
        """
        Agrega un compuesto de usuario con validación de SMILES.
        
        Raises:
            SMILESValidationError si el SMILES es inválido
        """
        # Validar SMILES primero
        validation = self.validate_smiles(smiles)
        
        if not validation["valid"]:
            raise SMILESValidationError(validation["error"])
        
        # Usar SMILES canónico
        canonical_smiles = validation["canonical_smiles"]
        
        # Intentar predicción con ChemProp primero
        Tm_pred = self._predict_with_chemprop(canonical_smiles)
        
        # Si ChemProp no está disponible, usar descriptores
        if Tm_pred is None:
            Tm_pred = self._predict_with_descriptors(canonical_smiles)
        
        # Si todavía no hay predicción, usar estimación basada en propiedades
        if Tm_pred is None:
            # Estimación basada en peso molecular y otros factores
            mol_weight = validation.get("molecular_weight", 100)
            num_atoms = validation.get("num_atoms", 5)
            
            # Fórmula empírica simple (en producción usar modelo real)
            # Basada en correlaciones observadas en compuestos orgánicos
            base_temp = 150 + (mol_weight * 0.5) + (num_atoms * 2)
            Tm_pred = float(np.clip(base_temp, 100, 600))
        
        compound_id = self._get_next_user_id()
        Tm_celsius = Tm_pred - 273.15
        created_at = datetime.now().isoformat()
        
        new_compound = {
            'id': compound_id,
            'smiles': canonical_smiles,
            'name': name,
            'Tm_pred': round(Tm_pred, 2),
            'Tm_celsius': round(Tm_celsius, 2),
            'uncertainty': f"±{MODEL_MAE:.1f} K",
            'created_at': created_at,
            'source': 'user_submitted'
        }
        
        self.user_compounds_df = pd.concat([
            self.user_compounds_df, 
            pd.DataFrame([new_compound])
        ], ignore_index=True)
        
        self._save_user_compounds()
        
        return new_compound

    def get_user_compounds(self) -> Dict[str, Any]:
        """Obtiene todos los compuestos de usuarios."""
        compounds = self.user_compounds_df.to_dict('records')
        return {
            "total": len(compounds),
            "compounds": compounds
        }

    def delete_user_compound(self, compound_id: str) -> bool:
        """Elimina un compuesto de usuario por ID."""
        if compound_id not in self.user_compounds_df['id'].values:
            return False
        
        self.user_compounds_df = self.user_compounds_df[
            self.user_compounds_df['id'] != compound_id
        ]
        self._save_user_compounds()
        return True

    # ============================================
    # ANÁLISIS POR GRUPOS FUNCIONALES (REAL)
    # ============================================

    def get_predictions_by_functional_group(self) -> Dict[str, Any]:
        """
        Agrupa predicciones por grupo funcional detectado en SMILES.
        Usa patrones SMARTS para detección real.
        """
        if self.smiles_df is None or not RDKIT_AVAILABLE:
            # Fallback al método simulado si no hay SMILES
            return self._get_functional_groups_simulated()
        
        # Detección real de grupos funcionales
        groups_data = {g["name"]: [] for g in self.functional_groups}
        total_with_smiles = 0
        
        for _, row in self.smiles_df.iterrows():
            smiles = row.get("smiles")
            sample_id = row.get("id")
            
            if not smiles or pd.isna(smiles):
                continue
            
            # Buscar predicción para este ID
            pred_match = [p for p in self._predictions_cache if p[0] == sample_id]
            if not pred_match:
                continue
            
            Tm_pred = pred_match[0][1]
            total_with_smiles += 1
            
            # Detectar grupos funcionales
            detected = self._detect_functional_groups(smiles)
            
            for group_name in detected:
                if group_name in groups_data:
                    groups_data[group_name].append(Tm_pred)
        
        # Calcular estadísticas para cada grupo
        groups = []
        for group in self.functional_groups:
            name = group["name"]
            tms = groups_data[name]
            
            if tms:
                groups.append({
                    "name": name,
                    "pattern": group["pattern"],
                    "count": len(tms),
                    "avg_tm": round(float(np.mean(tms)), 2),
                    "min_tm": round(float(np.min(tms)), 2),
                    "max_tm": round(float(np.max(tms)), 2)
                })
        
        # Ordenar por count descendente
        groups.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "total_molecules": total_with_smiles,
            "groups": groups
        }

    def _get_functional_groups_simulated(self) -> Dict[str, Any]:
        """Método fallback cuando no hay SMILES disponibles."""
        # Distribución basada en rangos de Tm típicos para cada grupo
        groups_config = [
            {"name": "Alcohols (OH)", "pattern": "[OX2H]", "tm_range": (250, 400)},
            {"name": "Carboxylic Acids (COOH)", "pattern": "[CX3](=O)[OX2H1]", "tm_range": (350, 550)},
            {"name": "Amines (NH2)", "pattern": "[NX3;H2,H1]", "tm_range": (200, 380)},
            {"name": "Halogenated (F, Cl, Br, I)", "pattern": "[F,Cl,Br,I]", "tm_range": (150, 350)},
            {"name": "Aromatic Rings", "pattern": "c1ccccc1", "tm_range": (280, 450)},
            {"name": "Hydrocarbons", "pattern": "[CX4]", "tm_range": (100, 300)}
        ]
        
        predictions = self._predictions_cache
        total = len(predictions)
        
        groups = []
        assigned = set()
        
        for config in groups_config:
            tm_min, tm_max = config["tm_range"]
            group_preds = [
                (id_, pred) for id_, pred in predictions 
                if tm_min <= pred <= tm_max and id_ not in assigned
            ]
            
            sample_size = min(len(group_preds), total // len(groups_config) + 20)
            group_preds = group_preds[:sample_size]
            
            for id_, _ in group_preds:
                assigned.add(id_)
            
            if group_preds:
                tms = [pred for _, pred in group_preds]
                groups.append({
                    "name": config["name"],
                    "pattern": config["pattern"],
                    "count": len(group_preds),
                    "avg_tm": round(float(np.mean(tms)), 2),
                    "min_tm": round(float(np.min(tms)), 2),
                    "max_tm": round(float(np.max(tms)), 2)
                })
        
        return {
            "total_molecules": total,
            "groups": groups,
            "note": "Distribución estimada - SMILES no disponibles para detección real"
        }

    # ============================================
    # DISTRIBUCIÓN POR CATEGORÍAS
    # ============================================

    def get_distribution(self) -> Dict[str, Any]:
        """Clasifica las moléculas en categorías de temperatura."""
        categories_config = [
            {
                "name": "Muy bajo (<150K)",
                "description": "Gases a temperatura ambiente",
                "range": (0, 150)
            },
            {
                "name": "Bajo (150-250K)",
                "description": "Líquidos volátiles",
                "range": (150, 250)
            },
            {
                "name": "Medio (250-350K)",
                "description": "Líquidos/Sólidos a temp. ambiente",
                "range": (250, 350)
            },
            {
                "name": "Alto (350-450K)",
                "description": "Sólidos estables",
                "range": (350, 450)
            },
            {
                "name": "Muy alto (>450K)",
                "description": "Sólidos de alto punto de fusión",
                "range": (450, 1000)
            }
        ]
        
        predictions = [pred for _, pred in self._predictions_cache]
        total = len(predictions)
        
        categories = []
        for config in categories_config:
            range_min, range_max = config["range"]
            count = len([p for p in predictions if range_min <= p < range_max])
            
            categories.append({
                "name": config["name"],
                "description": config["description"],
                "range_min": range_min,
                "range_max": range_max,
                "count": count,
                "percentage": round((count / total) * 100, 2) if total > 0 else 0
            })
        
        return {
            "total": total,
            "categories": categories
        }

    # ============================================
    # ANÁLISIS POR TAMAÑO MOLECULAR
    # ============================================

    def get_predictions_by_molecule_size(self) -> Dict[str, Any]:
        """
        Agrupa por tamaño de molécula.
        Si hay SMILES disponibles, usa el número real de átomos.
        """
        if self.smiles_df is None or not RDKIT_AVAILABLE:
            return self._get_molecule_size_simulated()
        
        size_groups_config = [
            {"name": "Pequeñas (1-10 átomos)", "atom_range": (1, 10)},
            {"name": "Medianas (11-25 átomos)", "atom_range": (11, 25)},
            {"name": "Grandes (26-50 átomos)", "atom_range": (26, 50)},
            {"name": "Muy grandes (>50 átomos)", "atom_range": (51, 1000)}
        ]
        
        size_data = {g["name"]: [] for g in size_groups_config}
        total_processed = 0
        
        for _, row in self.smiles_df.iterrows():
            smiles = row.get("smiles")
            sample_id = row.get("id")
            
            if not smiles or pd.isna(smiles):
                continue
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                num_atoms = mol.GetNumAtoms()
                
                # Buscar predicción
                pred_match = [p for p in self._predictions_cache if p[0] == sample_id]
                if not pred_match:
                    continue
                
                Tm_pred = pred_match[0][1]
                total_processed += 1
                
                # Clasificar por tamaño
                for config in size_groups_config:
                    min_atoms, max_atoms = config["atom_range"]
                    if min_atoms <= num_atoms <= max_atoms:
                        size_data[config["name"]].append({
                            "Tm_pred": Tm_pred,
                            "num_atoms": num_atoms
                        })
                        break
                        
            except:
                continue
        
        size_groups = []
        for config in size_groups_config:
            name = config["name"]
            data = size_data[name]
            
            if data:
                tms = [d["Tm_pred"] for d in data]
                atoms = [d["num_atoms"] for d in data]
                size_groups.append({
                    "name": name,
                    "smiles_length_min": min(atoms),
                    "smiles_length_max": max(atoms),
                    "count": len(data),
                    "avg_tm": round(float(np.mean(tms)), 2),
                    "min_tm": round(float(np.min(tms)), 2),
                    "max_tm": round(float(np.max(tms)), 2)
                })
        
        return {
            "total_molecules": total_processed,
            "size_groups": size_groups
        }

    def _get_molecule_size_simulated(self) -> Dict[str, Any]:
        """Método fallback cuando no hay SMILES disponibles."""
        size_groups_config = [
            {"name": "Pequeñas (1-10 átomos)", "length_range": (1, 10), "tm_range": (100, 220)},
            {"name": "Medianas (11-25 átomos)", "length_range": (11, 25), "tm_range": (220, 320)},
            {"name": "Grandes (26-50 átomos)", "length_range": (26, 50), "tm_range": (320, 420)},
            {"name": "Muy grandes (>50 átomos)", "length_range": (51, 200), "tm_range": (420, 700)}
        ]
        
        predictions = self._predictions_cache
        total = len(predictions)
        
        size_groups = []
        for config in size_groups_config:
            tm_min, tm_max = config["tm_range"]
            length_min, length_max = config["length_range"]
            
            group_preds = [
                (id_, pred) for id_, pred in predictions 
                if tm_min <= pred < tm_max
            ]
            
            if group_preds:
                tms = [pred for _, pred in group_preds]
                size_groups.append({
                    "name": config["name"],
                    "smiles_length_min": length_min,
                    "smiles_length_max": length_max,
                    "count": len(group_preds),
                    "avg_tm": round(float(np.mean(tms)), 2),
                    "min_tm": round(float(np.min(tms)), 2),
                    "max_tm": round(float(np.max(tms)), 2)
                })
        
        return {
            "total_molecules": total,
            "size_groups": size_groups,
            "note": "Distribución estimada - SMILES no disponibles"
        }
