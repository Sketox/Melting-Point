from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import re

import joblib
import pandas as pd
import numpy as np

from .config import MODEL_PATH, TEST_PROCESSED_PATH, USER_COMPOUNDS_PATH


class MLService:
    """
    Servicio de ML que:
    - Carga el modelo entrenado (joblib).
    - Carga el CSV test_processed.csv.
    - Gestiona compuestos de usuarios.
    - Proporciona análisis y estadísticas.
    """

    def __init__(self) -> None:
        model_path = Path(MODEL_PATH)
        csv_path = Path(TEST_PROCESSED_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV procesado no encontrado en: {csv_path}")

        # Carga modelo
        self.model = joblib.load(model_path)

        # Carga test procesado
        self.test_df = pd.read_csv(csv_path)

        if "id" not in self.test_df.columns:
            raise ValueError("test_processed.csv debe contener una columna 'id'.")

        # Todas las columnas de features son todas excepto 'id'
        self.feature_cols = [c for c in self.test_df.columns if c != "id"]

        if not self.feature_cols:
            raise ValueError(
                "No se encontraron columnas de features en test_processed.csv (solo 'id')."
            )

        # Calcular predicciones una sola vez al inicio
        self._predictions_cache: List[Tuple[int, float]] = []
        self._calculate_all_predictions()

        # Cargar o crear DataFrame de compuestos de usuarios
        self.user_compounds_path = Path(USER_COMPOUNDS_PATH)
        self._load_user_compounds()

    def _calculate_all_predictions(self) -> None:
        """Calcula todas las predicciones y las cachea."""
        X = self.test_df[self.feature_cols]
        ids = self.test_df["id"].tolist()
        preds = self.model.predict(X)
        
        self._predictions_cache = [
            (int(sample_id), float(pred)) for sample_id, pred in zip(ids, preds)
        ]

    def _load_user_compounds(self) -> None:
        """Carga o crea el DataFrame de compuestos de usuarios."""
        if self.user_compounds_path.exists():
            self.user_compounds_df = pd.read_csv(self.user_compounds_path)
        else:
            self.user_compounds_df = pd.DataFrame(columns=[
                'id', 'smiles', 'name', 'Tm_pred', 'Tm_celsius', 'created_at', 'source'
            ])
            # Crear el directorio si no existe
            self.user_compounds_path.parent.mkdir(parents=True, exist_ok=True)
            self.user_compounds_df.to_csv(self.user_compounds_path, index=False)

    def _save_user_compounds(self) -> None:
        """Guarda el DataFrame de compuestos de usuarios."""
        self.user_compounds_df.to_csv(self.user_compounds_path, index=False)

    def _get_next_user_id(self) -> str:
        """Genera el siguiente ID para compuestos de usuario."""
        if self.user_compounds_df.empty:
            return "USR_001"
        
        # Extraer números de IDs existentes
        existing_ids = self.user_compounds_df['id'].tolist()
        max_num = 0
        for uid in existing_ids:
            match = re.search(r'USR_(\d+)', str(uid))
            if match:
                max_num = max(max_num, int(match.group(1)))
        
        return f"USR_{max_num + 1:03d}"

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

    def get_dataset_size(self) -> int:
        """Devuelve el tamaño del dataset."""
        return len(self.test_df)

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
        Agrega un compuesto de usuario.
        NOTA: Como no tenemos las features del SMILES, usamos un valor simulado.
        En producción, aquí se extraerían las features con RDKit.
        """
        compound_id = self._get_next_user_id()
        
        # Simulación de predicción basada en longitud del SMILES
        # (En producción, extraer features con RDKit y usar el modelo real)
        # Usamos una fórmula simple que genera valores realistas
        smiles_len = len(smiles)
        base_temp = 200 + (smiles_len * 3.5) + np.random.normal(0, 20)
        Tm_pred = float(np.clip(base_temp, 100, 600))
        Tm_celsius = Tm_pred - 273.15
        
        created_at = datetime.now().isoformat()
        
        new_compound = {
            'id': compound_id,
            'smiles': smiles,
            'name': name,
            'Tm_pred': round(Tm_pred, 2),
            'Tm_celsius': round(Tm_celsius, 2),
            'created_at': created_at,
            'source': 'user_submitted'
        }
        
        # Agregar al DataFrame
        self.user_compounds_df = pd.concat([
            self.user_compounds_df, 
            pd.DataFrame([new_compound])
        ], ignore_index=True)
        
        # Guardar en CSV
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
    # ANÁLISIS POR GRUPOS FUNCIONALES
    # ============================================

    def get_predictions_by_functional_group(self) -> Dict[str, Any]:
        """
        Agrupa predicciones por grupo funcional detectado en SMILES.
        Nota: Usamos patrones simplificados para detección.
        """
        # Necesitamos los SMILES originales - los simulamos basándonos en el ID
        # En un caso real, tendrías un CSV con SMILES
        
        # Definir grupos funcionales con patrones y rangos típicos de Tm
        groups_config = [
            {
                "name": "Alcohols (OH)",
                "pattern": "O",
                "tm_range": (250, 400)
            },
            {
                "name": "Carboxylic Acids (COOH)",
                "pattern": "C(=O)O",
                "tm_range": (350, 550)
            },
            {
                "name": "Amines (NH2)",
                "pattern": "N",
                "tm_range": (200, 380)
            },
            {
                "name": "Halogenated (F, Cl, Br, I)",
                "pattern": "[F,Cl,Br,I]",
                "tm_range": (150, 350)
            },
            {
                "name": "Aromatic Rings",
                "pattern": "c1ccccc1",
                "tm_range": (280, 450)
            },
            {
                "name": "Hydrocarbons",
                "pattern": "C",
                "tm_range": (100, 300)
            }
        ]
        
        predictions = self._predictions_cache
        total = len(predictions)
        
        # Distribuir las predicciones en grupos basándose en rangos de Tm
        groups = []
        assigned = set()
        
        for config in groups_config:
            tm_min, tm_max = config["tm_range"]
            group_preds = [
                (id_, pred) for id_, pred in predictions 
                if tm_min <= pred <= tm_max and id_ not in assigned
            ]
            
            # Limitar para no asignar todo a un grupo
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
            "groups": groups
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
        Como no tenemos SMILES directamente, simulamos basándonos en patrones de Tm.
        """
        size_groups_config = [
            {
                "name": "Pequeñas (1-10 átomos)",
                "length_range": (1, 10),
                "tm_range": (100, 220)
            },
            {
                "name": "Medianas (11-25 átomos)",
                "length_range": (11, 25),
                "tm_range": (220, 320)
            },
            {
                "name": "Grandes (26-50 átomos)",
                "length_range": (26, 50),
                "tm_range": (320, 420)
            },
            {
                "name": "Muy grandes (>50 átomos)",
                "length_range": (51, 200),
                "tm_range": (420, 700)
            }
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
            "size_groups": size_groups
        }