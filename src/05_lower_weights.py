"""
Genera submissions con pesos más bajos de ChemProp
Tendencia: menos ChemProp = mejor resultado
Probamos: 30%, 32%, 35%, 38%
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
TEST_RAW = PROJECT_ROOT / "data" / "raw" / "test.csv"

print("Cargando predicciones...")

# Cargar predicciones
cp_preds = pd.read_csv(DATA_PROCESSED / "chemprop_predictions.csv")
cp_preds = cp_preds["Tm"].values if "Tm" in cp_preds.columns else cp_preds.iloc[:, 0].values

xgb_preds = pd.read_csv(SUBMISSION_DIR / "submission_xgboost.csv")["Tm"].values
lgbm_preds = pd.read_csv(SUBMISSION_DIR / "submission_lightgbm.csv")["Tm"].values

# GB ensemble (65% XGB, 35% LGBM)
gb_preds = 0.65 * xgb_preds + 0.35 * lgbm_preds

test_df = pd.read_csv(TEST_RAW)

# Pesos más bajos
weights = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38]

print(f"\nGenerando {len(weights)} submissions...")

for w_cp in weights:
    w_gb = 1 - w_cp
    ensemble = w_cp * cp_preds + w_gb * gb_preds
    
    filename = f"submission_cp{int(w_cp*100)}.csv"
    filepath = SUBMISSION_DIR / filename
    
    pd.DataFrame({
        "id": test_df["id"],
        "Tm": ensemble
    }).to_csv(filepath, index=False)
    
    print(f"  ✓ {filename} (ChemProp={w_cp:.0%}, GB={w_gb:.0%})")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅ SUBMISSIONS GENERADOS                                                    ║
║                                                                              ║
║  Pesos: 25%, 28%, 30%, 32%, 35%, 38%                                        ║
║                                                                              ║
║  Si la tendencia continúa, el óptimo estará entre 30-38%                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")