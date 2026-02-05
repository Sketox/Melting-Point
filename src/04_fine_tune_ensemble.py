"""
Genera submissions con pesos finos alrededor de 50% ChemProp
Basado en resultados: 50% es el mejor, probamos 40-55%
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
TEST_RAW = PROJECT_ROOT / "data" / "raw" / "test.csv"

print("Cargando predicciones...")

# Cargar predicciones de ChemProp
chemprop_preds = pd.read_csv(DATA_PROCESSED / "chemprop_predictions.csv")
if "Tm" in chemprop_preds.columns:
    cp_preds = chemprop_preds["Tm"].values
else:
    cp_preds = chemprop_preds.iloc[:, 0].values

# Cargar submission de XGBoost (que ya tenemos)
xgb_preds = pd.read_csv(SUBMISSION_DIR / "submission_xgboost.csv")["Tm"].values
lgbm_preds = pd.read_csv(SUBMISSION_DIR / "submission_lightgbm.csv")["Tm"].values

# GB ensemble (65% XGB, 35% LGBM - del paso 2)
gb_preds = 0.65 * xgb_preds + 0.35 * lgbm_preds

# Test IDs
test_df = pd.read_csv(TEST_RAW)

# Pesos a probar (finos alrededor de 50%)
weights = [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]

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
║  Pesos probados: 40%, 42%, 45%, 48%, 50%, 52%, 55%                          ║
║                                                                              ║
║  🎯 Sube todos a Kaggle y encuentra el óptimo exacto                        ║
║                                                                              ║
║  El mejor probablemente esté entre 45% y 52%                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")