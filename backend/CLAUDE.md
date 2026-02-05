# MeltingPoint Backend - CLAUDE.md

## Descripción
Backend FastAPI para predicción de puntos de fusión moleculares usando **Ensemble (XGB+LGB+CAT) + ChemProp D-MPNN**.

## 🎯 Rendimiento del Modelo

| Configuración | MAE (K) | Estado |
|---------------|---------|--------|
| ChemProp solo | 28.85 | ✅ Disponible |
| Ensemble solo | 26.64 | ✅ Disponible |
| **Ensemble + ChemProp** | **22.80** | ⭐ **Mejor (Kaggle)** |

## Instalación en Nueva Computadora

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. ⚠️ OBLIGATORIO: Aplicar parche para PyTorch 2.6+
python patch_chemprop_torch.py

# 4. Entrenar ensemble (si no existe)
cd ../src
python train_ensemble_production.py
cd ../backend

# 5. Ejecutar servidor
uvicorn app.main:app --reload --port 8000
```

### Verificar instalación correcta
Debes ver en los logs:
```
INFO: ChemProp 1.6.1 detectado correctamente.
INFO: ChemProp habilitado con 5 checkpoints.
INFO: Ensemble cargado con 15 modelos.
INFO: Modo COMBINADO activo (MAE ~22.80 K)
```

## Stack Tecnológico
- **Framework**: FastAPI 0.121+
- **ML Models**: 
  - ChemProp 1.6.1 (D-MPNN)
  - XGBoost, LightGBM, CatBoost (Ensemble)
- **Química**: RDKit 2025.9+
- **Data**: pandas, numpy, joblib
- **Python**: 3.11+

## Estructura
```
backend/
├── app/
│   ├── main.py           # FastAPI endpoints
│   ├── ml_service.py     # Predicciones (Ensemble + ChemProp)
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Configuración
├── models/
│   ├── model_chemprop/   # 5 folds ChemProp
│   │   ├── fold_0/model_0/model.pt
│   │   ├── fold_1/model_0/model.pt
│   │   ├── fold_2/model_0/model.pt
│   │   ├── fold_3/model_0/model.pt
│   │   └── fold_4/model_0/model.pt
│   ├── ensemble_predictor.joblib  # ⬅️ XGB+LGB+CAT
│   ├── best_params_paso6.json     # Hiperparámetros
│   └── model.joblib               # Fallback sklearn
├── patch_chemprop_torch.py
└── requirements.txt
```

## Modelo Híbrido

### ChemProp D-MPNN
| Parámetro | Valor |
|-----------|-------|
| Hidden Size | 300 |
| Depth | 6 |
| Folds | 5 |
| MAE | 28.85 K |

### Ensemble (XGB + LGB + CAT)
| Modelo | Peso | MAE Individual |
|--------|------|----------------|
| XGBoost | 35% | ~28.5 K |
| LightGBM | 30% | ~29.5 K |
| CatBoost | 35% | ~28.8 K |
| **Ensemble** | - | **26.64 K** |

### Combinación Óptima
```
Predicción = 20% × ChemProp + 80% × Ensemble
MAE = 22.80 K (Kaggle)
```

## Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Info del modelo (MAE, configuración) |
| POST | `/validate-smiles` | Validar SMILES |
| POST | `/compounds` | Crear compuesto + predicción |
| GET | `/compounds` | Listar compuestos usuario |
| DELETE | `/compounds/{id}` | Eliminar compuesto |
| GET | `/stats` | Estadísticas |
| GET | `/predict-all` | Todas las predicciones |

## Ejemplo de Uso

```bash
# Crear compuesto (Water)
curl -X POST "http://localhost:8000/compounds" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O", "name": "Water"}'

# Respuesta esperada (con modelo combinado):
{
  "id": "USR_001",
  "smiles": "O",
  "name": "Water",
  "Tm_pred": 272.17,           # Real: 273.15 K ✓
  "Tm_celsius": -0.98,
  "uncertainty": "±23 K",
  "method": "combined (cp=20%)"
}
```

## requirements.txt
```txt
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
pandas>=2.0.0
numpy==1.26.4
scikit-learn>=1.3.0
joblib>=1.3.0
rdkit>=2023.03.1
chemprop==1.6.1
torch>=2.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
```

## Troubleshooting

| Problema | Solución |
|----------|----------|
| "weights_only load failed" | `python patch_chemprop_torch.py` |
| MAE 28.85 (no 22.80) | Falta ensemble: `cd ../src && python train_ensemble_production.py` |
| "Ensemble no cargado" | Verificar que existe `models/ensemble_predictor.joblib` |
| Predicción lenta (10-30s) | Normal para ChemProp |

## Docs
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc