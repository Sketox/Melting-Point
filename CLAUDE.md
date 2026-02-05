# MeltingPoint - Kaggle Competition + Production API

## 🎯 ESTADO FINAL

| Info | Valor |
|------|-------|
| **Mejor score Kaggle** | **MAE 22.80** ⭐ |
| **Configuración óptima** | 20% ChemProp + 80% Ensemble |
| **Top 1 del leaderboard** | MAE 4.74 |
| **Backend API** | ✅ Funcionando (MAE ~22.80) |

---

## 📋 Descripción del Proyecto

Competencia de Kaggle para predecir el **punto de fusión molecular (Tm)** a partir de representaciones SMILES y features de grupos funcionales.

- **Competencia**: [Thermophysical Property: Melting Point](https://www.kaggle.com/competitions/melting-point)
- **Métrica**: MAE (Mean Absolute Error) en Kelvin
- **Backend**: API REST con FastAPI para predicciones en tiempo real

---

## 🏆 MODELO FINAL

### Arquitectura Híbrida

```
                    SMILES Input
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      ┌─────────────┐          ┌─────────────┐
      │  ChemProp   │          │  Ensemble   │
      │  D-MPNN     │          │ XGB+LGB+CAT │
      │  (GNN)      │          │ (Fingerprints)│
      └─────────────┘          └─────────────┘
            │                         │
            │ (20%)             (80%) │
            └────────────┬────────────┘
                         ▼
                  Final Prediction
                   MAE ~22.80 K
```

### Rendimiento por Componente

| Modelo | MAE (K) | Contribución |
|--------|---------|--------------|
| ChemProp solo | 28.85 | 20% |
| Ensemble solo | 26.64 | 80% |
| **Combinado** | **22.80** | ⭐ **Kaggle** |

### Features del Ensemble (2,757 total)

| Tipo | Cantidad | Descripción |
|------|----------|-------------|
| Morgan FP (ECFP4) | 2,048 | Subestructuras circulares |
| MACCS Keys | 167 | Patrones químicos predefinidos |
| RDKit Descriptors | ~200 | Propiedades físico-químicas |
| SMILES features | 13 | Longitud, anillos, heteroátomos |
| Group features | 337 | Grupos funcionales del dataset |

---

## 📊 Dataset

| Conjunto | Muestras | Columnas |
|----------|----------|----------|
| Train | 2,662 | 427 (id, SMILES, Tm, 424 Group features) |
| Test | 666 | 426 (sin Tm) |

**Target (Tm)**: 53.5 K - 897.1 K (media: 278.26 K)

---

## 🗂️ ESTRUCTURA DEL PROYECTO

```
MeltingPoint/
├── backend/                      # ⭐ API de Producción
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── ml_service.py        # Predicciones (Ensemble + ChemProp)
│   │   ├── schemas.py           # Pydantic models
│   │   └── config.py            # Configuración
│   ├── models/
│   │   ├── model_chemprop/      # 5 folds ChemProp entrenados
│   │   │   ├── fold_0/model_0/model.pt
│   │   │   └── ...
│   │   ├── ensemble_predictor.joblib  # ⭐ XGB+LGB+CAT
│   │   ├── best_params_paso6.json     # Hiperparámetros Optuna
│   │   └── model.joblib               # Fallback sklearn
│   ├── patch_chemprop_torch.py  # Parche PyTorch 2.6+
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── train.csv            # 2,662 muestras
│   │   └── test.csv             # 666 muestras
│   └── processed/
│       ├── chemprop_predictions.csv
│       └── test_processed.csv
│
├── src/                          # ⭐ Scripts de Entrenamiento
│   ├── train_ensemble_production.py  # ⭐ Entrena ensemble para backend
│   ├── 01_chemprop_max_precision.py
│   ├── 04_advanced_models.py
│   ├── 08_best_of_both.py       # PASO 6 (mejor resultado)
│   └── ...
│
├── submissions/
│   ├── submission_paso6_cp20.csv  # ⭐ MEJOR (MAE 22.80)
│   └── ...
│
└── CLAUDE.md                     # Este archivo
```

---

## 🚀 INSTALACIÓN Y USO

### 1. Entrenar Ensemble para Producción

```bash
cd src
python train_ensemble_production.py
```

Esto genera: `backend/models/ensemble_predictor.joblib`

### 2. Iniciar Backend API

```bash
cd backend

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate      # Windows

# Instalar dependencias
pip install -r requirements.txt

# ⚠️ OBLIGATORIO: Parche para PyTorch 2.6+
python patch_chemprop_torch.py

# Ejecutar API
uvicorn app.main:app --reload --port 8000
```

### 3. Verificar Instalación

Los logs deben mostrar:
```
INFO: ChemProp 1.6.1 detectado correctamente.
INFO: ChemProp habilitado con 5 checkpoints.
INFO: Ensemble cargado con 15 modelos.
INFO: Modo COMBINADO activo (MAE ~22.80 K)
```

### 4. Usar la API

```bash
# Health check
curl http://localhost:8000/health

# Predecir melting point
curl -X POST "http://localhost:8000/compounds" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O", "name": "Water"}'

# Respuesta:
# {
#   "Tm_pred": 272.17,      # Real: 273.15 K ✓
#   "Tm_celsius": -0.98,
#   "uncertainty": "±23 K",
#   "method": "combined (cp=20%)"
# }
```

### Documentación API
- **Swagger**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📈 HISTORIAL DE LA COMPETENCIA

### Cronología de Mejoras

| Paso | Descripción | MAE Kaggle | Cambio |
|------|-------------|------------|--------|
| 1 | ChemProp D-MPNN | 25.05 | Baseline |
| 2 | Ensemble básico (cp50) | 23.51 | -1.54 |
| 3 | Optimización pesos (cp35) | 23.40 | -0.11 |
| 4 | Morgan FP + CatBoost | 22.94 | -0.46 |
| 5 | Optuna + más FP | 23.43 | +0.49 ❌ Overfitting |
| **6** | **Optuna + features PASO 4** | **22.80** ⭐ | -0.14 |

### Lección Principal

> **Más features ≠ mejor**. El modelo con 2,757 features superó al de 5,833.
> El overfitting ocurre cuando OOF mejora pero Kaggle empeora.

---

## 🔬 DETALLES TÉCNICOS

### ChemProp D-MPNN

| Parámetro | Valor |
|-----------|-------|
| Hidden Size | 300 |
| Depth | 6 |
| Dropout | 0.1 |
| Epochs | 50 |
| Folds | 5 |
| MAE | 28.85 K |

### Ensemble (PASO 6)

| Modelo | Peso Óptimo | OOF MAE |
|--------|-------------|---------|
| CatBoost | 52.7% | 27.07 |
| LightGBM | 29.8% | 27.50 |
| XGBoost | 17.5% | 27.22 |
| **Ensemble** | - | **26.64** |

### Combinación Óptima

```python
# Mejor configuración (MAE 22.80)
prediction = 0.20 * chemprop + 0.80 * ensemble
```

---

## 🛠️ DEPENDENCIAS

### Backend (requirements.txt)

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
optuna>=3.0.0
```

### Entrenamiento adicional

```bash
pip install optuna tqdm
```

---

## ⚠️ TROUBLESHOOTING

| Problema | Solución |
|----------|----------|
| "weights_only load failed" | `python patch_chemprop_torch.py` |
| MAE 28.85 (no 22.80) | Falta ensemble: `cd src && python train_ensemble_production.py` |
| "Ensemble no cargado" | Verificar `backend/models/ensemble_predictor.joblib` |
| Predicción lenta (10-30s) | Normal para ChemProp, primera vez carga modelos |
| Predicción da 161 K (no 272 K para agua) | Parche ChemProp no aplicado |

---

## 📝 LECCIONES APRENDIDAS

### ✅ Lo que SÍ funcionó
1. **Morgan Fingerprints (2048 bits)** - Capturan subestructuras
2. **CatBoost** - Mejor modelo individual
3. **Ensemble 3 modelos** - XGB + LGB + CAT
4. **ChemProp al 20%** - Información complementaria
5. **Optuna con features controladas** - Sin overfitting

### ❌ Lo que NO funcionó
1. **Neural Network en ensemble** - MAE 29.72, empeoraba
2. **Demasiados fingerprints** - 5,833 features = overfitting
3. **ChemProp > 35%** - Demasiado peso empeora
4. **ChemProp solo** - MAE 25.05, peor que ensemble

---

## 📚 REFERENCIAS

### Papers
- [ChemProp D-MPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
- [Morgan Fingerprints (ECFP)](https://www.rdkit.org/docs/)
- [Melting Point Prediction](https://pubs.acs.org/doi/10.1021/ci0500132)

### Documentación
- [RDKit](https://www.rdkit.org/docs/)
- [CatBoost](https://catboost.ai/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 👤 Autor

**Sketo**  
Competencia: Thermophysical Property: Melting Point  
Mejor resultado: **MAE 22.80**  
API Backend: **Funcionando** ✅  
Fecha: Febrero 2026