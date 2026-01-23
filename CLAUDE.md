# CLAUDE.md - MeltingPoint Backend

## 📋 Resumen del Proyecto

**Nombre:** MeltingPoint API  
**Tipo:** Backend REST API con FastAPI  
**Propósito:** Predecir puntos de fusión (Tm) de compuestos orgánicos para la competencia Kaggle "Thermophysical Property: Melting Point"  
**Competencia:** https://www.kaggle.com/competitions/melting-point

## 🎯 Objetivo de la Competencia

Construir modelos de ML que predigan el punto de fusión en **Kelvin (K)** para compuestos orgánicos dados sus descriptores moleculares (representados en formato SMILES).

## 🏗️ Arquitectura

```
MeltingPoint/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app + endpoints
│   │   ├── ml_service.py    # Servicio de predicción ML
│   │   ├── schemas.py       # Pydantic models
│   │   └── config.py        # Configuración de rutas
│   └── models/
│       ├── model.joblib     # Modelo RandomForest serializado
│       └── model_chemprop/  # Modelo ChemProp (5 folds)
├── data/
│   ├── raw/                 # train.csv, test.csv
│   └── processed/           # test_processed.csv con features
├── src/                     # Scripts de entrenamiento
├── notebooks/               # Jupyter notebooks
└── .venv/                   # Entorno virtual Python
```

## 🔧 Stack Tecnológico

- **Framework:** FastAPI 0.100+
- **ML:** scikit-learn (RandomForestRegressor), ChemProp (D-MPNN)
- **Data:** Pandas, Joblib
- **Validación:** Pydantic
- **Server:** Uvicorn

## 📡 Endpoints Actuales (3)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check del servidor |
| POST | `/predict-by-id` | Predicción por ID del dataset |
| GET | `/predict-all` | Todas las predicciones del test set |

## 🧠 Modelo ML

- **Input:** Features extraídas de SMILES (descriptores moleculares)
- **Output:** Punto de fusión en Kelvin (K)
- **Archivo de datos:** `test_processed.csv` con columna `id` + features
- **Modelo:** `model.joblib` cargado al startup

## ⚙️ Configuración

```python
# config.py
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TEST_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"
```

## 🚀 Comandos

```bash
# Desde MeltingPoint/backend/
uvicorn app.main:app --reload --port 8000

# Docs: http://localhost:8000/docs
```

## 📊 Datos

- **Train:** ~2,500 moléculas con Tm conocido
- **Test:** 667 moléculas para predicción
- **Features:** Descriptores moleculares RDKit extraídos de SMILES

## 🔗 Conexión con Frontend

- CORS habilitado para `localhost:3000`
- Frontend consume `/predict-all` y `/predict-by-id`

## 📝 Notas para Desarrollo

- El modelo se carga una sola vez al iniciar (`@app.on_event("startup")`)
- Los IDs del dataset van desde 1 hasta 667
- Las predicciones están en Kelvin, el frontend convierte a Celsius
- El MLService busca el ID en el DataFrame y predice con el modelo cargado