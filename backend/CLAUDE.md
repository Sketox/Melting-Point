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

## 🎯 Sistema de Toma de Decisiones

El backend soporta un sistema completo de toma de decisiones con tres fuentes de datos:

| Fuente | Color | Cantidad | Descripción |
|--------|-------|----------|-------------|
| **Train** | 🟢 Verde | 2,662 | Datos reales con Tm medido experimentalmente |
| **Test** | 🔵 Azul | 666 | Predicciones del modelo (MAE ~22.80 K) |
| **User** | 🟠 Naranja | Variable | Compuestos agregados por el usuario |

### Interpretación de Incertidumbre

- **MAE del modelo**: ±22.80 K (intervalo de confianza)
- **Significado práctico**: Una predicción de 350 K significa que el Tm real está probablemente entre 327-373 K
- **Para decisiones críticas**: Considerar el rango completo de incertidumbre

### Datasets Procesados

```
data/processed/
├── dataset_train.csv    # 2,662 filas (id, smiles, Tm real, source='train')
└── dataset_test.csv     # 666 filas (id, smiles, Tm predicho, source='test')
```

## Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Info del modelo (MAE, configuración) |
| GET | `/data-all` | **Todos los datos (train+test+user) con fuente** |
| GET | `/compound-name` | **Nombre del compuesto desde PubChem** |
| POST | `/validate-smiles` | Validar SMILES |
| POST | `/compounds` | Crear compuesto + predicción |
| GET | `/compounds` | Listar compuestos usuario |
| DELETE | `/compounds/{id}` | Eliminar compuesto |
| GET | `/stats` | Estadísticas |
| GET | `/predict-all` | Todas las predicciones (test only) |
| GET | `/predictions/by-functional-group` | **Análisis por grupos funcionales** |
| GET | `/predictions/by-molecule-size` | Análisis por tamaño molecular |
| GET | `/predictions/distribution` | Distribución por categorías de Tm |

## 📊 Análisis por Grupos Funcionales

### ¿Por qué es importante?

El endpoint `/predictions/by-functional-group` es clave para la toma de decisiones porque:

1. **Base científica**: Los grupos funcionales determinan las interacciones intermoleculares
   - **Puentes de hidrógeno**: OH, NH2, COOH aumentan Tm
   - **π-stacking**: Grupos aromáticos aumentan Tm
   - **Polaridad**: Afecta la red cristalina

2. **Uso práctico para decisiones**:
   - Comparar tu compuesto con otros del mismo grupo
   - Verificar si la predicción es consistente con la estructura
   - Identificar si tu compuesto está en un rango típico

3. **Cómo defenderlo**:
   > "El análisis por grupos funcionales permite validar predicciones comparando
   > con compuestos de estructura similar. Si tu molécula tiene grupo OH,
   > puedes ver el rango típico de Tm para alcoholes y verificar que
   > la predicción sea consistente."

### Ejemplo de uso

```python
# Consultar promedios por grupo funcional
GET /predictions/by-functional-group

# Respuesta incluye:
{
  "groups": [
    {"name": "Hydroxyl (OH)", "count": 450, "avg_tm": 320.5, ...},
    {"name": "Amine (NH2)", "count": 280, "avg_tm": 315.2, ...},
    ...
  ]
}
```

## 📈 Interpretación del MAE

### ¿Por qué usamos MAE de Kaggle (22.80 K) y no el de entrenamiento?

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **MAE Kaggle** | 22.80 K | Error en datos NO vistos (test set real) |
| MAE ChemProp OOF | 28.85 K | Error en validación cruzada |
| MAE Ensemble OOF | 26.64 K | Error en validación cruzada |

**El MAE de Kaggle es más válido porque**:
1. Mide el error en datos completamente nuevos
2. No hay riesgo de overfitting
3. Es la métrica oficial de la competencia
4. Representa el rendimiento real de generalización

**Cómo comunicarlo**:
> "La incertidumbre de ±22.80 K está validada en el test set de Kaggle,
> que representa datos que el modelo nunca vio durante el entrenamiento.
> Esto es una estimación conservadora del error esperado en nuevos compuestos."

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

# Obtener nombre de compuesto desde PubChem
curl "http://localhost:8000/compound-name?smiles=CCO"
# Respuesta: {"smiles": "CCO", "name": "ethanol", "source": "pubchem"}

# Obtener todos los datos (train+test+user)
curl "http://localhost:8000/data-all"
# Respuesta: [{"id": 1, "smiles": "...", "Tm_pred": 350.5, "source": "train"}, ...]
```

## Guía de Uso para Decisiones

### Cuándo Confiar en las Predicciones

| Escenario | Recomendación |
|-----------|---------------|
| Predicción cerca de datos train | ✅ Mayor confianza |
| Predicción en extremos (< 100 K o > 800 K) | ⚠️ Menos datos de referencia |
| Molécula muy diferente al dataset | ⚠️ Extrapolar con cautela |
| Decisión crítica de seguridad | 🔬 Verificar experimentalmente |

### Flujo de Trabajo Recomendado

1. **Validar SMILES** → `/validate-smiles`
2. **Verificar nombre** → `/compound-name` (PubChem)
3. **Comparar con dataset** → Ver distribución en `/data-all`
4. **Predecir** → `/compounds` (crea registro con predicción)
5. **Interpretar** → Considerar ±22.80 K de incertidumbre

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