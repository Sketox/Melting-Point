# Estructura del Backend

Este directorio contiene el código principal de la API FastAPI para predicción de puntos de fusión.

## 📁 Estructura

```
app/
├── auth/                    # Módulo de Autenticación (opcional, MongoDB)
│   ├── __init__.py
│   ├── mongodb_client.py   # Cliente async de MongoDB
│   ├── auth_schemas.py     # Schemas Pydantic (User, Token, etc.)
│   ├── auth_service.py     # Lógica de autenticación (JWT, passwords)
│   ├── auth_routes.py      # Endpoints: /auth/*
│   └── user_predictions_routes.py
│
├── main.py                 # Aplicación FastAPI principal
├── ml_service.py           # Servicio de ML (predicciones, validación)
├── schemas.py              # Schemas generales del API
├── config.py               # Configuración de la aplicación
└── README.md               # Este archivo
```

## 🧪 Módulo Principal

### `main.py`

Aplicación FastAPI principal que:
- Configura CORS
- Define endpoints de ML y datos
- Conecta a MongoDB al startup (opcional)
- Incluye integración con PubChem para nombres de compuestos

### `ml_service.py`

Servicio de Machine Learning:
- Carga datasets procesados (train + test)
- Valida SMILES con RDKit
- Genera predicciones (ChemProp + Ensemble)
- Gestiona compuestos de usuario

### `schemas.py`

Schemas Pydantic:
- `PredictResponse`, `StatsResponse`
- `CompoundResponse`, `ValidateSmilesResponse`
- `DataItemResponse` (train/test/user)
- `CompoundNameResponse` (PubChem)

## 📊 Sistema de Datos

El backend maneja tres fuentes de datos:

| Fuente | Archivo | Descripción |
|--------|---------|-------------|
| **Train** | `dataset_train.csv` | 2,662 compuestos con Tm REAL medido |
| **Test** | `dataset_test.csv` | 666 compuestos con Tm PREDICHO |
| **User** | `user_compounds.csv` | Compuestos agregados por el usuario |

## 🎯 Endpoints Principales

### Datos y Predicciones
```
GET  /health              - Estado del sistema
GET  /model-info          - Info del modelo (MAE, configuración)
GET  /data-all            - Todos los datos (train+test+user)
GET  /predict-all         - Todas las predicciones test
GET  /stats               - Estadísticas del dataset
```

### Validación y Nombres
```
POST /validate-smiles     - Validar estructura SMILES
GET  /compound-name       - Nombre desde PubChem (con cache)
```

### Compuestos de Usuario
```
POST   /compounds         - Crear compuesto + predicción
GET    /compounds         - Listar compuestos usuario
DELETE /compounds/{id}    - Eliminar compuesto
```

### Analytics
```
GET /predictions/range              - Filtrar por rango de Tm
GET /predictions/distribution       - Distribución por categorías
GET /predictions/by-functional-group - Análisis por grupos funcionales
GET /predictions/by-molecule-size   - Análisis por tamaño molecular
```

## 🔬 Endpoint de Grupos Funcionales

**¿Por qué es útil?**

El endpoint `/predictions/by-functional-group` analiza qué grupos funcionales están presentes en las moléculas y cómo afectan el punto de fusión.

**Justificación científica:**
- Los grupos funcionales determinan las **interacciones intermoleculares**
- Grupos polares (OH, NH2) aumentan Tm por **puentes de hidrógeno**
- Grupos aromáticos aumentan Tm por **π-stacking**
- Útil para comparar tu compuesto con moléculas de estructura similar

**Ejemplo de uso para decisiones:**
1. Tu compuesto tiene grupo hidroxilo (OH)
2. Consultas el promedio de Tm para compuestos con OH
3. Comparas si tu predicción está dentro del rango esperado
4. Mayor confianza si tu Tm cae en el rango típico del grupo

## 🚀 Iniciar el Servidor

```bash
cd backend
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# IMPORTANTE: Aplicar parche para PyTorch 2.6+
python patch_chemprop_torch.py

uvicorn app.main:app --reload --port 8000
```

Docs interactivos: http://localhost:8000/docs

## 📦 Dependencias Principales

```bash
pip install -r requirements.txt
```

- **FastAPI** - Framework web
- **Pydantic** - Validación de datos
- **RDKit** - Química computacional
- **ChemProp** - Modelo D-MPNN
- **XGBoost, LightGBM** - Ensemble
- **pandas, numpy, scikit-learn** - ML y datos
- **httpx** - Cliente HTTP async (PubChem)

## 🔧 Configuración

### Variables de Entorno (`.env`)

```bash
# MongoDB (Opcional - para auth)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=melting_point_db

# JWT (Opcional - para auth)
JWT_SECRET_KEY=your-secret-key
```

## 📝 Notas

- **MongoDB**: Completamente opcional, el backend funciona sin él
- **PubChem**: Cache en memoria para evitar llamadas repetidas
- **Incertidumbre**: MAE del modelo combinado es ±22.80 K (Kaggle)
