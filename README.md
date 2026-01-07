# 🧪 Melting Point API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)

**API REST para predicción de puntos de fusión moleculares**

[Instalación](#-instalación) • [Endpoints](#-endpoints) • [Uso](#-uso) • [Modelo](#-modelo) • [Estructura](#-estructura)

</div>

---

## 📋 Descripción

API REST desarrollada con **FastAPI** que proporciona predicciones de puntos de fusión (Tm) para moléculas del dataset de la competencia [Kaggle Melting Point](https://www.kaggle.com/competitions/melting-point).

### ✨ Características

- 🚀 **Alto rendimiento** - FastAPI con soporte asíncrono
- 📖 **Documentación automática** - Swagger UI y ReDoc integrados
- 🔒 **Validación de datos** - Esquemas Pydantic para request/response
- 🔄 **CORS habilitado** - Listo para conectar con frontends
- 🧠 **ML integrado** - Modelo pre-entrenado cargado al iniciar

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.10 o superior
- pip (gestor de paquetes)

### Paso a paso

```bash
# 1. Navegar al directorio del backend
cd MeltingPoint/backend

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows (PowerShell):
venv\Scripts\activate
# Windows (CMD):
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar el servidor
uvicorn app.main:app --reload --port 8000
```

### Verificar instalación

```bash
# El servidor debería mostrar:
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Abre en tu navegador: **http://localhost:8000/docs**

---


### Instalar dependencias manualmente

```bash
pip install fastapi uvicorn pandas joblib scikit-learn pydantic
```

---

## 🔌 Endpoints

### Base URL

```
http://localhost:8000
```

---

### 🏠 Root

Verifica que el servidor está corriendo.

```http
GET /
```

**Response:**
```json
{
  "message": "Melting Point API",
  "status": "running",
  "docs": "/docs"
}
```

---

### 💚 Health Check

Verifica el estado del servidor y la disponibilidad del modelo.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

**cURL:**
```bash
curl http://localhost:8000/health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

---

### 🔮 Predict by ID

Obtiene la predicción del punto de fusión para un ID específico del dataset de test.

```http
POST /predict-by-id
Content-Type: application/json
```

**Request Body:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | integer | ID de la molécula en el dataset (requerido) |

```json
{
  "id": 69
}
```

**Response:**
```json
{
  "id": 69,
  "Tm_pred": 123.69
}
```

**Errores:**
| Código | Descripción |
|--------|-------------|
| 404 | ID no encontrado en el dataset |
| 500 | Modelo no inicializado |

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict-by-id" \
  -H "Content-Type: application/json" \
  -d '{"id": 42}'
```

**PowerShell:**
```powershell
$body = @{ id = 42 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict-by-id" -Method Post -Body $body -ContentType "application/json"
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict-by-id",
    json={"id": 42}
)
print(response.json())
# {"id": 42, "Tm_pred": 234.76}
```

---

### 📊 Predict All

Obtiene las predicciones de Tm para **todos** los IDs del dataset de test.

```http
GET /predict-all
```

**Response:**
```json
[
  { "id": 1, "Tm_pred": 341.51 },
  { "id": 2, "Tm_pred": 372.55 },
  { "id": 3, "Tm_pred": 205.82 },
  ...
]
```

**cURL:**
```bash
curl http://localhost:8000/predict-all
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict-all"
```

**Python:**
```python
import requests

response = requests.get("http://localhost:8000/predict-all")
predictions = response.json()

print(f"Total predicciones: {len(predictions)}")
for pred in predictions[:5]:
    print(f"ID {pred['id']}: {pred['Tm_pred']:.2f} K")
```

---

## 📖 Documentación Interactiva

FastAPI genera documentación automática:

| URL | Descripción |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** - Interfaz interactiva para probar endpoints |
| http://localhost:8000/redoc | **ReDoc** - Documentación en formato legible |
| http://localhost:8000/openapi.json | **OpenAPI Schema** - Especificación JSON |

---

## 🧠 Modelo

### Información del Modelo

| Parámetro | Valor |
|-----------|-------|
| **Algoritmo** | RandomForestRegressor / ChemProp |
| **Input** | Features procesadas de SMILES |
| **Output** | Punto de fusión en Kelvin (K) |
| **Archivo** | `models/model.joblib` |

### Pipeline de Predicción

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Request   │     │    MLService     │     │   Response  │
│   { id: n } │ ──► │  1. Buscar ID    │ ──► │  { Tm_pred }│
│             │     │  2. Extraer feat │     │             │
│             │     │  3. Predecir     │     │             │
└─────────────┘     └──────────────────┘     └─────────────┘
```

### Carga del Modelo

El modelo se carga automáticamente al iniciar la aplicación:

```python
@app.on_event("startup")
def startup_event() -> None:
    global ml_service
    ml_service = MLService()  # Carga modelo y datos
```

---

## 📁 Estructura

```
backend/
│
├── 📁 app/
│   ├── __init__.py          # Inicializador del módulo
│   ├── main.py              # Aplicación FastAPI y endpoints
│   ├── ml_service.py        # Servicio de Machine Learning
│   ├── schemas.py           # Esquemas Pydantic (request/response)
│   └── config.py            # Configuración de rutas
│
├── 📁 models/
│   ├── model.joblib         # Modelo entrenado serializado
│   └── 📁 model_chemprop/   # Modelo ChemProp (alternativo)
│       ├── fold_0/
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       ├── fold_4/
│       └── args.json
│
├── requirements.txt         # Dependencias Python
├── .gitignore              # Archivos ignorados por Git
└── README.md               # Este archivo
```

---

## ⚙️ Configuración

### Archivo `config.py`

```python
from pathlib import Path

# Directorio base (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorio raíz del proyecto (MeltingPoint/)
PROJECT_ROOT = BASE_DIR.parent

# Ruta al modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

# Ruta al CSV de test procesado
TEST_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "test_processed.csv"
```


---

## 🔒 CORS

El backend tiene CORS habilitado para permitir conexiones desde el frontend:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js
        "http://127.0.0.1:3000",
        "*",                         # Desarrollo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🛠️ Desarrollo

### Ejecutar en modo desarrollo

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Flag | Descripción |
|------|-------------|
| `--reload` | Recarga automática al detectar cambios |
| `--host 0.0.0.0` | Acepta conexiones externas |
| `--port 8000` | Puerto del servidor |


### Con Gunicorn (Linux)

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## 🧪 Testing

### Probar endpoints manualmente

```bash
# Health check
curl http://localhost:8000/health

# Predicción por ID
curl -X POST http://localhost:8000/predict-by-id \
  -H "Content-Type: application/json" \
  -d '{"id": 1}'

# Todas las predicciones (primeras 3)
curl http://localhost:8000/predict-all | python -m json.tool | head -20
```


---

## ❗ Solución de Problemas

### Error: `ModuleNotFoundError: No module named 'app'`

**Causa:** Estás ejecutando desde el directorio incorrecto.

**Solución:**
```bash
cd MeltingPoint/backend
uvicorn app.main:app --reload
```

---

### Error: `FileNotFoundError: Modelo no encontrado`

**Causa:** El archivo `model.joblib` no existe.

**Solución:** Verifica que existe:
```bash
ls models/model.joblib
```

---

### Error: `CORS policy blocked`

**Causa:** El frontend no puede conectar por restricciones CORS.

**Solución:** Verifica que `main.py` tiene el middleware CORS configurado.

---

### Error: `Connection refused`

**Causa:** El servidor no está corriendo.

**Solución:**
```bash
# Verificar que uvicorn está corriendo
curl http://localhost:8000/health
```

---

## 📊 Esquemas de Datos

### PredictByIdRequest

```python
class PredictByIdRequest(BaseModel):
    id: int  # ID de la molécula
```

### PredictResponse

```python
class PredictResponse(BaseModel):
    id: int        # ID de la molécula
    Tm_pred: float # Punto de fusión predicho (Kelvin)
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

<div align="center">

**Desarrollado para la competencia Kaggle Melting Point** 🧪

[⬆ Volver arriba](#-melting-point-api)

</div>
