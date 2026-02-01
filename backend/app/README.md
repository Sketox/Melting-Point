# Estructura del Backend

Este directorio contiene el código principal de la API FastAPI para predicción de puntos de fusión.

## 📁 Estructura Organizada

```
app/
├── auth/                    # Módulo de Autenticación y MongoDB
│   ├── __init__.py         # Exports del módulo
│   ├── mongodb_client.py   # Cliente async de MongoDB
│   ├── auth_schemas.py     # Schemas Pydantic (User, Token, etc.)
│   ├── auth_service.py     # Lógica de autenticación (JWT, passwords)
│   ├── auth_routes.py      # Endpoints: /auth/register, /auth/login, etc.
│   └── user_predictions_routes.py  # Endpoints: /user-predictions/*
│
├── supabase/               # Módulo de Supabase (opcional)
│   ├── __init__.py         # Exports del módulo
│   ├── supabase_client.py  # Cliente de Supabase
│   ├── supabase_service.py # Servicios de datos con Supabase
│   └── supabase_routes.py  # Endpoints: /api/v2/*
│
├── main.py                 # Aplicación FastAPI principal
├── ml_service.py           # Servicio de ML (predicciones, validación)
├── schemas.py              # Schemas generales del API
├── config.py               # Configuración de la aplicación
└── README.md               # Este archivo
```

## 🔐 Módulo Auth (`app.auth`)

**Propósito:** Sistema completo de autenticación con MongoDB y JWT.

### Archivos:

- **`mongodb_client.py`**: Conexión async a MongoDB, colecciones, índices
- **`auth_schemas.py`**: Modelos Pydantic para usuarios, tokens, predicciones
- **`auth_service.py`**: Lógica de negocio (hash passwords, JWT, validación)
- **`auth_routes.py`**: 8 endpoints de autenticación
- **`user_predictions_routes.py`**: CRUD de predicciones por usuario

### Uso:

```python
from app.auth import (
    auth_router,
    user_predictions_router,
    get_async_database,
    AuthService
)
```

### Endpoints:

```
POST   /auth/register           - Registrar nuevo usuario
POST   /auth/login              - Login con email/password
GET    /auth/me                 - Obtener usuario actual (requiere token)
POST   /auth/logout             - Cerrar sesión
PUT    /auth/change-password    - Cambiar password
PUT    /auth/profile            - Actualizar perfil
DELETE /auth/account            - Eliminar cuenta
GET    /auth/stats              - Estadísticas del usuario

POST   /user-predictions/       - Guardar predicción
GET    /user-predictions/       - Listar mis predicciones
GET    /user-predictions/{id}   - Obtener una predicción
PUT    /user-predictions/{id}   - Actualizar predicción
DELETE /user-predictions/{id}   - Eliminar predicción
GET    /user-predictions/search/by-smiles - Buscar por SMILES
```

## ☁️ Módulo Supabase (`app.supabase`)

**Propósito:** Integración con Supabase PostgreSQL (opcional, en mantenimiento).

### Archivos:

- **`supabase_client.py`**: Cliente Supabase (singleton, lazy init)
- **`supabase_service.py`**: Lógica de consultas a Supabase
- **`supabase_routes.py`**: Endpoints v2 del API

### Uso:

```python
from app.supabase import supabase_router
```

### Estado:

⚠️ **Opcional** - Si no configuras `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` en `.env`, el módulo no se carga pero el backend funciona normalmente.

### Endpoints:

```
GET /api/v2/predictions          - Todas las predicciones (desde Supabase)
GET /api/v2/predictions/{id}     - Predicción por ID
GET /api/v2/stats                - Estadísticas
GET /api/v2/distribution         - Distribución de temperaturas
GET /api/v2/compounds            - Listar compuestos
POST /api/v2/compounds           - Crear compuesto
DELETE /api/v2/compounds/{id}    - Eliminar compuesto
```

## 🧪 Archivos Principales

### `main.py`

Aplicación FastAPI principal que:
- Configura CORS
- Incluye routers de auth y supabase
- Define endpoints de ML (/predict-by-id, /stats, /validate-smiles, etc.)
- Conecta a MongoDB al startup

### `ml_service.py`

Servicio de Machine Learning:
- Carga modelo ChemProp
- Valida SMILES con RDKit
- Genera predicciones
- Gestiona compuestos de usuario (CSV)

### `schemas.py`

Schemas Pydantic generales:
- `PredictResponse`, `StatsResponse`
- `CompoundResponse`, `ValidateSmilesResponse`
- Requests y responses de endpoints ML

## 🔧 Configuración

### Variables de Entorno (`.env`)

```bash
# MongoDB (Requerido para auth)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=melting_point_db

# JWT (Requerido para auth)
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Supabase (Opcional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

## 🚀 Iniciar el Servidor

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Docs interactivos: http://localhost:8000/docs

## 📦 Dependencias

```bash
pip install -r requirements.txt
```

Principales:
- **FastAPI** - Framework web
- **Pydantic** - Validación de datos
- **PyMongo/Motor** - MongoDB async
- **python-jose** - JWT
- **passlib** - Hash de passwords
- **Supabase** - Cliente Supabase (opcional)
- **RDKit** - Química computacional
- **ChemProp** - Modelo de ML
- **pandas, numpy, scikit-learn** - ML y datos

## 🔒 Seguridad

- Passwords hasheados con bcrypt
- JWT con expiración configurable
- Validación de datos con Pydantic
- Índices únicos en MongoDB (email, username)
- CORS configurado

## 🧪 Testing

Verificar dependencias:
```bash
python test_dependencies.py
```

Verificar imports:
```bash
python -c "from app.main import app; print('OK')"
```

## 📝 Notas

- **MongoDB**: Puede ser local o MongoDB Atlas (cloud)
- **Supabase**: Completamente opcional, el backend funciona sin él
- **Organización**: Módulos separados para mejor mantenibilidad
- **Async**: MongoDB usa Motor para operaciones asíncronas
