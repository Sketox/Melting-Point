"""
Módulo de autenticación con MongoDB.

Incluye:
- Conexión a MongoDB (mongodb_client)
- Esquemas de autenticación (auth_schemas)
- Servicios de autenticación (auth_service)
- Rutas de autenticación (auth_routes)
- Rutas de predicciones de usuario (user_predictions_routes)
"""

from .mongodb_client import (
    get_async_database,
    create_indexes,
    test_mongodb_connection,
    close_mongodb_connection,
    Collections
)

from .auth_schemas import (
    PyObjectId,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    Token,
    UserInDB,
    UserPredictionInDB,
    UpdateProfileRequest,
    ChangePasswordRequest,
    LoginResponse,
    RegisterResponse
)

from .auth_service import (
    AuthService,
)

from .auth_routes import router as auth_router
from .user_predictions_routes import router as user_predictions_router

__all__ = [
    # MongoDB
    "get_async_database",
    "create_indexes",
    "test_mongodb_connection",
    "close_mongodb_connection",
    "Collections",
    
    # Schemas
    "PyObjectId",
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserResponse",
    "Token",
    "UserInDB",
    "UserPredictionInDB",
    "UpdateProfileRequest",
    "ChangePasswordRequest",
    "LoginResponse",
    "RegisterResponse",
    
    # Services
    "AuthService",
    
    # Routers
    "auth_router",
    "user_predictions_router",
]
