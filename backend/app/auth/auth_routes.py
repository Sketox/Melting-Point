"""
Endpoints de autenticación
Registro, login, logout, perfil
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import timedelta
import logging

from .auth_schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    RegisterResponse,
    LoginResponse,
    UserResponse,
    Token,
    ChangePasswordRequest,
    UpdateProfileRequest
)
from .auth_service import (
    AuthService,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .mongodb_client import get_async_database, Collections

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["🔐 Authentication"]
)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="📝 Registrar Usuario",
    description="Crea una nueva cuenta de usuario con autenticación JWT."
)
async def register(user_data: UserRegisterRequest):
    """
    Registra un nuevo usuario en el sistema.
    
    **Validaciones automáticas:**
    - Username único (3-50 caracteres, sin espacios)
    - Email válido y único
    - Password segura (mínimo 8 caracteres, 1 mayúscula, 1 número)
    
    **Request body:**
    ```json
    {
        "username": "usuario123",
        "email": "usuario@email.com",
        "password": "Password123",
        "full_name": "Juan Pérez"
    }
    ```
    
    **Response (201 Created):**
    ```json
    {
        "user": {
            "id": "507f1f77bcf86cd799439011",
            "username": "usuario123",
            "email": "usuario@email.com",
            "full_name": "Juan Pérez",
            "created_at": "2026-02-01T10:30:00",
            "predictions_count": 0,
            "is_active": true
        },
        "token": {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "token_type": "bearer",
            "expires_in": 1800
        },
        "message": "Usuario registrado exitosamente"
    }
    ```
    
    **Errores comunes:**
    - 400: Username o email ya existe
    - 422: Validación fallida (password débil, email inválido)
    """
    try:
        # Crear usuario
        user = await AuthService.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        # Crear token JWT
        access_token = AuthService.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        # Guardar sesión
        await AuthService.save_session(str(user.id), access_token)
        
        # Convertir a UserResponse
        user_response = UserResponse(
            _id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            bio=user.bio,
            created_at=user.created_at,
            predictions_count=0,
            is_active=True
        )
        
        # Token info
        token = Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # en segundos
        )
        
        logger.info(f"✓ Usuario registrado: {user.username}")
        
        return RegisterResponse(
            user=user_response,
            token=token,
            message="Usuario registrado exitosamente"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error en registro: {e}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar usuario: {str(e)}"
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="🔐 Iniciar Sesión",
    description="Autentica al usuario y retorna un token JWT."
)
async def login(login_data: UserLoginRequest):
    """
    Inicia sesión con email y contraseña.
    
    **Request body:**
    ```json
    {
        "email": "usuario@email.com",
        "password": "Password123"
    }
    ```
    
    **Response (200 OK):**
    ```json
    {
        "user": {
            "id": "507f1f77bcf86cd799439011",
            "username": "usuario123",
            "email": "usuario@email.com",
            "predictions_count": 5,
            "is_active": true
        },
        "token": {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "token_type": "bearer",
            "expires_in": 1800
        },
        "message": "Login exitoso"
    }
    ```
    
    **Errores:**
    - 401: Credenciales incorrectas
    - 404: Usuario no encontrado
    
    **Duración del token:** 30 minutos
    """
    # Autenticar usuario
    user = await AuthService.authenticate_user(login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Crear token JWT
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    # Guardar sesión
    await AuthService.save_session(str(user.id), access_token)
    
    # Convertir a UserResponse
    user_response = UserResponse(
        _id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        bio=user.bio,
        created_at=user.created_at,
        predictions_count=user.predictions_count,
        is_active=user.is_active
    )
    
    # Token info
    token = Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    
    logger.info(f"✓ Login exitoso: {user.username}")
    
    return LoginResponse(
        user=user_response,
        token=token,
        message="Login exitoso"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: UserResponse = Depends(get_current_user)):
    """
    👤 Obtener perfil del usuario actual
    
    Requiere autenticación (token JWT en header Authorization).
    Retorna la información del usuario autenticado.
    """
    return current_user


@router.post("/logout")
async def logout(current_user: UserResponse = Depends(get_current_user)):
    """
    🚪 Cerrar sesión
    
    Invalida el token actual del usuario.
    """
    # TODO: Obtener el token del header para invalidarlo
    # Por ahora solo retornamos mensaje de éxito
    # En producción, deberías invalidar el token en la BD
    
    logger.info(f"✓ Logout: {current_user.username}")
    
    return {
        "message": "Sesión cerrada exitosamente",
        "username": current_user.username
    }


@router.put("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    🔑 Cambiar contraseña
    
    Requiere autenticación y la contraseña actual.
    """
    db = get_async_database()
    
    # Obtener usuario completo con contraseña hasheada
    user = await AuthService.get_user_by_id(current_user.id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )
    
    # Verificar contraseña actual
    if not AuthService.verify_password(password_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Contraseña actual incorrecta"
        )
    
    # Actualizar contraseña
    new_hashed_password = AuthService.hash_password(password_data.new_password)
    
    from bson import ObjectId
    await db[Collections.USERS].update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"hashed_password": new_hashed_password}}
    )
    
    logger.info(f"✓ Contraseña cambiada: {current_user.username}")
    
    return {"message": "Contraseña actualizada exitosamente"}


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    profile_data: UpdateProfileRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    ✏️ Actualizar perfil
    
    Permite actualizar nombre completo y biografía.
    """
    db = get_async_database()
    from bson import ObjectId
    from datetime import datetime
    
    # Preparar datos de actualización
    update_data = {}
    if profile_data.full_name is not None:
        update_data["full_name"] = profile_data.full_name
    if profile_data.bio is not None:
        update_data["bio"] = profile_data.bio
    
    update_data["updated_at"] = datetime.utcnow()
    
    # Actualizar en BD
    await db[Collections.USERS].update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": update_data}
    )
    
    # Obtener usuario actualizado
    user = await AuthService.get_user_by_id(current_user.id)
    
    logger.info(f"✓ Perfil actualizado: {current_user.username}")
    
    return UserResponse(
        _id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        bio=user.bio,
        created_at=user.created_at,
        predictions_count=user.predictions_count,
        is_active=user.is_active
    )


@router.delete("/account")
async def delete_account(current_user: UserResponse = Depends(get_current_user)):
    """
    🗑️ Eliminar cuenta
    
    Elimina la cuenta del usuario y todas sus predicciones.
    Esta acción es irreversible.
    """
    db = get_async_database()
    from bson import ObjectId
    
    # Eliminar predicciones del usuario
    await db[Collections.USER_PREDICTIONS].delete_many({"user_id": current_user.id})
    
    # Eliminar sesiones del usuario
    await db[Collections.SESSIONS].delete_many({"user_id": current_user.id})
    
    # Eliminar usuario
    await db[Collections.USERS].delete_one({"_id": ObjectId(current_user.id)})
    
    logger.info(f"✓ Cuenta eliminada: {current_user.username}")
    
    return {
        "message": "Cuenta eliminada exitosamente",
        "username": current_user.username
    }


@router.get("/stats")
async def get_user_stats(current_user: UserResponse = Depends(get_current_user)):
    """
    📊 Estadísticas del usuario
    
    Retorna estadísticas sobre las predicciones del usuario.
    """
    db = get_async_database()
    
    # Contar predicciones
    total_predictions = await db[Collections.USER_PREDICTIONS].count_documents({
        "user_id": current_user.id
    })
    
    # Obtener predicción más reciente
    latest_prediction = await db[Collections.USER_PREDICTIONS].find_one(
        {"user_id": current_user.id},
        sort=[("created_at", -1)]
    )
    
    # Contar favoritos
    favorites_count = await db[Collections.USER_PREDICTIONS].count_documents({
        "user_id": current_user.id,
        "is_favorite": True
    })
    
    return {
        "username": current_user.username,
        "total_predictions": total_predictions,
        "favorites_count": favorites_count,
        "latest_prediction": latest_prediction,
        "member_since": current_user.created_at.isoformat()
    }
