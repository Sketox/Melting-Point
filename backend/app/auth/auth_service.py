"""
Servicio de autenticación
Manejo de JWT, hashing de contraseñas, validación
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .mongodb_client import get_async_database, Collections
from .auth_schemas import UserInDB, UserResponse, SessionInDB, UserActivityLog

logger = logging.getLogger(__name__)

# Configuración de seguridad desde variables de entorno
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev_secret_key_change_in_production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Context para hash de contraseñas con bcrypt (12 rounds por defecto)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme para FastAPI
security = HTTPBearer()


class AuthService:
    """Servicio de autenticación"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hashea una contraseña usando bcrypt
        Bcrypt tiene límite de 72 bytes, se trunca automáticamente
        """
        # Truncar a 72 bytes para evitar error de bcrypt
        password_bytes = password.encode('utf-8')[:72]
        password_truncated = password_bytes.decode('utf-8', errors='ignore')
        return pwd_context.hash(password_truncated)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verifica una contraseña contra su hash
        Trunca a 72 bytes para consistencia con bcrypt
        """
        # Truncar a 72 bytes igual que en hash_password
        password_bytes = plain_password.encode('utf-8')[:72]
        password_truncated = password_bytes.decode('utf-8', errors='ignore')
        return pwd_context.verify(password_truncated, hashed_password)
    
    @staticmethod
    def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Crea un token JWT
        
        Args:
            data: Datos a encodear en el token (ej: {"sub": user_id})
            expires_delta: Tiempo de expiración personalizado
            
        Returns:
            Token JWT firmado
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Dict:
        """
        Decodifica y valida un token JWT
        
        Args:
            token: Token JWT
            
        Returns:
            Payload del token
            
        Raises:
            HTTPException: Si el token es inválido o ha expirado
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"Error al decodificar token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[UserInDB]:
        """Obtiene un usuario por email"""
        db = get_async_database()
        user_dict = await db[Collections.USERS].find_one({"email": email.lower()})
        
        if user_dict:
            return UserInDB(**user_dict)
        return None
    
    @staticmethod
    async def get_user_by_username(username: str) -> Optional[UserInDB]:
        """Obtiene un usuario por username"""
        db = get_async_database()
        user_dict = await db[Collections.USERS].find_one({"username": username.lower()})
        
        if user_dict:
            return UserInDB(**user_dict)
        return None
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[UserInDB]:
        """Obtiene un usuario por ID"""
        db = get_async_database()
        from bson import ObjectId
        
        try:
            user_dict = await db[Collections.USERS].find_one({"_id": ObjectId(user_id)})
            if user_dict:
                return UserInDB(**user_dict)
        except Exception as e:
            logger.error(f"Error al obtener usuario: {e}")
        
        return None
    
    @staticmethod
    async def create_user(username: str, email: str, password: str, full_name: Optional[str] = None) -> UserInDB:
        """
        Crea un nuevo usuario
        
        Args:
            username: Nombre de usuario
            email: Email del usuario
            password: Contraseña en texto plano (se hasheará)
            full_name: Nombre completo (opcional)
            
        Returns:
            Usuario creado
            
        Raises:
            HTTPException: Si el usuario o email ya existe
        """
        try:
            db = get_async_database()
            
            # Verificar si el email ya existe
            existing_email = await AuthService.get_user_by_email(email)
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El email ya está registrado"
                )
            
            # Verificar si el username ya existe
            existing_username = await AuthService.get_user_by_username(username)
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El nombre de usuario ya está en uso"
                )
            
            # Crear usuario
            user = UserInDB(
                username=username.lower(),
                email=email.lower(),
                hashed_password=AuthService.hash_password(password),
                full_name=full_name,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
                predictions_count=0
            )
            
            # Convertir a dict para MongoDB
            user_dict = user.model_dump(by_alias=True, exclude={"id"})
            
            # Insertar en MongoDB
            result = await db[Collections.USERS].insert_one(user_dict)
            
            # Asignar el ID generado
            user.id = result.inserted_id
            logger.info(f"✓ Usuario creado: {username} ({email})")
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al crear usuario en MongoDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al crear usuario: {str(e)}"
            )
    
    @staticmethod
    async def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
        """
        Autentica un usuario
        
        Args:
            email: Email del usuario
            password: Contraseña en texto plano
            
        Returns:
            Usuario si las credenciales son válidas, None si no
        """
        user = await AuthService.get_user_by_email(email)
        
        if not user:
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Usuario inactivo"
            )
        
        if not AuthService.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    @staticmethod
    async def save_session(user_id: str, token: str) -> SessionInDB:
        """Guarda una sesión en la base de datos"""
        db = get_async_database()
        
        session = SessionInDB(
            user_id=user_id,
            token=token,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            is_active=True
        )
        
        session_dict = session.model_dump(by_alias=True, exclude={"id"})
        result = await db[Collections.SESSIONS].insert_one(session_dict)
        
        session.id = result.inserted_id
        return session
    
    @staticmethod
    async def log_activity(
        user_id: str,
        username: str,
        action: str,
        resource: str,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Registra actividad del usuario en la base de datos
        
        Args:
            user_id: ID del usuario
            username: Username del usuario
            action: Acción realizada (create, update, delete, login, etc)
            resource: Recurso afectado (profile, password, prediction, etc)
            details: Detalles adicionales opcionales
            ip_address: IP del cliente
            user_agent: User agent del navegador
        """
        try:
            db = get_async_database()
            
            activity_log = UserActivityLog(
                user_id=user_id,
                username=username,
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=datetime.utcnow()
            )
            
            log_dict = activity_log.model_dump(by_alias=True)
            await db[Collections.ACTIVITY_LOGS].insert_one(log_dict)
            
            logger.info(f"📝 Activity logged: {username} - {action} {resource}")
        except Exception as e:
            # No queremos que falle la operación principal si falla el log
            logger.error(f"Error logging activity: {e}")
    
    @staticmethod
    async def invalidate_session(token: str):
        """Invalida una sesión (logout)"""
        db = get_async_database()
        await db[Collections.SESSIONS].update_one(
            {"token": token},
            {"$set": {"is_active": False}}
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserResponse:
    """
    Dependency para obtener el usuario actual desde el token JWT
    
    Usar en endpoints protegidos:
    @app.get("/protected")
    async def protected_route(current_user: UserResponse = Depends(get_current_user)):
        return {"user": current_user.username}
    """
    token = credentials.credentials
    
    # Decodificar token
    payload = AuthService.decode_token(token)
    user_id: str = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar que la sesión esté activa
    db = get_async_database()
    session = await db[Collections.SESSIONS].find_one({
        "token": token,
        "is_active": True
    })
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión inválida o expirada",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar que no haya expirado
    if session["expires_at"] < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Obtener usuario
    user = await AuthService.get_user_by_id(user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )
    
    # Convertir a UserResponse
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


# Dependency opcional (permite que el usuario no esté autenticado)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[UserResponse]:
    """
    Dependency opcional para obtener el usuario actual
    Retorna None si no hay token
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
