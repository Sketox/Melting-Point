"""
Schemas para autenticación y usuarios
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId para Pydantic v2"""
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.chain_schema([
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.validate),
            ])
        ],
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: str(x)
        ))

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")


# ============================================
# REQUEST SCHEMAS
# ============================================

class UserRegisterRequest(BaseModel):
    """Schema para registro de usuario"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72, description="Password (8-72 caracteres)")
    full_name: Optional[str] = Field(None, max_length=100)
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username debe ser alfanumérico (se permiten _ y -)')
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password debe tener al menos 8 caracteres')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password debe contener al menos un número')
        if not any(char.isupper() for char in v):
            raise ValueError('Password debe contener al menos una mayúscula')
        return v


class UserLoginRequest(BaseModel):
    """Schema para login de usuario"""
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    """Schema para cambio de contraseña"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @field_validator('new_password')
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password debe tener al menos 8 caracteres')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password debe contener al menos un número')
        if not any(char.isupper() for char in v):
            raise ValueError('Password debe contener al menos una mayúscula')
        return v


class UpdateProfileRequest(BaseModel):
    """Schema para actualizar perfil"""
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)


# ============================================
# RESPONSE SCHEMAS
# ============================================

class Token(BaseModel):
    """Schema para token JWT"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # segundos


class UserResponse(BaseModel):
    """Schema para datos de usuario (sin contraseña)"""
    id: str = Field(alias="_id")
    username: str
    email: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    predictions_count: int = 0
    is_active: bool = True
    
    model_config = {
        "populate_by_name": True,
        "json_encoders": {ObjectId: str}
    }


class LoginResponse(BaseModel):
    """Schema para respuesta de login"""
    user: UserResponse
    token: Token
    message: str = "Login exitoso"


class RegisterResponse(BaseModel):
    """Schema para respuesta de registro"""
    user: UserResponse
    token: Token
    message: str = "Usuario registrado exitosamente"


# ============================================
# DATABASE MODELS
# ============================================

class UserInDB(BaseModel):
    """Modelo de usuario en la base de datos"""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    username: str
    email: str
    hashed_password: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_verified: bool = False
    predictions_count: int = 0
    
    model_config = {
        "populate_by_name": True,
        "json_encoders": {ObjectId: str}
    }


class UserPredictionInDB(BaseModel):
    """Modelo de predicción de usuario en MongoDB"""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: str  # ObjectId del usuario
    username: str  # Para consultas rápidas
    smiles: str
    tm_pred: float
    tm_pred_celsius: float
    compound_name: Optional[str] = None
    notes: Optional[str] = None
    is_favorite: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[dict] = None
    
    model_config = {
        "populate_by_name": True,
        "json_encoders": {ObjectId: str}
    }


class SessionInDB(BaseModel):
    """Modelo de sesión en MongoDB"""
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: str
    token: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    is_active: bool = True
    
    model_config = {
        "populate_by_name": True,
        "json_encoders": {ObjectId: str}
    }
