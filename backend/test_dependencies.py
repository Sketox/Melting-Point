"""
Script de prueba para verificar que todas las dependencias están instaladas
y funcionando correctamente
"""

import sys
print("=" * 60)
print("VERIFICACIÓN DE DEPENDENCIAS")
print("=" * 60)

# Probar imports
def test_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - FALTA: {e}")
        return False

# Verificar dependencias principales
dependencies = [
    # Backend API
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pydantic", "Pydantic"),
    # Database - MongoDB
    ("pymongo", "PyMongo"),
    ("motor", "Motor (Async MongoDB)"),
    # Authentication & Security
    ("jose", "Python-JOSE (JWT)"),
    ("passlib", "Passlib (Password hashing)"),
    ("bcrypt", "Bcrypt"),
    ("email_validator", "Email Validator"),
    # Configuration
    ("dotenv", "Python-dotenv"),
    # Data Processing
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    # Machine Learning
    ("sklearn", "Scikit-learn"),
    ("joblib", "Joblib"),
    # Chemistry & ChemProp
    ("rdkit", "RDKit"),
    ("torch", "PyTorch"),
    ("chemprop", "ChemProp"),
    # Database - Supabase (opcional)
    ("supabase", "Supabase (opcional)"),
    ("httpx", "HTTPX"),
]

print("\n📦 Verificando imports...")
print("-" * 60)

all_ok = True
optional_failed = []

for module, name in dependencies:
    is_optional = "opcional" in name.lower()
    result = test_import(module, name)
    
    if not result:
        if is_optional:
            optional_failed.append(name)
        else:
            all_ok = False

# Verificar configuración
print("\n⚙️ Verificando configuración...")
print("-" * 60)

try:
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongodb_url = os.getenv("MONGODB_URL")
    jwt_secret = os.getenv("JWT_SECRET_KEY")
    
    if mongodb_url:
        print(f"✓ MONGODB_URL configurada: {mongodb_url}")
    else:
        print("✗ MONGODB_URL no configurada en .env")
        all_ok = False
    
    if jwt_secret and jwt_secret != "tu_clave_secreta_super_segura_cambiala_en_produccion":
        print("✓ JWT_SECRET_KEY configurada")
    elif jwt_secret:
        print("⚠ JWT_SECRET_KEY usando valor por defecto (cambiar en producción)")
    else:
        print("✗ JWT_SECRET_KEY no configurada en .env")
        all_ok = False
        
except Exception as e:
    print(f"✗ Error al cargar .env: {e}")
    all_ok = False

# Verificar módulo de autenticación (app.auth)
print("\n🗄️ Verificando módulo de autenticación (app.auth)...")
print("-" * 60)

try:
    from app.auth import get_async_database, Collections
    print("✓ app.auth.mongodb_client - OK")
except Exception as e:
    print(f"✗ app.auth.mongodb_client - ERROR: {e}")
    all_ok = False

try:
    from app.auth import AuthService
    print("✓ app.auth.auth_service - OK")
except Exception as e:
    print(f"✗ app.auth.auth_service - ERROR: {e}")
    all_ok = False

try:
    from app.auth import UserRegisterRequest, UserLoginRequest
    print("✓ app.auth.auth_schemas - OK")
except Exception as e:
    print(f"✗ app.auth.auth_schemas - ERROR: {e}")
    all_ok = False

try:
    from app.auth import auth_router
    print("✓ app.auth.auth_routes - OK")
except Exception as e:
    print(f"✗ app.auth.auth_routes - ERROR: {e}")
    all_ok = False

try:
    from app.auth import user_predictions_router
    print("✓ app.auth.user_predictions_routes - OK")
except Exception as e:
    print(f"✗ app.auth.user_predictions_routes - ERROR: {e}")
    all_ok = False

# Verificar módulo Supabase (opcional)
print("\n☁️ Verificando módulo Supabase (app.supabase)...")
print("-" * 60)

try:
    from app.supabase import supabase_router
    print("✓ app.supabase.supabase_routes - OK")
except Exception as e:
    print(f"⚠ app.supabase - No disponible (opcional): {e}")
    # No marca como fallo porque Supabase es opcional

# Resultado final
print("\n" + "=" * 60)
if all_ok:
    print("✅ TODAS LAS VERIFICACIONES PASARON")
    print("=" * 60)
    
    if optional_failed:
        print("\n⚠️ Dependencias opcionales no disponibles:")
        for dep in optional_failed:
            print(f"   - {dep}")
    
    print("\n📝 Próximos pasos:")
    print("1. Instalar MongoDB:")
    print("   - Windows: https://www.mongodb.com/try/download/community")
    print("   - O usar MongoDB Atlas (cloud): https://www.mongodb.com/cloud/atlas")
    print("\n2. Iniciar el servidor:")
    print("   cd backend")
    print("   uvicorn app.main:app --reload")
    print("\n3. Probar endpoints de auth:")
    print("   http://localhost:8000/docs")
    print("\n4. Backend funcionando en:")
    print("   http://localhost:8000")
    sys.exit(0)
else:
    print("❌ ALGUNAS VERIFICACIONES FALLARON")
    print("=" * 60)
    print("\n⚠️ Por favor instala las dependencias faltantes:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
