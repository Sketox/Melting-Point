"""
Script de prueba para el sistema de autenticación
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_register():
    """Prueba el registro de usuario"""
    print("\n🧪 Probando registro de usuario...")
    
    payload = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPass123",
        "full_name": "Usuario de Prueba"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 201:
        print("✅ Registro exitoso!")
        return response.json()
    else:
        print("❌ Error en registro")
        return None

def test_login():
    """Prueba el login de usuario"""
    print("\n🧪 Probando login...")
    
    payload = {
        "email": "test@example.com",
        "password": "TestPass123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("✅ Login exitoso!")
        return response.json()
    else:
        print("❌ Error en login")
        return None

def test_get_me(token: str):
    """Prueba obtener usuario actual"""
    print("\n🧪 Probando GET /auth/me...")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("✅ Autenticación exitosa!")
        return response.json()
    else:
        print("❌ Error en autenticación")
        return None

def test_password_hashing():
    """Prueba el hashing de contraseñas"""
    print("\n🧪 Probando bcrypt...")
    
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    password = "TestPass123"
    hashed = pwd_context.hash(password)
    
    print(f"Password: {password}")
    print(f"Hash: {hashed}")
    print(f"Verificación: {pwd_context.verify(password, hashed)}")
    
    if pwd_context.verify(password, hashed):
        print("✅ Bcrypt funciona correctamente!")
    else:
        print("❌ Error en bcrypt")

if __name__ == "__main__":
    print("="*50)
    print("TEST DE AUTENTICACIÓN")
    print("="*50)
    
    # Test 1: Bcrypt
    test_password_hashing()
    
    # Test 2: Registro
    register_result = test_register()
    
    if register_result:
        token = register_result.get("token", {}).get("access_token")
        
        # Test 3: Obtener usuario
        if token:
            test_get_me(token)
    
    # Test 4: Login
    login_result = test_login()
    
    if login_result:
        token = login_result.get("token", {}).get("access_token")
        
        # Test 5: Obtener usuario después del login
        if token:
            test_get_me(token)
    
    print("\n" + "="*50)
    print("TESTS COMPLETADOS")
    print("="*50)
