# Guía de Configuración e Instalación de Supabase

## 📋 Requisitos Previos
- Cuenta en Supabase (https://supabase.com - gratis)
- Python 3.8+
- Node.js 18+

## 🚀 Paso 1: Crear Proyecto en Supabase

1. Ve a [https://app.supabase.com](https://app.supabase.com)
2. Haz clic en "New Project"
3. Completa:
   - **Project Name**: `melting-point` (o el nombre que prefieras)
   - **Database Password**: Guarda esta contraseña en un lugar seguro
   - **Region**: Selecciona la más cercana (ej: `South America (São Paulo)`)
4. Espera 2-3 minutos mientras Supabase crea tu base de datos

## 🔑 Paso 2: Obtener Credenciales

1. En tu proyecto de Supabase, ve a **Settings** → **API**
2. Copia las siguientes claves:
   - **Project URL** (ejemplo: `https://xxxxx.supabase.co`)
   - **anon public key** (clave pública para el frontend)
   - **service_role key** (clave privada para el backend - ¡NUNCA expongas en frontend!)

## ⚙️ Paso 3: Configurar Variables de Entorno

### Backend (FastAPI)

1. Crea el archivo `.env` en la carpeta `backend/`:
```bash
cd backend
cp ../database/.env.example .env
```

2. Edita `.env` y completa:
```env
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu_anon_public_key
SUPABASE_SERVICE_KEY=tu_service_role_key
```

### Frontend (Next.js)

1. Crea el archivo `.env.local` en la carpeta raíz de Next.js:
```bash
cd ../Melting-Point-Presentation
nano .env.local
```

2. Agrega:
```env
NEXT_PUBLIC_SUPABASE_URL=https://tu-proyecto.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=tu_anon_public_key
```

> **⚠️ IMPORTANTE**: Las variables con prefijo `NEXT_PUBLIC_` son expuestas al navegador. ¡NUNCA pongas la `service_role` key aquí!

## 📊 Paso 4: Crear el Esquema de Base de Datos

1. Ve a tu proyecto en Supabase
2. Click en **SQL Editor** (icono de `</>` en el menú lateral)
3. Abre el archivo `database/schema.sql` de este proyecto
4. Copia todo el contenido
5. Pégalo en el SQL Editor de Supabase
6. Click en **RUN** (o presiona `Ctrl+Enter`)

Deberías ver el mensaje: "Success. No rows returned"

## 🔄 Paso 5: Migrar los Datos CSV

### Instalar Dependencias Python

```bash
cd database
pip install supabase-py python-dotenv pandas
```

### Ejecutar Migración

```bash
python migrate_data.py
```

El script te preguntará si deseas limpiar las tablas primero. Responde:
- **Primera vez**: `n` (no)
- **Re-migración**: `s` (sí)

El proceso tardará ~30 segundos y mostrará:
```
✅ MIGRACIÓN COMPLETADA EXITOSAMENTE
Estadísticas:
  - Compuestos: 3330
  - Predicciones: 667
```

## 🔧 Paso 6: Instalar Dependencias en Backend

```bash
cd ../backend
pip install -r requirements.txt
pip install supabase-py python-dotenv
```

Actualiza `requirements.txt`:
```bash
echo "supabase-py==2.3.4" >> requirements.txt
echo "python-dotenv==1.0.0" >> requirements.txt
```

## 🌐 Paso 7: Instalar Dependencias en Frontend

```bash
cd ../Melting-Point-Presentation
npm install @supabase/supabase-js
```

## ✅ Paso 8: Verificar Conexión

### Verificar Backend

```bash
cd backend
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
result = supabase.table('compounds').select('id').limit(1).execute()
print('✅ Conexión exitosa!' if result.data else '❌ Error de conexión')
"
```

### Verificar Frontend

Crea un archivo de prueba `test-supabase.js`:
```javascript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
)

const { data, error } = await supabase.from('compounds').select('id').limit(1)
console.log(data ? '✅ Conexión exitosa!' : '❌ Error:', error)
```

## 📖 Paso 9: Consultas de Ejemplo

### Ver todos los compuestos
```sql
SELECT * FROM compounds LIMIT 10;
```

### Ver predicciones con errores
```sql
SELECT * FROM predictions_full 
WHERE tm_real IS NOT NULL 
ORDER BY absolute_error DESC 
LIMIT 10;
```

### Estadísticas del modelo
```sql
SELECT * FROM model_statistics;
```

### Distribución de temperaturas
```sql
SELECT * FROM temperature_distribution;
```

## 🔒 Seguridad (Opcional)

### Habilitar Row Level Security (RLS)

Si planeas agregar autenticación de usuarios:

```sql
-- En SQL Editor de Supabase
ALTER TABLE user_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own predictions"
ON user_predictions FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own predictions"
ON user_predictions FOR INSERT
WITH CHECK (auth.uid() = user_id);
```

## 🐛 Solución de Problemas

### Error: "relation does not exist"
→ Ejecuta el archivo `schema.sql` en el SQL Editor de Supabase

### Error: "Invalid API key"
→ Verifica que copiaste la clave correcta de Settings → API

### Error en migración: "duplicate key value"
→ Ejecuta la migración con la opción de limpiar tablas (`s`)

### Frontend no se conecta
→ Asegúrate de que las variables empiecen con `NEXT_PUBLIC_`
→ Reinicia el servidor de desarrollo (`npm run dev`)

## 📚 Recursos Adicionales

- [Documentación de Supabase](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [Supabase JS Client](https://github.com/supabase/supabase-js)

## 🎉 ¡Listo!

Tu proyecto ahora está conectado a Supabase. Los próximos pasos serán:
1. Actualizar el backend de FastAPI para usar Supabase
2. Actualizar el frontend para obtener datos de la API
3. Implementar autenticación (opcional)
