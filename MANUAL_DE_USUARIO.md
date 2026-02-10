# Manual de Usuario — MeltingPoint Dashboard

**Predicción de Puntos de Fusión Moleculares**
Arquitectura Híbrida ChemProp D-MPNN + Ensemble (XGBoost + LightGBM)

---

## Tabla de Contenido

1. [Introducción](#1-introducción)
2. [Navegación General](#2-navegación-general)
3. [Sistema de Colores](#3-sistema-de-colores)
4. [Registro e Inicio de Sesión](#4-registro-e-inicio-de-sesión)
   - 4.1 [Crear una Cuenta](#41-crear-una-cuenta)
   - 4.2 [Iniciar Sesión](#42-iniciar-sesión)
   - 4.3 [Editar Perfil y Cambiar Contraseña](#43-editar-perfil-y-cambiar-contraseña)
   - 4.4 [Cerrar Sesión](#44-cerrar-sesión)
5. [Página de Inicio](#5-página-de-inicio)
6. [Predicciones](#6-predicciones)
   - 6.1 [Panel de Estadísticas](#61-panel-de-estadísticas)
   - 6.2 [Buscar por ID](#62-buscar-por-id)
   - 6.3 [Agregar un Compuesto Personalizado](#63-agregar-un-compuesto-personalizado)
   - 6.4 [Eliminar un Compuesto de Usuario](#64-eliminar-un-compuesto-de-usuario)
   - 6.5 [Tabla de Datos](#65-tabla-de-datos)
   - 6.6 [Filtrar por Fuente](#66-filtrar-por-fuente)
   - 6.7 [Buscar por Texto](#67-buscar-por-texto)
   - 6.8 [Filtrar por Rango de Temperatura](#68-filtrar-por-rango-de-temperatura)
   - 6.9 [Ordenar la Tabla](#69-ordenar-la-tabla)
   - 6.10 [Copiar Datos de un Compuesto](#610-copiar-datos-de-un-compuesto)
   - 6.11 [Exportar a CSV](#611-exportar-a-csv)
7. [Analytics](#7-analytics)
   - 7.1 [Resumen del Dataset](#71-resumen-del-dataset)
   - 7.2 [Box Plot Comparativo](#72-box-plot-comparativo)
   - 7.3 [Filtro Global de Fuente](#73-filtro-global-de-fuente)
   - 7.4 [Distribución por Temperatura](#74-distribución-por-temperatura)
   - 7.5 [Complejidad vs. Tm (Scatter Plot)](#75-complejidad-vs-tm-scatter-plot)
   - 7.6 [Grupos Funcionales](#76-grupos-funcionales)
   - 7.7 [Tamaño Molecular vs. Punto de Fusión](#77-tamaño-molecular-vs-punto-de-fusión)
   - 7.8 [Guía de Interpretación](#78-guía-de-interpretación)
8. [Modelo](#8-modelo)
9. [Acerca de](#9-acerca-de)
10. [Preguntas Frecuentes (FAQ)](#10-preguntas-frecuentes-faq)

---

## 1. Introducción

MeltingPoint Dashboard es una aplicación web para la **predicción del punto de fusión molecular (Tm)** a partir de estructuras químicas en notación SMILES. Utiliza un modelo híbrido de aprendizaje automático que combina:

- **ChemProp D-MPNN** (20%): Red neuronal de grafos dirigidos que aprende directamente de la estructura molecular.
- **Ensemble XGBoost + LightGBM** (80%): Modelos de gradient boosting que utilizan descriptores moleculares calculados.

**Precisión validada en Kaggle: MAE = 22.80 K** (Error Absoluto Medio).

### ¿Qué puedes hacer con esta aplicación?

- Consultar el punto de fusión de **2,662 compuestos reales** (datos de entrenamiento).
- Explorar **666 predicciones** del modelo sobre compuestos de prueba.
- **Predecir el punto de fusión** de cualquier molécula ingresando su cadena SMILES.
- Visualizar distribuciones, tendencias y análisis de grupos funcionales con gráficos interactivos.
- Guardar tus propias predicciones y compuestos personalizados (requiere cuenta).
- Exportar datos filtrados a formato CSV.

---

## 2. Navegación General

La aplicación cuenta con una **barra de navegación fija** en la parte superior que se mantiene visible en todas las páginas. Contiene los siguientes enlaces:

| Enlace | Página | Descripción |
|--------|--------|-------------|
| **Home** | `/` | Página de inicio con resumen general |
| **Predictions** | `/predictions` | Tabla de datos, búsqueda y gestión de compuestos |
| **Analytics** | `/analytics` | Gráficos y visualizaciones interactivas |
| **Model** | `/model` | Información técnica del modelo de ML |
| **About** | `/about` | Información del proyecto y la competición |

Además, en la esquina superior derecha encontrarás:

- **Enlace a Kaggle**: Acceso directo a la competición original.
- **Enlace a GitHub**: Repositorio del código fuente.
- **Login / Menú de usuario**: Para iniciar sesión o gestionar tu cuenta.

En dispositivos móviles, la navegación se convierte en un **menú hamburguesa** que se despliega al hacer clic.

La barra de navegación se vuelve más opaca al hacer scroll hacia abajo para mejorar la legibilidad.

---

## 3. Sistema de Colores

Toda la aplicación utiliza un sistema de colores consistente para identificar el **origen de cada dato**. Es fundamental entender este sistema para interpretar correctamente la información:

| Fuente | Color | Etiqueta | Significado |
|--------|-------|----------|-------------|
| **Train** | 🟢 Verde (`#4ade80`) | "Real" | Valor de Tm **medido experimentalmente** en laboratorio. Dato confiable. |
| **Test** | 🔵 Azul (`#60a5fa`) | "Predicción" | Valor de Tm **predicho por el modelo**. Tiene una incertidumbre de ±22.80 K. |
| **User** | 🟠 Naranja (`#f5a623`) | "Usuario" | Compuesto **agregado por ti**. Tm predicho por el modelo con incertidumbre de ±22.80 K. |

Este código de colores se aplica en:
- Las etiquetas (badges) de la tabla de datos.
- Los puntos y barras de todos los gráficos.
- Las tarjetas de estadísticas.
- Los resultados de búsqueda.

---

## 4. Registro e Inicio de Sesión

La aplicación permite navegar libremente sin cuenta (modo visitante). Puedes explorar todos los datos, gráficos y análisis. Sin embargo, necesitas registrarte para **agregar compuestos personalizados**.

### 4.1 Crear una Cuenta

1. Hacer clic en **"Login"** en la barra de navegación superior.
2. En la página de inicio de sesión, hacer clic en el enlace **"Crear cuenta"**.
3. Completar el formulario de registro:

| Campo | Obligatorio | Requisitos |
|-------|:-----------:|------------|
| Nombre de usuario | Sí | 3-50 caracteres. Solo letras, números, `_` y `-`. |
| Nombre completo | No | Campo opcional. |
| Correo electrónico | Sí | Formato válido (ej: usuario@correo.com). |
| Contraseña | Sí | Mínimo 8 caracteres, al menos 1 mayúscula y 1 número. |

4. A medida que escribes la contraseña, verás indicadores en tiempo real:
   - ✅ Verde = requisito cumplido.
   - ❌ Rojo = requisito pendiente.
5. El botón **"Crear Cuenta"** se habilitará cuando todos los requisitos estén cumplidos.
6. Al registrarte exitosamente, serás redirigido a la página principal con tu sesión activa.

### 4.2 Iniciar Sesión

1. Hacer clic en **"Login"** en la barra de navegación.
2. Ingresar tu **correo electrónico** y **contraseña**.
3. Hacer clic en **"Iniciar Sesión"**.
4. Si las credenciales son correctas, serás redirigido a la página principal.

> **Nota:** Si ves un mensaje de error, verifica que el correo y la contraseña sean correctos.

### 4.3 Editar Perfil y Cambiar Contraseña

1. Hacer clic en tu **nombre de usuario** en la esquina superior derecha.
2. Seleccionar **"Editar perfil"** en el menú desplegable.
3. Puedes modificar: nombre de usuario, correo electrónico, nombre completo y biografía.
4. Para cambiar la contraseña, desplazarse a la sección correspondiente e ingresar la contraseña actual y la nueva.
5. Hacer clic en **"Guardar cambios"** o **"Cambiar contraseña"** según corresponda.

### 4.4 Cerrar Sesión

1. Hacer clic en tu nombre de usuario en la barra de navegación.
2. Seleccionar **"Cerrar sesión"**.

---

## 5. Página de Inicio

Al abrir la aplicación llegarás a la página principal, que ofrece un panorama general del sistema:

- **Indicador de conexión**: En la parte superior, un punto verde pulsante confirma que la aplicación está funcionando correctamente.
- **Sección hero**: Título del proyecto con botones de acceso rápido a **Predicciones** y **Acerca de**.
- **Tarjetas de estadísticas**: Resumen rápido con:
  - Total de compuestos de entrenamiento (2,662 reales).
  - Total de compuestos de prueba (666 predichos).
  - MAE del modelo (22.80 K).
  - Tipo de modelo (Híbrido).
- **Destacado del modelo**: Tarjeta mostrando la arquitectura híbrida con un ejemplo práctico: Agua (H₂O) — predicción 272.17 K vs. valor real 273.15 K.
- **Características principales**: Cuatro tarjetas describiendo las funcionalidades clave (predicción, modelo híbrido, visualizaciones, toma de decisiones).
- **Enlaces rápidos**: Accesos directos a todas las secciones de la aplicación.

---

## 6. Predicciones

La página de **Predicciones** (`/predictions`) es el centro de operaciones principal. Desde aquí puedes explorar todos los datos, buscar compuestos, agregar los tuyos y exportar información.

### 6.1 Panel de Estadísticas

En la parte superior se muestran 4 tarjetas con los conteos actuales:

| Tarjeta | Color | Contenido |
|---------|-------|-----------|
| Total | Gris | Número total de compuestos en el sistema |
| Train (Real) | Verde | 2,662 compuestos medidos experimentalmente |
| Test (Predicción) | Azul | 666 compuestos con Tm predicho |
| User (Usuario) | Naranja | Compuestos que tú has agregado |

### 6.2 Buscar por ID

1. En la sección **"Buscar por ID"**, ingresar el número de ID del compuesto.
2. Hacer clic en **"Buscar"** o presionar Enter.
3. Se mostrará una tarjeta con:
   - ID y etiqueta de fuente (Train/Test/User).
   - Temperatura de fusión en **Kelvin** y **Celsius**.
   - Cadena SMILES del compuesto.
   - Indicación de incertidumbre: **±22.80 K** para predicciones, o **"Medido"** para datos de entrenamiento.

### 6.3 Agregar un Compuesto Personalizado

> **Requisito:** Debes haber iniciado sesión. Si no lo has hecho, verás un botón de **"Iniciar sesión"** en esta sección.

1. Hacer clic en el botón **"Nuevo"** para expandir el formulario.
2. Ingresar un **nombre** para el compuesto (opcional pero recomendado).
3. Ingresar la **cadena SMILES** del compuesto.
4. La validación en tiempo real te indicará:
   - ✅ **SMILES válido**: Se muestra el número de átomos y el peso molecular.
   - ❌ **SMILES inválido**: Se muestra un mensaje de error.
5. Si el compuesto existe en PubChem, la aplicación sugerirá automáticamente su nombre. Puedes hacer clic en la sugerencia para usarlo.
6. Hacer clic en **"Guardar"**.
7. El modelo generará la predicción y se mostrará un mensaje de éxito con:
   - Nombre del compuesto.
   - Temperatura predicha en K y °C.
   - Incertidumbre del modelo (±22.80 K).
8. El compuesto aparecerá en la tabla con etiqueta naranja ("Usuario") y en tu lista personal.

#### Ejemplos de cadenas SMILES para probar

| Compuesto | SMILES |
|-----------|--------|
| Agua | `O` |
| Etanol | `CCO` |
| Benceno | `c1ccccc1` |
| Aspirina | `CC(=O)Oc1ccccc1C(=O)O` |
| Cafeína | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |
| Glucosa | `OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O` |

### 6.4 Eliminar un Compuesto de Usuario

1. Debajo del formulario de agregar, verás la lista de tus compuestos personalizados.
2. Cada compuesto tiene un botón de **eliminar** (ícono de papelera).
3. Hacer clic en el botón para eliminarlo permanentemente.

> **Nota:** Solo puedes eliminar compuestos que tú hayas creado.

### 6.5 Tabla de Datos

La tabla principal muestra todos los compuestos del sistema. Cada fila contiene:

| Columna | Descripción |
|---------|-------------|
| **ID** | Número identificador del compuesto |
| **Nombre** | Nombre del compuesto (si está disponible) |
| **SMILES** | Estructura molecular en notación SMILES (se trunca si es muy larga) |
| **Tm (K)** | Temperatura de fusión en Kelvin y Celsius, coloreada según la fuente |
| **Fuente** | Etiqueta Train/Test/User con el color correspondiente |
| **Acc.** | Botón para copiar los datos del compuesto |

La tabla incluye **paginación** en la parte inferior con controles para navegar entre páginas.

### 6.6 Filtrar por Fuente

Encima de la tabla verás botones de filtro:

- **Todos**: Muestra los 3 conjuntos de datos combinados.
- **Train**: Solo compuestos de entrenamiento (verdes).
- **Test**: Solo predicciones del modelo (azules).
- **User**: Solo tus compuestos personalizados (naranjas).

Hacer clic en el botón deseado para filtrar la tabla instantáneamente. El contador de compuestos se actualiza con el filtro.

### 6.7 Buscar por Texto

El campo de búsqueda filtra la tabla en tiempo real. Puedes buscar por:
- ID del compuesto
- Nombre del compuesto
- Cadena SMILES
- Valor de Tm

Simplemente escribe en el campo y la tabla se filtra automáticamente.

### 6.8 Filtrar por Rango de Temperatura

1. Hacer clic en el botón **"Rango de Tm"** para expandir el filtro.
2. Ajustar el rango usando el **slider doble** o ingresando valores exactos en los campos de texto.
3. La tabla se actualiza automáticamente mostrando solo los compuestos dentro del rango seleccionado.
4. Verás un indicador con la cantidad de compuestos filtrados y el porcentaje del total.
5. Hacer clic en **"Resetear"** para eliminar el filtro de temperatura.

### 6.9 Ordenar la Tabla

Hacer clic en el encabezado de una columna para ordenar la tabla. Un segundo clic invierte el orden:

- **ID**: Orden numérico.
- **Tm (K)**: Orden por temperatura de fusión.
- **Fuente**: Agrupa por tipo de dato.

Las flechas en el encabezado indican la dirección del ordenamiento actual.

### 6.10 Copiar Datos de un Compuesto

Cada fila tiene un botón de **copiar** (ícono de portapapeles) en la columna "Acc.":
- Al hacer clic, se copia la cadena SMILES del compuesto al portapapeles.
- El ícono cambia brevemente a un ✓ verde para confirmar la acción.

### 6.11 Exportar a CSV

1. Hacer clic en el botón **"Exportar CSV"** ubicado junto al título de la tabla.
2. Se descargará un archivo `.csv` con todos los datos actualmente visibles (respetando todos los filtros aplicados).
3. El archivo puede abrirse en Excel, Google Sheets o cualquier herramienta de análisis de datos.

---

## 7. Analytics

La página de **Analytics** (`/analytics`) ofrece visualizaciones interactivas para explorar los datos y apoyar la toma de decisiones. Incluye un botón de **refrescar** y un indicador de conexión en la parte superior.

### 7.1 Resumen del Dataset

Tres tarjetas de resumen estadístico, una por cada fuente de datos:

**Train (Real)** — Borde verde:
- Cantidad de compuestos.
- Media, Mediana, Mínimo, Máximo, Desviación Estándar, Rango Intercuartílico (IQR).

**Test (Predicciones)** — Borde azul:
- Cantidad de compuestos.
- Mismas estadísticas + nota de incertidumbre (MAE ±22.80 K).

**User (Personalizados)** — Borde naranja:
- Cantidad de tus compuestos.
- Estadísticas si tienes compuestos; mensaje de invitación si no tienes ninguno.

### 7.2 Box Plot Comparativo

Debajo de las tarjetas, un **boxplot visual** compara las tres distribuciones lado a lado:
- **Bigotes**: Valores mínimo y máximo.
- **Caja**: Primer cuartil (Q1) a tercer cuartil (Q3).
- **Línea central**: Mediana.
- Coloreado por fuente (verde, azul, naranja).

Esto permite ver de un vistazo cómo se comparan las distribuciones de temperatura entre los tres conjuntos de datos.

### 7.3 Filtro Global de Fuente

Los botones **Todos / Train / Test / Usuario** en esta sección filtran simultáneamente todos los gráficos de la página. Esto permite analizar cada fuente de datos de forma independiente.

Se muestra la cantidad de compuestos correspondientes al filtro activo.

### 7.4 Distribución por Temperatura

Gráfico de **barras apiladas** que muestra cuántos compuestos hay en cada rango de temperatura:

| Rango | Categoría |
|-------|-----------|
| < 150 K | Temperatura muy baja |
| 150–200 K | Temperatura baja |
| 200–250 K | Temperatura media-baja |
| 250–300 K | Temperatura ambiente |
| 300–350 K | Temperatura media-alta |
| 350–400 K | Temperatura alta |
| 400–500 K | Temperatura muy alta |
| > 500 K | Temperatura extrema |

Las barras están coloreadas según la fuente (verde/azul/naranja). **Pasa el cursor** sobre una barra para ver el desglose detallado por fuente.

### 7.5 Complejidad vs. Tm (Scatter Plot)

Gráfico de **dispersión** que muestra la relación entre la complejidad molecular y el punto de fusión:

- **Eje X**: Longitud de la cadena SMILES (proxy de complejidad molecular).
- **Eje Y**: Tm en Kelvin.
- **Puntos coloreados** por fuente (verde, azul, naranja).
- **Leyenda**: Train, Test, Usuario.

Pasa el cursor sobre un punto para ver: ID, Tm exacto, y vista previa de la cadena SMILES.

> **Interpretación:** En general, moléculas más complejas (SMILES más largos) tienden a tener puntos de fusión más altos debido a mayores fuerzas intermoleculares.

### 7.6 Grupos Funcionales

Gráfico de **barras horizontales con línea de tendencia** que muestra los 10 grupos funcionales más frecuentes:

- **Barras (rosa)**: Cantidad de compuestos que contienen cada grupo funcional.
- **Línea (naranja)**: Temperatura promedio de fusión de los compuestos con ese grupo.

Grupos funcionales analizados incluyen: OH (alcoholes), NH₂ (aminas), COOH (ácidos carboxílicos), halógenos, aromáticos, entre otros.

Pasa el cursor sobre una barra para ver: nombre del grupo, cantidad de compuestos, Tm promedio, y rango mín-máx.

> **Interpretación:** Los grupos polares capaces de formar puentes de hidrógeno (OH, COOH, NH₂) tienden a aumentar el punto de fusión.

### 7.7 Tamaño Molecular vs. Punto de Fusión

Gráfico **combinado de área + línea** con dos ejes:

- **Área (cyan, eje izquierdo)**: Cantidad de compuestos en cada categoría de tamaño.
- **Línea (naranja, eje derecho)**: Tm promedio por categoría.

Categorías de tamaño molecular (basadas en longitud de SMILES):

| Categoría | Rango |
|-----------|-------|
| Muy pequeño | 1–10 caracteres |
| Pequeño | 11–20 caracteres |
| Mediano | 21–35 caracteres |
| Grande | 36–50 caracteres |
| Muy grande | 51–75 caracteres |
| Enorme | >75 caracteres |

Pasa el cursor para ver el conteo de compuestos y el Tm promedio de cada categoría.

> **Interpretación:** Moléculas más grandes tienen más fuerzas de Van der Waals y generalmente mayor punto de fusión. Esta relación es útil para estimar si una predicción es razonable.

### 7.8 Guía de Interpretación

Al final de la página de Analytics se muestra una guía para la toma de decisiones:

- **Train (verde)**: Valores medidos experimentalmente — referencia confiable.
- **Test (azul)**: Predicciones del modelo con incertidumbre de **±22.80 K**.
- **User (naranja)**: Tus compuestos — compara con el dataset para evaluar confiabilidad.
- Las predicciones dentro del rango del dataset son **más confiables** que las extrapolaciones.

---

## 8. Modelo

La página de **Modelo** (`/model`) proporciona información técnica detallada sobre cómo funciona el sistema de predicción:

- **Especificaciones del modelo**: Tipo de arquitectura, dimensiones ocultas (300), profundidad (6 capas), dropout (10%), épocas de entrenamiento (50).

- **Validación cruzada (5-Fold)**: Tabla detallada con el MAE de cada fold de entrenamiento. Puedes pasar el cursor sobre cada fold para ver sus métricas.

- **Métricas finales**:

| Modelo | MAE (K) |
|--------|---------|
| **Híbrido (20% ChemProp + 80% Ensemble)** | **22.80 K** |
| ChemProp solo | 28.85 K |
| Ensemble solo | 26.64 K |

- **Pipeline de predicción**: Diagrama visual de 5 pasos que explica cómo se procesa una molécula: Entrada SMILES → Grafo molecular → Message passing → Readout → Predicción híbrida.

- **Características moleculares**: Listado de las propiedades atómicas (número atómico, grado, carga formal, quiralidad, hibridación, aromaticidad) y de enlace (tipo, conjugación, pertenencia a anillo, estereoquímica) que el modelo utiliza.

- **Ventajas del enfoque**: No requiere ingeniería de features manual, entiende la estructura molecular, es eficiente y competitivo.

---

## 9. Acerca de

La página **Acerca de** (`/about`) presenta:

- **Información de la competición** de Kaggle: "Thermophysical Property: Melting Point" con enlace directo.
- **Aplicaciones prácticas**: Diseño de fármacos, ciencia de materiales, screening virtual, reducción de costos experimentales.
- **Cómo funciona**: Flujo en 3 pasos (Entrada de SMILES → Procesamiento con modelo híbrido → Predicción con incertidumbre).
- **Línea temporal del proyecto**: 6 fases completadas (análisis, ChemProp, ensemble, híbrido, dashboard, producción).
- **Stack tecnológico**: Next.js 14, FastAPI, ChemProp, XGBoost, LightGBM, RDKit, Tailwind CSS, Recharts.

---

## 10. Preguntas Frecuentes (FAQ)

### ¿Necesito crear una cuenta para usar la aplicación?

No. Puedes explorar todos los datos, visualizaciones y análisis sin cuenta. Solo necesitas registrarte para **agregar compuestos personalizados**.

### ¿Qué tan preciso es el modelo?

El modelo tiene un **MAE de 22.80 K** validado en datos no vistos de Kaggle. Esto significa que, en promedio, las predicciones difieren del valor real en ±22.80 grados Kelvin. Para la mayoría de aplicaciones de screening, esta precisión es suficiente para filtrar candidatos.

### ¿Qué es una cadena SMILES?

SMILES (Simplified Molecular Input Line Entry System) es una notación de texto para representar estructuras moleculares. Cada molécula tiene una representación SMILES. Puedes obtener la cadena SMILES de cualquier compuesto en bases de datos como [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

### ¿Las temperaturas de los datos Train son predicciones del modelo?

No. Los datos de entrenamiento (verdes) tienen valores de Tm **medidos experimentalmente** en laboratorio. Solo los datos Test (azules) y User (naranjas) son predicciones del modelo.

### ¿Por qué algunos compuestos no tienen nombre?

La aplicación consulta la API de PubChem para obtener nombres. Si un compuesto no está registrado en PubChem o su SMILES no coincide con ningún registro, aparecerá sin nombre. Puedes asignar un nombre manualmente al agregar compuestos.

### ¿Cómo interpreto la incertidumbre de ±22.80 K?

Si el modelo predice un Tm de 350 K, el valor real probablemente se encuentra entre **327.20 K y 372.80 K**. Esta estimación se basa en el Error Absoluto Medio del modelo validado en Kaggle.

### ¿Puedo exportar los datos para usarlos en otra herramienta?

Sí. En la página de Predicciones, el botón **"Exportar CSV"** descarga un archivo con todos los datos visibles (respetando los filtros activos). El archivo se puede abrir en Excel, Google Sheets, Python (pandas), R, o cualquier herramienta de análisis.

### ¿Los gráficos de Analytics se actualizan en tiempo real?

Los gráficos se cargan al abrir la página. Si agregas nuevos compuestos, haz clic en el botón de **refrescar** en la página de Analytics para actualizar las visualizaciones.

---

*Manual de Usuario — MeltingPoint Dashboard v2.0*
*Febrero 2026*
