# MeltingPoint Backend - CLAUDE.md

## Descripción
Backend FastAPI para predicción de puntos de fusión moleculares usando modelo ChemProp D-MPNN.
Competencia Kaggle: "Thermophysical Property: Melting Point"

## Stack Tecnológico
- **Framework**: FastAPI 0.104+
- **ML**: ChemProp (D-MPNN), scikit-learn
- **Química**: RDKit (validación SMILES, descriptores moleculares)
- **Data**: pandas, numpy
- **Server**: uvicorn

## Estructura del Proyecto
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, endpoints, CORS
│   ├── ml_service.py     # Lógica ML, predicciones, validación SMILES
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Configuración, rutas, constantes
├── data/
│   └── test_processed.csv  # Dataset de predicciones (666 moléculas)
├── models/                 # Modelos ChemProp (opcional)
├── requirements.txt
└── CLAUDE.md
```

## Modelo ML
- **Arquitectura**: ChemProp D-MPNN (Directed Message Passing Neural Network)
- **MAE**: 28.85 ± 3.16 K (5-fold cross-validation)
- **Dataset**: 2,662 moléculas de entrenamiento, 666 de test
- **Input**: SMILES molecular
- **Output**: Temperatura de fusión en Kelvin

## Endpoints API (14 total)

### Info
- `GET /health` - Estado del servidor y modelo
- `GET /model-info` - Métricas del modelo (MAE, arquitectura)

### Validación
- `POST /validate-smiles` - Validar estructura SMILES antes de usar
  - Request: `{"smiles": "CCO"}`
  - Response: `{"valid": true, "canonical_smiles": "CCO", "num_atoms": 3, "molecular_weight": 46.07, "error": null}`

### Predicciones
- `GET /predict-all` - Todas las predicciones del test set
- `POST /predict-by-id` - Predicción por ID de molécula

### Analytics
- `GET /stats` - Estadísticas descriptivas (mean, std, min, max, quartiles)
- `GET /predictions/range` - Filtrar por rango de temperatura
- `GET /predictions/distribution` - Distribución por categorías de Tm
- `GET /predictions/by-functional-group` - Agrupación por grupos funcionales
- `GET /predictions/by-molecule-size` - Agrupación por tamaño molecular

### Compuestos de Usuario
- `GET /compounds` - Listar compuestos guardados
- `POST /compounds` - Crear compuesto (valida SMILES, predice Tm)
  - Request: `{"smiles": "CCO", "name": "Ethanol"}`
  - Response incluye predicción con incertidumbre
- `DELETE /compounds/{id}` - Eliminar compuesto

## Validación SMILES (RDKit)
El backend usa RDKit para validar SMILES:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValidationError("SMILES inválido")
```

Información retornada:
- `valid`: boolean
- `canonical_smiles`: SMILES canónico
- `num_atoms`: número de átomos
- `molecular_weight`: peso molecular (g/mol)
- `error`: mensaje de error si inválido

## Grupos Funcionales (SMARTS)
```python
FUNCTIONAL_GROUPS = {
    "Alcoholes": "[OX2H]",
    "Ácidos Carboxílicos": "[CX3](=O)[OX2H1]",
    "Aminas": "[NX3;H2,H1;!$(NC=O)]",
    "Halogenados": "[F,Cl,Br,I]",
    "Aromáticos": "c1ccccc1",
    "Hidrocarburos": "[CX4]"
}
```

## Configuración
```python
# config.py
MODEL_MAE = 28.85        # Error absoluto medio
MODEL_MAE_STD = 3.16     # Desviación estándar del MAE
DATA_PATH = "data/test_processed.csv"
```

## Comandos de Desarrollo
```bash
# Instalar dependencias
pip install -r requirements.txt --break-system-packages

# Instalar RDKit (requerido para validación)
pip install rdkit --break-system-packages

# Ejecutar servidor
uvicorn app.main:app --reload --port 8000

# Ver docs
# http://localhost:8000/docs (Swagger)
# http://localhost:8000/redoc (ReDoc)
```

## Dependencias Clave
```
fastapi>=0.104.0
uvicorn>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
rdkit>=2023.03.1      # CRÍTICO para validación SMILES
pydantic>=2.0.0
python-multipart>=0.0.6
```

## Notas Importantes
1. **Incertidumbre**: Todas las predicciones incluyen ±28.85 K de incertidumbre
2. **Validación**: El endpoint `/compounds` valida SMILES antes de aceptar
3. **Moléculas simples**: El modelo fue entrenado con moléculas orgánicas complejas, predicciones para moléculas simples (agua, metano) pueden ser inexactas
4. **CORS**: Configurado para permitir localhost:3000 (frontend Next.js)

## Errores Comunes
- **400 Bad Request**: SMILES inválido o caracteres no permitidos
- **404 Not Found**: ID de molécula no existe en el dataset
- **500 Internal Server Error**: Error en predicción (revisar logs)
