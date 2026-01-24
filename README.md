# 🧪 Melting Point API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChemProp](https://img.shields.io/badge/ChemProp-1.6.1-orange?style=for-the-badge)
![RDKit](https://img.shields.io/badge/RDKit-2025.9-blue?style=for-the-badge)

**REST API for molecular melting point prediction using ChemProp D-MPNN**

[Installation](#-installation) • [Endpoints](#-endpoints) • [Model](#-model) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Description

REST API built with **FastAPI** that provides melting point (Tm) predictions for molecules using a trained **ChemProp D-MPNN** model with 5-fold cross-validation. Developed for the [Kaggle Thermophysical Property Competition](https://www.kaggle.com/competitions/playground-series-s5e6).

### ✨ Features

- 🚀 **High Performance** - FastAPI with async support
- 🧠 **ChemProp D-MPNN** - Graph neural network specialized for molecular properties
- 🔬 **RDKit Integration** - Complete SMILES validation and molecular descriptors
- 📊 **Analytics Endpoints** - Statistics, distributions, and functional group analysis
- 👤 **User Compounds** - Add custom molecules with real-time predictions
- 📖 **Auto Documentation** - Swagger UI and ReDoc included

---

## 🚀 Installation

### Prerequisites

- Python 3.11+ (tested on 3.14)
- pip

### Step by Step

```bash
# 1. Navigate to backend directory
cd MeltingPoint/backend

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. ⚠️ REQUIRED: Apply PyTorch 2.6+ compatibility patch
python patch_chemprop_torch.py

# 6. Run the server
uvicorn app.main:app --reload --port 8000
```

### Verify Installation

You should see in the logs:
```
INFO: ChemProp 1.6.1 detected correctly.
INFO: ChemProp enabled with 5 checkpoints (ensemble).
INFO: Application startup complete.
```

Open in browser: **http://localhost:8000/docs**

---

## 📦 Requirements

```txt
# Backend API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Data & ML
pandas>=2.0.0
numpy==1.26.4
scikit-learn>=1.3.0
joblib>=1.3.0

# Chemistry
rdkit>=2023.03.1
chemprop==1.6.1
torch>=2.0.0
```

---

## 🧠 Model

### ChemProp D-MPNN Architecture

| Parameter | Value |
|-----------|-------|
| **Type** | D-MPNN (Directed Message Passing Neural Network) |
| **Hidden Size** | 300 |
| **Depth** | 6 layers |
| **Dropout** | 0.1 |
| **Epochs** | 50 |
| **Batch Size** | 32 |
| **Validation** | 5-Fold Cross-Validation |

### Performance Metrics

| Fold | Validation MAE | Test MAE | Best Epoch |
|------|----------------|----------|------------|
| 0 | 29.63 K | 26.15 K | 30 |
| 1 | 29.26 K | 27.64 K | 41 |
| 2 | 27.22 K | 35.03 K | 45 |
| 3 | 26.57 K | 27.35 K | 48 |
| 4 | 31.31 K | 28.09 K | 28 |
| **Overall** | - | **28.85 ± 3.16 K** | - |

### Example Predictions

| Molecule | SMILES | Predicted | Actual |
|----------|--------|-----------|--------|
| Water | `O` | 272.17 K | 273.15 K ✓ |
| Ethanol | `CCO` | ~159 K | 159 K ✓ |

---

## 📌 API Endpoints

### Base URL
```
http://localhost:8000
```

### Info & Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model-info` | Model metrics and configuration |

### SMILES Validation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/validate-smiles` | Validate SMILES structure with RDKit |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/predict-all` | All test set predictions |
| POST | `/predict-by-id` | Prediction by molecule ID |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats` | Descriptive statistics (mean, std, min, max, median, q25, q75) |
| GET | `/predictions/range` | Filter by temperature range |
| GET | `/predictions/distribution` | Distribution by temperature categories |
| GET | `/predictions/by-functional-group` | Analysis by functional groups |
| GET | `/predictions/by-molecule-size` | Analysis by molecular size |

### User Compounds

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/compounds` | List saved compounds |
| POST | `/compounds` | Create compound (validates SMILES + predicts Tm) |
| DELETE | `/compounds/{id}` | Delete compound |

---

## 🔧 Project Structure

```
backend/
├── 📁 app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app and endpoints
│   ├── ml_service.py     # ML service with ChemProp integration
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Configuration and paths
│
├── 📁 data/
│   └── user_compounds.csv  # User compounds (auto-generated)
│
├── 📁 models/
│   ├── model.joblib        # Sklearn fallback model
│   └── 📁 model_chemprop/  # Trained ChemProp model (5 folds)
│       ├── fold_0/model_0/model.pt
│       ├── fold_1/model_0/model.pt
│       ├── fold_2/model_0/model.pt
│       ├── fold_3/model_0/model.pt
│       ├── fold_4/model_0/model.pt
│       └── args.json
│
├── patch_chemprop_torch.py  # ⚠️ Required for PyTorch 2.6+
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## ⚠️ Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "Weights only load failed" | PyTorch 2.6+ compatibility | Run `python patch_chemprop_torch.py` |
| Water predicts 161 K (not 272 K) | ChemProp not working | Verify patch was applied, check logs |
| Slow predictions (10-30s) | Normal ChemProp behavior | Model loads from disk each time |
| "multiprocessing_context" error | Windows incompatibility | Already handled with `num_workers=0` |

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Validate SMILES
curl -X POST "http://localhost:8000/validate-smiles" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O"}'

# Create compound (triggers ChemProp prediction)
curl -X POST "http://localhost:8000/compounds" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O", "name": "Water"}'
# Expected: ~272.17 K

# Get statistics
curl http://localhost:8000/stats
```

---

## 📖 Interactive Documentation

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** - Interactive testing |
| http://localhost:8000/redoc | **ReDoc** - Readable documentation |

---

## 📄 License

MIT License

---

<div align="center">

**Developed for the Kaggle Thermophysical Property Competition** 🧪

</div>