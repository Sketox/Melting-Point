# 🧪 Melting Point API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChemProp](https://img.shields.io/badge/ChemProp-1.6.1-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue?style=for-the-badge)
![MAE](https://img.shields.io/badge/MAE-22.80_K-green?style=for-the-badge)

**REST API for molecular melting point prediction using Hybrid ML Model**

[Installation](#-installation) • [Model](#-model) • [Endpoints](#-api-endpoints) • [Troubleshooting](#️-troubleshooting)

</div>

---

## 📋 Description

REST API built with **FastAPI** that provides melting point (Tm) predictions for molecules using a **hybrid model** combining:

- **ChemProp D-MPNN** - Graph Neural Network for molecular structure
- **Ensemble (XGBoost + LightGBM + CatBoost)** - Gradient Boosting on molecular fingerprints

Developed for the [Kaggle Thermophysical Property Competition](https://www.kaggle.com/competitions/playground-series-s5e6).

### ✨ Features

- 🚀 **High Performance** - FastAPI with async support
- 🧠 **Hybrid Model** - GNN + GBDT ensemble achieving **MAE 22.80 K**
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
INFO: ChemProp enabled with 5 checkpoints.
INFO: Ensemble loaded with 15 models.
INFO: COMBINED mode active (MAE ~22.80 K)
```

Open in browser: **http://localhost:8000/docs**

---

## 🧠 Model

### Hybrid Architecture

```
                    SMILES Input
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      ┌─────────────┐          ┌─────────────┐
      │  ChemProp   │          │  Ensemble   │
      │  D-MPNN     │          │ XGB+LGB+CAT │
      │  (GNN)      │          │(Fingerprints)│
      └─────────────┘          └─────────────┘
            │                         │
            │ (20%)             (80%) │
            └────────────┬────────────┘
                         ▼
                  Final Prediction
                   MAE ~22.80 K
```

### Model Components

| Component | Type | MAE | Weight |
|-----------|------|-----|--------|
| ChemProp | D-MPNN (Graph Neural Network) | 28.85 K | 20% |
| XGBoost | Gradient Boosting | ~28.5 K | ~17.5% |
| LightGBM | Gradient Boosting | ~29.5 K | ~29.8% |
| CatBoost | Gradient Boosting | ~27.0 K | ~52.7% |
| **Ensemble** | Weighted Average | **26.64 K** | **80%** |
| **Combined** | ChemProp + Ensemble | **22.80 K** | ⭐ |

### Features Used (2,757 total)

| Type | Count | Description |
|------|-------|-------------|
| Morgan FP (ECFP4) | 2,048 | Circular substructures |
| MACCS Keys | 167 | Predefined chemical patterns |
| RDKit Descriptors | ~200 | Physicochemical properties |
| SMILES features | 13 | Length, rings, heteroatoms |
| Group features | 337 | Functional groups from dataset |

### Performance Comparison

| Configuration | MAE (K) | Notes |
|---------------|---------|-------|
| ChemProp only | 28.85 | GNN baseline |
| Ensemble only | 26.64 | GBDT baseline |
| **Combined (20% CP)** | **22.80** | ⭐ **Best (Kaggle)** |

### Example Predictions

| Molecule | SMILES | Predicted | Actual | Error |
|----------|--------|-----------|--------|-------|
| Water | `O` | 272.17 K | 273.15 K | 0.98 K ✓ |
| Ethanol | `CCO` | ~159 K | 159 K | ~0 K ✓ |
| Benzene | `c1ccccc1` | ~279 K | 278.5 K | ~0.5 K ✓ |

---

## 📦 Requirements

```txt
# Backend API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Data Processing
pandas>=2.0.0
numpy==1.26.4

# Machine Learning
scikit-learn>=1.3.0
joblib>=1.3.0
xgboost>=2.0.0      # NEW
lightgbm>=4.0.0     # NEW
catboost>=1.2.0     # NEW

# Chemistry
rdkit>=2023.03.1
chemprop==1.6.1
torch>=2.0.0
```

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
| GET | `/stats` | Descriptive statistics |
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

## 📂 Project Structure

```
backend/
├── 📁 app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app and endpoints
│   ├── ml_service.py     # ML service (Ensemble + ChemProp)
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Configuration and paths
│
├── 📁 data/
│   └── user_compounds.csv  # User compounds (auto-generated)
│
├── 📁 models/
│   ├── ensemble_predictor.joblib  # ⭐ XGB+LGB+CAT ensemble (~100 MB)
│   ├── model.joblib               # Sklearn fallback (~5 MB)
│   ├── best_params_paso6.json     # Optuna hyperparameters
│   └── 📁 model_chemprop/         # ChemProp 5 folds (~50 MB)
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
| MAE 28.85 (not 22.80) | Ensemble not loaded | Check `ensemble_predictor.joblib` exists |
| Water predicts 161 K (not 272 K) | ChemProp patch not applied | Run `python patch_chemprop_torch.py` |
| Slow first prediction (10-30s) | Normal behavior | Models load from disk on first call |
| "Ensemble not loaded" | Missing file | Download from repo or retrain |

### Retraining the Ensemble (if needed)

```bash
cd ../src
python train_ensemble_production.py
# Takes ~10-15 minutes
# Generates: backend/models/ensemble_predictor.joblib
```

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","model_loaded":true,"dataset_size":666}

# Validate SMILES
curl -X POST "http://localhost:8000/validate-smiles" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O"}'

# Create compound (triggers hybrid prediction)
curl -X POST "http://localhost:8000/compounds" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O", "name": "Water"}'

# Expected response:
# {
#   "id": "USR_001",
#   "smiles": "O",
#   "name": "Water",
#   "Tm_pred": 272.17,
#   "Tm_celsius": -0.98,
#   "uncertainty": "±23 K",
#   "method": "combined (cp=20%)"
# }
```

---

## 📖 Interactive Documentation

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** - Interactive testing |
| http://localhost:8000/redoc | **ReDoc** - API documentation |

---

## 🔄 Deploying to Another Computer

1. **Clone the repository** (includes models in Git LFS or directly)
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Apply patch**: `python patch_chemprop_torch.py`
4. **Run server**: `uvicorn app.main:app --reload --port 8000`

Required model files (~155 MB total):
- `models/ensemble_predictor.joblib` (~100 MB)
- `models/model_chemprop/` (~50 MB)
- `models/model.joblib` (~5 MB)

---

## 📄 License

MIT License

---

<div align="center">

**Developed for the Kaggle Thermophysical Property Competition** 🧪

**Best Score: MAE 22.80 K** ⭐

</div>