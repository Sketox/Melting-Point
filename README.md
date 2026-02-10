# Melting Point API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ChemProp](https://img.shields.io/badge/ChemProp-1.6.1-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue?style=for-the-badge)
![MAE](https://img.shields.io/badge/MAE-22.80_K-green?style=for-the-badge)

**REST API for molecular melting point prediction using Hybrid ML Model**

[Quick Start](#-quick-start) | [Model](#-model) | [Endpoints](#-api-endpoints) | [Frontend](#-frontend-dashboard) | [Troubleshooting](#-troubleshooting)

</div>

---

## Description

REST API built with **FastAPI** that provides melting point (Tm) predictions for molecules using a **hybrid model** combining:

- **ChemProp D-MPNN** - Graph Neural Network for molecular structure
- **Ensemble (XGBoost + LightGBM)** - Gradient Boosting on molecular fingerprints

Developed for the [Kaggle Thermophysical Property Competition](https://www.kaggle.com/competitions/playground-series-s5e6).

### Features

- **High Performance** - FastAPI with async support
- **Hybrid Model** - GNN + GBDT ensemble achieving **MAE 22.80 K**
- **RDKit Integration** - Complete SMILES validation and molecular descriptors
- **Analytics Endpoints** - Statistics, distributions, and functional group analysis
- **User Compounds** - Add custom molecules with real-time predictions
- **Auto Documentation** - Swagger UI and ReDoc included

---

## Quick Start

### Prerequisites

- **Python 3.11+** (tested on 3.11, 3.12, 3.13, 3.14)
- **pip**
- **Node.js 18+** (for the frontend dashboard)

### Backend Setup

```bash
# 1. Navigate to backend directory
cd MeltingPoint/backend

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\activate
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 4. (Optional) Install PyTorch CPU-only first to save ~2GB
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install dependencies
pip install -r requirements.txt

# 6. REQUIRED: Apply PyTorch 2.6+ compatibility patch for ChemProp
python patch_chemprop_torch.py

# 7. Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
# 1. Navigate to frontend directory (from project root)
cd melting-point-dashboard

# 2. Install dependencies
npm install

# 3. Run in development mode
npm run dev

# 4. Open http://localhost:3000
```

### Verify Installation

You should see in the backend logs:
```
INFO: ChemProp 1.6.1 detected correctly.
INFO: ChemProp enabled with 5 checkpoints.
INFO: Ensemble loaded with 15 models.
INFO: COMBINED mode active (MAE ~22.80 K)
```

Open in browser: **http://localhost:8000/docs** (API) | **http://localhost:3000** (Dashboard)

---

## Model

### Hybrid Architecture

```
                    SMILES Input
                         |
            +------------+------------+
            v                         v
      +-------------+          +-------------+
      |  ChemProp   |          |  Ensemble   |
      |  D-MPNN     |          |  XGB + LGB  |
      |  (GNN)      |          |(Fingerprints)|
      +-------------+          +-------------+
            |                         |
            | (20%)             (80%) |
            +------------+------------+
                         v
                  Final Prediction
                   MAE ~22.80 K
```

### Model Components

| Component | Type | MAE | Weight |
|-----------|------|-----|--------|
| ChemProp | D-MPNN (Graph Neural Network) | 28.85 K | 20% |
| XGBoost | Gradient Boosting | ~28.5 K | 55% of ensemble |
| LightGBM | Gradient Boosting | ~29.5 K | 45% of ensemble |
| **Ensemble** | Weighted Average | **27.59 K** | **80%** |
| **Combined** | ChemProp + Ensemble | **22.80 K** | Best |

> **Note**: CatBoost was used in the original competition submission (MAE 22.80 K).
> The current ensemble uses XGB+LGB only (no CatBoost dependency needed).
> CatBoost can optionally be installed for marginal improvement.

### Features Used (2,757 total)

| Type | Count | Description |
|------|-------|-------------|
| Morgan FP (ECFP4) | 2,048 | Circular substructures |
| MACCS Keys | 167 | Predefined chemical patterns |
| RDKit Descriptors | ~200 | Physicochemical properties |
| SMILES features | 13 | Length, rings, heteroatoms |
| Group features | 337 | Functional groups from dataset |

### Example Predictions

| Molecule | SMILES | Predicted | Actual | Error |
|----------|--------|-----------|--------|-------|
| Water | `O` | 272.17 K | 273.15 K | 0.98 K |
| Ethanol | `CCO` | ~159 K | 159 K | ~0 K |
| Benzene | `c1ccccc1` | ~279 K | 278.5 K | ~0.5 K |

---

## API Endpoints

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

### Data & Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/data-all` | All data (train+test+user) with source field |
| GET | `/predict-all` | All test set predictions |
| POST | `/predict-by-id` | Prediction by molecule ID |
| POST | `/validate-smiles` | Validate SMILES structure with RDKit |
| GET | `/compound-name` | Get compound name from PubChem |

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

## Project Structure

```
MeltingPoint/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── main.py                   # FastAPI endpoints
│   │   ├── ml_service.py             # ML predictions (Ensemble + ChemProp)
│   │   ├── schemas.py                # Pydantic models
│   │   └── config.py                 # Configuration
│   ├── models/
│   │   ├── model_chemprop/           # ChemProp 5 folds (~50 MB)
│   │   ├── ensemble_predictor.joblib # XGB+LGB ensemble (~100 MB)
│   │   ├── model.joblib              # Sklearn fallback (~5 MB)
│   │   └── best_params_paso6.json    # Hyperparameters
│   ├── data/
│   │   └── user_compounds.csv        # User compounds (auto-generated)
│   ├── patch_chemprop_torch.py       # Required for PyTorch 2.6+
│   ├── requirements.txt
│   └── requirements-deploy.txt
│
├── data/
│   ├── raw/
│   │   ├── train.csv                 # 2,662 samples
│   │   └── test.csv                  # 666 samples
│   └── processed/
│
├── src/                              # Training Scripts
│   ├── train_ensemble_production.py  # Retrain ensemble for backend
│   ├── train_ensemble_no_catboost.py # XGB+LGB only (no VS 2022 needed)
│   └── ...
│
├── submissions/
│   └── submission_paso6_cp20.csv     # Best (MAE 22.80)
│
└── melting-point-dashboard/          # Next.js Frontend (separate repo)
    ├── src/
    ├── package.json
    └── README.md
```

---

## Frontend Dashboard

The frontend is in the `melting-point-dashboard/` directory (at the same level as `MeltingPoint/`).

See [melting-point-dashboard/README.md](../melting-point-dashboard/README.md) for details.

---

## Deploying to Another Computer

### Step-by-step

1. **Clone/copy the project** (includes trained models)
2. **Backend**:
   ```bash
   cd MeltingPoint/backend
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   python patch_chemprop_torch.py
   uvicorn app.main:app --reload --port 8000
   ```
3. **Frontend**:
   ```bash
   cd melting-point-dashboard
   npm install
   npm run dev
   ```

### Required Model Files (~155 MB total)

- `backend/models/ensemble_predictor.joblib` (~100 MB)
- `backend/models/model_chemprop/` (~50 MB, 5 fold checkpoints)
- `backend/models/model.joblib` (~5 MB, fallback)

### If Ensemble is Missing

```bash
cd MeltingPoint/src
python train_ensemble_no_catboost.py   # No CatBoost needed
# Generates: backend/models/ensemble_predictor.joblib
```

### Common Issues on New Machines

| Problem | Cause | Solution |
|---------|-------|----------|
| `numpy` install fails | Version pinned too strictly | Use `numpy>=1.26.0,<2.0.0` |
| `catboost` install fails | Requires VS 2022 Build Tools | Not needed - ensemble uses XGB+LGB |
| `torch` takes forever / too large | Downloads CUDA version (~2.5 GB) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `rdkit` not found | Not on PyPI for all platforms | `pip install rdkit` or `conda install -c conda-forge rdkit` |
| `chemprop` conflicts | Needs specific versions | Install after torch |
| "Weights only load failed" | PyTorch 2.6+ compatibility | Run `python patch_chemprop_torch.py` |
| MAE 28.85 (not 22.80) | Ensemble not loaded | Check `ensemble_predictor.joblib` exists |
| Water predicts 161 K | ChemProp patch not applied | Run `python patch_chemprop_torch.py` |
| Slow first prediction | Normal - models load from disk | Wait 10-30s on first call |

---

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Validate SMILES
curl -X POST "http://localhost:8000/validate-smiles" \
  -H "Content-Type: application/json" \
  -d "{\"smiles\": \"O\"}"

# Create compound (triggers hybrid prediction)
curl -X POST "http://localhost:8000/compounds" \
  -H "Content-Type: application/json" \
  -d "{\"smiles\": \"O\", \"name\": \"Water\"}"
```

---

## Interactive Documentation

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** - Interactive testing |
| http://localhost:8000/redoc | **ReDoc** - API documentation |

---

## License

MIT License

---

<div align="center">

**Developed for the Kaggle Thermophysical Property Competition**

**Best Score: MAE 22.80 K**

</div>
