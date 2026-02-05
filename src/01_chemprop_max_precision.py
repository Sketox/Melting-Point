"""
═══════════════════════════════════════════════════════════════════════════════
                PASO 1: ChemProp D-MPNN - MÁXIMA PRECISIÓN
                MeltingPoint Kaggle Competition
                
                *** PARA CHEMPROP v1.6.1 - WINDOWS ***
                
                ANTES DE EJECUTAR:
                pip uninstall chemprop -y
                pip install chemprop==1.6.1
═══════════════════════════════════════════════════════════════════════════════

Autor: Sketo
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
TRAIN_RAW = DATA_RAW / "train.csv"
TEST_RAW = DATA_RAW / "test.csv"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHEMPROP_TRAIN = DATA_PROCESSED / "chemprop_train.csv"
CHEMPROP_TEST = DATA_PROCESSED / "chemprop_test.csv"

MODEL_DIR = PROJECT_ROOT / "backend" / "models" / "chemprop_max"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PARA MÁXIMA PRECISIÓN
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "hidden_size": 600,
    "depth": 5,
    "ffn_hidden_size": 600,
    "ffn_num_layers": 3,
    "dropout": 0.2,              # Aumentado para compensar falta de weight_decay
    "aggregation": "mean",
    "epochs": 150,
    "batch_size": 50,
    "warmup_epochs": 2.0,
    "init_lr": 0.0001,
    "max_lr": 0.001,
    "final_lr": 0.0001,
    "ensemble_size": 10,
    "num_folds": 1,
    "split_type": "random",
    "split_sizes": [0.9, 0.05, 0.05],
    "features_generator": "rdkit_2d_normalized",
    "metric": "mae",
    "dataset_type": "regression",
}


def check_chemprop_version():
    """Verifica que ChemProp v1.x esté instalado."""
    try:
        import chemprop
        version = chemprop.__version__
        print(f"\n  ChemProp versión: {version}")
        
        if version.startswith("2"):
            print(f"""
  ❌ Tienes ChemProp v{version} (versión 2.x)
  
  Este script requiere v1.6.1. Ejecuta:
  
      pip uninstall chemprop -y
      pip install chemprop==1.6.1
  
  Y vuelve a ejecutar este script.
            """)
            return False
        return True
    except ImportError:
        print("\n  ❌ ChemProp no está instalado.")
        print("     Ejecuta: pip install chemprop==1.6.1")
        return False


def check_gpu():
    """Verifica GPU."""
    print("\n" + "="*70)
    print("  VERIFICANDO SISTEMA")
    print("="*70)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  ✓ CUDA: {torch.version.cuda}")
            return True
        else:
            print("\n  ⚠ GPU no detectada, usará CPU (más lento)")
            return False
    except ImportError:
        print("\n  ⚠ PyTorch no instalado")
        return False


def run_chemprop_command(executable: str, args: list, description: str = "") -> int:
    """
    Ejecuta un comando de ChemProp.
    Usa shell=True para encontrar el ejecutable en el PATH del entorno virtual.
    """
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print("="*70)
    
    # Construir comando completo como string para shell=True
    cmd = [executable] + args
    cmd_str = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    
    print(f"\n  Ejecutando ChemProp...")
    print(f"  (esto puede tardar 1-3 horas)\n")
    print("-"*70)
    
    try:
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True  # Necesario para encontrar el ejecutable en Windows
        )
        
        for line in process.stdout:
            print(f"  {line}", end="")
        
        process.wait()
        print("-"*70)
        
        if process.returncode != 0:
            print(f"\n  ❌ Error (código {process.returncode})")
            return process.returncode
        
        print(f"\n  ✓ Completado")
        return 0
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        return 1


def prepare_data():
    """Prepara los datos en formato ChemProp."""
    print("\n" + "="*70)
    print("  PASO 1.1: Preparando datos")
    print("="*70)
    
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(TRAIN_RAW)
    test_df = pd.read_csv(TEST_RAW)
    
    print(f"\n  Train: {len(train_df)} moléculas")
    print(f"  Test:  {len(test_df)} moléculas")
    
    # Formato ChemProp
    pd.DataFrame({
        "smiles": train_df["SMILES"], 
        "Tm": train_df["Tm"]
    }).to_csv(CHEMPROP_TRAIN, index=False)
    
    pd.DataFrame({
        "smiles": test_df["SMILES"]
    }).to_csv(CHEMPROP_TEST, index=False)
    
    print(f"\n  ✓ Datos preparados")
    print(f"    {CHEMPROP_TRAIN}")
    print(f"    {CHEMPROP_TEST}")


def train():
    """Entrena el modelo ChemProp."""
    print("\n" + "="*70)
    print("  PASO 1.2: Entrenando ChemProp")
    print("="*70)
    
    cfg = CONFIG
    
    print(f"""
    📋 Configuración de MÁXIMA PRECISIÓN:
    
    • Epochs:       {cfg['epochs']}
    • Ensemble:     {cfg['ensemble_size']} modelos
    • Hidden size:  {cfg['hidden_size']}
    • Depth:        {cfg['depth']}
    • Dropout:      {cfg['dropout']}
    • Features:     RDKit 2D normalized
    
    ⏳ Tiempo estimado: 1-3 horas en RTX 2060
    """)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Argumentos para chemprop_train
    args = [
        "--data_path", str(CHEMPROP_TRAIN),
        "--dataset_type", cfg["dataset_type"],
        "--target_columns", "Tm",
        "--save_dir", str(MODEL_DIR),
        "--hidden_size", str(cfg["hidden_size"]),
        "--depth", str(cfg["depth"]),
        "--ffn_hidden_size", str(cfg["ffn_hidden_size"]),
        "--ffn_num_layers", str(cfg["ffn_num_layers"]),
        "--dropout", str(cfg["dropout"]),
        "--aggregation", cfg["aggregation"],
        "--epochs", str(cfg["epochs"]),
        "--batch_size", str(cfg["batch_size"]),
        "--warmup_epochs", str(cfg["warmup_epochs"]),
        "--init_lr", str(cfg["init_lr"]),
        "--max_lr", str(cfg["max_lr"]),
        "--final_lr", str(cfg["final_lr"]),
        "--ensemble_size", str(cfg["ensemble_size"]),
        "--num_folds", str(cfg["num_folds"]),
        "--split_type", cfg["split_type"],
        "--split_sizes", str(cfg["split_sizes"][0]), str(cfg["split_sizes"][1]), str(cfg["split_sizes"][2]),
        "--features_generator", cfg["features_generator"],
        "--no_features_scaling",  # Requerido cuando usas rdkit_2d_normalized
        "--metric", cfg["metric"],
        "--save_smiles_splits",
        "--save_preds",
    ]
    
    result = run_chemprop_command("chemprop_train", args, "Entrenando modelo...")
    
    if result != 0:
        print("\n  ❌ Error durante el entrenamiento.")
        print("\n  Posibles soluciones:")
        print("    1. Verifica que tienes ChemProp v1.6.1:")
        print("       pip install chemprop==1.6.1")
        print("    2. Revisa que los archivos de datos existen")
        sys.exit(1)


def predict():
    """Genera predicciones."""
    print("\n" + "="*70)
    print("  PASO 1.3: Generando predicciones")
    print("="*70)
    
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = DATA_PROCESSED / "chemprop_predictions.csv"
    
    args = [
        "--test_path", str(CHEMPROP_TEST),
        "--checkpoint_dir", str(MODEL_DIR),
        "--preds_path", str(preds_path),
        "--features_generator", CONFIG["features_generator"],
        "--no_features_scaling",  # Requerido cuando usas rdkit_2d_normalized
    ]
    
    result = run_chemprop_command("chemprop_predict", args, "Generando predicciones...")
    
    if result != 0:
        print("\n  ❌ Error durante la predicción.")
        sys.exit(1)
    
    # Crear submission
    print("\n  📝 Creando archivo de submission...")
    
    test_df = pd.read_csv(TEST_RAW)
    preds_df = pd.read_csv(preds_path)
    
    if "Tm" in preds_df.columns:
        predictions = preds_df["Tm"]
    else:
        predictions = preds_df.iloc[:, 0]
    
    submission = pd.DataFrame({
        "id": test_df["id"],
        "Tm": predictions
    })
    
    submission_path = SUBMISSION_DIR / "submission_chemprop.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n  ✓ Submission guardado: {submission_path}")
    
    print(f"\n  📊 Estadísticas de predicciones:")
    print(f"      Min:    {predictions.min():.2f} K")
    print(f"      Max:    {predictions.max():.2f} K")
    print(f"      Mean:   {predictions.mean():.2f} K")
    print(f"      Std:    {predictions.std():.2f} K")


def show_results():
    """Muestra resultados si están disponibles."""
    test_scores = MODEL_DIR / "test_scores.csv"
    if test_scores.exists():
        print("\n" + "="*70)
        print("  📊 RESULTADOS DE VALIDACIÓN")
        print("="*70)
        scores = pd.read_csv(test_scores)
        print(f"\n{scores.to_string(index=False)}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🧪 PASO 1: ChemProp D-MPNN - MÁXIMA PRECISIÓN                        ║
║        MeltingPoint Kaggle Competition                                       ║
║                                                                              ║
║        Versión: ChemProp v1.6.1                                             ║
║        GPU: RTX 2060 (6GB)                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar sistema
    check_gpu()
    
    # Verificar ChemProp
    if not check_chemprop_version():
        sys.exit(1)
    
    # Ejecutar pipeline
    prepare_data()
    train()
    predict()
    show_results()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ✅ PASO 1 COMPLETADO                                                        ║
║                                                                              ║
║  Archivos generados:                                                         ║
║    • Modelo: backend/models/chemprop_max/                                   ║
║    • Submission: submissions/submission_chemprop.csv                         ║
║                                                                              ║
║  🎯 SIGUIENTE:                                                               ║
║    1. Sube submission_chemprop.csv a Kaggle                                 ║
║    2. Anota el score que te da                                              ║
║    3. Regresa y continuamos con PASO 2: Ensemble                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()