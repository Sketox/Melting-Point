import subprocess
from pathlib import Path

import pandas as pd


def run_cmd(cmd: list[str]) -> None:
    """Ejecuta un comando de consola y muestra lo que hace."""
    print("\n▶ Ejecutando comando:")
    print("  ", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    print(f"✅ Comando terminado con código {result.returncode}\n")


def main():
    # 1. Detectar raíz del proyecto
    project_root = Path(__file__).resolve().parents[1]

    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test.csv"

    model_dir = project_root / "backend" / "models" / "model_chemprop"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Cargando train.csv desde: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"📂 Cargando test.csv desde: {test_path}")
    test_df = pd.read_csv(test_path)

    # 2. Preparar CSVs en el formato que espera ChemProp
    #    Train: smiles + target
    chemprop_train_path = processed_dir / "chemprop_train.csv"
    chemprop_test_path = processed_dir / "chemprop_test.csv"

    train_df_small = pd.DataFrame({
        "smiles": train_df["SMILES"],
        "target": train_df["Tm"],
    })
    train_df_small.to_csv(chemprop_train_path, index=False)

    test_df_small = pd.DataFrame({
        "smiles": test_df["SMILES"],
    })
    test_df_small.to_csv(chemprop_test_path, index=False)

    print(f"✅ Guardado chemprop_train.csv en: {chemprop_train_path}")
    print(f"✅ Guardado chemprop_test.csv en:  {chemprop_test_path}")

    # 3. Entrenar modelo D-MPNN con chemprop_train (CLI)
    #    Aquí puedes ajustar parámetros igual que en línea de comandos.
    cmd_train = [
        "chemprop_train",
        "--data_path", str(chemprop_train_path),
        "--dataset_type", "regression",
        "--target_columns", "target",
        "--save_dir", str(model_dir),
        "--epochs", "50",
        "--hidden_size", "300",
        "--depth", "6",
        "--dropout", "0.1",
        "--batch_size", "32",
        "--num_folds", "5",
        "--split_type", "random",
        "--metric", "mae",
    ]

    print("🚀 Entrenando modelo D-MPNN (ChemProp)...")
    run_cmd(cmd_train)

    # 4. Predecir sobre test.csv con chemprop_predict
    preds_path = processed_dir / "chemprop_test_preds.csv"

    cmd_predict = [
        "chemprop_predict",
        "--test_path", str(chemprop_test_path),
        "--checkpoint_dir", str(model_dir),
        "--preds_path", str(preds_path),
    ]

    print("🧮 Generando predicciones para test.csv ...")
    run_cmd(cmd_predict)

    # 5. Cargar predicciones y combinarlas con los IDs de Kaggle
    preds_df = pd.read_csv(preds_path)

    # ChemProp guarda una columna por cada target; como solo tenemos "target",
    # será la primera columna.
    if "target" in preds_df.columns:
        tm_pred = preds_df["target"]
    else:
        # fallback por si la columna viene sin nombre claro
        tm_pred = preds_df.iloc[:, 0]

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "Tm_pred": tm_pred,
    })

    out_path = processed_dir / "test_chemprop_predictions.csv"
    submission_df.to_csv(out_path, index=False)

    print(f"💾 Predicciones finales guardadas en: {out_path}")
    print("🎉 Entrenamiento ChemProp + predicción completados correctamente.")


if __name__ == "__main__":
    main()
