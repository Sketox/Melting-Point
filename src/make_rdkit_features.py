from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# --- Rutas básicas (ajusta si tus variables se llaman distinto) ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

OUTPUT_TRAIN = PROCESSED_DIR / "train_rdkit.csv"
OUTPUT_TEST = PROCESSED_DIR / "test_rdkit.csv"


def smiles_to_features(smiles: str) -> dict:
    """Convierte un SMILES a un diccionario de descriptores RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Si falla el parseo, devolvemos NaNs para rellenar luego
        return {
            "rdkit_MolWt": float("nan"),
            "rdkit_ExactMolWt": float("nan"),
            "rdkit_MolLogP": float("nan"),
            "rdkit_TPSA": float("nan"),
            "rdkit_NumHDonors": float("nan"),
            "rdkit_NumHAcceptors": float("nan"),
            "rdkit_NumRotatableBonds": float("nan"),
            "rdkit_RingCount": float("nan"),
            "rdkit_FractionCSP3": float("nan"),
        }

    return {
        "rdkit_MolWt": Descriptors.MolWt(mol),
        "rdkit_ExactMolWt": rdMolDescriptors.CalcExactMolWt(mol),
        "rdkit_MolLogP": Descriptors.MolLogP(mol),
        "rdkit_TPSA": rdMolDescriptors.CalcTPSA(mol),
        "rdkit_NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "rdkit_NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rdkit_NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rdkit_RingCount": rdMolDescriptors.CalcNumRings(mol),
        "rdkit_FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def add_rdkit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas rdkit_*** a un DataFrame que tiene la columna 'SMILES'."""
    feat_list = df["SMILES"].apply(smiles_to_features)
    feat_df = pd.DataFrame(list(feat_list))
    # Rellenamos NaNs con la media de cada columna
    feat_df = feat_df.fillna(feat_df.mean())
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def main() -> None:
    print("📂 Cargando train y test crudos...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("🧪 Generando descriptores RDKit para train...")
    train_df_rdkit = add_rdkit_features(train_df)

    print("🧪 Generando descriptores RDKit para test...")
    test_df_rdkit = add_rdkit_features(test_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"💾 Guardando train con RDKit en: {OUTPUT_TRAIN}")
    train_df_rdkit.to_csv(OUTPUT_TRAIN, index=False)

    print(f"💾 Guardando test con RDKit en: {OUTPUT_TEST}")
    test_df_rdkit.to_csv(OUTPUT_TEST, index=False)

    print("✅ RDKit features generadas correctamente.")


if __name__ == "__main__":
    main()
