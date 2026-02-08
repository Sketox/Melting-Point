"""
Script para agregar nombres de compuestos a los CSVs usando PubChem.

PubChem tiene rate limit de 5 requests por segundo, pero para ser seguros
usamos 5 segundos entre cada request para no ser bloqueados.

El script:
1. Lee train.csv y test.csv
2. Para cada SMILES, busca el nombre en PubChem
3. Guarda los CSVs actualizados con la columna 'name'
4. Muestra progreso y guarda checkpoint cada 50 compuestos
"""

import pandas as pd
import requests
import time
import os
from pathlib import Path
import json

# Configuracion
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_FILE = Path(__file__).parent / "pubchem_cache.json"
RATE_LIMIT_SECONDS = 5  # 5 segundos entre requests para no saturar PubChem
CHECKPOINT_EVERY = 50   # Guardar cada 50 compuestos

def load_cache():
    """Carga el cache de nombres ya consultados."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Guarda el cache de nombres."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_compound_name_pubchem(smiles: str) -> str:
    """
    Obtiene el nombre de un compuesto desde PubChem usando SMILES.

    Args:
        smiles: SMILES del compuesto

    Returns:
        Nombre del compuesto o cadena vacia si no se encuentra
    """
    try:
        # URL encode del SMILES
        import urllib.parse
        encoded_smiles = urllib.parse.quote(smiles, safe='')

        # Buscar en PubChem por SMILES
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/property/IUPACName/JSON"

        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties']
                if props and 'IUPACName' in props[0]:
                    return props[0]['IUPACName']

        # Intentar obtener nombre comun (synonym) si no hay IUPAC
        url_synonyms = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/synonyms/JSON"
        response2 = requests.get(url_synonyms, timeout=30)

        if response2.status_code == 200:
            data2 = response2.json()
            if 'InformationList' in data2 and 'Information' in data2['InformationList']:
                info = data2['InformationList']['Information']
                if info and 'Synonym' in info[0] and info[0]['Synonym']:
                    # Retornar el primer sinonimo (suele ser el mas comun)
                    return info[0]['Synonym'][0]

        return ""

    except Exception as e:
        print(f"    Error consultando PubChem: {e}")
        return ""

def add_names_to_csv(csv_path: Path, cache: dict, output_suffix: str = "_with_names") -> pd.DataFrame:
    """
    Agrega nombres a un CSV.

    Args:
        csv_path: Ruta al CSV
        cache: Diccionario de cache SMILES -> nombre
        output_suffix: Sufijo para el archivo de salida

    Returns:
        DataFrame con la columna 'name' agregada
    """
    print(f"\n{'='*60}")
    print(f"Procesando: {csv_path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    total = len(df)

    # Verificar si ya tiene columna name
    if 'name' not in df.columns:
        df['name'] = ''

    # Contar cuantos faltan
    missing_names = df[df['name'] == ''].index.tolist()
    print(f"Total compuestos: {total}")
    print(f"Con nombre: {total - len(missing_names)}")
    print(f"Sin nombre: {len(missing_names)}")

    if len(missing_names) == 0:
        print("Todos los compuestos ya tienen nombre!")
        return df

    print(f"\nIniciando consultas a PubChem (rate limit: {RATE_LIMIT_SECONDS}s)...")
    print(f"Tiempo estimado: {(len(missing_names) * RATE_LIMIT_SECONDS) / 60:.1f} minutos")
    print("-" * 40)

    new_names_count = 0
    cache_hits = 0

    for i, idx in enumerate(missing_names, 1):
        smiles = df.loc[idx, 'SMILES']
        compound_id = df.loc[idx, 'id']

        # Verificar cache primero
        if smiles in cache:
            name = cache[smiles]
            cache_hits += 1
        else:
            # Consultar PubChem
            name = get_compound_name_pubchem(smiles)
            cache[smiles] = name

            # Respetar rate limit solo para requests nuevos
            if i < len(missing_names):  # No esperar despues del ultimo
                time.sleep(RATE_LIMIT_SECONDS)

        df.loc[idx, 'name'] = name

        if name:
            new_names_count += 1
            status = "OK"
        else:
            status = "NOT FOUND"

        # Mostrar progreso
        progress = (i / len(missing_names)) * 100
        print(f"[{i:4d}/{len(missing_names)}] {progress:5.1f}% | ID {compound_id:5d} | {status:10s} | {name[:40] if name else '-'}...")

        # Checkpoint cada N compuestos
        if i % CHECKPOINT_EVERY == 0:
            output_path = csv_path.parent / f"{csv_path.stem}{output_suffix}.csv"
            df.to_csv(output_path, index=False)
            save_cache(cache)
            print(f"    >> Checkpoint guardado ({i} compuestos)")

    # Guardar final
    output_path = csv_path.parent / f"{csv_path.stem}{output_suffix}.csv"
    df.to_csv(output_path, index=False)
    save_cache(cache)

    print(f"\n{'='*40}")
    print(f"Resumen para {csv_path.name}:")
    print(f"  - Nombres encontrados: {new_names_count}")
    print(f"  - Cache hits: {cache_hits}")
    print(f"  - No encontrados: {len(missing_names) - new_names_count - cache_hits}")
    print(f"  - Guardado en: {output_path}")
    print(f"{'='*40}")

    return df

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     PubChem Compound Name Generator                       ║
    ║     Rate Limit: 5 segundos entre requests                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Cargar cache
    cache = load_cache()
    print(f"Cache cargado: {len(cache)} compuestos en memoria")

    # Procesar train.csv
    train_path = DATA_DIR / "train.csv"
    if train_path.exists():
        add_names_to_csv(train_path, cache)
    else:
        print(f"No se encontro: {train_path}")

    # Procesar test.csv
    test_path = DATA_DIR / "test.csv"
    if test_path.exists():
        add_names_to_csv(test_path, cache)
    else:
        print(f"No se encontro: {test_path}")

    print(f"\n\nProceso completado!")
    print(f"Cache total: {len(cache)} compuestos")
    print(f"Archivos generados:")
    print(f"  - {DATA_DIR / 'train_with_names.csv'}")
    print(f"  - {DATA_DIR / 'test_with_names.csv'}")

if __name__ == "__main__":
    main()
