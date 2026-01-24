"""
patch_chemprop_torch.py - Parche para ChemProp compatible con PyTorch 2.6+

PyTorch 2.6 cambió weights_only=True por defecto en torch.load(),
lo que rompe la carga de checkpoints de ChemProp.

Este script parcha los archivos de ChemProp para usar weights_only=False.

Ejecutar UNA VEZ después de instalar chemprop:
    python patch_chemprop_torch.py
"""

import os
import sys
import re

def find_chemprop_path():
    """Encuentra el directorio de instalación de chemprop."""
    for path in sys.path:
        chemprop_path = os.path.join(path, 'chemprop')
        if os.path.exists(chemprop_path) and os.path.isdir(chemprop_path):
            return chemprop_path
    
    # Intentar con pip show
    try:
        import subprocess
        result = subprocess.run(['pip', 'show', 'chemprop'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Location:'):
                    location = line.split(':', 1)[1].strip()
                    return os.path.join(location, 'chemprop')
    except:
        pass
    
    return None

def patch_file(filepath, replacements):
    """Aplica reemplazos a un archivo."""
    if not os.path.exists(filepath):
        return False, "Archivo no existe"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = 0
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes_made += 1
    
    if changes_made > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, f"{changes_made} cambios aplicados"
    elif any(new in original_content for _, new in replacements):
        return True, "Ya estaba parcheado"
    else:
        return False, "Patrón no encontrado"

def main():
    print("=" * 60)
    print("ChemProp PyTorch 2.6+ Compatibility Patch")
    print("=" * 60)
    print()
    
    chemprop_path = find_chemprop_path()
    
    if not chemprop_path:
        print("ERROR: No se encontró la instalación de chemprop")
        print("Asegúrate de que chemprop está instalado: pip install chemprop")
        return 1
    
    print(f"Encontrado chemprop en: {chemprop_path}")
    print()
    
    # Archivos a parchear y sus reemplazos
    patches = {
        # utils.py - función load_checkpoint y load_args
        'utils.py': [
            # Parche para load_checkpoint
            (
                'state = torch.load(path, map_location=map_location)',
                'state = torch.load(path, map_location=map_location, weights_only=False)'
            ),
            # Parche para load_args
            (
                'torch.load(path, map_location=lambda storage, loc: storage)',
                'torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)'
            ),
            # Versión alternativa que puede aparecer
            (
                "torch.load(path, map_location=lambda storage, loc: storage)['args']",
                "torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)['args']"
            ),
        ],
        # train/make_predictions.py - puede tener torch.load también
        'train/make_predictions.py': [
            (
                'torch.load(path, map_location=lambda storage, loc: storage)',
                'torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)'
            ),
        ],
    }
    
    success_count = 0
    fail_count = 0
    
    for rel_path, replacements in patches.items():
        filepath = os.path.join(chemprop_path, rel_path)
        print(f"Parcheando: {rel_path}")
        
        success, message = patch_file(filepath, replacements)
        
        if success:
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ⚠ {message}")
            fail_count += 1
    
    print()
    print("=" * 60)
    
    if success_count > 0:
        print(f"✓ Parche completado! ({success_count} archivos modificados)")
        print()
        print("Ahora reinicia el servidor:")
        print("  uvicorn app.main:app --reload")
        print()
        print("Y prueba crear un compuesto - debería usar tu modelo ChemProp.")
        return 0
    else:
        print("⚠ No se pudieron aplicar los parches")
        print()
        print("Alternativa: hacer downgrade de PyTorch:")
        print("  pip install torch==2.5.1")
        return 1

if __name__ == '__main__':
    sys.exit(main())