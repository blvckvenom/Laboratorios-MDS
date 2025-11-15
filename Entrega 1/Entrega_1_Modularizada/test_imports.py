"""
Script de prueba para validar todos los imports del proyecto.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

print("="*80)
print("VALIDACIÓN DE IMPORTS DEL PROYECTO")
print("="*80)

errors = []
success_count = 0

# Lista de módulos a probar
modules_to_test = [
    # Config
    ("config", ["paths"]),

    # Data
    ("data", ["loader", "quality"]),

    # Preprocessing
    ("preprocessing", ["cleaner", "geographic"]),

    # EDA
    ("eda", ["clientes", "productos", "temporal", "comportamiento", "correlaciones"]),

    # Features
    ("features", ["cliente_features", "producto_features", "interaccion_features",
                  "temporal_features", "transformers"]),

    # Models
    ("models", ["baseline", "comparacion", "optimizer", "xgboost_final"]),

    # Evaluation
    ("evaluation", ["metrics", "importance"]),

    # Interpretability
    ("interpretability", ["shap_analysis", "visualizations"]),

    # Utils
    ("utils", ["plots"]),
]

# Probar cada módulo
for package, modules in modules_to_test:
    print(f"\n[*] Probando package: {package}")

    for module in modules:
        try:
            full_module_name = f"{package}.{module}"
            exec(f"import {full_module_name}")
            print(f"  [OK] {full_module_name}")
            success_count += 1
        except Exception as e:
            error_msg = f"  [ERROR] {full_module_name}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

# Resumen
print("\n" + "="*80)
print("RESUMEN DE VALIDACION")
print("="*80)
print(f"\n[OK] Modulos exitosos: {success_count}")
print(f"[ERROR] Modulos con errores: {len(errors)}")

if errors:
    print("\nErrores encontrados:")
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print("\n*** TODOS LOS IMPORTS SON VALIDOS! ***")
    sys.exit(0)
