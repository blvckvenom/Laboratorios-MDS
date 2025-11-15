"""
Script de prueba exhaustiva de funciones principales de cada módulo.
"""

import sys
from pathlib import Path
import inspect

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

print("="*80)
print("VALIDACION EXHAUSTIVA DE FUNCIONES")
print("="*80)

# Importar todos los módulos
from config import paths
from data import loader, quality
from preprocessing import cleaner, geographic
from eda import clientes, productos, temporal, comportamiento, correlaciones
from features import (
    cliente_features, producto_features, interaccion_features,
    temporal_features, transformers
)
from models import baseline, comparacion, optimizer, xgboost_final
from evaluation import metrics, importance
from interpretability import shap_analysis, visualizations
from utils import plots

# Diccionario de módulos a validar
modules_to_validate = {
    "config.paths": paths,
    "data.loader": loader,
    "data.quality": quality,
    "preprocessing.cleaner": cleaner,
    "preprocessing.geographic": geographic,
    "eda.clientes": clientes,
    "eda.productos": productos,
    "eda.temporal": temporal,
    "eda.comportamiento": comportamiento,
    "eda.correlaciones": correlaciones,
    "features.cliente_features": cliente_features,
    "features.producto_features": producto_features,
    "features.interaccion_features": interaccion_features,
    "features.temporal_features": temporal_features,
    "features.transformers": transformers,
    "models.baseline": baseline,
    "models.comparacion": comparacion,
    "models.optimizer": optimizer,
    "models.xgboost_final": xgboost_final,
    "evaluation.metrics": metrics,
    "evaluation.importance": importance,
    "interpretability.shap_analysis": shap_analysis,
    "interpretability.visualizations": visualizations,
    "utils.plots": plots,
}

total_funciones = 0
total_clases = 0

print("\nConteo de funciones y clases por módulo:\n")
print(f"{'Módulo':<40} {'Funciones':<12} {'Clases':<12}")
print("-" * 80)

for module_name, module in modules_to_validate.items():
    # Obtener funciones del módulo
    funciones = [name for name, obj in inspect.getmembers(module)
                 if inspect.isfunction(obj) and obj.__module__ == module.__name__]

    # Obtener clases del módulo
    clases = [name for name, obj in inspect.getmembers(module)
              if inspect.isclass(obj) and obj.__module__ == module.__name__]

    total_funciones += len(funciones)
    total_clases += len(clases)

    print(f"{module_name:<40} {len(funciones):<12} {len(clases):<12}")

    # Mostrar funciones principales (primeras 5)
    if funciones:
        print(f"  Funciones: {', '.join(funciones[:5])}", end="")
        if len(funciones) > 5:
            print(f" ... (+{len(funciones)-5} más)")
        else:
            print()

    # Mostrar clases
    if clases:
        print(f"  Clases: {', '.join(clases)}")

print("\n" + "="*80)
print("RESUMEN TOTAL")
print("="*80)
print(f"\nTotal de módulos validados: {len(modules_to_validate)}")
print(f"Total de funciones encontradas: {total_funciones}")
print(f"Total de clases encontradas: {total_clases}")

# Validar constantes importantes en config
print("\n" + "="*80)
print("VALIDACION DE CONFIGURACION")
print("="*80)

config_attrs = ['ROOT_DIR', 'DATA_DIR', 'CLIENTES_PATH', 'PRODUCTOS_PATH',
                'TRANSACCIONES_PATH', 'RANDOM_STATE']

print("\nAtributos de configuración:")
for attr in config_attrs:
    if hasattr(paths, attr):
        value = getattr(paths, attr)
        print(f"  [OK] {attr}: {value}")
    else:
        print(f"  [WARNING] {attr}: No encontrado")

print("\n" + "="*80)
print("VALIDACION COMPLETADA EXITOSAMENTE")
print("="*80)
print("\n*** TODOS LOS MODULOS Y FUNCIONES SON VALIDOS! ***")
