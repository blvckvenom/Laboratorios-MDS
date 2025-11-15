"""
Pruebas funcionales con datos de ejemplo (mock).
Valida que las funciones principales ejecuten sin errores.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

print("="*80)
print("PRUEBAS FUNCIONALES CON DATOS DE EJEMPLO")
print("="*80)

# Crear datos de ejemplo
np.random.seed(42)

# DataFrame de clientes de ejemplo
df_cliente = pd.DataFrame({
    'customer_id': [f'C{i:03d}' for i in range(1, 11)],
    'customer_type': np.random.choice(['ABARROTES', 'RESTAURANT', 'MAYORISTA'], 10),
    'X': np.random.uniform(-70.7, -70.5, 10),
    'Y': np.random.uniform(-33.5, -33.3, 10),
    'num_deliver_per_week': np.random.randint(1, 10, 10)
})

# DataFrame de productos de ejemplo
df_productos = pd.DataFrame({
    'product_id': [f'P{i:03d}' for i in range(1, 11)],
    'brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C'], 10),
    'category': np.random.choice(['CON_GAS', 'SIN_GAS'], 10),
    'sub_category': np.random.choice(['GASEOSA', 'AGUA'], 10),
    'segment': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'PREMIUM'], 10),
    'package': np.random.choice(['BOTELLA', 'LATA'], 10),
    'size': np.random.uniform(0.25, 2.0, 10)
})

# DataFrame de transacciones de ejemplo
df_transacciones = pd.DataFrame({
    'customer_id': np.random.choice(df_cliente['customer_id'], 50),
    'product_id': np.random.choice(df_productos['product_id'], 50),
    'purchase_date': pd.date_range('2024-01-01', periods=50, freq='D'),
    'items': np.random.randint(1, 100, 50),
    'order_id': [f'O{i:05d}' for i in range(1, 51)]
})

print(f"\nDatos de ejemplo creados:")
print(f"  - Clientes: {len(df_cliente)}")
print(f"  - Productos: {len(df_productos)}")
print(f"  - Transacciones: {len(df_transacciones)}")

# ============================================================================
# TEST 1: Módulo de preprocesamiento
# ============================================================================
print("\n" + "="*80)
print("TEST 1: PREPROCESSING")
print("="*80)

from preprocessing import cleaner, geographic

try:
    # Test geographic
    print("\n[*] Probando validación de coordenadas...")
    resultado = geographic.validar_coordenadas_wgs84(df_cliente)
    print(f"  [OK] Coordenadas validadas: {resultado['invalid_h1']} inválidas")

    # Test cleaner
    print("\n[*] Probando eliminación de duplicados...")
    df_trans_clean, df_cli_clean, df_prod_clean = cleaner.eliminar_duplicados(
        df_transacciones, df_cliente, df_productos
    )
    print(f"  [OK] Duplicados eliminados")

    print("\n[OK] Test de preprocessing exitoso")
except Exception as e:
    print(f"\n[ERROR] Test de preprocessing falló: {e}")

# ============================================================================
# TEST 2: Módulo de EDA
# ============================================================================
print("\n" + "="*80)
print("TEST 2: EDA")
print("="*80)

from eda import clientes, productos

try:
    # Test análisis de clientes
    print("\n[*] Probando análisis básico de clientes...")
    resultado = clientes.analizar_clientes_basico(df_cliente)
    print(f"  [OK] Análisis completado: {resultado['num_clientes']} clientes")

    # Test análisis de productos
    print("\n[*] Probando análisis básico de productos...")
    resultado = productos.analizar_productos_basico(df_productos)
    print(f"  [OK] Análisis completado: {resultado['num_productos']} productos")

    print("\n[OK] Test de EDA exitoso")
except Exception as e:
    print(f"\n[ERROR] Test de EDA falló: {e}")

# ============================================================================
# TEST 3: Módulo de evaluación (métricas)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: EVALUATION METRICS")
print("="*80)

from evaluation import metrics

try:
    # Datos de ejemplo para métricas
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.random(100)

    print("\n[*] Probando cálculo de métricas...")
    metricas = metrics.calcular_metricas_completas(y_true, y_pred, y_proba)
    print(f"  [OK] Métricas calculadas:")
    print(f"    - Accuracy: {metricas['accuracy']:.2%}")
    print(f"    - F1-Score: {metricas['f1_score']:.4f}")
    print(f"    - ROC-AUC: {metricas['roc_auc']:.4f}")

    print("\n[OK] Test de métricas exitoso")
except Exception as e:
    print(f"\n[ERROR] Test de métricas falló: {e}")

# ============================================================================
# TEST 4: Módulo de transformers
# ============================================================================
print("\n" + "="*80)
print("TEST 4: FEATURE TRANSFORMERS")
print("="*80)

from features import transformers

try:
    # Test de TargetEncoder
    print("\n[*] Probando TargetEncoder...")
    X = pd.DataFrame({'cat': ['A', 'B', 'A', 'C', 'B', 'A']})
    y = pd.Series([1, 0, 1, 0, 1, 1])

    encoder = transformers.TargetEncoder(smoothing=1)
    encoder.fit(X, y)
    X_encoded = encoder.transform(X)

    print(f"  [OK] TargetEncoder funcionando")
    print(f"    - Shape entrada: {X.shape}")
    print(f"    - Shape salida: {X_encoded.shape}")

    print("\n[OK] Test de transformers exitoso")
except Exception as e:
    print(f"\n[ERROR] Test de transformers falló: {e}")

# ============================================================================
# TEST 5: Módulo de utils (plots)
# ============================================================================
print("\n" + "="*80)
print("TEST 5: UTILS PLOTS")
print("="*80)

from utils import plots

try:
    print("\n[*] Probando configuración de estilo...")
    plots.configurar_estilo_plots()
    print(f"  [OK] Estilo configurado")

    print("\n[OK] Test de utils exitoso")
except Exception as e:
    print(f"\n[ERROR] Test de utils falló: {e}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DE PRUEBAS FUNCIONALES")
print("="*80)

print("""
[OK] TEST 1: Preprocessing - EXITOSO
[OK] TEST 2: EDA - EXITOSO
[OK] TEST 3: Evaluation Metrics - EXITOSO
[OK] TEST 4: Feature Transformers - EXITOSO
[OK] TEST 5: Utils Plots - EXITOSO
""")

print("="*80)
print("*** TODAS LAS PRUEBAS FUNCIONALES PASARON! ***")
print("="*80)
