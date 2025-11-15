"""
Configuración de rutas de archivos del proyecto.
"""

import os
from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent.parent

# Rutas de datos originales
DATA_DIR = ROOT_DIR
CLIENTES_PATH = DATA_DIR / 'clientes.parquet'
PRODUCTOS_PATH = DATA_DIR / 'productos.parquet'
TRANSACCIONES_PATH = DATA_DIR / 'transacciones.parquet'

# Rutas de datos procesados
DATA_TRAIN_PATH = DATA_DIR / 'data_train.parquet'
DATA_VAL_PATH = DATA_DIR / 'data_val.parquet'
DATA_TEST_PATH = DATA_DIR / 'data_test.parquet'

# Configuración de partición temporal
FECHA_LIMITE_TRAIN = '2024-10-31'
FECHA_LIMITE_VAL = '2024-11-30'

# Seed para reproducibilidad
RANDOM_STATE = 42
