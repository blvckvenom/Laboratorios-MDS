"""
Funciones para cargar los datasets del proyecto.
"""

import pandas as pd
from config.paths import CLIENTES_PATH, PRODUCTOS_PATH, TRANSACCIONES_PATH


def cargar_clientes():
    """
    Carga el dataset de clientes desde archivo parquet.

    Returns:
        pd.DataFrame: DataFrame con información de clientes
    """
    df_cliente = pd.read_parquet(CLIENTES_PATH)
    return df_cliente


def cargar_productos():
    """
    Carga el dataset de productos desde archivo parquet.

    Returns:
        pd.DataFrame: DataFrame con catálogo de productos
    """
    df_productos = pd.read_parquet(PRODUCTOS_PATH)
    return df_productos


def cargar_transacciones():
    """
    Carga el dataset de transacciones desde archivo parquet.

    Returns:
        pd.DataFrame: DataFrame con historial de transacciones
    """
    df_transacciones = pd.read_parquet(TRANSACCIONES_PATH)
    return df_transacciones


def cargar_todos_los_datos():
    """
    Carga los tres datasets principales del proyecto.

    Returns:
        tuple: (df_cliente, df_productos, df_transacciones)
    """
    print("Cargando datos...")
    df_cliente = cargar_clientes()
    print(f"✓ Clientes cargados: {df_cliente.shape}")

    df_productos = cargar_productos()
    print(f"✓ Productos cargados: {df_productos.shape}")

    df_transacciones = cargar_transacciones()
    print(f"✓ Transacciones cargadas: {df_transacciones.shape}")

    return df_cliente, df_productos, df_transacciones


def explorar_dataset(df, nombre="Dataset"):
    """
    Muestra información básica de un dataset.

    Args:
        df (pd.DataFrame): DataFrame a explorar
        nombre (str): Nombre del dataset para visualización
    """
    pd.set_option('display.max_columns', None)

    print(f"\n{'='*60}")
    print(f"EXPLORACIÓN: {nombre}")
    print(f"{'='*60}")

    print(f"\nPrimeras filas de {nombre}:")
    print(df.head())

    print(f"\nTipos de datos de {nombre}:")
    print(df.dtypes)

    print(f"\nResumen estadístico de {nombre}:")
    print(df.describe())

    print(f"\nValores nulos en {nombre}:")
    print(df.isnull().sum())

    print(f"\nDimensiones de {nombre}: {df.shape}")
