"""
Funciones para crear features basados en productos.
"""

import pandas as pd
import numpy as np


def crear_features_producto(df_transacciones_agg):
    """
    Crea features agregados de producto.

    Args:
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        pd.DataFrame: Features de producto
    """
    print("[Creando features de producto...]")

    producto_stats = df_transacciones_agg.groupby('product_id').agg({
        'fecha_dt': 'count',
        'customer_id': 'nunique',
        'items': 'sum'
    }).reset_index()

    producto_stats.columns = [
        'product_id', 'total_ventas_global', 'clientes_unicos_global', 'items_vendidos_global'
    ]

    # Ranking de popularidad
    producto_stats['popularidad_rank'] = producto_stats['items_vendidos_global'].rank(
        ascending=False, method='dense'
    ).astype(int)

    print(f"  Features de producto creados: {len(producto_stats.columns) - 1}")

    return producto_stats


def agregar_features_producto(df, producto_stats):
    """
    Agrega features de producto al DataFrame.

    Args:
        df: DataFrame al que agregar features
        producto_stats: DataFrame con estadísticas de producto

    Returns:
        pd.DataFrame: DataFrame con features agregados
    """
    df_fe = df.merge(producto_stats, on='product_id', how='left')

    return df_fe


def aplicar_features_producto(df_train_fe, df_val_fe, df_test_fe, df_transacciones_agg):
    """
    Aplica features de producto a los datasets.

    Args:
        df_train_fe: DataFrame de entrenamiento con features
        df_val_fe: DataFrame de validación con features
        df_test_fe: DataFrame de test con features
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        tuple: (df_train_fe, df_val_fe, df_test_fe)
    """
    print("\n[2/6] Creando features agregados de producto...")

    # Crear features de producto
    producto_stats = crear_features_producto(df_transacciones_agg)

    # Agregar a cada dataset
    df_train_fe = agregar_features_producto(df_train_fe, producto_stats)
    df_val_fe = agregar_features_producto(df_val_fe, producto_stats)
    df_test_fe = agregar_features_producto(df_test_fe, producto_stats)

    return df_train_fe, df_val_fe, df_test_fe
