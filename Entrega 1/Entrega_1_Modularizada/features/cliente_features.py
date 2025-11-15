"""
Funciones para crear features basados en clientes (RFM - Recency, Frequency, Monetary).
"""

import pandas as pd
import numpy as np


def crear_features_cliente_rfm(df_transacciones_agg):
    """
    Crea features agregados de cliente basados en RFM.

    Args:
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        pd.DataFrame: Features de cliente
    """
    print("[Creando features RFM de cliente...]")

    cliente_stats = df_transacciones_agg.groupby('customer_id').agg({
        'fecha_dt': ['min', 'max', 'count'],
        'product_id': 'nunique',
        'items': ['sum', 'mean']
    }).reset_index()

    cliente_stats.columns = [
        'customer_id', 'primera_compra_global', 'ultima_compra_global',
        'total_ordenes_global', 'productos_unicos_global',
        'items_totales_global', 'items_promedio_global'
    ]

    print(f"  Features de cliente creados: {len(cliente_stats.columns) - 1}")

    return cliente_stats


def agregar_features_cliente_temporales(df, cliente_stats, fecha_referencia_col='fecha_referencia'):
    """
    Agrega features de cliente con cálculos temporales.

    Args:
        df: DataFrame al que agregar features
        cliente_stats: DataFrame con estadísticas de cliente
        fecha_referencia_col: Nombre de la columna de fecha de referencia

    Returns:
        pd.DataFrame: DataFrame con features agregados
    """
    # Merge
    df_fe = df.merge(cliente_stats, on='customer_id', how='left')

    # Calcular días desde primera y última compra
    df_fe['dias_desde_primera_compra'] = (
        df_fe[fecha_referencia_col] - df_fe['primera_compra_global']
    ).dt.days.fillna(0).clip(lower=0)

    df_fe['dias_desde_ultima_compra'] = (
        df_fe[fecha_referencia_col] - df_fe['ultima_compra_global']
    ).dt.days.fillna(999).clip(lower=0)

    # Frecuencia de compra diaria
    df_fe['frecuencia_compra_diaria'] = df_fe['total_ordenes_global'] / (df_fe['dias_desde_primera_compra'] + 1)

    # Diversidad de productos
    df_fe['diversidad_productos'] = df_fe['productos_unicos_global'] / (df_fe['total_ordenes_global'] + 1)

    print(f"  Features cliente temporales agregados: 4")

    return df_fe


def aplicar_features_cliente(df_train, df_val, df_test, df_transacciones_agg):
    """
    Aplica features de cliente a los datasets de train, validation y test.

    Args:
        df_train: DataFrame de entrenamiento
        df_val: DataFrame de validación
        df_test: DataFrame de test
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        tuple: (df_train_fe, df_val_fe, df_test_fe)
    """
    print("\n[1/6] Creando features agregados de cliente (RFM)...")

    # Crear features de cliente
    cliente_stats = crear_features_cliente_rfm(df_transacciones_agg)

    # Agregar a cada dataset
    df_train_fe = agregar_features_cliente_temporales(df_train, cliente_stats)
    df_val_fe = agregar_features_cliente_temporales(df_val, cliente_stats)
    df_test_fe = agregar_features_cliente_temporales(df_test, cliente_stats)

    return df_train_fe, df_val_fe, df_test_fe
