"""
Funciones para crear features de interacción cliente-producto.
"""

import pandas as pd
import numpy as np


def crear_features_interaccion(df_transacciones_agg):
    """
    Crea features de interacción cliente-producto.

    Args:
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        pd.DataFrame: Features de interacción
    """
    print("[Creando features de interacción cliente-producto...]")

    interaccion_stats = df_transacciones_agg.groupby(['customer_id', 'product_id']).agg({
        'fecha_dt': ['count', 'max'],
        'items': 'mean'
    }).reset_index()

    interaccion_stats.columns = [
        'customer_id', 'product_id', 'veces_comprado_global',
        'ultima_compra_producto_global', 'items_promedio_producto'
    ]

    print(f"  Features de interacción creados: {len(interaccion_stats.columns) - 2}")

    return interaccion_stats


def agregar_features_interaccion_temporales(df, interaccion_stats, fecha_referencia_col='fecha_referencia'):
    """
    Agrega features de interacción con cálculos temporales.

    Args:
        df: DataFrame al que agregar features
        interaccion_stats: DataFrame con estadísticas de interacción
        fecha_referencia_col: Nombre de la columna de fecha de referencia

    Returns:
        pd.DataFrame: DataFrame con features agregados
    """
    # Merge
    df_fe = df.merge(interaccion_stats, on=['customer_id', 'product_id'], how='left')

    # Calcular features derivados
    df_fe['compro_este_producto_antes'] = (df_fe['veces_comprado_global'] > 0).astype(int)

    df_fe['dias_desde_ultima_compra_producto'] = (
        df_fe[fecha_referencia_col] - df_fe['ultima_compra_producto_global']
    ).dt.days.fillna(999).clip(lower=0)

    # Rellenar valores faltantes
    df_fe['veces_comprado_global'] = df_fe['veces_comprado_global'].fillna(0)
    df_fe['items_promedio_producto'] = df_fe['items_promedio_producto'].fillna(0)

    print(f"  Features de interacción agregados: 4")

    return df_fe


def aplicar_features_interaccion(df_train_fe, df_val_fe, df_test_fe, df_transacciones_agg):
    """
    Aplica features de interacción a los datasets.

    Args:
        df_train_fe: DataFrame de entrenamiento con features
        df_val_fe: DataFrame de validación con features
        df_test_fe: DataFrame de test con features
        df_transacciones_agg: DataFrame de transacciones agregadas

    Returns:
        tuple: (df_train_fe, df_val_fe, df_test_fe)
    """
    print("\n[3/6] Creando features de interacción cliente-producto...")

    # Crear features de interacción
    interaccion_stats = crear_features_interaccion(df_transacciones_agg)

    # Agregar a cada dataset
    df_train_fe = agregar_features_interaccion_temporales(df_train_fe, interaccion_stats)
    df_val_fe = agregar_features_interaccion_temporales(df_val_fe, interaccion_stats)
    df_test_fe = agregar_features_interaccion_temporales(df_test_fe, interaccion_stats)

    return df_train_fe, df_val_fe, df_test_fe
