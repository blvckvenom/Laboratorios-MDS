"""
Funciones para crear features temporales.
"""

import pandas as pd
import numpy as np


def crear_features_temporales(df, fecha_col='fecha_referencia'):
    """
    Crea features temporales basados en la fecha de referencia.

    Args:
        df: DataFrame al que agregar features
        fecha_col: Nombre de la columna de fecha

    Returns:
        pd.DataFrame: DataFrame con features temporales
    """
    df_fe = df.copy()

    # Features básicos temporales
    df_fe['dia_semana'] = df_fe[fecha_col].dt.dayofweek
    df_fe['mes'] = df_fe[fecha_col].dt.month
    df_fe['trimestre'] = df_fe[fecha_col].dt.quarter
    df_fe['semana_del_año'] = df_fe[fecha_col].dt.isocalendar().week

    # Indicadores binarios
    df_fe['es_fin_semana'] = (df_fe['dia_semana'] >= 5).astype(int)
    df_fe['es_lunes_jueves'] = df_fe['dia_semana'].isin([0, 3]).astype(int)
    df_fe['es_temporada_alta'] = df_fe['mes'].isin([11, 12]).astype(int)
    df_fe['es_temporada_baja'] = df_fe['mes'].isin([5, 6, 7]).astype(int)

    # Encoding cíclico
    df_fe['mes_sin'] = np.sin(2 * np.pi * df_fe['mes'] / 12)
    df_fe['mes_cos'] = np.cos(2 * np.pi * df_fe['mes'] / 12)
    df_fe['dia_semana_sin'] = np.sin(2 * np.pi * df_fe['dia_semana'] / 7)
    df_fe['dia_semana_cos'] = np.cos(2 * np.pi * df_fe['dia_semana'] / 7)

    return df_fe


def aplicar_features_temporales(df_train_fe, df_val_fe, df_test_fe):
    """
    Aplica features temporales a los datasets.

    Args:
        df_train_fe: DataFrame de entrenamiento con features
        df_val_fe: DataFrame de validación con features
        df_test_fe: DataFrame de test con features

    Returns:
        tuple: (df_train_fe, df_val_fe, df_test_fe)
    """
    print("\n[4/6] Creando features temporales...")

    df_train_fe = crear_features_temporales(df_train_fe)
    df_val_fe = crear_features_temporales(df_val_fe)
    df_test_fe = crear_features_temporales(df_test_fe)

    print(f"  Features temporales creados: 14")

    return df_train_fe, df_val_fe, df_test_fe
