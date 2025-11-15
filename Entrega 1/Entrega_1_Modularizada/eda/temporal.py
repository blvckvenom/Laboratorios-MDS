"""
Funciones para análisis temporal de transacciones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_rango_temporal(df_transacciones):
    """
    Analiza el rango temporal de las transacciones.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        dict: Estadísticas temporales
    """
    print("=" * 60)
    print("ANÁLISIS TEMPORAL")
    print("=" * 60)

    # Asegurar que purchase_date es datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    fecha_min = df_transacciones['purchase_date'].min()
    fecha_max = df_transacciones['purchase_date'].max()
    dias_totales = (fecha_max - fecha_min).days + 1

    print(f"\nPeriodo de datos: {fecha_min.date()} a {fecha_max.date()}")
    print(f"Días cubiertos: {dias_totales}")

    # Transacciones por día
    trans_por_dia = df_transacciones.groupby(df_transacciones['purchase_date'].dt.date).size()
    print(f"\nTransacciones diarias:")
    print(f"  Promedio: {trans_por_dia.mean():.1f}")
    print(f"  Mediana: {trans_por_dia.median():.1f}")
    print(f"  Mínimo: {trans_por_dia.min()}")
    print(f"  Máximo: {trans_por_dia.max()}")

    return {
        'fecha_min': fecha_min,
        'fecha_max': fecha_max,
        'dias_totales': dias_totales,
        'trans_por_dia_promedio': trans_por_dia.mean()
    }


def analizar_patron_semanal(df_transacciones):
    """
    Analiza el patrón de transacciones por día de la semana.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.Series: Transacciones por día de la semana
    """
    print("\n" + "=" * 60)
    print("PATRÓN SEMANAL")
    print("=" * 60)

    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    # Obtener día de la semana
    df_transacciones['day_of_week'] = df_transacciones['purchase_date'].dt.day_name()

    # Orden de días
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    trans_por_dia_semana = df_transacciones['day_of_week'].value_counts().reindex(days_order)

    print("\nTransacciones por día de la semana:")
    print(trans_por_dia_semana)

    # Identificar días pico y valle
    dia_pico = trans_por_dia_semana.idxmax()
    dia_valle = trans_por_dia_semana.idxmin()

    print(f"\nDía pico: {dia_pico} ({trans_por_dia_semana[dia_pico]:,} transacciones)")
    print(f"Día valle: {dia_valle} ({trans_por_dia_semana[dia_valle]:,} transacciones)")

    return trans_por_dia_semana


def analizar_patron_mensual(df_transacciones):
    """
    Analiza el patrón de transacciones por mes.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.Series: Transacciones por mes
    """
    print("\n" + "=" * 60)
    print("PATRÓN MENSUAL")
    print("=" * 60)

    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    # Obtener mes
    df_transacciones['month'] = df_transacciones['purchase_date'].dt.month
    df_transacciones['month_name'] = df_transacciones['purchase_date'].dt.month_name()

    trans_por_mes = df_transacciones.groupby(['month', 'month_name']).size().reset_index(name='count')
    trans_por_mes = trans_por_mes.sort_values('month')

    print("\nTransacciones por mes:")
    for _, row in trans_por_mes.iterrows():
        print(f"{row['month_name']:12s}: {row['count']:,}")

    # Identificar meses pico y valle
    mes_pico_idx = trans_por_mes['count'].idxmax()
    mes_valle_idx = trans_por_mes['count'].idxmin()

    mes_pico = trans_por_mes.loc[mes_pico_idx, 'month_name']
    mes_valle = trans_por_mes.loc[mes_valle_idx, 'month_name']

    print(f"\nMes pico: {mes_pico} ({trans_por_mes.loc[mes_pico_idx, 'count']:,} transacciones)")
    print(f"Mes valle: {mes_valle} ({trans_por_mes.loc[mes_valle_idx, 'count']:,} transacciones)")

    return trans_por_mes


def visualizar_serie_temporal_diaria(df_transacciones, figsize=(14, 6)):
    """
    Visualiza la serie temporal de transacciones diarias.

    Args:
        df_transacciones: DataFrame de transacciones
        figsize: Tamaño de la figura
    """
    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    trans_diarias = df_transacciones.groupby(df_transacciones['purchase_date'].dt.date).size()

    plt.figure(figsize=figsize)
    plt.plot(trans_diarias.index, trans_diarias.values, linewidth=1, alpha=0.7)
    plt.title('Serie Temporal de Transacciones Diarias', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Número de Transacciones', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualizar_patron_semanal(df_transacciones, figsize=(12, 6)):
    """
    Visualiza el patrón de transacciones por día de la semana.

    Args:
        df_transacciones: DataFrame de transacciones
        figsize: Tamaño de la figura
    """
    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    df_transacciones['day_of_week'] = df_transacciones['purchase_date'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    trans_por_dia = df_transacciones['day_of_week'].value_counts().reindex(days_order)

    plt.figure(figsize=figsize)
    colors = ['red' if d in ['Monday', 'Thursday'] else 'steelblue' for d in days_order]
    plt.bar(trans_por_dia.index, trans_por_dia.values, color=colors, edgecolor='black')
    plt.title('Patrón Semanal de Transacciones', fontsize=14, fontweight='bold')
    plt.xlabel('Día de la Semana', fontsize=12)
    plt.ylabel('Número de Transacciones', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualizar_patron_mensual(df_transacciones, figsize=(12, 6)):
    """
    Visualiza el patrón de transacciones por mes.

    Args:
        df_transacciones: DataFrame de transacciones
        figsize: Tamaño de la figura
    """
    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    df_transacciones['month'] = df_transacciones['purchase_date'].dt.month
    trans_por_mes = df_transacciones.groupby('month').size()

    months_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    plt.figure(figsize=figsize)
    plt.bar(range(1, len(trans_por_mes) + 1), trans_por_mes.values,
            color='lightcoral', edgecolor='black')
    plt.title('Patrón Mensual de Transacciones', fontsize=14, fontweight='bold')
    plt.xlabel('Mes', fontsize=12)
    plt.ylabel('Número de Transacciones', fontsize=12)
    plt.xticks(range(1, len(trans_por_mes) + 1),
               [months_names[i-1] for i in trans_por_mes.index])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def analisis_temporal_completo(df_transacciones, visualizar=True):
    """
    Realiza un análisis temporal completo.

    Args:
        df_transacciones: DataFrame de transacciones
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Resumen completo del análisis temporal
    """
    resumen = {}

    # Análisis de rango
    resumen['rango'] = analizar_rango_temporal(df_transacciones)

    # Patrón semanal
    resumen['patron_semanal'] = analizar_patron_semanal(df_transacciones)

    # Patrón mensual
    resumen['patron_mensual'] = analizar_patron_mensual(df_transacciones)

    # Visualizaciones
    if visualizar:
        visualizar_serie_temporal_diaria(df_transacciones)
        visualizar_patron_semanal(df_transacciones)
        visualizar_patron_mensual(df_transacciones)

    return resumen
