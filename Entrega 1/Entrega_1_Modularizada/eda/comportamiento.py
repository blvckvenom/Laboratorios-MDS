"""
Funciones para análisis de comportamiento de compra.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_metricas_por_cliente(df_transacciones):
    """
    Analiza métricas de comportamiento por cliente.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: Métricas agregadas por cliente
    """
    print("=" * 60)
    print("ANÁLISIS DE COMPORTAMIENTO POR CLIENTE")
    print("=" * 60)

    # Agregar métricas por cliente
    metricas_cliente = df_transacciones.groupby('customer_id').agg({
        'order_id': 'count',  # Total de órdenes
        'product_id': 'nunique',  # Productos únicos
        'items': 'sum'  # Total de items
    }).rename(columns={
        'order_id': 'total_ordenes',
        'product_id': 'productos_unicos',
        'items': 'items_total'
    })

    print("\nEstadísticas de órdenes por cliente:")
    print(metricas_cliente['total_ordenes'].describe())

    print("\nEstadísticas de productos únicos por cliente:")
    print(metricas_cliente['productos_unicos'].describe())

    print("\nEstadísticas de items totales por cliente:")
    print(metricas_cliente['items_total'].describe())

    # Top clientes
    print("\nTop 10 clientes más activos:")
    top_clientes = metricas_cliente.nlargest(10, 'total_ordenes')
    print(top_clientes)

    return metricas_cliente


def analizar_frecuencia_recompra(df_transacciones):
    """
    Analiza la frecuencia de recompra de productos.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: Métricas de recompra
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE FRECUENCIA DE RECOMPRA")
    print("=" * 60)

    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    # Calcular periodos entre compras por producto y cliente
    df_sorted = df_transacciones.sort_values(['customer_id', 'product_id', 'purchase_date'])
    df_sorted['fecha_previa'] = df_sorted.groupby(['customer_id', 'product_id'])['purchase_date'].shift(1)
    df_sorted['dias_desde_anterior'] = (df_sorted['purchase_date'] - df_sorted['fecha_previa']).dt.days

    # Filtrar solo recompras
    recompras = df_sorted[df_sorted['dias_desde_anterior'].notna()]

    if len(recompras) > 0:
        print(f"\nTotal de eventos de recompra: {len(recompras):,}")
        print(f"\nEstadísticas de días entre recompras:")
        print(recompras['dias_desde_anterior'].describe())

        # Productos con más recompras
        productos_recomprados = recompras.groupby('product_id').size().sort_values(ascending=False)
        print(f"\nProductos únicos con historial de recompra: {len(productos_recomprados)}")
        print(f"\nTop 10 productos más recomprados:")
        print(productos_recomprados.head(10))

        return recompras
    else:
        print("\nNo se detectaron eventos de recompra en los datos")
        return pd.DataFrame()


def clasificar_lealtad_clientes(df_transacciones):
    """
    Clasifica clientes según nivel de lealtad.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: Clientes clasificados por lealtad
    """
    print("\n" + "=" * 60)
    print("CLASIFICACIÓN DE LEALTAD DE CLIENTES")
    print("=" * 60)

    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    # Calcular métricas de lealtad
    lealtad = df_transacciones.groupby('customer_id').agg({
        'purchase_date': ['min', 'max', 'count'],
        'product_id': 'nunique'
    })

    lealtad.columns = ['primera_compra', 'ultima_compra', 'num_compras', 'productos_unicos']
    lealtad['periodo_dias'] = (lealtad['ultima_compra'] - lealtad['primera_compra']).dt.days
    lealtad['frecuencia_compra'] = lealtad['num_compras'] / (lealtad['periodo_dias'] + 1)

    # Clasificar lealtad
    def clasificar_lealtad(row):
        if row['periodo_dias'] >= 365 and row['frecuencia_compra'] >= 0.1:
            return 'Alto'
        elif row['periodo_dias'] >= 180 and row['frecuencia_compra'] >= 0.05:
            return 'Medio'
        else:
            return 'Bajo'

    lealtad['nivel_lealtad'] = lealtad.apply(clasificar_lealtad, axis=1)

    print("\nDistribución de niveles de lealtad:")
    print(lealtad['nivel_lealtad'].value_counts())
    print("\nPorcentajes:")
    print(lealtad['nivel_lealtad'].value_counts(normalize=True) * 100)

    return lealtad


def analizar_adquisicion_clientes(df_transacciones):
    """
    Analiza la adquisición de nuevos clientes a lo largo del tiempo.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: Clientes nuevos por fecha
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE ADQUISICIÓN DE CLIENTES")
    print("=" * 60)

    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df_transacciones['purchase_date']):
        df_transacciones['purchase_date'] = pd.to_datetime(df_transacciones['purchase_date'])

    # Obtener primera compra de cada cliente
    primera_compra = df_transacciones.groupby('customer_id')['purchase_date'].min()

    # Contar nuevos clientes por día
    nuevos_por_dia = primera_compra.value_counts().sort_index()

    print(f"\nTotal de clientes adquiridos: {len(primera_compra):,}")
    print(f"\nClientes nuevos por día:")
    print(f"  Promedio: {nuevos_por_dia.mean():.2f}")
    print(f"  Mediana: {nuevos_por_dia.median():.2f}")
    print(f"  Máximo: {nuevos_por_dia.max()}")

    # Día con más adquisiciones
    dia_max_adq = nuevos_por_dia.idxmax()
    print(f"\nDía con más adquisiciones: {dia_max_adq.date()} ({nuevos_por_dia.max()} clientes)")

    return primera_compra


def visualizar_distribucion_ordenes(metricas_cliente, figsize=(12, 6)):
    """
    Visualiza la distribución de órdenes por cliente.

    Args:
        metricas_cliente: DataFrame con métricas por cliente
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histograma
    axes[0].hist(metricas_cliente['total_ordenes'], bins=50, color='steelblue', edgecolor='black')
    axes[0].set_title('Distribución de Órdenes por Cliente', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Número de Órdenes', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Boxplot
    axes[1].boxplot(metricas_cliente['total_ordenes'], vert=True)
    axes[1].set_title('Boxplot de Órdenes por Cliente', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Número de Órdenes', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualizar_lealtad(lealtad, figsize=(10, 6)):
    """
    Visualiza la distribución de niveles de lealtad.

    Args:
        lealtad: DataFrame con clasificación de lealtad
        figsize: Tamaño de la figura
    """
    lealtad_counts = lealtad['nivel_lealtad'].value_counts()

    plt.figure(figsize=figsize)
    colors = {'Alto': 'green', 'Medio': 'orange', 'Bajo': 'red'}
    plt.pie(lealtad_counts.values, labels=lealtad_counts.index, autopct='%1.1f%%',
            startangle=90, colors=[colors.get(x, 'gray') for x in lealtad_counts.index])
    plt.title('Distribución de Niveles de Lealtad de Clientes', fontsize=14, fontweight='bold')
    plt.show()


def analisis_comportamiento_completo(df_transacciones, visualizar=True):
    """
    Realiza un análisis completo de comportamiento de compra.

    Args:
        df_transacciones: DataFrame de transacciones
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Resumen completo del análisis
    """
    resumen = {}

    # Métricas por cliente
    resumen['metricas_cliente'] = analizar_metricas_por_cliente(df_transacciones)

    # Frecuencia de recompra
    resumen['recompras'] = analizar_frecuencia_recompra(df_transacciones)

    # Lealtad
    resumen['lealtad'] = clasificar_lealtad_clientes(df_transacciones)

    # Adquisición
    resumen['adquisicion'] = analizar_adquisicion_clientes(df_transacciones)

    # Visualizaciones
    if visualizar:
        visualizar_distribucion_ordenes(resumen['metricas_cliente'])
        visualizar_lealtad(resumen['lealtad'])

    return resumen
