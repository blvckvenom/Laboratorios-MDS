"""
Funciones para análisis de correlaciones y relaciones entre variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_correlaciones_clientes(df_cliente):
    """
    Analiza correlaciones entre variables numéricas de clientes.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        pd.DataFrame: Matriz de correlaciones
    """
    print("=" * 60)
    print("ANÁLISIS DE CORRELACIONES - CLIENTES")
    print("=" * 60)

    # Seleccionar variables numéricas
    vars_numericas = df_cliente.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nVariables numéricas disponibles: {vars_numericas}")

    # Calcular correlaciones
    correlaciones = df_cliente[vars_numericas].corr()

    print("\nMatriz de correlaciones:")
    print(correlaciones)

    return correlaciones


def analizar_relacion_cliente_comportamiento(df_transacciones, df_cliente):
    """
    Analiza la relación entre tipo de cliente y comportamiento de compra.

    Args:
        df_transacciones: DataFrame de transacciones
        df_cliente: DataFrame de clientes

    Returns:
        pd.DataFrame: Métricas por tipo de cliente
    """
    print("\n" + "=" * 60)
    print("RELACIÓN CLIENTE-COMPORTAMIENTO")
    print("=" * 60)

    # Unir datos
    df_merged = df_transacciones.merge(df_cliente[['customer_id', 'customer_type']], on='customer_id', how='left')

    # Agregar por tipo de cliente
    comportamiento_por_tipo = df_merged.groupby('customer_type').agg({
        'order_id': 'count',
        'product_id': 'nunique',
        'items': 'sum'
    }).rename(columns={
        'order_id': 'total_ordenes',
        'product_id': 'productos_unicos',
        'items': 'total_items'
    })

    # Calcular promedios
    clientes_por_tipo = df_cliente['customer_type'].value_counts()
    comportamiento_por_tipo['ordenes_promedio'] = comportamiento_por_tipo['total_ordenes'] / clientes_por_tipo
    comportamiento_por_tipo['items_promedio'] = comportamiento_por_tipo['total_items'] / clientes_por_tipo

    print("\nComportamiento por tipo de cliente:")
    print(comportamiento_por_tipo)

    return comportamiento_por_tipo


def analizar_preferencias_por_segmento(df_transacciones, df_productos, df_cliente):
    """
    Analiza las preferencias de segmento de producto según tipo de cliente.

    Args:
        df_transacciones: DataFrame de transacciones
        df_productos: DataFrame de productos
        df_cliente: DataFrame de clientes

    Returns:
        pd.DataFrame: Tabla cruzada de preferencias
    """
    print("\n" + "=" * 60)
    print("PREFERENCIAS POR SEGMENTO DE PRODUCTO")
    print("=" * 60)

    # Unir datos
    df_merged = df_transacciones.merge(
        df_cliente[['customer_id', 'customer_type']], on='customer_id', how='left'
    ).merge(
        df_productos[['product_id', 'segment']], on='product_id', how='left'
    )

    # Tabla cruzada
    preferencias = pd.crosstab(
        df_merged['customer_type'],
        df_merged['segment'],
        df_merged['items'],
        aggfunc='sum',
        margins=True
    )

    print("\nItems vendidos por tipo de cliente y segmento:")
    print(preferencias)

    return preferencias


def analizar_productos_mas_vendidos(df_transacciones, df_productos, top_n=20):
    """
    Analiza los productos más vendidos.

    Args:
        df_transacciones: DataFrame de transacciones
        df_productos: DataFrame de productos
        top_n: Número de productos top a mostrar

    Returns:
        pd.DataFrame: Top productos
    """
    print("\n" + "=" * 60)
    print(f"TOP {top_n} PRODUCTOS MÁS VENDIDOS")
    print("=" * 60)

    # Agregar items por producto
    ventas_por_producto = df_transacciones.groupby('product_id').agg({
        'items': 'sum',
        'order_id': 'count'
    }).rename(columns={
        'items': 'total_items',
        'order_id': 'num_ordenes'
    })

    # Unir con información de productos
    top_productos = ventas_por_producto.nlargest(top_n, 'total_items')
    top_productos = top_productos.merge(
        df_productos[['product_id', 'brand', 'segment', 'package', 'size']],
        left_index=True,
        right_on='product_id',
        how='left'
    )

    print(top_productos)

    return top_productos


def visualizar_correlaciones(correlaciones, figsize=(10, 8)):
    """
    Visualiza matriz de correlaciones.

    Args:
        correlaciones: DataFrame con correlaciones
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    sns.heatmap(correlaciones, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlaciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualizar_comportamiento_por_tipo(comportamiento_por_tipo, figsize=(14, 6)):
    """
    Visualiza el comportamiento por tipo de cliente.

    Args:
        comportamiento_por_tipo: DataFrame con métricas por tipo
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Órdenes promedio
    axes[0].bar(comportamiento_por_tipo.index, comportamiento_por_tipo['ordenes_promedio'],
                color='steelblue', edgecolor='black')
    axes[0].set_title('Órdenes Promedio por Tipo de Cliente', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tipo de Cliente', fontsize=10)
    axes[0].set_ylabel('Órdenes Promedio', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Items promedio
    axes[1].bar(comportamiento_por_tipo.index, comportamiento_por_tipo['items_promedio'],
                color='coral', edgecolor='black')
    axes[1].set_title('Items Promedio por Tipo de Cliente', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tipo de Cliente', fontsize=10)
    axes[1].set_ylabel('Items Promedio', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualizar_preferencias_segmento(preferencias, figsize=(12, 6)):
    """
    Visualiza las preferencias de segmento por tipo de cliente.

    Args:
        preferencias: DataFrame con tabla cruzada
        figsize: Tamaño de la figura
    """
    # Eliminar fila y columna de totales
    pref_sin_totales = preferencias.drop('All', axis=0).drop('All', axis=1)

    plt.figure(figsize=figsize)
    sns.heatmap(pref_sin_totales, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Preferencias de Segmento por Tipo de Cliente (Items Vendidos)', fontsize=14, fontweight='bold')
    plt.xlabel('Segmento de Producto', fontsize=12)
    plt.ylabel('Tipo de Cliente', fontsize=12)
    plt.tight_layout()
    plt.show()


def analisis_correlaciones_completo(df_transacciones, df_cliente, df_productos, visualizar=True):
    """
    Realiza un análisis completo de correlaciones.

    Args:
        df_transacciones: DataFrame de transacciones
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Resumen completo del análisis
    """
    resumen = {}

    # Correlaciones de clientes
    resumen['corr_clientes'] = analizar_correlaciones_clientes(df_cliente)

    # Relación cliente-comportamiento
    resumen['comportamiento_por_tipo'] = analizar_relacion_cliente_comportamiento(df_transacciones, df_cliente)

    # Preferencias por segmento
    resumen['preferencias_segmento'] = analizar_preferencias_por_segmento(df_transacciones, df_productos, df_cliente)

    # Productos más vendidos
    resumen['top_productos'] = analizar_productos_mas_vendidos(df_transacciones, df_productos)

    # Visualizaciones
    if visualizar:
        visualizar_correlaciones(resumen['corr_clientes'])
        visualizar_comportamiento_por_tipo(resumen['comportamiento_por_tipo'])
        visualizar_preferencias_segmento(resumen['preferencias_segmento'])

    return resumen
