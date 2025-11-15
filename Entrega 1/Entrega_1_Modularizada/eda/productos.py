"""
Funciones para análisis exploratorio de productos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_productos_basico(df_productos):
    """
    Análisis básico del catálogo de productos.

    Args:
        df_productos: DataFrame de productos

    Returns:
        dict: Estadísticas básicas de productos
    """
    print("=" * 60)
    print("ANÁLISIS DE PRODUCTOS")
    print("=" * 60)

    num_productos_unicos = df_productos['product_id'].nunique()
    print(f"\nNúmero de productos únicos: {num_productos_unicos:,}")

    # Distribución por categoría
    print("\nDistribución por categoría:")
    print(df_productos['category'].value_counts())

    # Distribución por subcategoría
    print("\nDistribución por subcategoría:")
    print(df_productos['sub_category'].value_counts())

    # Distribución por segmento
    print("\nDistribución por segmento:")
    print(df_productos['segment'].value_counts())

    # Distribución por tipo de envase
    print("\nDistribución por tipo de envase:")
    print(df_productos['package'].value_counts())

    # Número de marcas
    num_marcas = df_productos['brand'].nunique()
    print(f"\nNúmero de marcas únicas: {num_marcas}")

    return {
        'num_productos': num_productos_unicos,
        'num_marcas': num_marcas,
        'categorias': df_productos['category'].value_counts(),
        'segmentos': df_productos['segment'].value_counts()
    }


def analizar_tamaños_productos(df_productos):
    """
    Analiza la distribución de tamaños de productos.

    Args:
        df_productos: DataFrame de productos

    Returns:
        dict: Estadísticas de tamaños
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE TAMAÑOS DE PRODUCTOS")
    print("=" * 60)

    print("\nEstadísticas de tamaños (litros):")
    print(df_productos['size'].describe())

    print(f"\nTamaño mínimo: {df_productos['size'].min():.3f}L")
    print(f"Tamaño máximo: {df_productos['size'].max():.3f}L")
    print(f"Tamaño promedio: {df_productos['size'].mean():.3f}L")
    print(f"Tamaño mediana: {df_productos['size'].median():.3f}L")

    # Número de tamaños únicos
    tamaños_unicos = df_productos['size'].nunique()
    print(f"\nNúmero de tamaños únicos: {tamaños_unicos}")

    # Categorización de tamaños
    bins = [0, 0.5, 1.0, 3.0, np.inf]
    labels = ['0-0.5L', '0.5-1L', '1-3L', '>3L']
    df_productos['size_categoria'] = pd.cut(df_productos['size'], bins=bins, labels=labels)

    print("\nDistribución por categoría de tamaño:")
    print(df_productos['size_categoria'].value_counts().sort_index())

    return {
        'min': df_productos['size'].min(),
        'max': df_productos['size'].max(),
        'mean': df_productos['size'].mean(),
        'median': df_productos['size'].median(),
        'tamaños_unicos': tamaños_unicos
    }


def visualizar_distribucion_segmentos(df_productos, figsize=(12, 6)):
    """
    Visualiza la distribución de segmentos de productos.

    Args:
        df_productos: DataFrame de productos
        figsize: Tamaño de la figura
    """
    segment_counts = df_productos['segment'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gráfico de barras
    axes[0].bar(segment_counts.index, segment_counts.values, color='coral', edgecolor='black')
    axes[0].set_title('Distribución de Segmentos de Producto', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Segmento', fontsize=12)
    axes[0].set_ylabel('Cantidad', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Gráfico de torta
    axes[1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('Set2'))
    axes[1].set_title('Proporción de Segmentos', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def visualizar_distribucion_envases(df_productos, figsize=(10, 6)):
    """
    Visualiza la distribución de tipos de envase.

    Args:
        df_productos: DataFrame de productos
        figsize: Tamaño de la figura
    """
    package_counts = df_productos['package'].value_counts()

    plt.figure(figsize=figsize)
    plt.bar(package_counts.index, package_counts.values, color='skyblue', edgecolor='black')
    plt.title('Distribución de Tipos de Envase', fontsize=14, fontweight='bold')
    plt.xlabel('Tipo de Envase', fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualizar_distribucion_tamaños(df_productos, figsize=(12, 6)):
    """
    Visualiza la distribución de tamaños de productos.

    Args:
        df_productos: DataFrame de productos
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histograma
    axes[0].hist(df_productos['size'], bins=30, color='lightgreen', edgecolor='black')
    axes[0].set_title('Distribución de Tamaños (Litros)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tamaño (L)', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Boxplot
    axes[1].boxplot(df_productos['size'], vert=True)
    axes[1].set_title('Boxplot de Tamaños', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Tamaño (L)', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def analizar_marcas_top(df_productos, top_n=10):
    """
    Analiza las marcas más frecuentes.

    Args:
        df_productos: DataFrame de productos
        top_n: Número de marcas top a mostrar

    Returns:
        pd.Series: Top marcas
    """
    print(f"\nTop {top_n} marcas más frecuentes:")
    top_marcas = df_productos['brand'].value_counts().head(top_n)
    print(top_marcas)

    return top_marcas


def analisis_productos_completo(df_productos, visualizar=True):
    """
    Realiza un análisis completo de productos.

    Args:
        df_productos: DataFrame de productos
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Resumen completo del análisis
    """
    resumen = {}

    # Análisis básico
    resumen['basico'] = analizar_productos_basico(df_productos)

    # Análisis de tamaños
    resumen['tamaños'] = analizar_tamaños_productos(df_productos)

    # Top marcas
    resumen['top_marcas'] = analizar_marcas_top(df_productos)

    # Visualizaciones
    if visualizar:
        visualizar_distribucion_segmentos(df_productos)
        visualizar_distribucion_envases(df_productos)
        visualizar_distribucion_tamaños(df_productos)

    return resumen
