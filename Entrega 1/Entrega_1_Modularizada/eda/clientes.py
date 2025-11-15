"""
Funciones para análisis exploratorio de clientes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_clientes_basico(df_cliente):
    """
    Análisis básico del dataset de clientes.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        dict: Estadísticas básicas de clientes
    """
    print("=" * 60)
    print("ANÁLISIS DE CLIENTES")
    print("=" * 60)

    num_clientes_unicos = df_cliente['customer_id'].nunique()
    print(f"\nNúmero de clientes únicos: {num_clientes_unicos:,}")

    # Distribución por tipo de cliente
    print("\nDistribución por tipo de cliente:")
    tipo_dist = df_cliente['customer_type'].value_counts()
    print(tipo_dist)
    print("\nPorcentajes:")
    print(df_cliente['customer_type'].value_counts(normalize=True) * 100)

    # Estadísticas de entregas
    print("\nEstadísticas de entregas por semana:")
    print(df_cliente['num_deliver_per_week'].describe())

    return {
        'num_clientes': num_clientes_unicos,
        'tipo_distribucion': tipo_dist,
        'entregas_promedio': df_cliente['num_deliver_per_week'].mean()
    }


def analizar_distribucion_geografica(df_cliente):
    """
    Analiza la distribución geográfica de los clientes.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        dict: Estadísticas geográficas
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS GEOGRÁFICO")
    print("=" * 60)

    print("\nRango de coordenadas:")
    print(f"Longitud (X): [{df_cliente['X'].min():.4f}, {df_cliente['X'].max():.4f}]")
    print(f"Latitud (Y): [{df_cliente['Y'].min():.4f}, {df_cliente['Y'].max():.4f}]")

    print("\nCentroide geográfico:")
    centroid_x = df_cliente['X'].mean()
    centroid_y = df_cliente['Y'].mean()
    print(f"Centro: ({centroid_x:.4f}, {centroid_y:.4f})")

    # Dispersión geográfica
    std_x = df_cliente['X'].std()
    std_y = df_cliente['Y'].std()
    print(f"\nDispersión (std):")
    print(f"X: {std_x:.4f}")
    print(f"Y: {std_y:.4f}")

    return {
        'centroid': (centroid_x, centroid_y),
        'range_x': (df_cliente['X'].min(), df_cliente['X'].max()),
        'range_y': (df_cliente['Y'].min(), df_cliente['Y'].max()),
        'std_x': std_x,
        'std_y': std_y
    }


def visualizar_distribucion_tipos(df_cliente, figsize=(12, 6)):
    """
    Visualiza la distribución de tipos de clientes.

    Args:
        df_cliente: DataFrame de clientes
        figsize: Tamaño de la figura
    """
    tipo_counts = df_cliente['customer_type'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gráfico de barras
    axes[0].bar(tipo_counts.index, tipo_counts.values, color='steelblue', edgecolor='black')
    axes[0].set_title('Distribución de Tipos de Cliente', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tipo de Cliente', fontsize=12)
    axes[0].set_ylabel('Cantidad', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Gráfico de torta
    axes[1].pie(tipo_counts.values, labels=tipo_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('pastel'))
    axes[1].set_title('Proporción de Tipos de Cliente', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def visualizar_distribucion_geografica(df_cliente, figsize=(10, 8)):
    """
    Visualiza la distribución geográfica de los clientes.

    Args:
        df_cliente: DataFrame de clientes
        figsize: Tamaño de la figura
    """
    muestra = df_cliente[['X', 'Y']].sample(min(1500, len(df_cliente)), random_state=42)

    plt.figure(figsize=figsize)
    plt.scatter(muestra['X'], muestra['Y'], alpha=0.6, s=12, c='steelblue', edgecolor='white', linewidth=0.5)
    plt.title('Distribución Geográfica de Clientes', fontsize=14, fontweight='bold')
    plt.xlabel('Longitud (X)', fontsize=12)
    plt.ylabel('Latitud (Y)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Añadir información de rangos
    plt.text(0.02, 0.98, f'Rango X: [{df_cliente["X"].min():.1f}, {df_cliente["X"].max():.1f}]',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    plt.text(0.02, 0.90, f'Rango Y: [{df_cliente["Y"].min():.1f}, {df_cliente["Y"].max():.1f}]',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.show()


def analizar_entregas_por_tipo(df_cliente):
    """
    Analiza las entregas por semana según el tipo de cliente.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        pd.DataFrame: Estadísticas de entregas por tipo
    """
    print("\n" + "=" * 60)
    print("ENTREGAS POR TIPO DE CLIENTE")
    print("=" * 60)

    entregas_por_tipo = df_cliente.groupby('customer_type')['num_deliver_per_week'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])

    print(entregas_por_tipo)

    return entregas_por_tipo


def analisis_clientes_completo(df_cliente, visualizar=True):
    """
    Realiza un análisis completo de clientes.

    Args:
        df_cliente: DataFrame de clientes
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Resumen completo del análisis
    """
    resumen = {}

    # Análisis básico
    resumen['basico'] = analizar_clientes_basico(df_cliente)

    # Análisis geográfico
    resumen['geografico'] = analizar_distribucion_geografica(df_cliente)

    # Entregas por tipo
    resumen['entregas_por_tipo'] = analizar_entregas_por_tipo(df_cliente)

    # Visualizaciones
    if visualizar:
        visualizar_distribucion_tipos(df_cliente)
        visualizar_distribucion_geografica(df_cliente)

    return resumen
