"""
Funciones auxiliares para visualizaciones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def configurar_estilo_plots():
    """
    Configura el estilo global de los plots.
    """
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_distribucion_simple(data, columna, titulo=None, xlabel=None, figsize=(10, 6)):
    """
    Genera un histograma simple.

    Args:
        data: DataFrame
        columna: Columna a visualizar
        titulo: Título del gráfico
        xlabel: Etiqueta del eje X
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    plt.hist(data[columna], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(titulo or f'Distribución de {columna}', fontsize=14, fontweight='bold')
    plt.xlabel(xlabel or columna, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_barras_categorias(data, columna, titulo=None, figsize=(10, 6), top_n=None):
    """
    Genera un gráfico de barras para variables categóricas.

    Args:
        data: DataFrame
        columna: Columna categórica
        titulo: Título del gráfico
        figsize: Tamaño de la figura
        top_n: Mostrar solo top N categorías
    """
    conteos = data[columna].value_counts()

    if top_n:
        conteos = conteos.head(top_n)

    plt.figure(figsize=figsize)
    plt.bar(conteos.index, conteos.values, color='coral', edgecolor='black')
    plt.title(titulo or f'Distribución de {columna}', fontsize=14, fontweight='bold')
    plt.xlabel(columna, fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_series_temporal(data, fecha_col, valor_col, titulo=None, figsize=(14, 6)):
    """
    Genera un gráfico de serie temporal.

    Args:
        data: DataFrame
        fecha_col: Columna de fechas
        valor_col: Columna de valores
        titulo: Título del gráfico
        figsize: Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    plt.plot(data[fecha_col], data[valor_col], linewidth=1.5, alpha=0.8)
    plt.title(titulo or 'Serie Temporal', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel(valor_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlacion_heatmap(data, columnas=None, figsize=(10, 8)):
    """
    Genera un heatmap de correlaciones.

    Args:
        data: DataFrame
        columnas: Lista de columnas (si None, usa todas las numéricas)
        figsize: Tamaño de la figura
    """
    if columnas is None:
        columnas = data.select_dtypes(include=['number']).columns

    correlaciones = data[columnas].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(correlaciones, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlaciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_comparacion_modelos(resultados_df, metrica='F1-Score', figsize=(12, 6)):
    """
    Genera gráfico comparativo de modelos.

    Args:
        resultados_df: DataFrame con resultados de modelos
        metrica: Métrica a visualizar
        figsize: Tamaño de la figura
    """
    resultados_ordenados = resultados_df.sort_values(metrica, ascending=False)

    plt.figure(figsize=figsize)
    colores = ['green' if i == 0 else 'steelblue' for i in range(len(resultados_ordenados))]
    plt.bar(resultados_ordenados['Modelo'], resultados_ordenados[metrica],
            color=colores, edgecolor='black')
    plt.title(f'Comparación de Modelos - {metrica}', fontsize=14, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel(metrica, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def guardar_figura(nombre_archivo, dpi=300):
    """
    Guarda la figura actual.

    Args:
        nombre_archivo: Nombre del archivo (con extensión)
        dpi: Resolución
    """
    plt.savefig(nombre_archivo, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figura guardada: {nombre_archivo}")
