"""
Funciones para cálculo y visualización de métricas de evaluación.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)


def calcular_metricas_completas(y_true, y_pred, y_proba=None):
    """
    Calcula todas las métricas de evaluación.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_proba: Probabilidades (opcional, para ROC-AUC)

    Returns:
        dict: Diccionario con todas las métricas
    """
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        metricas['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metricas


def mostrar_metricas(metricas, titulo="Métricas del Modelo"):
    """
    Muestra las métricas de forma formateada.

    Args:
        metricas: Diccionario de métricas
        titulo: Título del reporte
    """
    print("\n" + "="*80)
    print(titulo)
    print("="*80)

    print(f"\nAccuracy:  {metricas['accuracy']:.2%}")
    print(f"Precision: {metricas['precision']:.2%}")
    print(f"Recall:    {metricas['recall']:.2%}")
    print(f"F1-Score:  {metricas['f1_score']:.4f}")

    if 'roc_auc' in metricas:
        print(f"ROC-AUC:   {metricas['roc_auc']:.4f}")


def visualizar_matriz_confusion(y_true, y_pred, labels=['No Compra', 'Compra'], figsize=(8, 6)):
    """
    Visualiza la matriz de confusión.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        labels: Etiquetas de las clases
        figsize: Tamaño de la figura
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Predicción', fontsize=12)
    plt.tight_layout()
    plt.show()


def visualizar_curva_roc(y_true, y_proba, figsize=(10, 6)):
    """
    Visualiza la curva ROC.

    Args:
        y_true: Valores reales
        y_proba: Probabilidades de la clase positiva
        figsize: Tamaño de la figura
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def comparar_metricas_modelos(metricas_dict, figsize=(14, 6)):
    """
    Compara métricas de múltiples modelos.

    Args:
        metricas_dict: Diccionario con nombre_modelo: metricas
        figsize: Tamaño de la figura
    """
    # Preparar datos
    df = pd.DataFrame(metricas_dict).T

    # Crear subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gráfico 1: F1-Score y ROC-AUC
    if 'f1_score' in df.columns and 'roc_auc' in df.columns:
        df[['f1_score', 'roc_auc']].plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
        axes[0].set_title('F1-Score y ROC-AUC por Modelo', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=10)
        axes[0].set_xlabel('Modelo', fontsize=10)
        axes[0].legend(['F1-Score', 'ROC-AUC'])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(df.index, rotation=45, ha='right')

    # Gráfico 2: Precision y Recall
    if 'precision' in df.columns and 'recall' in df.columns:
        df[['precision', 'recall']].plot(kind='bar', ax=axes[1], color=['lightgreen', 'salmon'])
        axes[1].set_title('Precision y Recall por Modelo', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=10)
        axes[1].set_xlabel('Modelo', fontsize=10)
        axes[1].legend(['Precision', 'Recall'])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticklabels(df.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def reporte_evaluacion_completo(y_true, y_pred, y_proba=None, titulo="Evaluación del Modelo", visualizar=True):
    """
    Genera un reporte de evaluación completo.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_proba: Probabilidades (opcional)
        titulo: Título del reporte
        visualizar: Si se deben generar visualizaciones

    Returns:
        dict: Métricas calculadas
    """
    # Calcular métricas
    metricas = calcular_metricas_completas(y_true, y_pred, y_proba)

    # Mostrar métricas
    mostrar_metricas(metricas, titulo)

    # Mostrar classification report
    print("\n" + "-"*80)
    print("Classification Report Detallado")
    print("-"*80)
    print(classification_report(y_true, y_pred, target_names=['No Compra (0)', 'Compra (1)']))

    # Visualizaciones
    if visualizar:
        visualizar_matriz_confusion(y_true, y_pred)

        if y_proba is not None:
            visualizar_curva_roc(y_true, y_proba)

    return metricas
