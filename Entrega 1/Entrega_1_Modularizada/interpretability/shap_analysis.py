"""
Funciones para análisis de interpretabilidad con SHAP.
"""

import numpy as np
import pandas as pd
import shap


def crear_explainer(modelo):
    """
    Crea un TreeExplainer de SHAP para el modelo.

    Args:
        modelo: Modelo entrenado (XGBoost, LightGBM, etc.)

    Returns:
        shap.TreeExplainer: Explainer SHAP
    """
    print("\nCreando SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(modelo)
    print("✓ Explainer creado")

    return explainer


def calcular_shap_values(explainer, X, max_samples=5000):
    """
    Calcula SHAP values para un dataset.

    Args:
        explainer: SHAP explainer
        X: Features para calcular SHAP values
        max_samples: Número máximo de muestras a procesar

    Returns:
        numpy array: SHAP values
    """
    print(f"\nCalculando SHAP values para {min(max_samples, len(X)):,} muestras...")

    X_sample = X[:max_samples] if len(X) > max_samples else X
    shap_values = explainer.shap_values(X_sample)

    print(f"✓ SHAP values calculados: shape {np.array(shap_values).shape}")

    return shap_values, X_sample


def obtener_importancia_shap(shap_values, feature_names=None, top_n=20):
    """
    Obtiene importancia de features basada en SHAP values.

    Args:
        shap_values: SHAP values calculados
        feature_names: Nombres de features
        top_n: Número de features top a retornar

    Returns:
        pd.DataFrame: Importancia ordenada por feature
    """
    # Importancia media absoluta
    importancia_shap = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importancia_shap))]

    df_shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importancia_shap
    }).sort_values('shap_importance', ascending=False)

    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURES - IMPORTANCIA SHAP")
    print('='*80)
    print(df_shap_importance.head(top_n).to_string(index=False))

    return df_shap_importance


def analizar_prediccion_individual(explainer, X, indice, feature_names=None):
    """
    Analiza una predicción individual con SHAP.

    Args:
        explainer: SHAP explainer
        X: Features
        indice: Índice de la muestra a analizar
        feature_names: Nombres de features

    Returns:
        dict: Análisis de la predicción
    """
    # Calcular SHAP para la muestra
    shap_values = explainer.shap_values(X[indice:indice+1])
    shap_values_sample = shap_values[0]

    # Obtener top features contribuyentes
    abs_shap = np.abs(shap_values_sample)
    top_indices = np.argsort(abs_shap)[::-1][:5]

    print(f"\nAnálisis de muestra #{indice}")
    print("-" * 60)
    print("Top 5 features contribuyentes:")

    for i, idx in enumerate(top_indices, 1):
        fname = feature_names[idx] if feature_names else f"feature_{idx}"
        valor = X[indice, idx] if hasattr(X, 'shape') else X.iloc[indice, idx]
        print(f"{i}. {fname}: valor={valor:.4f}, SHAP={shap_values_sample[idx]:.4f}")

    return {
        'shap_values': shap_values_sample,
        'top_indices': top_indices
    }


def analisis_shap_completo(modelo, X, max_samples=5000, feature_names=None, top_n=20):
    """
    Realiza un análisis SHAP completo.

    Args:
        modelo: Modelo entrenado
        X: Features
        max_samples: Número máximo de muestras
        feature_names: Nombres de features
        top_n: Número de features top

    Returns:
        tuple: (explainer, shap_values, X_sample, df_importance)
    """
    print("\n" + "="*90)
    print("ANÁLISIS DE INTERPRETABILIDAD CON SHAP")
    print("="*90)

    # Crear explainer
    explainer = crear_explainer(modelo)

    # Calcular SHAP values
    shap_values, X_sample = calcular_shap_values(explainer, X, max_samples)

    # Obtener importancia
    df_importance = obtener_importancia_shap(shap_values, feature_names, top_n)

    return explainer, shap_values, X_sample, df_importance
