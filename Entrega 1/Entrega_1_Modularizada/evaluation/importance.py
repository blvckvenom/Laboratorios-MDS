"""
Funciones para análisis de importancia de features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif


def calcular_importancia_xgboost(X_train, y_train, n_estimators=200, max_depth=6, random_state=42):
    """
    Calcula importancia de features usando XGBoost (gain).

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de estimadores
        max_depth: Profundidad máxima
        random_state: Seed

    Returns:
        tuple: (modelo, importancias)
    """
    print("\nCalculando importancia con XGBoost (gain)...")

    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        importance_type='gain',
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)
    importancias = xgb_model.feature_importances_

    print("✓ Importancia XGBoost calculada")

    return xgb_model, importancias


def calcular_permutation_importance(modelo, X_val, y_val, n_repeats=10, random_state=42):
    """
    Calcula Permutation Importance.

    Args:
        modelo: Modelo entrenado
        X_val: Features de validación
        y_val: Target de validación
        n_repeats: Número de repeticiones
        random_state: Seed

    Returns:
        numpy array: Importancias promedio
    """
    print("\nCalculando Permutation Importance...")

    perm_importance = permutation_importance(
        modelo, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importancias = perm_importance.importances_mean

    print("✓ Permutation Importance calculada")

    return importancias


def calcular_mutual_information(X_train, y_train, random_state=42):
    """
    Calcula Mutual Information entre features y target.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        random_state: Seed

    Returns:
        numpy array: Scores de mutual information
    """
    print("\nCalculando Mutual Information...")

    mi_scores = mutual_info_classif(
        X_train, y_train,
        random_state=random_state,
        n_jobs=-1
    )

    print("✓ Mutual Information calculada")

    return mi_scores


def normalizar_scores(scores):
    """
    Normaliza scores a rango [0, 1].

    Args:
        scores: Array de scores

    Returns:
        numpy array: Scores normalizados
    """
    scores = np.array(scores)
    if scores.max() > 0:
        return (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def consolidar_importancias(importancia_gain, importancia_perm, importancia_mi,
                            feature_names=None, pesos=(0.40, 0.40, 0.20)):
    """
    Consolida múltiples métricas de importancia.

    Args:
        importancia_gain: Importancias de XGBoost
        importancia_perm: Importancias de permutación
        importancia_mi: Mutual information
        feature_names: Nombres de features (opcional)
        pesos: Pesos para cada métrica (suma debe ser 1.0)

    Returns:
        pd.DataFrame: DataFrame con todas las importancias
    """
    print("\nConsolidando resultados de importancia...")

    # Normalizar
    gain_norm = normalizar_scores(importancia_gain)
    perm_norm = normalizar_scores(importancia_perm)
    mi_norm = normalizar_scores(importancia_mi)

    # Promedio ponderado
    importancia_combined = (
        pesos[0] * gain_norm +
        pesos[1] * perm_norm +
        pesos[2] * mi_norm
    )

    # Nombres de features
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importancia_gain))]

    # DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'xgb_gain': importancia_gain,
        'permutation': importancia_perm,
        'mutual_info': importancia_mi,
        'combined': importancia_combined
    }).sort_values('combined', ascending=False)

    print("✓ Importancias consolidadas")

    return df_importance


def mostrar_top_features(df_importance, top_n=20):
    """
    Muestra los top N features más importantes.

    Args:
        df_importance: DataFrame de importancias
        top_n: Número de features top a mostrar
    """
    print("\n" + "="*90)
    print(f"TOP {top_n} FEATURES MÁS IMPORTANTES")
    print("="*90)
    print(df_importance.head(top_n)[['feature', 'combined', 'xgb_gain', 'permutation']].to_string(index=False))


def visualizar_importancias(df_importance, metodo='combined', top_n=20, figsize=(10, 8)):
    """
    Visualiza importancias de features.

    Args:
        df_importance: DataFrame de importancias
        metodo: Método a visualizar ('combined', 'xgb_gain', 'permutation', 'mutual_info')
        top_n: Número de features top a visualizar
        figsize: Tamaño de la figura
    """
    top_features = df_importance.head(top_n).sort_values(metodo)

    plt.figure(figsize=figsize)
    plt.barh(top_features['feature'], top_features[metodo], color='steelblue', edgecolor='black')
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Features - {metodo.upper()}', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def comparar_metodos_importancia(df_importance, top_n=15, figsize=(14, 10)):
    """
    Compara los diferentes métodos de importancia.

    Args:
        df_importance: DataFrame de importancias
        top_n: Número de features top a comparar
        figsize: Tamaño de la figura
    """
    top_features = df_importance.head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # XGBoost Gain
    top_gain = top_features.sort_values('xgb_gain')
    axes[0, 0].barh(top_gain['feature'], top_gain['xgb_gain'], color='coral')
    axes[0, 0].set_title('XGBoost Gain Importance', fontweight='bold')
    axes[0, 0].set_xlabel('Importancia')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Permutation Importance
    top_perm = top_features.sort_values('permutation')
    axes[0, 1].barh(top_perm['feature'], top_perm['permutation'], color='lightgreen')
    axes[0, 1].set_title('Permutation Importance', fontweight='bold')
    axes[0, 1].set_xlabel('Importancia')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Mutual Information
    top_mi = top_features.sort_values('mutual_info')
    axes[1, 0].barh(top_mi['feature'], top_mi['mutual_info'], color='lightblue')
    axes[1, 0].set_title('Mutual Information', fontweight='bold')
    axes[1, 0].set_xlabel('Importancia')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Combined
    top_combined = top_features.sort_values('combined')
    axes[1, 1].barh(top_combined['feature'], top_combined['combined'], color='steelblue')
    axes[1, 1].set_title('Combined Importance', fontweight='bold')
    axes[1, 1].set_xlabel('Importancia')
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def analisis_importancia_completo(X_train, y_train, X_val, y_val,
                                 feature_names=None, top_n=20, visualizar=True):
    """
    Realiza un análisis completo de importancia de features.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        feature_names: Nombres de features
        top_n: Número de features top a mostrar
        visualizar: Si se deben generar visualizaciones

    Returns:
        tuple: (df_importance, modelo_xgb)
    """
    print("\n" + "="*90)
    print("ANÁLISIS DE IMPORTANCIA DE FEATURES")
    print("="*90)

    # Calcular importancias con diferentes métodos
    modelo_xgb, importancia_gain = calcular_importancia_xgboost(X_train, y_train)
    importancia_perm = calcular_permutation_importance(modelo_xgb, X_val, y_val)
    importancia_mi = calcular_mutual_information(X_train, y_train)

    # Consolidar
    df_importance = consolidar_importancias(
        importancia_gain, importancia_perm, importancia_mi, feature_names
    )

    # Mostrar top features
    mostrar_top_features(df_importance, top_n)

    # Visualizaciones
    if visualizar:
        comparar_metodos_importancia(df_importance, top_n)

    return df_importance, modelo_xgb
