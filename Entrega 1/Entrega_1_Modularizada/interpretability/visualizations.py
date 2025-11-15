"""
Funciones para visualizaciones de interpretabilidad con SHAP.
"""

import matplotlib.pyplot as plt
import shap


def plot_shap_summary(shap_values, X_sample, feature_names=None, max_display=20):
    """
    Genera SHAP summary plot (beeswarm).

    Args:
        shap_values: SHAP values calculados
        X_sample: Features de las muestras
        feature_names: Nombres de features
        max_display: Número máximo de features a mostrar
    """
    print("\nGenerando SHAP summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                     max_display=max_display, show=False)
    plt.title('SHAP Summary Plot - Impacto de Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values, feature_names=None, max_display=20):
    """
    Genera SHAP bar plot (importancia media).

    Args:
        shap_values: SHAP values calculados
        feature_names: Nombres de features
        max_display: Número máximo de features a mostrar
    """
    print("\nGenerando SHAP bar plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar',
                     max_display=max_display, show=False)
    plt.title('SHAP Bar Plot - Importancia de Features', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(explainer, X, indice, feature_names=None):
    """
    Genera SHAP waterfall plot para una predicción individual.

    Args:
        explainer: SHAP explainer
        X: Features
        indice: Índice de la muestra a visualizar
        feature_names: Nombres de features
    """
    print(f"\nGenerando SHAP waterfall plot para muestra #{indice}...")

    shap_values = explainer(X[indice:indice+1])

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[0], max_display=15, show=False)
    plt.title(f'SHAP Waterfall Plot - Muestra #{indice}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_force(explainer, X, indice):
    """
    Genera SHAP force plot para una predicción individual.

    Args:
        explainer: SHAP explainer
        X: Features
        indice: Índice de la muestra
    """
    print(f"\nGenerando SHAP force plot para muestra #{indice}...")

    shap_values = explainer(X[indice:indice+1])
    shap.force_plot(shap_values[0], matplotlib=True, show=False)
    plt.title(f'SHAP Force Plot - Muestra #{indice}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, X_sample, feature_idx, feature_names=None,
                         interaction_index='auto'):
    """
    Genera SHAP dependence plot para un feature específico.

    Args:
        shap_values: SHAP values calculados
        X_sample: Features de las muestras
        feature_idx: Índice del feature a visualizar
        feature_names: Nombres de features
        interaction_index: Índice del feature de interacción
    """
    print(f"\nGenerando SHAP dependence plot para feature {feature_idx}...")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, X_sample,
                        feature_names=feature_names,
                        interaction_index=interaction_index,
                        show=False)
    plt.title(f'SHAP Dependence Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualizaciones_shap_completas(explainer, shap_values, X_sample, feature_names=None,
                                   ejemplos_indices=[0, 1, 2], max_display=20):
    """
    Genera todas las visualizaciones SHAP principales.

    Args:
        explainer: SHAP explainer
        shap_values: SHAP values calculados
        X_sample: Features de las muestras
        feature_names: Nombres de features
        ejemplos_indices: Índices de ejemplos para waterfall plots
        max_display: Número máximo de features a mostrar
    """
    print("\n" + "="*90)
    print("GENERANDO VISUALIZACIONES SHAP")
    print("="*90)

    # Summary plot (beeswarm)
    plot_shap_summary(shap_values, X_sample, feature_names, max_display)

    # Bar plot
    plot_shap_bar(shap_values, feature_names, max_display)

    # Waterfall plots para ejemplos individuales
    for idx in ejemplos_indices[:3]:  # Máximo 3 ejemplos
        if idx < len(X_sample):
            plot_shap_waterfall(explainer, X_sample, idx, feature_names)

    print("\n✓ Visualizaciones SHAP generadas")
