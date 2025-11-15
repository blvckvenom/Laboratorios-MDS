"""
Script principal para ejecutar el pipeline completo de machine learning.

Este script orquesta todos los módulos del proyecto:
- Carga de datos
- Preprocesamiento
- EDA
- Feature Engineering
- Entrenamiento de modelos
- Evaluación
- Interpretabilidad
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Imports de módulos del proyecto
from config import paths
from data import loader, quality
from preprocessing import cleaner, geographic
from eda import clientes, productos, temporal, comportamiento, correlaciones
from features import (
    cliente_features, producto_features, interaccion_features,
    temporal_features, transformers
)
from models import baseline, comparacion, optimizer, xgboost_final
from evaluation import metrics, importance
from interpretability import shap_analysis, visualizations
from utils import plots


def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("="*100)
    print("PIPELINE DE MACHINE LEARNING - ENTREGA 1")
    print("="*100)

    # ========================================
    # 1. CARGA DE DATOS
    # ========================================
    print("\n" + "="*100)
    print("FASE 1: CARGA DE DATOS")
    print("="*100)

    df_cliente, df_productos, df_transacciones = loader.cargar_todos_los_datos()

    # Exploración básica
    loader.explorar_dataset(df_cliente, "Clientes")
    loader.explorar_dataset(df_productos, "Productos")
    loader.explorar_dataset(df_transacciones, "Transacciones")

    # ========================================
    # 2. PREPROCESAMIENTO
    # ========================================
    print("\n" + "="*100)
    print("FASE 2: PREPROCESAMIENTO")
    print("="*100)

    # Validar y corregir coordenadas geográficas
    geographic.validar_coordenadas_wgs84(df_cliente)
    df_cliente = geographic.corregir_coordenadas(df_cliente)

    # Aplicar limpieza completa
    df_cliente_clean, df_productos_clean, df_transacciones_agg = cleaner.aplicar_limpieza_completa(
        df_cliente, df_productos, df_transacciones
    )

    # ========================================
    # 3. ANÁLISIS DE CALIDAD
    # ========================================
    print("\n" + "="*100)
    print("FASE 3: ANÁLISIS DE CALIDAD")
    print("="*100)

    resumen_calidad = quality.analisis_calidad_completo(
        df_cliente_clean, df_productos_clean, df_transacciones_agg
    )

    # ========================================
    # 4. EDA (ANÁLISIS EXPLORATORIO)
    # ========================================
    print("\n" + "="*100)
    print("FASE 4: ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*100)

    # Análisis de clientes
    resumen_clientes = clientes.analisis_clientes_completo(df_cliente_clean, visualizar=False)

    # Análisis de productos
    resumen_productos = productos.analisis_productos_completo(df_productos_clean, visualizar=False)

    # Análisis temporal
    resumen_temporal = temporal.analisis_temporal_completo(df_transacciones_agg, visualizar=False)

    # Análisis de comportamiento
    resumen_comportamiento = comportamiento.analisis_comportamiento_completo(
        df_transacciones_agg, visualizar=False
    )

    # Análisis de correlaciones
    resumen_correlaciones = correlaciones.analisis_correlaciones_completo(
        df_transacciones_agg, df_cliente_clean, df_productos_clean, visualizar=False
    )

    print("\n✓ Análisis exploratorio completado")

    # ========================================
    # 5. FEATURE ENGINEERING
    # ========================================
    print("\n" + "="*100)
    print("FASE 5: FEATURE ENGINEERING")
    print("="*100)

    # NOTA: Esta sección requiere que los datos estén particionados en train/val/test
    # y que exista una estructura cliente-producto-semana con target
    # El código a continuación es un ejemplo de cómo se usarían los módulos:

    """
    # Ejemplo de uso (requiere datos particionados):

    # 1. Aplicar features de cliente
    df_train_fe, df_val_fe, df_test_fe = cliente_features.aplicar_features_cliente(
        df_train, df_val, df_test, df_transacciones_agg
    )

    # 2. Aplicar features de producto
    df_train_fe, df_val_fe, df_test_fe = producto_features.aplicar_features_producto(
        df_train_fe, df_val_fe, df_test_fe, df_transacciones_agg
    )

    # 3. Aplicar features de interacción
    df_train_fe, df_val_fe, df_test_fe = interaccion_features.aplicar_features_interaccion(
        df_train_fe, df_val_fe, df_test_fe, df_transacciones_agg
    )

    # 4. Aplicar features temporales
    df_train_fe, df_val_fe, df_test_fe = temporal_features.aplicar_features_temporales(
        df_train_fe, df_val_fe, df_test_fe
    )

    # 5. Aplicar transformaciones
    df_train_fe, df_val_fe, df_test_fe = transformers.aplicar_transformaciones(
        df_train_fe, df_val_fe, df_test_fe
    )

    # 6. Preparar datasets para modelado
    X_train, y_train, X_val, y_val, X_test, y_test = transformers.preparar_datasets_para_modelado(
        df_train_fe, df_val_fe, df_test_fe, feature_columns
    )
    """

    print("\n✓ Feature engineering configurado (requiere datos particionados)")

    # ========================================
    # 6. MODELADO
    # ========================================
    print("\n" + "="*100)
    print("FASE 6: MODELADO")
    print("="*100)

    """
    # Ejemplo de uso (requiere X_train, y_train, X_val, y_val):

    # 6.1 Baseline
    preprocessor = transformers.crear_pipeline_preprocesamiento(
        numeric_features, categorical_features_onehot,
        categorical_features_target, binary_features
    )

    modelo_baseline, metricas_baseline, tiempo_baseline = baseline.ejecutar_baseline_completo(
        preprocessor, X_train, y_train, X_val, y_val
    )

    # 6.2 Comparación de modelos
    df_resultados, modelos_entrenados = comparacion.comparar_modelos(
        preprocessor, X_train, y_train, X_val, y_val
    )

    # 6.3 Optimización con Optuna
    modelo_final, metricas_final, best_params, study = optimizer.pipeline_optimizacion_completo(
        X_train, y_train, X_val, y_val,
        metricas_baseline=metricas_baseline,
        metricas_original=df_resultados.iloc[0].to_dict(),
        n_trials=200
    )

    # 6.4 Guardar modelo
    xgboost_final.guardar_modelo(modelo_final, 'modelos', 'modelo_final_optimizado')
    xgboost_final.guardar_parametros(best_params, 'modelos', 'best_params')
    """

    print("\n✓ Módulos de modelado disponibles")

    # ========================================
    # 7. EVALUACIÓN
    # ========================================
    print("\n" + "="*100)
    print("FASE 7: EVALUACIÓN")
    print("="*100)

    """
    # Ejemplo de uso:

    # 7.1 Evaluar modelo
    metricas = metrics.reporte_evaluacion_completo(
        y_val, y_pred, y_proba, titulo="Evaluación del Modelo Final"
    )

    # 7.2 Análisis de importancia
    df_importance, modelo_xgb = importance.analisis_importancia_completo(
        X_train_transformed, y_train, X_val_transformed, y_val,
        feature_names=feature_names, visualizar=True
    )
    """

    print("\n✓ Módulos de evaluación disponibles")

    # ========================================
    # 8. INTERPRETABILIDAD
    # ========================================
    print("\n" + "="*100)
    print("FASE 8: INTERPRETABILIDAD")
    print("="*100)

    """
    # Ejemplo de uso:

    # 8.1 Análisis SHAP
    explainer, shap_values, X_sample, df_shap_importance = shap_analysis.analisis_shap_completo(
        modelo_final, X_val_transformed, feature_names=feature_names
    )

    # 8.2 Visualizaciones SHAP
    visualizations.visualizaciones_shap_completas(
        explainer, shap_values, X_sample, feature_names=feature_names
    )
    """

    print("\n✓ Módulos de interpretabilidad disponibles")

    # ========================================
    # RESUMEN FINAL
    # ========================================
    print("\n" + "="*100)
    print("RESUMEN DE EJECUCIÓN")
    print("="*100)

    print(f"\n✓ Datos cargados:")
    print(f"  - Clientes: {len(df_cliente_clean):,}")
    print(f"  - Productos: {len(df_productos_clean):,}")
    print(f"  - Transacciones: {len(df_transacciones_agg):,}")

    print(f"\n✓ Calidad de datos:")
    print(f"  - Problemas detectados: {resumen_calidad['total_problemas']}")
    print(f"  - Clientes sin compras: {resumen_calidad['clientes_sin_compras']}")
    print(f"  - Productos sin ventas: {resumen_calidad['productos_sin_ventas']}")

    print("\n" + "="*100)
    print("PIPELINE COMPLETADO")
    print("="*100)
    print("\nNOTA: Las fases de modelado requieren datos particionados en train/val/test")
    print("      y una estructura cliente-producto-semana con variable target.")
    print("\nTodos los módulos están disponibles para uso individual según necesidad.")


if __name__ == "__main__":
    # Configurar estilos de visualización
    plots.configurar_estilo_plots()

    # Ejecutar pipeline
    main()
