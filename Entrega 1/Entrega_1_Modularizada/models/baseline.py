"""
Modelo baseline: Regresión Logística.
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def crear_modelo_baseline(preprocessor, random_state=42):
    """
    Crea modelo baseline (Regresión Logística).

    Args:
        preprocessor: Pipeline de preprocesamiento
        random_state: Seed para reproducibilidad

    Returns:
        Pipeline: Modelo baseline
    """
    baseline_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])

    return baseline_model


def entrenar_baseline(baseline_model, X_train, y_train):
    """
    Entrena el modelo baseline.

    Args:
        baseline_model: Modelo baseline
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento

    Returns:
        tuple: (modelo_entrenado, tiempo_entrenamiento)
    """
    print("\n" + "="*80)
    print("ENTRENANDO MODELO BASELINE: Regresión Logística")
    print("="*80)

    inicio = time.time()
    baseline_model.fit(X_train, y_train)
    tiempo = time.time() - inicio

    print(f"\nTiempo de entrenamiento: {tiempo:.2f} segundos")

    return baseline_model, tiempo


def evaluar_baseline(modelo, X_val, y_val):
    """
    Evalúa el modelo baseline.

    Args:
        modelo: Modelo entrenado
        X_val: Features de validación
        y_val: Target de validación

    Returns:
        dict: Métricas de evaluación
    """
    print("\n" + "="*80)
    print("EVALUACIÓN DEL MODELO BASELINE")
    print("="*80)

    # Predicciones
    y_pred = modelo.predict(X_val)
    y_pred_proba = modelo.predict_proba(X_val)[:, 1]

    # Calcular métricas
    metricas = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1_score': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }

    # Imprimir métricas
    print("\nMétricas en validación:")
    print(f"  Accuracy:  {metricas['accuracy']:.2%}")
    print(f"  Precision: {metricas['precision']:.2%}")
    print(f"  Recall:    {metricas['recall']:.2%}")
    print(f"  F1-Score:  {metricas['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metricas['roc_auc']:.4f}")

    return metricas


def ejecutar_baseline_completo(preprocessor, X_train, y_train, X_val, y_val, random_state=42):
    """
    Ejecuta el flujo completo del baseline: crear, entrenar y evaluar.

    Args:
        preprocessor: Pipeline de preprocesamiento
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        random_state: Seed para reproducibilidad

    Returns:
        tuple: (modelo, metricas, tiempo)
    """
    # Crear modelo
    modelo = crear_modelo_baseline(preprocessor, random_state)

    # Entrenar
    modelo, tiempo = entrenar_baseline(modelo, X_train, y_train)

    # Evaluar
    metricas = evaluar_baseline(modelo, X_val, y_val)

    print(f"\n✓ Modelo baseline completado")
    print(f"  F1-Score: {metricas['f1_score']:.4f}")
    print(f"  ROC-AUC: {metricas['roc_auc']:.4f}")

    return modelo, metricas, tiempo
