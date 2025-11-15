"""
Optimización de hiperparámetros con Optuna.
"""

import time
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def crear_funcion_objetivo(X_train, y_train, X_val, y_val):
    """
    Crea la función objetivo para optimización con Optuna.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación

    Returns:
        function: Función objetivo
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        return f1

    return objective


def optimizar_hiperparametros(X_train, y_train, X_val, y_val, n_trials=200, random_state=42):
    """
    Optimiza hiperparámetros usando Optuna.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        n_trials: Número de trials de Optuna
        random_state: Seed para reproducibilidad

    Returns:
        tuple: (best_params, best_f1, study, tiempo)
    """
    print("\n" + "="*90)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    print("="*90)

    print(f"\nMétrica objetivo: F1-Score")
    print(f"Número de trials: {n_trials}")
    print(f"Justificación: Balance entre Precision y Recall crítico para el negocio")

    # Crear función objetivo
    objective = crear_funcion_objetivo(X_train, y_train, X_val, y_val)

    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )

    # Optimizar
    print(f"\nIniciando optimización (esto tomará varios minutos)...")
    tiempo_inicio = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    tiempo = time.time() - tiempo_inicio

    best_params = study.best_params
    best_f1 = study.best_value

    print(f"\n✓ Optimización completada en {tiempo/60:.2f} minutos")
    print(f"\nMejor F1-Score: {best_f1:.4f}")
    print(f"\nMejores parámetros:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")

    return best_params, best_f1, study, tiempo


def entrenar_modelo_optimizado(best_params, X_train, y_train):
    """
    Entrena el modelo final con los mejores parámetros.

    Args:
        best_params: Mejores parámetros encontrados
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento

    Returns:
        XGBClassifier: Modelo entrenado
    """
    print("\n" + "="*90)
    print("ENTRENAMIENTO DEL MODELO FINAL OPTIMIZADO")
    print("="*90)

    modelo_final = XGBClassifier(**best_params)
    modelo_final.fit(X_train, y_train)

    print("\n✓ Modelo final optimizado entrenado")

    return modelo_final


def evaluar_modelo_optimizado(modelo, X_val, y_val):
    """
    Evalúa el modelo optimizado.

    Args:
        modelo: Modelo entrenado
        X_val: Features de validación
        y_val: Target de validación

    Returns:
        dict: Métricas del modelo
    """
    y_pred = modelo.predict(X_val)
    y_proba = modelo.predict_proba(X_val)[:, 1]

    metricas = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba)
    }

    print("\nMétricas del modelo optimizado:")
    print(f"  Accuracy:  {metricas['accuracy']:.4f}")
    print(f"  Precision: {metricas['precision']:.4f}")
    print(f"  Recall:    {metricas['recall']:.4f}")
    print(f"  F1-Score:  {metricas['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metricas['roc_auc']:.4f}")

    return metricas


def comparar_modelos_completo(metricas_baseline, metricas_original, metricas_optimizado):
    """
    Compara los tres modelos: baseline, original y optimizado.

    Args:
        metricas_baseline: Métricas del baseline
        metricas_original: Métricas del modelo original
        metricas_optimizado: Métricas del modelo optimizado
    """
    print("\n" + "-"*90)
    print("COMPARACIÓN DE MODELOS")
    print("-"*90)

    print(f"Baseline:        F1={metricas_baseline['f1_score']:.4f}, ROC-AUC={metricas_baseline['roc_auc']:.4f}")
    print(f"XGB Original:    F1={metricas_original['F1-Score']:.4f}, ROC-AUC={metricas_original['ROC-AUC']:.4f}")
    print(f"XGB Optimizado:  F1={metricas_optimizado['f1_score']:.4f}, ROC-AUC={metricas_optimizado['roc_auc']:.4f}")

    mejora_vs_baseline = ((metricas_optimizado['f1_score'] - metricas_baseline['f1_score']) / metricas_baseline['f1_score']) * 100
    mejora_vs_original = ((metricas_optimizado['f1_score'] - metricas_original['F1-Score']) / metricas_original['F1-Score']) * 100

    print(f"\nMejora vs Baseline: {mejora_vs_baseline:+.2f}%")
    print(f"Mejora vs Original: {mejora_vs_original:+.2f}%")


def pipeline_optimizacion_completo(X_train, y_train, X_val, y_val,
                                   metricas_baseline=None, metricas_original=None,
                                   n_trials=200, random_state=42):
    """
    Pipeline completo de optimización: optimizar, entrenar y evaluar.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        metricas_baseline: Métricas del baseline (opcional)
        metricas_original: Métricas del modelo original (opcional)
        n_trials: Número de trials de Optuna
        random_state: Seed para reproducibilidad

    Returns:
        tuple: (modelo_final, metricas, best_params, study)
    """
    # Optimizar
    best_params, best_f1, study, tiempo_opt = optimizar_hiperparametros(
        X_train, y_train, X_val, y_val, n_trials, random_state
    )

    # Entrenar modelo final
    modelo_final = entrenar_modelo_optimizado(best_params, X_train, y_train)

    # Evaluar
    metricas = evaluar_modelo_optimizado(modelo_final, X_val, y_val)

    # Comparar si se proporcionaron métricas previas
    if metricas_baseline and metricas_original:
        comparar_modelos_completo(metricas_baseline, metricas_original, metricas)

    return modelo_final, metricas, best_params, study
