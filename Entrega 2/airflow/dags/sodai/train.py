from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import mlflow
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

from .config import TARGET_COL, DEFAULT_MODEL_PATH, DEFAULT_METRICS_PATH
from .mlflow_utils import setup_mlflow

# features numericas
NUMERIC_FEATURES = [
    'total_ordenes_global', 'productos_unicos_global', 'items_totales_global',
    'items_promedio_global', 'dias_desde_primera_compra', 'dias_desde_ultima_compra',
    'frecuencia_compra_diaria', 'diversidad_productos', 'total_ventas_global',
    'clientes_unicos_global', 'items_vendidos_global', 'popularidad_rank',
    'veces_comprado_global', 'dias_desde_ultima_compra_producto', 'items_promedio_producto',
    'size', 'size_log1p', 'segment_ordinal', 'X', 'Y', 'distancia_al_centro',
    'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 'semana_del_año',
    'num_deliver_per_week'
]

# features categoricas
CATEGORICAL_FEATURES = [
    'category', 'sub_category', 'package', 'size_categoria',
    'trimestre', 'dia_semana', 'mes', 'segment',
    'brand', 'customer_type'
]

# features binarias
BINARY_FEATURES = [
    'compro_este_producto_antes', 'es_fin_semana', 'es_lunes_jueves',
    'es_temporada_alta', 'es_temporada_baja'
]

# todas las features
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def optimize_hyperparameters(X_train, y_train, n_trials=20):
    """
    optimiza hiperparametros de xgboost usando optuna
    retorna los mejores parametros encontrados
    """

    def objective(trial):
        try:
            # definir espacio de busqueda para xgboost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
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
                'tree_method': 'hist',  # metodo rapido para datasets grandes
                'eval_metric': 'logloss'
            }

            # entrenar modelo con estos parametros
            # usar n_jobs=2 para evitar deadlock con airflow multiprocessing
            # enable_categorical debe pasarse explicitamente para que cross_val_score lo preserve
            model = XGBClassifier(n_jobs=2, enable_categorical=True, **params)

            # validacion cruzada para evaluar usando f1-score
            # nota: n_jobs=1 porque airflow ya usa multiprocessing
            scores = cross_val_score(
                model, X_train, y_train,
                cv=3,
                scoring='f1',
                n_jobs=1
            )

            f1 = scores.mean()

            return f1
        except Exception as e:
            print(f"error en trial {trial.number}: {e}")
            return 0.0  # retornar f1 bajo si falla el trial

    # silenciar logs de optuna para no saturar consola
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"optimizando hiperparametros con {n_trials} trials")

    # crear estudio de optuna para maximizar f1-score
    study = optuna.create_study(direction='maximize')
    # usar n_jobs=1 para evitar deadlock con airflow multiprocessing
    # timeout de 1 hora para prevenir cuelgues indefinidos
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=3600, show_progress_bar=True)

    print(f"mejor f1-score encontrado: {study.best_value:.4f}")
    print(f"mejores parametros: {study.best_params}")

    return study.best_params, study.best_value


def entrenar_modelo(df: pd.DataFrame, optimize=True, n_trials=20) -> None:
    """
    entrena un modelo de random forest usando el dataset modelado
    integra optuna para optimizacion y mlflow para tracking

    pasos:
    - validar datos de entrada
    - separar en train/test
    - optimizar hiperparametros con optuna (opcional)
    - entrenar modelo final
    - registrar todo en mlflow
    - guardar modelo y metricas
    """

    # configurar mlflow
    setup_mlflow()

    if df.empty:
        raise ValueError("el dataframe de entrada esta vacio")

    if TARGET_COL not in df.columns:
        raise ValueError(f"la columna objetivo '{TARGET_COL}' no esta en el dataframe")

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"faltan columnas de features: {missing_features}"
        )

    cols_needed = FEATURES + [TARGET_COL]
    df_clean = df[cols_needed].dropna().copy()

    if df_clean.empty:
        raise ValueError("despues de eliminar nans no quedan filas para entrenar")

    X = df_clean[FEATURES]
    y = df_clean[TARGET_COL]

    # convertir columnas categoricas a tipo 'category' para xgboost
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype('category')

    # separar en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"datos de prueba: {X_test.shape[0]} muestras")

    # iniciar run de mlflow
    with mlflow.start_run() as run:

        # optimizar hiperparametros si esta habilitado
        if optimize:
            print("\niniciando optimizacion de hiperparametros con optuna")
            best_params, best_cv_f1 = optimize_hyperparameters(
                X_train, y_train, n_trials=n_trials
            )

            # registrar parametros de optuna en mlflow
            mlflow.log_param("optimization_method", "optuna")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_metric("best_cv_f1", best_cv_f1)

        else:
            # usar parametros optimizados de la entrega 1
            best_params = {
                'n_estimators': 229,
                'max_depth': 7,
                'learning_rate': 0.031515986190846335,
                'min_child_weight': 8,
                'gamma': 0.29765412419010323,
                'subsample': 0.8851768860058578,
                'colsample_bytree': 0.9238634358906146,
                'colsample_bylevel': 0.7944120209318764,
                'reg_alpha': 8.593361252262834,
                'reg_lambda': 4.704043810822978,
                'scale_pos_weight': 2.1885367036674523,
                'random_state': 42,
                'tree_method': 'hist',  # metodo rapido para datasets grandes
                'eval_metric': 'logloss'
            }
            mlflow.log_param("optimization_method", "preoptimized")

        # entrenar modelo final con mejores parametros
        print("\nentrenando modelo final con mejores parametros")
        # usar n_jobs=4 para evitar deadlock con airflow multiprocessing
        # enable_categorical debe pasarse explicitamente
        model = XGBClassifier(n_jobs=4, enable_categorical=True, **best_params)
        model.fit(X_train, y_train)

        # evaluar en conjunto de prueba
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # calcular metricas de clasificacion
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        print(f"\nmetricas en test set:")
        print(f"  accuracy: {accuracy:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall: {recall:.4f}")
        print(f"  f1-score: {f1:.4f}")
        print(f"  roc-auc: {roc_auc:.4f}")

        # registrar parametros del modelo en mlflow
        for param_name, param_value in best_params.items():
            if param_name != 'n_jobs':  # no logear n_jobs
                mlflow.log_param(param_name, param_value)

        # registrar metricas en mlflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("n_train", X_train.shape[0])
        mlflow.log_metric("n_test", X_test.shape[0])
        mlflow.log_metric("n_features", len(FEATURES))

        # registrar feature importance
        feature_importance = dict(zip(FEATURES, model.feature_importances_))
        for feat, importance in feature_importance.items():
            mlflow.log_metric(f"feat_imp_{feat}", float(importance))

        # registrar modelo en mlflow
        mlflow.sklearn.log_model(model, "model")

        # guardar mapeo de categorias conocidas para validacion en inferencia
        categorical_mappings = {}
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                # obtener categorias unicas del training set
                categorical_mappings[col] = X_train[col].cat.categories.tolist()

        print(f"\ncategorias conocidas guardadas para {len(categorical_mappings)} columnas")

        # preparar diccionario de metricas para guardar localmente
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "features_used": FEATURES,
            "hyperparameters": best_params,
            "mlflow_run_id": run.info.run_id,
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "categorical_mappings": categorical_mappings
        }

        if optimize:
            metrics["optuna_best_cv_f1"] = best_cv_f1
            metrics["optuna_n_trials"] = n_trials

        # guardar modelo localmente
        model_path = Path(DEFAULT_MODEL_PATH)
        metrics_path = Path(DEFAULT_METRICS_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\nmodelo guardado en: {model_path}")
        print(f"metricas guardadas en: {metrics_path}")
        print(f"mlflow run id: {run.info.run_id}")
