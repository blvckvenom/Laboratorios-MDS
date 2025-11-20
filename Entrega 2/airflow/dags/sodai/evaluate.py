from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # usar backend sin gui para entorno servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import (
    DATASETS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    SHAP_DIR,
    TARGET_COL,
)


def generar_graficos_shap(model, X_sample, feature_names, output_dir):
    """
    genera graficos shap para interpretabilidad del modelo
    guarda summary plot y dependence plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("calculando valores shap (esto puede tardar)")

    # crear explainer de shap
    # usamos treexplainer para random forest (mas rapido)
    explainer = shap.TreeExplainer(model)

    # calcular shap values
    shap_values = explainer.shap_values(X_sample)

    # guardar shap values para uso posterior
    shap_data = {
        'shap_values': shap_values,
        'base_value': explainer.expected_value,
        'data': X_sample.values,
        'feature_names': feature_names
    }

    shap_path = output_dir / "shap_values.pkl"
    with open(shap_path, 'wb') as f:
        pickle.dump(shap_data, f)

    print(f"valores shap guardados en: {shap_path}")

    # generar summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"summary plot guardado en: {summary_path}")

    # generar feature importance plot basado en shap
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    importance_path = output_dir / "shap_feature_importance.png"
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"feature importance plot guardado en: {importance_path}")

    # generar dependence plots para las top 3 features mas importantes
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    for idx in top_features_idx:
        feat_name = feature_names[idx]

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        dep_path = output_dir / f"shap_dependence_{feat_name}.png"
        plt.savefig(dep_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"dependence plot para {feat_name} guardado en: {dep_path}")

    return {
        'shap_values_path': str(shap_path),
        'summary_plot': str(summary_path),
        'importance_plot': str(importance_path),
        'n_samples_analyzed': len(X_sample)
    }


def evaluar_modelo(
    dataset_name: str = "df_modelado.parquet",
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    generate_shap: bool = True,
    shap_sample_size: int = 500,
) -> dict:
    """
    evalua el modelo entrenado y genera interpretabilidad con shap

    pasos:
    - cargar dataset y modelo
    - calcular metricas de evaluacion (rmse, r2)
    - generar analisis shap si esta habilitado
    - actualizar archivo de metricas
    """

    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH

    data_path = DATASETS_DIR / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"no se encontro el dataset: {data_path}")

    df = pd.read_parquet(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"la columna objetivo '{TARGET_COL}' no esta en el dataframe")

    # cargar metricas existentes para obtener features usadas
    existing_metrics: dict = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}

    features_used_train = existing_metrics.get("features_used")

    if features_used_train is not None:
        feature_cols = [
            col for col in features_used_train
            if col in df.columns
        ]
    else:
        # fallback: usar todas las columnas excepto target y fecha
        feature_cols = [
            c for c in df.columns
            if c not in [TARGET_COL, "purchase_date"]
        ]

    if not feature_cols:
        raise ValueError("no se encontraron columnas de features para evaluar")

    print(f"evaluando modelo con {len(feature_cols)} features")

    y = df[TARGET_COL]
    X = df[feature_cols]

    # convertir columnas categoricas a tipo 'category' para xgboost
    categorical_cols = ['category', 'sub_category', 'package', 'size_categoria',
                        'trimestre', 'dia_semana', 'mes', 'segment', 'brand', 'customer_type']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    if not model_path.exists():
        raise FileNotFoundError(f"no se encontro el modelo: {model_path}")

    model = joblib.load(model_path)

    # calcular predicciones
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # calcular metricas de clasificacion
    accuracy_full = float(accuracy_score(y, y_pred))
    precision_full = float(precision_score(y, y_pred))
    recall_full = float(recall_score(y, y_pred))
    f1_full = float(f1_score(y, y_pred))
    roc_auc_full = float(roc_auc_score(y, y_proba))

    eval_metrics = {
        "accuracy_full": accuracy_full,
        "precision_full": precision_full,
        "recall_full": recall_full,
        "f1_full": f1_full,
        "roc_auc_full": roc_auc_full,
        "n_samples_eval": int(len(df)),
        "features_used_eval": feature_cols,
    }

    print(f"metricas de evaluacion:")
    print(f"  accuracy: {accuracy_full:.4f}")
    print(f"  precision: {precision_full:.4f}")
    print(f"  recall: {recall_full:.4f}")
    print(f"  f1-score: {f1_full:.4f}")
    print(f"  roc-auc: {roc_auc_full:.4f}")

    # generar analisis shap si esta habilitado
    if generate_shap:
        print("\ngenerando analisis de interpretabilidad con shap")

        # tomar muestra aleatoria para shap (mas rapido)
        if len(X) > shap_sample_size:
            X_sample = X.sample(n=shap_sample_size, random_state=42)
        else:
            X_sample = X

        try:
            # convertir columnas categoricas a codigos numericos para shap
            # shap no maneja bien el tipo 'category' de pandas con xgboost
            X_sample_shap = X_sample.copy()
            for col in categorical_cols:
                if col in X_sample_shap.columns and X_sample_shap[col].dtype.name == 'category':
                    X_sample_shap[col] = X_sample_shap[col].cat.codes

            shap_info = generar_graficos_shap(
                model,
                X_sample_shap,
                feature_cols,
                SHAP_DIR
            )

            eval_metrics["shap_analysis"] = shap_info

        except Exception as e:
            print(f"error al generar shap: {e}")
            eval_metrics["shap_analysis"] = {"error": str(e)}

    # combinar con metricas existentes
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {**existing_metrics, **eval_metrics}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nmetricas actualizadas en: {metrics_path}")

    return eval_metrics
