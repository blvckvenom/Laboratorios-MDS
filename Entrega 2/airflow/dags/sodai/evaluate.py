# sodai/evaluate.py

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .config import (
    DATASETS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    TARGET_COL,
)


def evaluar_modelo(
    dataset_name: str = "df_modelado.parquet",
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict:
    """
    Evalúa el modelo entrenado usando el dataset modelado.

    - Carga el dataset final desde artifacts/datasets.
    - Carga el modelo entrenado desde artifacts/models.
    - Usa las mismas features que se usaron en entrenamiento
      (leídas desde el JSON de métricas si es posible).
    - Calcula métricas globales (RMSE y R^2) sobre todo el dataset.
    - Actualiza el archivo JSON de métricas en artifacts/metrics.

    Devuelve el diccionario de métricas nuevas (las de evaluación).
    """

    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH

    data_path = DATASETS_DIR / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset de evaluación en: {data_path}")

    df = pd.read_parquet(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no está en el DataFrame")

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
        feature_cols = [
            c for c in df.columns
            if c not in [TARGET_COL, "purchase_date"]
        ]

    if not feature_cols:
        raise ValueError(
            "No se encontraron columnas de features para evaluar el modelo. "
            "Revisa que 'features_used' exista en el JSON de métricas o que el DataFrame tenga columnas válidas."
        )

    print("Evaluando modelo con las siguientes features:")
    print(feature_cols)

    y = df[TARGET_COL]
    X = df[feature_cols]

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {model_path}")

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    rmse_full = float(mean_squared_error(y, y_pred) ** 0.5)
    r2_full = float(r2_score(y, y_pred))

    eval_metrics = {
        "rmse_full": rmse_full,
        "r2_full": r2_full,
        "n_samples_eval": int(len(df)),
        "features_used_eval": feature_cols,
    }

    print("Métricas de evaluación (evaluate.py):")
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {**existing_metrics, **eval_metrics}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return eval_metrics
