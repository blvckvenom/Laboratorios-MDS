from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd

from .config import DEFAULT_MODEL_PATH, PREDICTIONS_DIR, TARGET_COL
from .data_io import cargar_dataset_modelado
from .train import FEATURES


def generar_predicciones(nombre: str = "df_modelado.parquet") -> Path:
    """
    genera predicciones binarias y probabilidades usando el modelo entrenado

    pasos:
    - carga df_modelado desde artifacts/datasets
    - carga el modelo xgboost entrenado desde artifacts/models
    - usa las mismas features que en el entrenamiento
    - genera predicciones binarias (0/1) y probabilidades (0-1)
    - agrega columnas 'prediction' y 'prediction_proba' al dataframe
    - guarda el resultado en artifacts/predictions/predicciones.parquet
    """

    df = cargar_dataset_modelado(nombre)

    if df.empty:
        raise ValueError("El DataFrame de modelamiento está vacío; no se pueden generar predicciones.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no se encuentra en el DataFrame.")

    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {model_path}")

    model = joblib.load(model_path)

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Faltan las siguientes columnas de features en el DataFrame para predecir: {missing_features}"
        )

    X = df[FEATURES]

    # convertir columnas categoricas a tipo 'category' para xgboost
    categorical_cols = ['category', 'sub_category', 'package', 'size_categoria',
                        'trimestre', 'dia_semana', 'mes', 'segment', 'brand', 'customer_type']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    # generar predicciones binarias y probabilidades
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["prediction"] = y_pred
    df_out["prediction_proba"] = y_proba

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    salida = PREDICTIONS_DIR / "predicciones.parquet"
    df_out.to_parquet(salida, index=False)

    print("primeras filas con prediccion y probabilidad:")
    print(df_out[["customer_id", "product_id", TARGET_COL, "prediction", "prediction_proba"]].head().to_string())

    # estadisticas de predicciones
    pred_positivos = int((y_pred == 1).sum())
    pred_negativos = int((y_pred == 0).sum())
    proba_media = float(y_proba.mean())
    proba_std = float(y_proba.std())

    print(f"\nestadisticas de predicciones:")
    print(f"  predicciones positivas: {pred_positivos} ({pred_positivos/len(y_pred):.2%})")
    print(f"  predicciones negativas: {pred_negativos} ({pred_negativos/len(y_pred):.2%})")
    print(f"  probabilidad media: {proba_media:.4f}")
    print(f"  probabilidad std: {proba_std:.4f}")

    resumen = {
        "n_samples": int(df_out.shape[0]),
        "predictions_positive": pred_positivos,
        "predictions_negative": pred_negativos,
        "probability_mean": proba_media,
        "probability_std": proba_std,
        "predictions_path": str(salida),
        "features_used_for_prediction": FEATURES,
    }
    print("\nresumen de predicciones guardado en:")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    return salida
