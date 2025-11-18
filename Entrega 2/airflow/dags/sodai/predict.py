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
    Genera predicciones usando el modelo entrenado y el dataset de modelamiento.

    Pasos:
    - Carga df_modelado desde artifacts/datasets.
    - Carga el modelo entrenado desde artifacts/models.
    - Usa las mismas FEATURES que en el entrenamiento.
    - Agrega una columna 'prediction' al DataFrame.
    - Guarda el resultado en artifacts/predictions/predicciones.parquet.
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

    y_pred = model.predict(X)

    df_out = df.copy()
    df_out["prediction"] = y_pred

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    salida = PREDICTIONS_DIR / "predicciones.parquet"
    df_out.to_parquet(salida, index=False)

    print("Primeras filas con predicción:")
    print(df_out.head().to_string())

    resumen = {
        "n_samples": int(df_out.shape[0]),
        "columns": df_out.columns.tolist(),
        "predictions_path": str(salida),
        "features_used_for_prediction": FEATURES,
    }
    print("Resumen de predicciones (predict.py):")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    return salida
