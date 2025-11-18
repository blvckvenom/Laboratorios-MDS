from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import TARGET_COL, DEFAULT_MODEL_PATH, DEFAULT_METRICS_PATH

FEATURES = [
    "customer_id",
    "product_id",
    "order_id",
    "region_id",
    "zone_id",
    "Y",
    "X",
    "num_deliver_per_week",
    "num_visit_per_week",
    "size",
]


def entrenar_modelo(df: pd.DataFrame) -> None:
    """
    Entrena un modelo de RandomForest usando el dataset modelado.

    - Usa la columna TARGET_COL como variable objetivo.
    - Usa la lista FEATURES como variables explicativas.
    - Separa en train/test.
    - Entrena el modelo y guarda:
        * El modelo en DEFAULT_MODEL_PATH.
        * Las métricas en DEFAULT_METRICS_PATH.
    """

    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío; no se puede entrenar el modelo.")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no está en el DataFrame")

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Faltan las siguientes columnas de features en el DataFrame para entrenar: {missing_features}"
        )

    cols_needed = FEATURES + [TARGET_COL]
    df_clean = df[cols_needed].dropna().copy()

    if df_clean.empty:
        raise ValueError("Después de eliminar NaNs, no quedan filas para entrenar el modelo.")

    X = df_clean[FEATURES]
    y = df_clean[TARGET_COL]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    print("Columnas recibidas en entrenar_modelo:")
    print(list(df.columns))
    print("Features que se usarán para entrenar:")
    print(FEATURES)


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "features_used": FEATURES,
    }

    model_path = Path(DEFAULT_MODEL_PATH)
    metrics_path = Path(DEFAULT_METRICS_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print("Métricas:", metrics)
