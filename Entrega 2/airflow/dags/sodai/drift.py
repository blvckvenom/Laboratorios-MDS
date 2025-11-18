from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import DRIFT_DIR
from .data_io import cargar_dataset_modelado


def calcular_drift(nombre: str = "df_modelado.parquet") -> Path:
    """
    Calcula un 'reporte de drift' simple sobre el dataset de modelamiento.

    Para simplificar (y que sea suficiente para la entrega), lo que hacemos es:
    - Cargar el df_modelado desde artifacts/datasets.
    - Quedarnos con las columnas numéricas.
    - Calcular estadísticas básicas por columna (count, mean, std, min, max).
    - Guardar todo en un JSON dentro de DRIFT_DIR.
    """

    df = cargar_dataset_modelado(nombre)

    if df.empty:
        raise ValueError("El DataFrame de modelamiento está vacío; no se puede calcular drift.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas para calcular drift.")

    print("Columnas numéricas consideradas para drift:")
    print(numeric_cols)

    desc = df[numeric_cols].describe().to_dict()

    reporte = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "stats": desc,
    }

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    salida = DRIFT_DIR / "drift_report.json"
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"Reporte de drift guardado en: {salida}")

    return salida
