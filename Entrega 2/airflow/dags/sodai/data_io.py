from pathlib import Path

import pandas as pd

from .config import (
    CLIENTES_PATH,
    PRODUCTOS_PATH,
    TRANSACCIONES_PATH,
    DATASETS_DIR,
)


def cargar_datos_crudos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clientes = pd.read_parquet(CLIENTES_PATH)
    productos = pd.read_parquet(PRODUCTOS_PATH)
    transacciones = pd.read_parquet(TRANSACCIONES_PATH)
    return clientes, productos, transacciones


def guardar_dataset_modelado(
    df: pd.DataFrame,
    nombre: str = "df_modelado.parquet",
) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    salida = DATASETS_DIR / nombre
    df.to_parquet(salida, index=False)
    return salida


def cargar_dataset_modelado(nombre: str = "df_modelado.parquet") -> pd.DataFrame:
    ruta = DATASETS_DIR / nombre
    return pd.read_parquet(ruta)
