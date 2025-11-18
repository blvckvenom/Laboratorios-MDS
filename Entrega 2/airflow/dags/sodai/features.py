import pandas as pd


def construir_df_modelado(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    transacciones: pd.DataFrame,
) -> pd.DataFrame:

    df = transacciones.merge(clientes, how="left", on="customer_id")

    df = df.merge(productos, how="left", on="product_id")

    return df
