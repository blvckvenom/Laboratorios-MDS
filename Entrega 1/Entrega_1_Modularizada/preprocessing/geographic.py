"""
Funciones para corrección y validación de coordenadas geográficas.
"""

import pandas as pd
import numpy as np


def validar_coordenadas_wgs84(df_cliente):
    """
    Valida que las coordenadas estén en formato WGS84.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        dict: Resultados de validación
    """
    total_cli = len(df_cliente)

    # Hipótesis 1: X=lon, Y=lat (convención estándar)
    valid_lon_h1 = df_cliente['X'].between(-180, 180, inclusive='both')
    valid_lat_h1 = df_cliente['Y'].between(-90, 90, inclusive='both')
    invalid_h1 = (~(valid_lon_h1 & valid_lat_h1)).sum()

    # Hipótesis 2: X=lat, Y=lon (intercambiadas)
    valid_lat_h2 = df_cliente['X'].between(-90, 90, inclusive='both')
    valid_lon_h2 = df_cliente['Y'].between(-180, 180, inclusive='both')
    invalid_h2 = (~(valid_lat_h2 & valid_lon_h2)).sum()

    print("\nValidación WGS84")
    print(f"H1 (X=lon, Y=lat) -> inválidos: {invalid_h1:,} ({invalid_h1/total_cli:.2%})")
    print(f"H2 (X=lat, Y=lon) -> inválidos: {invalid_h2:,} ({invalid_h2/total_cli:.2%})")

    if invalid_h1 == 0:
        geo_diagnostico = "OK WGS84 X=lon Y=lat"
    elif invalid_h2 == 0 or invalid_h2 < invalid_h1:
        geo_diagnostico = "SUGERENCIA SWAP XY"
    else:
        geo_diagnostico = "NO PARECE WGS84 REVISAR CRS"

    print("\nDiagnóstico geográfico:", geo_diagnostico)

    # Mostrar ejemplos inválidos
    muestra_geo_bad = df_cliente.loc[~(valid_lon_h1 & valid_lat_h1), ['customer_id', 'X', 'Y']]
    if len(muestra_geo_bad) > 0:
        print("\nEjemplos inválidos bajo H1:")
        print(muestra_geo_bad)

    return {
        'invalid_h1': invalid_h1,
        'invalid_h2': invalid_h2,
        'diagnostico': geo_diagnostico,
        'muestra_problematicos': muestra_geo_bad
    }


def corregir_coordenadas(df_cliente, ids_swap=None, id_nulo=None):
    """
    Corrige coordenadas geográficas aplicando swap y eliminando registros nulos.

    Args:
        df_cliente: DataFrame de clientes
        ids_swap: Lista de IDs que necesitan swap X<->Y
        id_nulo: ID con coordenada nula a eliminar

    Returns:
        pd.DataFrame: DataFrame corregido
    """
    # Valores por defecto basados en análisis
    if ids_swap is None:
        ids_swap = ['219231', '236766', '165126']
    if id_nulo is None:
        id_nulo = '203985'

    df_cliente = df_cliente.copy()

    # Crear máscaras
    mask_swap = df_cliente['customer_id'].astype(str).isin(ids_swap)
    mask_drop = df_cliente['customer_id'].astype(str).eq(id_nulo)

    print("\nPre-corrección")
    print(f"- Filas a swap: {mask_swap.sum()}")
    print(f"- Filas a eliminar por nulo: {mask_drop.sum()}")

    # Mostrar valores antes del swap
    print("\nValores antes del swap (preview):")
    print(df_cliente.loc[mask_swap, ['customer_id', 'X', 'Y']])

    # Aplicar swap X<->Y
    tmp_X = df_cliente.loc[mask_swap, 'X'].copy()
    df_cliente.loc[mask_swap, 'X'] = df_cliente.loc[mask_swap, 'Y'].values
    df_cliente.loc[mask_swap, 'Y'] = tmp_X.values

    # Eliminar fila con coordenada nula
    n_before = len(df_cliente)
    df_cliente = df_cliente.loc[~mask_drop].reset_index(drop=True)
    n_after = len(df_cliente)

    print("\nCorrección aplicada.")
    print(f"- Filas eliminadas: {n_before - n_after}")

    print("\nValores después del swap (verificación):")
    print(df_cliente.loc[df_cliente['customer_id'].astype(str).isin(ids_swap), ['customer_id', 'X', 'Y']])

    # Verificación posterior
    valid_lon_h1_post = df_cliente['X'].between(-180, 180, inclusive='both')
    valid_lat_h1_post = df_cliente['Y'].between(-90, 90, inclusive='both')
    invalid_h1_post = (~(valid_lon_h1_post & valid_lat_h1_post)).sum()

    valid_lat_h2_post = df_cliente['X'].between(-90, 90, inclusive='both')
    valid_lon_h2_post = df_cliente['Y'].between(-180, 180, inclusive='both')
    invalid_h2_post = (~(valid_lat_h2_post & valid_lon_h2_post)).sum()

    print("\nValidación WGS84 (post-corrección)")
    print(f"H1 (X=lon, Y=lat) -> inválidos: {invalid_h1_post:,} ({invalid_h1_post/len(df_cliente):.2%})")
    print(f"H2 (X=lat, Y=lon) -> inválidos: {invalid_h2_post:,} ({invalid_h2_post/len(df_cliente):.2%})")

    print("\nRangos X/Y (post-corrección)")
    print(df_cliente[['X', 'Y']].agg(['min', 'max', 'mean']).T)

    # Muestra de inválidos restantes
    restantes = df_cliente.loc[~(valid_lon_h1_post & valid_lat_h1_post), ['customer_id', 'X', 'Y']]
    if len(restantes) > 0:
        print("\nAún inválidos bajo H1 (revisar manualmente):")
        print(restantes)
    else:
        print("\nSin inválidos bajo H1 tras la corrección puntual.")

    return df_cliente


def obtener_coordenadas_invalidas(df_cliente):
    """
    Identifica registros con coordenadas inválidas.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        pd.DataFrame: Registros con coordenadas inválidas
    """
    valid_lon = df_cliente['X'].between(-180, 180, inclusive='both')
    valid_lat = df_cliente['Y'].between(-90, 90, inclusive='both')

    invalidos = df_cliente.loc[~(valid_lon & valid_lat), ['customer_id', 'X', 'Y']]

    if len(invalidos) > 0:
        print(f"\nCoordenadas inválidas encontradas: {len(invalidos)}")
        print(invalidos)
    else:
        print("\nTodas las coordenadas son válidas WGS84")

    return invalidos


def calcular_distancia_al_centro(df_cliente):
    """
    Calcula la distancia euclidiana de cada cliente al centroide geográfico.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        pd.Series: Distancias al centro
    """
    centroid_x = df_cliente['X'].mean()
    centroid_y = df_cliente['Y'].mean()

    distancias = np.sqrt(
        (df_cliente['X'] - centroid_x)**2 +
        (df_cliente['Y'] - centroid_y)**2
    )

    print(f"\nCentroide geográfico: ({centroid_x:.4f}, {centroid_y:.4f})")
    print(f"Distancia promedio al centro: {distancias.mean():.4f}")
    print(f"Distancia máxima al centro: {distancias.max():.4f}")

    return distancias
