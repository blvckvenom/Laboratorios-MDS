"""
Funciones para verificar la calidad e integridad de los datos.
"""

import pandas as pd
import numpy as np


def verificar_valores_nulos(datasets_dict):
    """
    Verifica valores nulos en múltiples datasets.

    Args:
        datasets_dict (dict): Diccionario con nombres y DataFrames

    Returns:
        dict: Resumen de valores nulos por dataset
    """
    print("\n1. Verificación de valores nulos post-preprocesamiento")
    print("-" * 50)

    total_nulos_global = 0
    resumen = {}

    for nombre, df in datasets_dict.items():
        print(f"\nDataset: {nombre}")
        nulos = df.isnull().sum()
        total_registros = len(df)

        if nulos.sum() == 0:
            print("  No se encontraron valores nulos")
            resumen[nombre] = 0
        else:
            print("  Valores nulos encontrados:")
            for columna, cantidad in nulos[nulos > 0].items():
                porcentaje = (cantidad / total_registros) * 100
                print(f"    - {columna}: {cantidad} ({porcentaje:.2f}%)")
                total_nulos_global += cantidad
            resumen[nombre] = nulos.sum()

    return total_nulos_global, resumen


def verificar_integridad_referencial(df_transacciones, df_cliente, df_productos):
    """
    Verifica la integridad referencial entre datasets.

    Args:
        df_transacciones: DataFrame de transacciones
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos

    Returns:
        tuple: (clientes_faltantes, productos_faltantes, clientes_sin_compras, productos_sin_ventas)
    """
    print(f"\n2. Validación de integridad referencial")
    print("-" * 50)

    # Verificar clientes
    clientes_en_transacciones = set(df_transacciones['customer_id'].unique())
    clientes_en_maestro = set(df_cliente['customer_id'].unique())
    clientes_faltantes = clientes_en_transacciones - clientes_en_maestro

    print(f"Total de clientes únicos en transacciones: {len(clientes_en_transacciones):,}")
    print(f"Total de clientes únicos en maestro: {len(clientes_en_maestro):,}")
    print(f"Clientes en transacciones sin datos maestros: {len(clientes_faltantes)}")

    if len(clientes_faltantes) > 0:
        print(f"  Clientes problemáticos: {list(clientes_faltantes)[:10]}")

    # Clientes sin compras
    clientes_sin_compras = clientes_en_maestro - clientes_en_transacciones
    print(f"Clientes sin transacciones: {len(clientes_sin_compras)} ({len(clientes_sin_compras)/len(clientes_en_maestro)*100:.1f}%)")

    # Verificar productos
    productos_en_transacciones = set(df_transacciones['product_id'].unique())
    productos_en_maestro = set(df_productos['product_id'].unique())
    productos_faltantes = productos_en_transacciones - productos_en_maestro

    print(f"\nTotal de productos únicos en transacciones: {len(productos_en_transacciones):,}")
    print(f"Total de productos únicos en maestro: {len(productos_en_maestro):,}")
    print(f"Productos en transacciones sin datos maestros: {len(productos_faltantes)}")

    if len(productos_faltantes) > 0:
        print(f"  Productos problemáticos: {list(productos_faltantes)[:10]}")

    # Productos sin ventas
    productos_sin_ventas = productos_en_maestro - productos_en_transacciones
    print(f"Productos sin ventas: {len(productos_sin_ventas)} ({len(productos_sin_ventas)/len(productos_en_maestro)*100:.1f}%)")

    return clientes_faltantes, productos_faltantes, clientes_sin_compras, productos_sin_ventas


def detectar_inconsistencias(df_transacciones, df_productos, df_cliente):
    """
    Detecta inconsistencias en los valores de los datasets.

    Args:
        df_transacciones: DataFrame de transacciones
        df_productos: DataFrame de productos
        df_cliente: DataFrame de clientes

    Returns:
        tuple: (items_negativos, tamaños_invalidos, entregas_negativas)
    """
    print(f"\n3. Detección de inconsistencias en valores")
    print("-" * 50)

    # Items negativos
    items_negativos = df_transacciones[df_transacciones['items'] < 0]
    print(f"Transacciones con items negativos: {len(items_negativos)} ({len(items_negativos)/len(df_transacciones)*100:.2f}%)")

    # Tamaños inválidos
    tamaños_invalidos = df_productos[df_productos['size'] <= 0]
    print(f"Productos con tamaño negativo o cero: {len(tamaños_invalidos)}")

    # Entregas negativas
    entregas_negativas = df_cliente[df_cliente['num_deliver_per_week'] < 0]
    print(f"Clientes con entregas negativas: {len(entregas_negativas)}")

    return items_negativos, tamaños_invalidos, entregas_negativas


def validar_coordenadas_geograficas(df_cliente):
    """
    Valida que las coordenadas estén en formato WGS84 válido.

    Args:
        df_cliente: DataFrame de clientes

    Returns:
        tuple: (coordenadas_validas_total, coordenadas_invalidas_total)
    """
    print(f"\n4. Validación de coordenadas geográficas WGS84 (post-corrección)")
    print("-" * 50)

    # Verificar rango WGS84
    valid_lon = df_cliente['X'].between(-180, 180, inclusive='both')
    valid_lat = df_cliente['Y'].between(-90, 90, inclusive='both')
    coordenadas_validas_total = (valid_lon & valid_lat).sum()
    coordenadas_invalidas_total = (~(valid_lon & valid_lat)).sum()

    print(f"Coordenadas válidas WGS84: {coordenadas_validas_total:,}/{len(df_cliente):,} ({coordenadas_validas_total/len(df_cliente):.2%})")
    print(f"Coordenadas inválidas: {coordenadas_invalidas_total}")

    if coordenadas_invalidas_total > 0:
        print("  Advertencia: Se encontraron coordenadas inválidas después de la corrección")
        coords_problematicas = df_cliente.loc[~(valid_lon & valid_lat), ['customer_id', 'X', 'Y']]
        print(coords_problematicas.head())

    return coordenadas_validas_total, coordenadas_invalidas_total


def verificar_duplicados(df_transacciones, df_cliente, df_productos):
    """
    Verifica duplicados en los datasets.

    Args:
        df_transacciones: DataFrame de transacciones
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos

    Returns:
        dict: Conteos de duplicados por tipo
    """
    print(f"\n5. Verificación de duplicados")
    print("-" * 50)

    resultados = {}

    # Duplicados en clientes
    resultados['duplicados_clientes_id'] = df_cliente['customer_id'].duplicated().sum()
    print(f"Clientes duplicados por ID: {resultados['duplicados_clientes_id']}")

    # Duplicados en productos
    resultados['duplicados_productos_id'] = df_productos['product_id'].duplicated().sum()
    print(f"Productos duplicados por ID: {resultados['duplicados_productos_id']}")

    # Duplicados exactos en transacciones
    resultados['duplicados_transacciones_completos'] = df_transacciones.duplicated().sum()
    print(f"Transacciones duplicadas (exactas): {resultados['duplicados_transacciones_completos']}")

    # Duplicados por combinación clave
    resultados['duplicados_trans_clave'] = df_transacciones.duplicated(
        subset=['customer_id', 'product_id', 'purchase_date'], keep=False
    ).sum()
    print(f"Transacciones con mismo cliente-producto-fecha: {resultados['duplicados_trans_clave']}")

    return resultados


def validar_coherencia_temporal(df_transacciones):
    """
    Valida la coherencia temporal de las transacciones.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        tuple: (fechas_futuras, fechas_antiguas, rango_dias)
    """
    print(f"\n6. Validación de coherencia temporal")
    print("-" * 50)

    fecha_min = df_transacciones['purchase_date'].min()
    fecha_max = df_transacciones['purchase_date'].max()
    rango_dias = (fecha_max - fecha_min).days + 1

    print(f"Rango de fechas: {fecha_min.date()} a {fecha_max.date()}")
    print(f"Días cubiertos: {rango_dias}")

    # Fechas futuras
    fecha_limite = pd.Timestamp('2024-12-31')
    fechas_futuras = df_transacciones[df_transacciones['purchase_date'] > fecha_limite]
    print(f"Transacciones con fechas posteriores a 2024-12-31: {len(fechas_futuras)}")

    # Fechas antiguas
    fecha_min_esperada = pd.Timestamp('2024-01-01')
    fechas_antiguas = df_transacciones[df_transacciones['purchase_date'] < fecha_min_esperada]
    print(f"Transacciones con fechas anteriores a 2024-01-01: {len(fechas_antiguas)}")

    return fechas_futuras, fechas_antiguas, rango_dias


def detectar_outliers(df_transacciones):
    """
    Detecta outliers en items usando el método IQR.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        tuple: (outliers_items, limite_inferior, limite_superior)
    """
    print(f"\n7. Estadísticas de distribución para detectar anomalías")
    print("-" * 50)

    Q1_items = df_transacciones['items'].quantile(0.25)
    Q3_items = df_transacciones['items'].quantile(0.75)
    IQR_items = Q3_items - Q1_items
    limite_inferior_items = Q1_items - 1.5 * IQR_items
    limite_superior_items = Q3_items + 1.5 * IQR_items

    outliers_items = df_transacciones[
        (df_transacciones['items'] < limite_inferior_items) |
        (df_transacciones['items'] > limite_superior_items)
    ]

    print(f"Outliers en items (método IQR):")
    print(f"  Total outliers: {len(outliers_items):,} ({len(outliers_items)/len(df_transacciones)*100:.2f}%)")
    print(f"  Rango normal esperado: [{limite_inferior_items:.1f}, {limite_superior_items:.1f}]")
    print(f"  Valor mínimo observado: {df_transacciones['items'].min()}")
    print(f"  Valor máximo observado: {df_transacciones['items'].max()}")

    return outliers_items, limite_inferior_items, limite_superior_items


def analisis_calidad_completo(df_cliente, df_productos, df_transacciones):
    """
    Realiza un análisis completo de calidad de datos.

    Args:
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos
        df_transacciones: DataFrame de transacciones

    Returns:
        dict: Resumen completo del análisis
    """
    print("=" * 60)
    print("Análisis de calidad de los datos")
    print("=" * 60)

    datasets = {
        'Clientes': df_cliente,
        'Productos': df_productos,
        'Transacciones': df_transacciones
    }

    # Ejecutar todas las verificaciones
    total_nulos_global, _ = verificar_valores_nulos(datasets)
    clientes_faltantes, productos_faltantes, clientes_sin_compras, productos_sin_ventas = verificar_integridad_referencial(
        df_transacciones, df_cliente, df_productos
    )
    items_negativos, tamaños_invalidos, entregas_negativas = detectar_inconsistencias(
        df_transacciones, df_productos, df_cliente
    )
    coordenadas_validas, coordenadas_invalidas = validar_coordenadas_geograficas(df_cliente)
    duplicados = verificar_duplicados(df_transacciones, df_cliente, df_productos)
    fechas_futuras, fechas_antiguas, rango_dias = validar_coherencia_temporal(df_transacciones)
    outliers_items, _, _ = detectar_outliers(df_transacciones)

    # Resumen
    print(f"\n8. Resumen de calidad de datos")
    print("=" * 60)

    total_problemas = (
        total_nulos_global +
        len(clientes_faltantes) +
        len(productos_faltantes) +
        len(items_negativos) +
        len(tamaños_invalidos) +
        len(entregas_negativas) +
        coordenadas_invalidas +
        duplicados['duplicados_clientes_id'] +
        duplicados['duplicados_productos_id'] +
        duplicados['duplicados_transacciones_completos'] +
        len(fechas_futuras) +
        len(fechas_antiguas)
    )

    print(f"\nProblemas críticos detectados:")
    print(f"  - Valores nulos: {total_nulos_global}")
    print(f"  - Integridad referencial (clientes): {len(clientes_faltantes)}")
    print(f"  - Integridad referencial (productos): {len(productos_faltantes)}")
    print(f"  - Items negativos: {len(items_negativos)}")
    print(f"  - Tamaños inválidos: {len(tamaños_invalidos)}")
    print(f"  - Entregas negativas: {len(entregas_negativas)}")
    print(f"  - Coordenadas inválidas: {coordenadas_invalidas}")
    print(f"  - Duplicados (clientes): {duplicados['duplicados_clientes_id']}")
    print(f"  - Duplicados (productos): {duplicados['duplicados_productos_id']}")
    print(f"  - Duplicados (transacciones): {duplicados['duplicados_transacciones_completos']}")
    print(f"  - Fechas fuera de rango: {len(fechas_futuras) + len(fechas_antiguas)}")
    print(f"\nTotal de problemas críticos: {total_problemas}")

    if total_problemas == 0:
        print("\nEstado: Los datos están en buen estado tras el preprocesamiento")
    else:
        print(f"\nAdvertencia: Se encontraron {total_problemas} problemas que requieren atención")

    return {
        'total_problemas': total_problemas,
        'clientes_sin_compras': len(clientes_sin_compras),
        'productos_sin_ventas': len(productos_sin_ventas),
        'items_negativos': len(items_negativos),
        'outliers': len(outliers_items)
    }
