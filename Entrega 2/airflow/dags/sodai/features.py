from __future__ import annotations

import numpy as np
import pandas as pd


def limpiar_datos(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    transacciones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    limpia datos aplicando correcciones identificadas en el eda

    - elimina transacciones con items negativos
    - elimina clientes sin transacciones
    - marca productos sin historial de ventas
    - aplica correcciones geograficas
    """

    # eliminar transacciones con items negativos
    transacciones_clean = transacciones[transacciones['items'] >= 0].copy()

    # eliminar clientes sin transacciones
    clientes_con_transacciones = transacciones_clean['customer_id'].unique()
    clientes_clean = clientes[clientes['customer_id'].isin(clientes_con_transacciones)].copy()

    # marcar productos sin historial de ventas
    productos_con_ventas = transacciones_clean['product_id'].unique()
    productos_clean = productos.copy()
    productos_clean['tiene_historial_ventas'] = productos_clean['product_id'].isin(productos_con_ventas).astype(int)

    return clientes_clean, productos_clean, transacciones_clean


def agregar_transacciones_diarias(transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    consolida transacciones multiples del mismo cliente-producto-dia
    sumando items y contando ordenes
    """

    transacciones['fecha_dt'] = pd.to_datetime(transacciones['purchase_date'])

    transacciones_agg = transacciones.groupby(
        ['customer_id', 'product_id', 'fecha_dt'],
        as_index=False
    ).agg({
        'items': 'sum',
        'order_id': 'nunique'
    })

    # agregar columna de semana para modelado
    transacciones_agg['semana'] = transacciones_agg['fecha_dt'].dt.to_period('W-MON')

    return transacciones_agg


def crear_variable_objetivo(transacciones_agg: pd.DataFrame) -> pd.DataFrame:
    """
    crea variable objetivo binaria: compra_siguiente_semana

    para cada combinacion cliente-producto-semana, verifica si hubo compra
    en la semana siguiente

    retorna df con columnas: customer_id, product_id, semana_actual, target
    """

    # agregar a nivel semanal
    combinaciones_observadas = transacciones_agg.groupby(
        ['customer_id', 'product_id', 'semana'],
        as_index=False
    ).agg({
        'items': 'sum',
        'fecha_dt': 'min'
    })

    combinaciones_observadas['semana_actual'] = combinaciones_observadas['semana']
    combinaciones_observadas['semana_siguiente'] = combinaciones_observadas['semana_actual'].apply(lambda x: x + 1)

    # crear conjunto de compras realizadas
    compras_realizadas = set(
        zip(
            combinaciones_observadas['customer_id'],
            combinaciones_observadas['product_id'],
            combinaciones_observadas['semana_actual']
        )
    )

    # verificar compra en semana siguiente
    def verificar_compra_siguiente(row):
        tupla = (row['customer_id'], row['product_id'], row['semana_siguiente'])
        return 1 if tupla in compras_realizadas else 0

    combinaciones_observadas['compra_siguiente_semana'] = combinaciones_observadas.apply(verificar_compra_siguiente, axis=1)

    # renombrar para claridad
    combinaciones_observadas = combinaciones_observadas.rename(columns={'fecha_dt': 'fecha_semana'})

    return combinaciones_observadas


def crear_features_cliente(
    df_objetivo: pd.DataFrame,
    transacciones_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    crea features de comportamiento historico del cliente (rfm)

    - total_ordenes_global: numero total de ordenes del cliente
    - productos_unicos_global: productos distintos comprados
    - items_totales_global: suma de items comprados
    - dias_desde_primera_compra: dias desde primera transaccion
    - dias_desde_ultima_compra: dias desde ultima transaccion
    - frecuencia_compra_diaria: ordenes por dia
    - diversidad_productos: ratio productos/ordenes
    """

    # calcular estadisticas globales por cliente
    cliente_stats = transacciones_agg.groupby('customer_id').agg({
        'fecha_dt': ['min', 'max', 'count'],
        'product_id': 'nunique',
        'items': ['sum', 'mean']
    }).reset_index()

    cliente_stats.columns = [
        'customer_id', 'primera_compra_global', 'ultima_compra_global',
        'total_ordenes_global', 'productos_unicos_global',
        'items_totales_global', 'items_promedio_global'
    ]

    # merge con df objetivo
    df_fe = df_objetivo.merge(cliente_stats, on='customer_id', how='left')

    # calcular features derivados
    df_fe['dias_desde_primera_compra'] = (
        df_fe['fecha_semana'] - df_fe['primera_compra_global']
    ).dt.days.fillna(0).clip(lower=0)

    df_fe['dias_desde_ultima_compra'] = (
        df_fe['fecha_semana'] - df_fe['ultima_compra_global']
    ).dt.days.fillna(999).clip(lower=0)

    df_fe['frecuencia_compra_diaria'] = df_fe['total_ordenes_global'] / (df_fe['dias_desde_primera_compra'] + 1)
    df_fe['diversidad_productos'] = df_fe['productos_unicos_global'] / (df_fe['total_ordenes_global'] + 1)

    return df_fe


def crear_features_producto(
    df_fe: pd.DataFrame,
    transacciones_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    crea features de popularidad y ventas del producto

    - total_ventas_global: numero de transacciones del producto
    - clientes_unicos_global: clientes que compraron el producto
    - items_vendidos_global: suma de items vendidos
    - popularidad_rank: ranking de popularidad
    """

    producto_stats = transacciones_agg.groupby('product_id').agg({
        'fecha_dt': 'count',
        'customer_id': 'nunique',
        'items': 'sum'
    }).reset_index()

    producto_stats.columns = ['product_id', 'total_ventas_global', 'clientes_unicos_global', 'items_vendidos_global']
    producto_stats['popularidad_rank'] = producto_stats['items_vendidos_global'].rank(ascending=False, method='dense').astype(int)

    df_fe = df_fe.merge(producto_stats, on='product_id', how='left')

    return df_fe


def crear_features_interaccion(
    df_fe: pd.DataFrame,
    transacciones_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    crea features de interaccion cliente-producto

    - compro_este_producto_antes: flag binario
    - veces_comprado_global: cuantas veces compro este producto
    - dias_desde_ultima_compra_producto: dias desde ultima compra de este producto
    - items_promedio_producto: promedio de items por compra de este producto
    """

    interaccion_stats = transacciones_agg.groupby(['customer_id', 'product_id']).agg({
        'fecha_dt': ['count', 'max'],
        'items': 'mean'
    }).reset_index()

    interaccion_stats.columns = [
        'customer_id', 'product_id', 'veces_comprado_global',
        'ultima_compra_producto_global', 'items_promedio_producto'
    ]

    df_fe = df_fe.merge(interaccion_stats, on=['customer_id', 'product_id'], how='left')

    # crear flag binario
    df_fe['compro_este_producto_antes'] = (df_fe['veces_comprado_global'] > 0).astype(int)

    # calcular dias desde ultima compra del producto
    df_fe['dias_desde_ultima_compra_producto'] = (
        df_fe['fecha_semana'] - df_fe['ultima_compra_producto_global']
    ).dt.days.fillna(999).clip(lower=0)

    # rellenar nulos
    df_fe['veces_comprado_global'] = df_fe['veces_comprado_global'].fillna(0)
    df_fe['items_promedio_producto'] = df_fe['items_promedio_producto'].fillna(0)

    return df_fe


def crear_features_temporales(df_fe: pd.DataFrame) -> pd.DataFrame:
    """
    crea features temporales y estacionales

    - dia_semana, mes, trimestre, semana_del_año
    - indicadores binarios: fin_semana, lunes_jueves, temporada_alta/baja
    - encoding ciclico: mes_sin, mes_cos, dia_semana_sin, dia_semana_cos
    """

    df_fe['dia_semana'] = df_fe['fecha_semana'].dt.dayofweek
    df_fe['mes'] = df_fe['fecha_semana'].dt.month
    df_fe['trimestre'] = df_fe['fecha_semana'].dt.quarter
    df_fe['semana_del_año'] = df_fe['fecha_semana'].dt.isocalendar().week.astype(int)

    # indicadores binarios
    df_fe['es_fin_semana'] = (df_fe['dia_semana'] >= 5).astype(int)
    df_fe['es_lunes_jueves'] = df_fe['dia_semana'].isin([0, 3]).astype(int)
    df_fe['es_temporada_alta'] = df_fe['mes'].isin([11, 12]).astype(int)
    df_fe['es_temporada_baja'] = df_fe['mes'].isin([5, 6, 7]).astype(int)

    # encoding ciclico
    df_fe['mes_sin'] = np.sin(2 * np.pi * df_fe['mes'] / 12)
    df_fe['mes_cos'] = np.cos(2 * np.pi * df_fe['mes'] / 12)
    df_fe['dia_semana_sin'] = np.sin(2 * np.pi * df_fe['dia_semana'] / 7)
    df_fe['dia_semana_cos'] = np.cos(2 * np.pi * df_fe['dia_semana'] / 7)

    return df_fe


def aplicar_transformaciones_producto(df_fe: pd.DataFrame) -> pd.DataFrame:
    """
    aplica transformaciones sobre variables de producto

    - size_log1p: transformacion logaritmica de size
    - size_categoria: categorizacion de size en 5 grupos
    - segment_ordinal: encoding ordinal de segment
    - distancia_al_centro: distancia geografica al centroide
    """

    # transformacion logaritmica
    df_fe['size_log1p'] = np.log1p(df_fe['size'])

    # categorizacion
    df_fe['size_categoria'] = pd.cut(
        df_fe['size'],
        bins=[0, 0.33, 0.66, 1.5, 3.0, np.inf],
        labels=['individual', 'personal', 'familiar_pequeno', 'familiar_grande', 'granel']
    )

    # encoding ordinal
    segment_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'PREMIUM': 3}
    df_fe['segment_ordinal'] = df_fe['segment'].map(segment_order)

    # distancia geografica
    centroid_x = df_fe['X'].mean()
    centroid_y = df_fe['Y'].mean()
    df_fe['distancia_al_centro'] = np.sqrt(
        (df_fe['X'] - centroid_x)**2 + (df_fe['Y'] - centroid_y)**2
    )

    return df_fe


def construir_df_modelado(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    transacciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    funcion principal que construye el dataset de modelado con todas las transformaciones

    pasos:
    1. limpia datos
    2. agrega transacciones diarias
    3. crea variable objetivo compra_siguiente_semana
    4. merge con datos de clientes y productos
    5. crea features de cliente, producto, interaccion y temporales
    6. aplica transformaciones de producto

    retorna dataframe listo para modelado con target y features
    """

    print("construyendo dataset de modelado...")

    # paso 1: limpieza
    print("  - limpiando datos...")
    clientes_clean, productos_clean, transacciones_clean = limpiar_datos(clientes, productos, transacciones)

    # paso 2: agregacion diaria
    print("  - agregando transacciones diarias...")
    transacciones_agg = agregar_transacciones_diarias(transacciones_clean)

    # paso 3: crear variable objetivo
    print("  - creando variable objetivo compra_siguiente_semana...")
    df_objetivo = crear_variable_objetivo(transacciones_agg)

    # paso 4: merge con clientes y productos
    print("  - uniendo con datos de clientes y productos...")
    df_modelado = df_objetivo.merge(
        clientes_clean[['customer_id', 'customer_type', 'X', 'Y', 'num_deliver_per_week']],
        on='customer_id',
        how='left'
    )

    df_modelado = df_modelado.merge(
        productos_clean[['product_id', 'brand', 'category', 'sub_category', 'segment', 'package', 'size']],
        on='product_id',
        how='left'
    )

    # paso 5: crear features
    print("  - creando features de cliente...")
    df_modelado = crear_features_cliente(df_modelado, transacciones_agg)

    print("  - creando features de producto...")
    df_modelado = crear_features_producto(df_modelado, transacciones_agg)

    print("  - creando features de interaccion...")
    df_modelado = crear_features_interaccion(df_modelado, transacciones_agg)

    print("  - creando features temporales...")
    df_modelado = crear_features_temporales(df_modelado)

    # paso 6: transformaciones de producto
    print("  - aplicando transformaciones de producto...")
    df_modelado = aplicar_transformaciones_producto(df_modelado)

    # eliminar columnas auxiliares
    columnas_eliminar = [
        'primera_compra_global', 'ultima_compra_global',
        'ultima_compra_producto_global', 'semana_siguiente'
    ]
    df_modelado = df_modelado.drop(columns=[col for col in columnas_eliminar if col in df_modelado.columns])

    print(f"\ndataset de modelado creado: {df_modelado.shape}")
    print(f"balance del target: {df_modelado['compra_siguiente_semana'].mean():.2%}")

    return df_modelado
