"""
Funciones para limpieza de datos.
"""

import pandas as pd
import numpy as np


def eliminar_duplicados(df_transacciones, df_cliente, df_productos):
    """
    Elimina registros duplicados de los datasets.

    Args:
        df_transacciones: DataFrame de transacciones
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos

    Returns:
        tuple: (df_transacciones_filtrado, df_cliente_filtrado, df_productos_filtrado)
    """
    print("Eliminando duplicados...")

    # Transacciones
    inicial_trans = len(df_transacciones)
    df_transacciones_filtrado = df_transacciones.drop_duplicates()
    eliminados_trans = inicial_trans - len(df_transacciones_filtrado)
    print(f"Transacciones duplicadas eliminadas: {eliminados_trans}")

    # Clientes
    inicial_clientes = len(df_cliente)
    df_clientes_filtrado = df_cliente.drop_duplicates()
    eliminados_clientes = inicial_clientes - len(df_clientes_filtrado)
    print(f"Clientes duplicados eliminados: {eliminados_clientes}")

    # Productos
    inicial_productos = len(df_productos)
    df_productos_filtrado = df_productos.drop_duplicates()
    eliminados_productos = inicial_productos - len(df_productos_filtrado)
    print(f"Productos duplicados eliminados: {eliminados_productos}")

    return df_transacciones_filtrado, df_clientes_filtrado, df_productos_filtrado


def eliminar_items_negativos(df_transacciones):
    """
    Elimina transacciones con items negativos.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: DataFrame limpio sin items negativos
    """
    print(f"\nRegistros iniciales en transacciones: {len(df_transacciones):,}")
    df_transacciones_clean = df_transacciones[df_transacciones['items'] >= 0].copy()
    items_negativos_eliminados = len(df_transacciones) - len(df_transacciones_clean)
    print(f"Transacciones con items negativos eliminadas: {items_negativos_eliminados:,}")
    print(f"Registros restantes: {len(df_transacciones_clean):,}")

    return df_transacciones_clean


def eliminar_clientes_sin_transacciones(df_cliente, df_transacciones):
    """
    Elimina clientes que no tienen transacciones.

    Args:
        df_cliente: DataFrame de clientes
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: DataFrame de clientes filtrado
    """
    clientes_con_transacciones = df_transacciones['customer_id'].unique()
    print(f"\nClientes iniciales en maestro: {len(df_cliente):,}")
    df_cliente_clean = df_cliente[df_cliente['customer_id'].isin(clientes_con_transacciones)].copy()
    clientes_eliminados = len(df_cliente) - len(df_cliente_clean)
    print(f"Clientes sin transacciones eliminados: {clientes_eliminados}")
    print(f"Clientes restantes: {len(df_cliente_clean):,}")

    return df_cliente_clean


def marcar_productos_sin_ventas(df_productos, df_transacciones):
    """
    Marca productos sin historial de ventas.

    Args:
        df_productos: DataFrame de productos
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: DataFrame de productos con columna 'tiene_historial_ventas'
    """
    productos_con_ventas = df_transacciones['product_id'].unique()
    df_productos_clean = df_productos.copy()
    df_productos_clean['tiene_historial_ventas'] = df_productos_clean['product_id'].isin(productos_con_ventas).astype(int)

    productos_sin_ventas = (df_productos_clean['tiene_historial_ventas'] == 0).sum()
    print(f"\nProductos sin historial de ventas: {productos_sin_ventas} ({productos_sin_ventas/len(df_productos_clean)*100:.1f}%)")
    print(f"Productos con historial de ventas: {len(productos_con_ventas)} ({len(productos_con_ventas)/len(df_productos_clean)*100:.1f}%)")

    return df_productos_clean


def consolidar_transacciones_por_dia(df_transacciones):
    """
    Consolida transacciones múltiples del mismo día para el mismo cliente-producto.

    Args:
        df_transacciones: DataFrame de transacciones

    Returns:
        pd.DataFrame: DataFrame consolidado
    """
    print("\nConsolidando transacciones múltiples del mismo día")
    print("-" * 80)

    df_transacciones_clean = df_transacciones.copy()
    df_transacciones_clean['fecha_dt'] = pd.to_datetime(df_transacciones_clean['purchase_date'])
    transacciones_iniciales = len(df_transacciones_clean)

    # Agregar por cliente-producto-fecha sumando items
    df_transacciones_agg = df_transacciones_clean.groupby(
        ['customer_id', 'product_id', 'fecha_dt'],
        as_index=False
    ).agg({
        'items': 'sum',
        'order_id': 'nunique'
    })

    print(f"Transacciones antes de agregación: {transacciones_iniciales:,}")
    print(f"Transacciones después de agregación: {len(df_transacciones_agg):,}")
    print(f"Reducción: {transacciones_iniciales - len(df_transacciones_agg):,} registros")

    return df_transacciones_agg


def aplicar_limpieza_completa(df_cliente, df_productos, df_transacciones):
    """
    Aplica todas las operaciones de limpieza en el orden correcto.

    Args:
        df_cliente: DataFrame de clientes
        df_productos: DataFrame de productos
        df_transacciones: DataFrame de transacciones

    Returns:
        tuple: (df_cliente_clean, df_productos_clean, df_transacciones_clean)
    """
    print("=" * 60)
    print("APLICANDO LIMPIEZA COMPLETA DE DATOS")
    print("=" * 60)

    # 1. Eliminar items negativos
    print("\n1. Eliminando items negativos")
    df_transacciones_clean = eliminar_items_negativos(df_transacciones)

    # 2. Eliminar clientes sin transacciones
    print("\n2. Eliminando clientes sin transacciones")
    df_cliente_clean = eliminar_clientes_sin_transacciones(df_cliente, df_transacciones_clean)

    # 3. Marcar productos sin ventas
    print("\n3. Marcando productos sin historial de ventas")
    df_productos_clean = marcar_productos_sin_ventas(df_productos, df_transacciones_clean)

    # 4. Consolidar transacciones
    df_transacciones_agg = consolidar_transacciones_por_dia(df_transacciones_clean)

    print("\n" + "=" * 60)
    print("LIMPIEZA COMPLETA")
    print("=" * 60)

    return df_cliente_clean, df_productos_clean, df_transacciones_agg
