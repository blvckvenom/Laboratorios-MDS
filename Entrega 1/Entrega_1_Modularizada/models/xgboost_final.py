"""
Utilidades para el modelo XGBoost final.
"""

import pickle
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def guardar_modelo(modelo, ruta, nombre='modelo_final'):
    """
    Guarda el modelo entrenado.

    Args:
        modelo: Modelo a guardar
        ruta: Directorio donde guardar
        nombre: Nombre del archivo

    Returns:
        Path: Ruta completa del archivo guardado
    """
    ruta_dir = Path(ruta)
    ruta_dir.mkdir(parents=True, exist_ok=True)

    archivo = ruta_dir / f"{nombre}.pkl"

    with open(archivo, 'wb') as f:
        pickle.dump(modelo, f)

    print(f"\n✓ Modelo guardado en: {archivo}")

    return archivo


def cargar_modelo(archivo):
    """
    Carga un modelo guardado.

    Args:
        archivo: Ruta del archivo del modelo

    Returns:
        Modelo cargado
    """
    with open(archivo, 'rb') as f:
        modelo = pickle.load(f)

    print(f"\n✓ Modelo cargado desde: {archivo}")

    return modelo


def guardar_parametros(parametros, ruta, nombre='best_params'):
    """
    Guarda los parámetros del modelo.

    Args:
        parametros: Diccionario de parámetros
        ruta: Directorio donde guardar
        nombre: Nombre del archivo

    Returns:
        Path: Ruta completa del archivo guardado
    """
    ruta_dir = Path(ruta)
    ruta_dir.mkdir(parents=True, exist_ok=True)

    archivo = ruta_dir / f"{nombre}.json"

    with open(archivo, 'w') as f:
        json.dump(parametros, f, indent=4)

    print(f"✓ Parámetros guardados en: {archivo}")

    return archivo


def mostrar_reporte_clasificacion(y_true, y_pred):
    """
    Muestra el reporte de clasificación.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
    """
    print("\n" + "-"*90)
    print("CLASSIFICATION REPORT")
    print("-"*90)
    print(classification_report(y_true, y_pred, target_names=['No Compra (0)', 'Compra (1)']))


def mostrar_matriz_confusion(y_true, y_pred):
    """
    Muestra la matriz de confusión.

    Args:
        y_true: Valores reales
        y_pred: Predicciones
    """
    cm = confusion_matrix(y_true, y_pred)

    print("-"*90)
    print("MATRIZ DE CONFUSIÓN")
    print("-"*90)
    print(f"\n              Predicho")
    print(f"            No(0)  Si(1)")
    print(f"Real No(0) {cm[0,0]:>6,} {cm[0,1]:>6,}")
    print(f"     Si(1) {cm[1,0]:>6,} {cm[1,1]:>6,}")
    print()


def predecir_con_modelo(modelo, X):
    """
    Realiza predicciones con el modelo.

    Args:
        modelo: Modelo entrenado
        X: Features para predecir

    Returns:
        tuple: (predicciones, probabilidades)
    """
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    return predicciones, probabilidades


def evaluar_modelo_final(modelo, X_test, y_test):
    """
    Evalúa el modelo final en el conjunto de test.

    Args:
        modelo: Modelo entrenado
        X_test: Features de test
        y_test: Target de test

    Returns:
        tuple: (predicciones, probabilidades)
    """
    print("\n" + "="*90)
    print("EVALUACIÓN DEL MODELO FINAL EN TEST")
    print("="*90)

    predicciones, probabilidades = predecir_con_modelo(modelo, X_test)

    # Mostrar reportes
    mostrar_reporte_clasificacion(y_test, predicciones)
    mostrar_matriz_confusion(y_test, predicciones)

    return predicciones, probabilidades
