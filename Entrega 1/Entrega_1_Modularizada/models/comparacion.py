"""
Comparación de múltiples modelos de clasificación.
"""

import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def crear_clasificadores(preprocessor, random_state=42):
    """
    Crea diccionario con diferentes clasificadores para comparar.

    Args:
        preprocessor: Pipeline de preprocesamiento
        random_state: Seed para reproducibilidad

    Returns:
        dict: Diccionario de modelos
    """
    modelos = {
        'KNN': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='minkowski',
                n_jobs=-1
            ))
        ]),

        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50,
                class_weight='balanced',
                random_state=random_state
            ))
        ]),

        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=25,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ))
        ]),

        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=random_state,
                max_iter=1000
            ))
        ]),

        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.39,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            ))
        ]),

        'LightGBM': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            ))
        ])
    }

    return modelos


def entrenar_y_evaluar_modelo(nombre, modelo, X_train, y_train, X_val, y_val):
    """
    Entrena y evalúa un modelo.

    Args:
        nombre: Nombre del modelo
        modelo: Modelo a entrenar
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación

    Returns:
        dict: Resultados del modelo
    """
    print(f"\nEntrenando {nombre}...")

    # Entrenar
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    # Predecir
    y_pred = modelo.predict(X_val)
    y_pred_proba = modelo.predict_proba(X_val)[:, 1]

    # Métricas
    resultados = {
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'F1-Score': f1_score(y_val, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_val, y_pred_proba),
        'Tiempo (s)': tiempo
    }

    print(f"  F1-Score: {resultados['F1-Score']:.4f} | ROC-AUC: {resultados['ROC-AUC']:.4f} | Tiempo: {tiempo:.2f}s")

    return resultados, modelo


def comparar_modelos(preprocessor, X_train, y_train, X_val, y_val, random_state=42):
    """
    Compara múltiples modelos de clasificación.

    Args:
        preprocessor: Pipeline de preprocesamiento
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        random_state: Seed para reproducibilidad

    Returns:
        tuple: (df_resultados, modelos_entrenados)
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS DE CLASIFICACIÓN")
    print("="*80)

    # Crear modelos
    modelos = crear_clasificadores(preprocessor, random_state)

    resultados_lista = []
    modelos_entrenados = {}

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        resultados, modelo_entrenado = entrenar_y_evaluar_modelo(
            nombre, modelo, X_train, y_train, X_val, y_val
        )
        resultados_lista.append(resultados)
        modelos_entrenados[nombre] = modelo_entrenado

    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados_lista)
    df_resultados = df_resultados.sort_values('F1-Score', ascending=False)

    # Mostrar tabla de resultados
    print("\n" + "="*80)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("="*80)
    print(df_resultados.to_string(index=False))

    # Mostrar el mejor modelo
    mejor_modelo = df_resultados.iloc[0]
    print(f"\n🏆 Mejor modelo: {mejor_modelo['Modelo']}")
    print(f"   F1-Score: {mejor_modelo['F1-Score']:.4f}")
    print(f"   ROC-AUC: {mejor_modelo['ROC-AUC']:.4f}")

    return df_resultados, modelos_entrenados


def obtener_mejor_modelo(df_resultados, modelos_entrenados, metrica='F1-Score'):
    """
    Obtiene el mejor modelo según una métrica.

    Args:
        df_resultados: DataFrame con resultados
        modelos_entrenados: Diccionario con modelos entrenados
        metrica: Métrica a usar para seleccionar el mejor

    Returns:
        tuple: (nombre_mejor, modelo_mejor)
    """
    mejor_fila = df_resultados.sort_values(metrica, ascending=False).iloc[0]
    nombre_mejor = mejor_fila['Modelo']
    modelo_mejor = modelos_entrenados[nombre_mejor]

    return nombre_mejor, modelo_mejor
