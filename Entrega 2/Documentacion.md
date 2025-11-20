# Pipeline de Airflow – SodAI (Documentación)

Este bloque describe el pipeline de `Airflow` implementado en esta entrega del laboratorio, incluyendo la estructura del DAG, la función de cada tarea, el flujo completo del pipeline y la lógica para trabajar con nuevos datos, *drift* y reentrenamiento del modelo.

**Última Ejecución Exitosa**: `manual__2025-11-19T23:49:04+00:00`
**Estado**: SUCCESS
**Duración**: ~7 minutos
**Métricas del Modelo**: F1=0.7190, ROC-AUC=0.8438, Recall=0.8855

---

## 1. Descripción general del DAG

El DAG principal se llama:

- **`sodai_training_and_scoring`**

Este DAG orquesta un pipeline completo de MLOps que incluye:

1. Construir el dataset de modelamiento a partir de los datos crudos (`clientes`, `productos`, `transacciones`).
2. Detectar automáticamente *data drift* comparando con datos de referencia anteriores.
3. Reentrenar el modelo solo si se detecta drift (flujo condicional).
4. Optimizar hiperparámetros con Optuna y registrar experimentos en MLflow.
5. Evaluar el modelo y generar interpretabilidad con SHAP.
6. Generar predicciones usando el modelo más reciente.

El DAG está definido en el archivo:

- `dags/sodai_pipeline_dag.py`

y utiliza funciones auxiliares dentro del paquete:

- `dags/sodai/`  
  (por ejemplo `data_io.py`, `features.py`, `train.py`, `evaluate.py`, `drift.py`, `predict.py`).

---

## 2. Descripción de cada tarea del DAG

A continuación se describe la funcionalidad de cada tarea y cómo se relacionan entre sí.

| Task ID               | Función interna          | Descripción                                                                                                   | Depende de             |
|-----------------------|--------------------------|---------------------------------------------------------------------------------------------------------------|------------------------|
| `build_dataset`       | `_build_dataset`         | Carga los datos crudos (`clientes.parquet`, `productos.parquet`, `transacciones.parquet`) y construye el dataset de modelamiento con ingeniería de features avanzada (52 features totales: 27 numéricas, 10 categóricas, 5 binarias). El resultado se guarda como `df_modelado.parquet` en `artifacts/datasets/` y se pasa la ruta por XCom. | —                      |
| `train_model`         | `_train_model`           | Entrena un modelo XGBoost clasificador binario con optimización de hiperparámetros usando Optuna (10 trials). Registra experimentos en MLflow y guarda el mejor modelo en `artifacts/models/sodai_model.joblib` junto con métricas, feature importance y categorical mappings en `artifacts/metrics/sodai_metrics.json`. | `build_dataset`        |
| `evaluate_model`      | `_evaluate_model`        | Evalúa el modelo entrenado sobre el dataset completo calculando métricas de clasificación (accuracy, precision, recall, F1-score, ROC-AUC). Genera análisis de interpretabilidad con SHAP (summary plots, feature importance plots, dependence plots) guardados en `artifacts/shap/`. | `train_model`          |
| `check_drift`         | `_check_drift`           | Detecta data drift comparando distribuciones entre el dataset de referencia y el actual usando PSI (Population Stability Index) y pruebas Kolmogorov-Smirnov. Genera reporte detallado en `artifacts/drift/drift_report.json` con scores por columna y visualizaciones. | `evaluate_model`       |
| `generate_predictions`| `_generate_predictions`  | Genera predicciones binarias (0/1) y probabilidades (0-1) usando el modelo XGBoost entrenado. Agrega columnas `prediction` y `prediction_proba` al dataset y guarda el resultado en `artifacts/predictions/predicciones.parquet` con estadísticas de distribución. | `check_drift`          |

Las tareas se ejecutan de forma lineal con la siguiente cadena de dependencias:

build_dataset → train_model → evaluate_model → check_drift → generate_predictions

--- 

## 3. Diagrama de flujo del pipeline completo  

A continuación se muestra un diagrama de flujo lógico del pipeline (en notación tipo diagrama de bloques). Si se renderiza con Mermaid, puede visualizarse gráficamente:

flowchart LR
    A[Datos crudos<br/>clientes / productos / transacciones] --> B[build_dataset<br/>construir_df_modelado]
    B --> C[train_model<br/>entrenar_modelo]
    C --> D[evaluate_model<br/>evaluar_modelo]
    D --> E[check_drift<br/>calcular_drift]
    E --> F[generate_predictions<br/>generar_predicciones]

    C --> M[Modelo entrenado<br/>sodai_model.joblib]
    D --> N[Métricas<br/>sodai_metrics.json]
    E --> O[Reporte drift<br/>drift_report.json]
    F --> P[Predicciones<br/>predicciones.parquet]

Este diagrama resume:

La entrada del pipeline: los archivos *.parquet de datos crudos en la carpeta data/.

El paso intermedio: construcción del dataset de modelamiento y entrenamiento del modelo.

Los artefactos generados: modelo, métricas, reporte de drift y predicciones.

--- 


## 4. Representación visual del DAG en la interfaz de Airflow

En la interfaz web de Airflow (http://localhost:8080), el DAG se visualiza en la vista Graph, mostrando las tareas en secuencia:

build_dataset → train_model → evaluate_model → check_drift → generate_predictions

Se incluye un pantallazo de esa vista, por ejemplo:
![DAG sodai_training_and_scoring en Airflow](.\ssairflow.png)

Esta captura de pantalla deja evidencia de:

La estructura del DAG.

El orden de ejecución de las tareas.

El estado de cada tarea (por ejemplo success/failed durante pruebas).

--- 


## 5. Lógica para futuros datos, detección de drift y reentrenamiento
### 5.1 Integración de futuros datos

El pipeline está diseñado para trabajar con nuevos conjuntos de datos sin modificar el código del DAG:

Los datos crudos se leen desde rutas configuradas en config.py:

CLIENTES_PATH = RAW_DIR / "clientes.parquet"
PRODUCTOS_PATH = RAW_DIR / "productos.parquet"
TRANSACCIONES_PATH = RAW_DIR / "transacciones.parquet"

RAW_DIR apunta a la carpeta de datos (/opt/airflow/data dentro del contenedor), que corresponde a airflow/data/ en la máquina local.

Para integrar un nuevo dataset basta con:

Generar nuevos archivos clientes.parquet, productos.parquet y/o transacciones.parquet (por ejemplo filtrando una semana distinta del histórico).

Reemplazar estos archivos en la carpeta airflow/data/.

Disparar nuevamente el DAG sodai_training_and_scoring.

En cada ejecución:

build_dataset vuelve a construir desde cero el dataset de modelamiento con los datos actuales.

train_model vuelve a entrenar el modelo usando ese dataset actualizado.

De esta manera, el reentrenamiento está acoplado directamente a la llegada de nuevos datos sin necesidad de cambiar el código.

### 5.2 Reentrenamiento del modelo

La tarea train_model:

1. Carga el dataset modelado (df_modelado.parquet) desde artifacts/datasets/.

2. Define 52 features de entrenamiento organizadas en:
   - **27 features numéricas**: agregaciones de cliente (total_ordenes_global, dias_desde_primera_compra, frecuencia_compra_diaria), agregaciones de producto (total_ventas_global, popularidad_rank), features de interacción (veces_comprado_global, dias_desde_ultima_compra_producto), coordenadas geográficas (X, Y, distancia_al_centro), temporales cíclicas (mes_sin, mes_cos, dia_semana_sin, dia_semana_cos), y básicas (size, size_log1p, segment_ordinal, semana_del_año, num_deliver_per_week)
   - **10 features categóricas**: category, sub_category, package, size_categoria, trimestre, dia_semana, mes, segment, brand, customer_type
   - **5 features binarias**: compro_este_producto_antes, es_fin_semana, es_lunes_jueves, es_temporada_alta, es_temporada_baja

3. Usa la columna binaria `compro` como variable objetivo (clasificación binaria: 0=no compró, 1=compró).

4. Optimiza hiperparámetros con Optuna:
   - 10 trials de búsqueda bayesiana
   - Validación cruzada 3-fold para seleccionar el mejor conjunto de parámetros
   - Registra todos los experimentos en MLflow con tracking de métricas y artifacts

5. Entrena un modelo XGBoost clasificador con:
   - Soporte nativo para features categóricas (enable_categorical=True)
   - Hiperparámetros optimizados: n_estimators, max_depth, learning_rate, min_child_weight, gamma, subsample, colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight
   - División train/test 80/20 estratificada

6. Guarda:
   - El modelo entrenado en `artifacts/models/sodai_model.joblib`
   - Métricas completas en `artifacts/metrics/sodai_metrics.json`: accuracy, precision, recall, f1_score, roc_auc, feature importance, categorical mappings, hiperparámetros óptimos, MLflow run ID

Cada vez que se ejecuta el DAG con datos nuevos:

- El modelo se reentrena automáticamente usando Optuna para optimizar hiperparámetros.
- Las métricas se recalculan y se registran en MLflow, permitiendo comparar experimentos.
- Los categorical mappings se actualizan con las categorías presentes en los nuevos datos.

### 5.3 Detección de data drift

La tarea check_drift implementa análisis estadístico riguroso para detectar cambios en las distribuciones:

1. Carga el dataset modelado actual y el de referencia (dataset anterior).

2. Selecciona las 27 columnas numéricas relevantes:
   - Agregaciones de cliente: total_ordenes_global, productos_unicos_global, items_totales_global, items_promedio_global, dias_desde_primera_compra, dias_desde_ultima_compra, frecuencia_compra_diaria, diversidad_productos
   - Agregaciones de producto: total_ventas_global, clientes_unicos_global, items_vendidos_global, popularidad_rank
   - Features de interacción: veces_comprado_global, dias_desde_ultima_compra_producto, items_promedio_producto
   - Features geográficas y básicas: X, Y, distancia_al_centro, size, size_log1p, segment_ordinal, num_deliver_per_week
   - Features temporales: mes_sin, mes_cos, dia_semana_sin, dia_semana_cos, semana_del_año

3. Aplica dos pruebas estadísticas complementarias:
   - **PSI (Population Stability Index)**: Mide el cambio en la distribución discretizando en bins. Umbrales: PSI < 0.1 (sin drift), 0.1-0.25 (drift moderado), > 0.25 (drift significativo)
   - **Kolmogorov-Smirnov (KS)**: Prueba no paramétrica que detecta diferencias en distribuciones continuas con p-value < 0.05 indicando drift significativo

4. Genera visualizaciones comparativas:
   - Histogramas antes/después para cada feature
   - Gráficos de distribución con KDE (Kernel Density Estimation)
   - Heatmap de drift scores por columna

5. Guarda reporte completo en `artifacts/drift/drift_report.json` con:
   - Score PSI por columna
   - Estadística KS y p-value por columna
   - Estadísticas descriptivas (media, std, min, max, percentiles)
   - Diagnóstico general: "no_drift", "drift_moderado" o "drift_significativo"
   - Timestamp y número de muestras comparadas

Este mecanismo permite:

- Monitorear automáticamente cambios en las distribuciones de entrada.
- Alertar cuando se detecta drift significativo que podría afectar el rendimiento del modelo.
- Tomar decisiones informadas sobre cuándo reentrenar el modelo.

### 5.4 Generación de predicciones con los datos más recientes

Finalmente, la tarea generate_predictions:

1. Carga el modelo XGBoost entrenado desde `artifacts/models/sodai_model.joblib`.

2. Carga el dataset actual (`df_modelado.parquet`) con las 52 features.

3. Convierte las 10 columnas categóricas a tipo 'category' para compatibilidad con XGBoost:
   - category, sub_category, package, size_categoria, trimestre, dia_semana, mes, segment, brand, customer_type

4. Genera predicciones usando las mismas 52 features del entrenamiento:
   - **Predicción binaria** (`prediction`): clase predicha (0=no comprará, 1=comprará)
   - **Probabilidad** (`prediction_proba`): score de probabilidad entre 0 y 1

5. Calcula estadísticas de las predicciones:
   - Conteo de predicciones positivas vs negativas
   - Distribución porcentual
   - Media y desviación estándar de las probabilidades

6. Guarda el resultado en `artifacts/predictions/predicciones.parquet` con:
   - Todas las columnas originales del dataset
   - Columna `prediction` con la clase predicha
   - Columna `prediction_proba` con la probabilidad
   - Metadatos de las features usadas

De este modo, cada ejecución del DAG con nuevos datos genera:

- Un modelo reentrenado con hiperparámetros optimizados por Optuna.
- Métricas de evaluación actualizadas y registradas en MLflow.
- Análisis de interpretabilidad con SHAP (summary plots, feature importance).
- Reporte de drift comparando con datos de referencia.
- Archivo de predicciones actualizado con clases y probabilidades.

Todo el pipeline se puede volver a correr de manera reproducible simplemente actualizando los archivos de datos en `airflow/data/` y disparando de nuevo el DAG desde la interfaz web de Airflow.

