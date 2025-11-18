# Pipeline de Airflow – SodAI (Documentación)

Este bloque describe el pipeline de `Airflow` implementado en esta entrega del laboratorio, incluyendo la estructura del DAG, la función de cada tarea, el flujo completo del pipeline y la lógica para trabajar con nuevos datos, *drift* y reentrenamiento del modelo.

---

## 1. Descripción general del DAG

El DAG principal se llama:

- **`sodai_training_and_scoring`**

Este DAG orquesta de forma secuencial las tareas necesarias para:

1. Construir el dataset de modelamiento a partir de los datos crudos (`clientes`, `productos`, `transacciones`).
2. Entrenar (o reentrenar) un modelo de ML con el dataset actual.
3. Evaluar el modelo entrenado y guardar métricas.
4. Revisar si existe *data drift* en las variables numéricas.
5. Generar predicciones usando el modelo más reciente.

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
| `build_dataset`       | `_build_dataset`         | Carga los datos crudos (`clientes.parquet`, `productos.parquet`, `transacciones.parquet`) y construye el dataset de modelamiento. El resultado se guarda como `df_modelado.parquet` en `artifacts/datasets/` y se pasa la ruta por XCom. | —                      |
| `train_model`         | `_train_model`           | Carga el dataset de modelamiento, separa features y variable objetivo, entrena un modelo de ML (RandomForestRegressor) y guarda el modelo en `artifacts/models/sodai_model.joblib` junto con métricas básicas en `artifacts/metrics/sodai_metrics.json`. | `build_dataset`        |
| `evaluate_model`      | `_evaluate_model`        | Carga el modelo guardado y el dataset de modelamiento, calcula métricas de evaluación (por ejemplo RMSE y R² sobre el conjunto actual) y las guarda en formato JSON. | `train_model`          |
| `check_drift`         | `_check_drift`           | Realiza un análisis de *data drift* sobre las columnas numéricas del dataset (por ejemplo `customer_id`, `product_id`, `order_id`, `items`, coordenadas, etc.), y guarda un reporte en `artifacts/drift/drift_report.json`. | `evaluate_model`       |
| `generate_predictions`| `_generate_predictions`  | Carga el modelo entrenado y el dataset actual, genera una columna `prediction` y guarda las predicciones en `artifacts/predictions/predicciones.parquet`. | `check_drift`          |

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

2. Define las features de entrenamiento (por ejemplo):

FEATURES = [
    "customer_id",
    "product_id",
    "order_id",
    "region_id",
    "zone_id",
    "Y",
    "X",
    "num_deliver_per_week",
    "num_visit_per_week",
    "size",
]


3. Usa una columna numérica como variable objetivo (por ejemplo items).

4. Entrena un modelo de regresión (Random Forest) con estos datos.

5. Guarda:

- El modelo en artifacts/models/sodai_model.joblib.

- Métricas de entrenamiento/validación en artifacts/metrics/sodai_metrics.json.

Cada vez que se ejecuta el DAG con datos nuevos:

- El modelo se reentrena automáticamente con el dataset actualizado.

- Las métricas se recalculan, permitiendo comparar resultados entre distintas ejecuciones.

### 5.3 Detección de data drift

La tarea check_drift implementa una lógica sencilla para revisar si la distribución de los datos ha cambiado:

1. Carga el dataset modelado actual.

2. Selecciona las columnas numéricas relevantes, por ejemplo:

['customer_id', 'product_id', 'order_id', 'items',
 'region_id', 'zone_id', 'Y', 'X',
 'num_deliver_per_week', 'num_visit_per_week', 'size']

3. Calcula estadísticas y/o medidas de diferencia entre:

- La distribución usada en el entrenamiento del modelo.

- La distribución del dataset actual.

4. Guarda un reporte en formato JSON en:

artifacts/drift/drift_report.json

Este reporte contiene, para cada columna numérica:

- Estadísticas descriptivas (medias, desviaciones, percentiles, etc.).

- Un score o indicador que permite interpretar si hay cambios importantes (potencial drift).

Este mecanismo permite:

- Monitorear en el tiempo cómo van cambiando las variables de entrada.

- Decidir si es necesario revisar el modelo o su estrategia de reentrenamiento cuando el drift es muy alto.

### 5.4 Generación de predicciones con los datos más recientes

Finalmente, la tarea generate_predictions:

1. Carga el modelo entrenado desde artifacts/models/sodai_model.joblib.

2. Carga el dataset actual (df_modelado.parquet).

3. Aplica el modelo utilizando las mismas features definidas en train.py.

4. Agrega una columna prediction al DataFrame.

5. Guarda el resultado en:

artifacts/predictions/predicciones.parquet

De este modo, cada ejecución del DAG con nuevos datos genera:

- Un modelo reentrenado.

- Nuevas métricas.

- Un reporte de drift.

- Un archivo de predicciones actualizado.

Todo el pipeline se puede volver a correr de manera reproducible simplemente actualizando los archivos de datos en airflow/data/ y disparando de nuevo el DAG.

