from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


def _build_dataset(**context):
    """construye dataset de modelado desde datos crudos"""
    from sodai import data_io, features

    clientes, productos, transacciones = data_io.cargar_datos_crudos()
    df_modelado = features.construir_df_modelado(
        clientes=clientes,
        productos=productos,
        transacciones=transacciones,
    )

    salida = data_io.guardar_dataset_modelado(df_modelado)

    ti = context["ti"]
    ti.xcom_push(key="dataset_path", value=str(salida))


def _check_drift_decision(**context):
    """
    verifica si hay drift y decide si reentrenar
    retorna task_id hacia donde bifurcar el flujo
    """
    from sodai import drift

    # primero calculamos drift
    drift.calcular_drift()

    # luego verificamos si hay drift significativo
    hay_drift = drift.hay_drift_significativo()

    if hay_drift:
        print("drift detectado - se procedera a reentrenar modelo")
        return "train_model"
    else:
        print("no hay drift significativo - se usara modelo existente")
        return "skip_training"


def _train_model(**context):
    """entrena modelo con optimizacion de hiperparametros"""
    from pathlib import Path
    from sodai import data_io, train

    ti = context["ti"]
    dataset_path = ti.xcom_pull(
        task_ids="build_dataset",
        key="dataset_path",
    )

    if dataset_path:
        df = data_io.cargar_dataset_modelado(Path(dataset_path).name)
    else:
        df = data_io.cargar_dataset_modelado()

    # entrenar con optimizacion usando optuna
    # configuracion ajustada para evitar deadlock: n_jobs=1 en optuna, n_jobs=2 en xgboost trials
    # 10 trials para optimizacion rapida sin cuelgues
    train.entrenar_modelo(df, optimize=True, n_trials=10)


def _evaluate_model(**context):
    """evalua modelo y genera interpretabilidad con shap"""
    from sodai import evaluate

    # evaluar modelo y generar analisis shap
    # usa sample de 200 para mejor interpretabilidad
    evaluate.evaluar_modelo(generate_shap=True, shap_sample_size=200)


def _generate_predictions(**context):
    """genera predicciones con el modelo actual"""
    from sodai import predict

    predict.generar_predicciones()


default_args = {
    "owner": "data_team",
    "retries": 1,
}

with DAG(
    dag_id="sodai_training_and_scoring",
    description="pipeline sodai con mlflow, optuna, shap y deteccion de drift",
    default_args=default_args,
    schedule_interval=None,  # ejecutar manualmente
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["sodai", "ml", "production"],
) as dag:

    # construir dataset
    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=_build_dataset,
    )

    # verificar drift y decidir si reentrenar
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift_decision,
    )

    # entrenar modelo (solo si hay drift)
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    # skip training (si no hay drift)
    skip_training = EmptyOperator(
        task_id="skip_training",
    )

    # evaluar modelo (se ejecuta siempre despues de entrenar o skip)
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
        trigger_rule="none_failed_min_one_success",  # ejecutar si train o skip terminaron
    )

    # generar predicciones
    generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=_generate_predictions,
    )

    # definir flujo del dag con branching condicional
    # build_dataset -> check_drift -> [train_model o skip_training]
    # -> evaluate_model -> generate_predictions
    build_dataset >> check_drift
    check_drift >> [train_model, skip_training]
    [train_model, skip_training] >> evaluate_model
    evaluate_model >> generate_predictions
