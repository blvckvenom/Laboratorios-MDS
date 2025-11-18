from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _build_dataset(**context):
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


def _train_model(**context):
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

    train.entrenar_modelo(df)


def _evaluate_model(**context):
    from sodai import evaluate

    evaluate.evaluar_modelo()


def _check_drift(**context):
    from sodai import drift

    drift.calcular_drift()


def _generate_predictions(**context):
    from sodai import predict

    predict.generar_predicciones()


default_args = {
    "owner": "vergara",
    "retries": 0,
}

with DAG(
    dag_id="sodai_training_and_scoring",
    description="Pipeline SodAI: dataset, entrenamiento, evaluación, drift y predicciones",
    default_args=default_args,
    schedule_interval=None,  # lo gatillamos a mano
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["sodai", "ml"],
) as dag:
    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=_build_dataset,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=_generate_predictions,
    )

    build_dataset >> train_model >> evaluate_model >> check_drift >> generate_predictions
