# dags/dag_lineal.py

from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Importa funciones
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface



DAG_ID = "hiring_lineal"

with DAG(
    dag_id=DAG_ID,
    description="Pipeline lineal: descarga -> split -> train -> gradio",
    start_date=pendulum.datetime(2024, 10, 1, tz="UTC"),
    schedule=None,           
    catchup=False,           
    max_active_runs=1,
    default_view="graph",
    render_template_as_native_obj=True,
    tags=["lab9", "airflow", "hiring"],
) as dag:

    start = EmptyOperator(task_id="start_pipeline")

    # 1) Crear carpetas /opt/airflow/{{ ds_nodash }}/{raw,splits,models}
    create_dirs = BashOperator(
    task_id="create_folders",
    bash_command=(
        "export PYTHONPATH=/opt/airflow/dags && "
        "python - <<'PY'\n"
        "from hiring_functions import create_folders\n"
        "print('[create_folders] ds_nodash={{ ds_nodash }}')\n"
        "create_folders(ds_nodash='{{ ds_nodash }}', base_dir='/opt/airflow')\n"
        "print('[create_folders] OK')\n"
        "PY"
    ),
    )

    # 2) Descargar data_1.csv a la carpeta raw de la corrida
    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            "mkdir -p /opt/airflow/{{ ds_nodash }}/raw && "
            "curl -fsSL -o /opt/airflow/{{ ds_nodash }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    # 3) Split hold-out y guardar en splits/
    do_split = BashOperator(
    task_id="split_data",
    bash_command=(
        "export PYTHONPATH=/opt/airflow/dags && "
        "python - <<'PY'\n"
        "from hiring_functions import split_data\n"
        "print('[split_data] ds_nodash={{ ds_nodash }}')\n"
        "split_data(ds_nodash='{{ ds_nodash }}', base_dir='/opt/airflow', filename='data_1.csv')\n"
        "print('[split_data] OK')\n"
        "PY"
    ),
    )

    # 4) Preprocesar + entrenar RandomForest y guardar modelo en models/
    train = BashOperator(
    task_id="preprocess_and_train",
    bash_command=(
        "export PYTHONPATH=/opt/airflow/dags && "
        "python - <<'PY'\n"
        "from hiring_functions import preprocess_and_train\n"
        "print('[preprocess_and_train] ds_nodash={{ ds_nodash }}')\n"
        "preprocess_and_train(ds_nodash='{{ ds_nodash }}', base_dir='/opt/airflow')\n"
        "print('[preprocess_and_train] OK')\n"
        "PY"
    ),
    )

    # 5) Levantar interfaz Gradio usando el modelo de esta corrida
    serve_gradio = BashOperator(
    task_id="gradio_interface",
    bash_command=(
        "export PYTHONPATH=/opt/airflow/dags && "
        "python - <<'PY'\n"
        "from hiring_functions import gradio_interface\n"
        "print('[gradio] ds_nodash={{ ds_nodash }}')\n"
        "gradio_interface(ds_nodash='{{ ds_nodash }}', base_dir='/opt/airflow', port=7870)\n"
        "PY"
        ),
    )

    start >> create_dirs >> download_data >> do_split >> train >> serve_gradio
