# dags/dag_dynamic.py

from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Importa funciones
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models

DAG_ID = 'hiring_dynamic'

def choose_branch(execution_date_str: str, **_):
    # Branch: si fecha < 2024-11-01 => solo data_1; si no, data_1 y data_2
    cutoff = datetime(2024, 11, 1)
    dt = datetime.fromisoformat(execution_date_str.replace('Z','').replace(' ', 'T')) if 'T' in execution_date_str else datetime.fromisoformat(execution_date_str)
    return 'download_data_1_and_2' if dt >= cutoff else 'download_data_1_only'

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2024, 10, 1),
    schedule='0 15 5 * *',   # 5 de cada mes 15:00 UTC
    catchup=True,            
    default_args={'owner': 'airflow'},
    tags=['lab9','dynamic']
) as dag:

    start = EmptyOperator(task_id='start')

    # Crea carpetas
    t_create = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'ds_nodash': '{{ ds_nodash }}', 'base_dir': '/opt/airflow'}
    )

    # Branching por fecha de ejecucion
    branch = BranchPythonOperator(
        task_id='branch_download',
        python_callable=choose_branch,
        op_kwargs={'execution_date_str': '{{ ds }}'}
    )

    # Descargas 
    download_data_1_only = BashOperator(
        task_id='download_data_1_only',
        bash_command=(
            "curl -L -f -o /opt/airflow/runs/{{ ds_nodash }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )

    download_data_1_and_2 = BashOperator(
        task_id='download_data_1_and_2',
        bash_command=(
            "set -e; "
            "curl -L -f -o /opt/airflow/runs/{{ ds_nodash }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv; "
            "curl -L -f -o /opt/airflow/runs/{{ ds_nodash }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        )
    )

    # Merge con trigger
    t_merge = PythonOperator(
        task_id='load_and_merge',
        python_callable=load_and_merge,
        op_kwargs={'ds_nodash': '{{ ds_nodash }}', 'base_dir': '/opt/airflow'},
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    # Split
    t_split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'ds_nodash': '{{ ds_nodash }}', 'base_dir': '/opt/airflow'}
    )

    # Entrenamientos en paralelo
    t_rf = PythonOperator(
        task_id='train_rf',
        python_callable=train_model,
        op_kwargs={
            'ds_nodash': '{{ ds_nodash }}',
            'base_dir': '/opt/airflow',
            'model_name': 'rf',
        }
    )
    t_logreg = PythonOperator(
        task_id='train_logreg',
        python_callable=train_model,
        op_kwargs={
            'ds_nodash': '{{ ds_nodash }}',
            'base_dir': '/opt/airflow',
            'model_name': 'logreg',
        }
    )
    t_gboost = PythonOperator(
        task_id='train_gboost',
        python_callable=train_model,
        op_kwargs={
            'ds_nodash': '{{ ds_nodash }}',
            'base_dir': '/opt/airflow',
            'model_name': 'gboost',
        }
    )

    # Evaluacion: espera que los 3 entrenen
    t_eval = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'ds_nodash': '{{ ds_nodash }}', 'base_dir': '/opt/airflow'},
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    # Grafo
    start >> t_create >> branch
    branch >> download_data_1_only >> t_merge
    branch >> download_data_1_and_2 >> t_merge
    t_merge >> t_split >> [t_rf, t_logreg, t_gboost] >> t_eval
