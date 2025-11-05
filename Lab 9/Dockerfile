FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR ${AIRFLOW_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*


ENV AIRFLOW_VERSION=2.9.3
ENV PYTHON_VERSION=3.10
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" --constraint \
    "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN pip install --no-cache-dir \
    pandas==2.2.2 scikit-learn==1.5.1 joblib==1.4.2 pyarrow==17.0.0 \
    gradio==4.44.0

COPY dags/ ${AIRFLOW_HOME}/dags/

RUN airflow db migrate && \
    airflow users create --username admin --firstname Air --lastname Flow \
    --role Admin --email admin@example.com --password admin


EXPOSE 8080

CMD bash -lc "airflow scheduler & airflow webserver --port 8080"
