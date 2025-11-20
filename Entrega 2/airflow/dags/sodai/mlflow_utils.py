from __future__ import annotations

import mlflow
from pathlib import Path

from .config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow():
    """
    configura mlflow con el tracking uri y experimento
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"mlflow tracking uri: {MLFLOW_TRACKING_URI}")
    print(f"mlflow experiment: {MLFLOW_EXPERIMENT_NAME}")


def log_params_and_metrics(params: dict, metrics: dict):
    """
    registra parametros y metricas en mlflow
    """
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


def log_model(model, artifact_path: str = "model"):
    """
    registra el modelo en mlflow usando sklearn
    """
    mlflow.sklearn.log_model(model, artifact_path)


def log_artifacts(local_dir: str | Path):
    """
    registra artifacts desde un directorio local
    """
    mlflow.log_artifacts(str(local_dir))


def get_best_run_id(experiment_name: str = None) -> str | None:
    """
    obtiene el run_id del mejor modelo segun f1_score
    retorna none si no hay runs
    """
    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT_NAME

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        return None

    return runs[0].info.run_id


def load_best_model(experiment_name: str = None):
    """
    carga el mejor modelo registrado en mlflow
    """
    run_id = get_best_run_id(experiment_name)

    if run_id is None:
        raise ValueError("no hay runs disponibles en mlflow")

    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)
