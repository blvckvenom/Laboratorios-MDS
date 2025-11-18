from pathlib import Path

try:
    from airflow.models import Variable
except Exception:
    Variable = None


def _get_var(name: str, default: str) -> str:
    if Variable is None:
        return default
    try:
        return Variable.get(name, default)
    except Exception:
        return default

BASE_DATA_DIR = Path(_get_var("SODAI_BASE_DATA_DIR", "/opt/airflow/data"))
ARTIFACTS_DIR = Path(_get_var("SODAI_ARTIFACTS_DIR", "/opt/airflow/artifacts"))

RAW_DIR = BASE_DATA_DIR

DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
DRIFT_DIR = ARTIFACTS_DIR / "drift"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

for _dir in [DATASETS_DIR, MODELS_DIR, METRICS_DIR, DRIFT_DIR, PREDICTIONS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

CLIENTES_PATH = RAW_DIR / "clientes.parquet"
PRODUCTOS_PATH = RAW_DIR / "productos.parquet"
TRANSACCIONES_PATH = RAW_DIR / "transacciones.parquet"

TARGET_COL = "items"

DEFAULT_MODEL_PATH = MODELS_DIR / "sodai_model.joblib"
DEFAULT_METRICS_PATH = METRICS_DIR / "sodai_metrics.json"
