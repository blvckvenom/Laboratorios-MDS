from pathlib import Path
import os

# paths de artifacts
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "sodai_model.joblib"))
METRICS_PATH = os.getenv("METRICS_PATH", str(METRICS_DIR / "sodai_metrics.json"))

# configuracion de servidor
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
