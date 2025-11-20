from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import time

from models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from inference import ModelPredictor
from config import MODEL_PATH, METRICS_PATH

# variable global para el predictor
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: cargar modelo
    global predictor
    predictor = ModelPredictor(MODEL_PATH, METRICS_PATH)
    predictor.load()
    print("modelo cargado exitosamente")
    yield
    # shutdown
    print("cerrando aplicacion")

app = FastAPI(
    title="sodai drinks prediction api",
    description="api para prediccion de compras de productos",
    version="1.0.0",
    lifespan=lifespan
)

# cors para permitir requests desde gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["general"])
async def root():
    return {
        "service": "sodai drinks prediction api",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["general"])
async def health_check():
    return {
        "status": "healthy" if predictor and predictor.model else "unhealthy",
        "model_loaded": predictor is not None and predictor.model is not None,
        "timestamp": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH)
    }

@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
async def get_model_info():
    if not predictor or not predictor.metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="modelo no esta cargado"
        )

    return {
        "metrics": {
            "accuracy": predictor.metrics['accuracy'],
            "precision": predictor.metrics['precision'],
            "recall": predictor.metrics['recall'],
            "f1_score": predictor.metrics['f1_score'],
            "roc_auc": predictor.metrics['roc_auc']
        },
        "features": predictor.feature_names,
        "hyperparameters": predictor.metrics['hyperparameters'],
        "mlflow_run_id": predictor.metrics['mlflow_run_id'],
        "feature_importance": predictor.feature_importance
    }

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: PredictionRequest):
    if not predictor or not predictor.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="modelo no esta cargado"
        )

    try:
        # convertir pydantic model a dict para procesamiento
        request_dict = request.model_dump()

        # realizar prediccion
        result = predictor.predict(request_dict)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"error al generar prediccion: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(request: BatchPredictionRequest):
    if not predictor or not predictor.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="modelo no esta cargado"
        )

    start_time = time.time()
    results = []

    try:
        for pred_request in request.predictions:
            request_dict = pred_request.model_dump()
            result = predictor.predict(request_dict)
            results.append(result)

        processing_time = time.time() - start_time

        return {
            "results": results,
            "total_processed": len(results),
            "processing_time_seconds": round(processing_time, 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"error en prediccion batch: {str(e)}"
        )

@app.post("/features/compute", tags=["utils"])
async def compute_features(request: PredictionRequest):
    """
    endpoint auxiliar para ver features calculadas
    util para debugging y entender el modelo
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="predictor no esta cargado"
        )

    try:
        request_dict = request.model_dump()
        features_df = predictor._compute_features(request_dict)

        return {
            "features": features_df.to_dict(orient='records')[0],
            "feature_count": len(features_df.columns)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"error al calcular features: {str(e)}"
        )
