from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date

class CustomerData(BaseModel):
    customer_type: str = "REGULAR"
    X: float = 0.0
    Y: float = 0.0
    num_deliver_per_week: int = Field(ge=0, le=7, default=2)

class ProductData(BaseModel):
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str
    size: float = Field(gt=0.0)

class HistoricalData(BaseModel):
    # features de cliente
    total_ordenes_global: Optional[int] = None
    productos_unicos_global: Optional[int] = None
    items_totales_global: Optional[int] = None
    items_promedio_global: Optional[float] = None
    dias_desde_primera_compra: Optional[int] = None
    dias_desde_ultima_compra: Optional[int] = None
    frecuencia_compra_diaria: Optional[float] = None
    diversidad_productos: Optional[float] = None

    # features de producto
    total_ventas_global: Optional[int] = None
    clientes_unicos_global: Optional[int] = None
    items_vendidos_global: Optional[int] = None
    popularidad_rank: Optional[int] = None

    # features de interaccion
    veces_comprado_global: Optional[int] = None
    dias_desde_ultima_compra_producto: Optional[int] = None
    items_promedio_producto: Optional[float] = None
    compro_este_producto_antes: Optional[int] = None

class PredictionRequest(BaseModel):
    customer_id: str
    product_id: str
    fecha_prediccion: date = Field(default_factory=date.today)

    customer: CustomerData
    product: ProductData
    historical: Optional[HistoricalData] = None

    @field_validator('fecha_prediccion')
    @classmethod
    def validate_fecha(cls, v):
        if v > date.today():
            raise ValueError('fecha no puede ser futura')
        return v

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    interpretation: Literal["baja", "media", "alta", "muy alta"]
    confidence_level: Literal["baja", "media", "alta"]
    recommendation: str
    top_features: list[FeatureImportance]
    model_metadata: dict

class BatchPredictionRequest(BaseModel):
    predictions: list[PredictionRequest] = Field(max_length=1000)

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_processed: int
    processing_time_seconds: float

class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    timestamp: str
    model_path: str

class ModelInfoResponse(BaseModel):
    metrics: dict
    features: list[str]
    hyperparameters: dict
    mlflow_run_id: str
    feature_importance: dict
