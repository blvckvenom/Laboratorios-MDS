# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import os
import pickle
import pandas as pd 

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("artifacts", "models", "best_model.pkl"))

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

class MedicionAgua(BaseModel):
    ph: float = Field(..., description="pH")
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

app = FastAPI(title="API Potabilidad del Agua", version="1.0")

@app.get("/")
def home():
    """
    Home: descripción breve del servicio.
    """
    return {
        "proyecto": "Potabilidad del Agua",
        "descripcion": "Clasificador binario (0/1) entrenado con XGBoost para predecir si una medición de agua es potable.",
        "entrada": {
            "type": "JSON",
            "features": FEATURES
        },
        "salida": {
            "potabilidad": "0 = No potable, 1 = Potable"
        },
        "endpoints": {
            "predict": "POST /potabilidad/",
            "docs": "/docs"
        }
    }

@app.post("/potabilidad/")
def predecir_potabilidad(x: MedicionAgua):
    fila = {f: getattr(x, f) for f in FEATURES}
    X = pd.DataFrame([fila], columns=FEATURES)

    y_pred = int(MODEL.predict(X)[0])
    return {"potabilidad": y_pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
