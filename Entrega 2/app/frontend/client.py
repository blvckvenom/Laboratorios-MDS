import requests
from typing import Dict, Any
import os

class APIClient:
    """cliente para comunicarse con backend fastapi"""

    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = os.getenv("BACKEND_URL", "http://backend:8000")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """realiza prediccion individual"""
        url = f"{self.base_url}/predict"
        response = self.session.post(url, json=request_data, timeout=30)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, predictions: list) -> Dict[str, Any]:
        """realiza predicciones en lote"""
        url = f"{self.base_url}/predict/batch"
        request_data = {"predictions": predictions}
        response = self.session.post(url, json=request_data, timeout=60)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """verifica salud del backend"""
        url = f"{self.base_url}/health"
        response = self.session.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """obtiene informacion del modelo"""
        url = f"{self.base_url}/model-info"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
