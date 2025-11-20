import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json

class ModelPredictor:
    def __init__(self, model_path: str, metrics_path: str):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.model = None
        self.metrics = None
        self.feature_names = None
        self.feature_importance = None

    def load(self):
        """carga modelo y metricas"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"modelo no encontrado: {self.model_path}")

        self.model = joblib.load(self.model_path)

        with open(self.metrics_path, 'r') as f:
            self.metrics = json.load(f)

        self.feature_names = self.metrics['features_used']
        self.feature_importance = self.metrics['feature_importance']
        self.categorical_mappings = self.metrics.get('categorical_mappings', {})

        print(f"modelo cargado desde {self.model_path}")
        print(f"features: {len(self.feature_names)}")
        if self.categorical_mappings:
            print(f"categorical mappings cargados para {len(self.categorical_mappings)} columnas")

    def _compute_features(self, request_data: dict) -> pd.DataFrame:
        """
        calcula las 52 features a partir del request
        aplica misma logica que features.py del pipeline
        """

        # extraer datos del request
        customer = request_data['customer']
        product = request_data['product']
        historical = request_data.get('historical', {}) or {}
        fecha = pd.to_datetime(request_data['fecha_prediccion'])

        # inicializar diccionario de features
        features = {}

        # --- features de cliente ---
        # usar valores historicos si existen, sino defaults conservadores
        features['total_ordenes_global'] = historical.get('total_ordenes_global', 1)
        features['productos_unicos_global'] = historical.get('productos_unicos_global', 1)
        features['items_totales_global'] = historical.get('items_totales_global', 1)
        features['items_promedio_global'] = historical.get('items_promedio_global', 1.0)
        features['dias_desde_primera_compra'] = historical.get('dias_desde_primera_compra', 30)
        features['dias_desde_ultima_compra'] = historical.get('dias_desde_ultima_compra', 7)
        features['frecuencia_compra_diaria'] = historical.get('frecuencia_compra_diaria',
                                                               features['total_ordenes_global'] / max(features['dias_desde_primera_compra'], 1))
        features['diversidad_productos'] = historical.get('diversidad_productos',
                                                          features['productos_unicos_global'] / max(features['total_ordenes_global'], 1))

        # --- features de producto ---
        features['total_ventas_global'] = historical.get('total_ventas_global', 100)
        features['clientes_unicos_global'] = historical.get('clientes_unicos_global', 50)
        features['items_vendidos_global'] = historical.get('items_vendidos_global', 200)
        features['popularidad_rank'] = historical.get('popularidad_rank', 100)

        # --- features de interaccion ---
        features['veces_comprado_global'] = historical.get('veces_comprado_global', 0)
        features['dias_desde_ultima_compra_producto'] = historical.get('dias_desde_ultima_compra_producto', 999)
        features['items_promedio_producto'] = historical.get('items_promedio_producto', 1.0)
        features['compro_este_producto_antes'] = historical.get('compro_este_producto_antes',
                                                                 1 if features['veces_comprado_global'] > 0 else 0)

        # --- features de producto (basicas) ---
        features['size'] = product['size']
        features['size_log1p'] = np.log1p(product['size'])

        # segment ordinal
        segment_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'PREMIUM': 3}
        features['segment_ordinal'] = segment_map[product['segment']]

        # coordenadas geograficas
        features['X'] = customer['X']
        features['Y'] = customer['Y']

        # distancia al centro (usando centroide aproximado)
        centroid_x = 0.0
        centroid_y = 0.0
        features['distancia_al_centro'] = np.sqrt((features['X'] - centroid_x)**2 +
                                                    (features['Y'] - centroid_y)**2)

        # num_deliver_per_week
        features['num_deliver_per_week'] = customer['num_deliver_per_week']

        # --- features temporales ---
        features['dia_semana'] = fecha.dayofweek
        features['mes'] = fecha.month
        features['trimestre'] = fecha.quarter
        features['semana_del_año'] = fecha.isocalendar()[1]

        # encoding ciclico
        features['mes_sin'] = np.sin(2 * np.pi * features['mes'] / 12)
        features['mes_cos'] = np.cos(2 * np.pi * features['mes'] / 12)
        features['dia_semana_sin'] = np.sin(2 * np.pi * features['dia_semana'] / 7)
        features['dia_semana_cos'] = np.cos(2 * np.pi * features['dia_semana'] / 7)

        # --- features binarias ---
        features['es_fin_semana'] = 1 if features['dia_semana'] >= 5 else 0
        features['es_lunes_jueves'] = 1 if features['dia_semana'] in [0, 3] else 0
        features['es_temporada_alta'] = 1 if features['mes'] in [11, 12] else 0
        features['es_temporada_baja'] = 1 if features['mes'] in [5, 6, 7] else 0

        # --- features categoricas ---
        features['category'] = product['category']
        features['sub_category'] = product['sub_category']
        features['package'] = product['package']
        features['segment'] = product['segment']
        features['brand'] = product['brand']
        features['customer_type'] = customer['customer_type']

        # size_categoria
        size = product['size']
        if size <= 0.33:
            features['size_categoria'] = 'individual'
        elif size <= 0.66:
            features['size_categoria'] = 'personal'
        elif size <= 1.5:
            features['size_categoria'] = 'familiar_pequeno'
        elif size <= 3.0:
            features['size_categoria'] = 'familiar_grande'
        else:
            features['size_categoria'] = 'granel'

        # crear dataframe con features en orden correcto
        df = pd.DataFrame([features])

        # convertir categoricas a tipo category y validar valores conocidos
        categorical_cols = ['category', 'sub_category', 'package', 'size_categoria',
                            'trimestre', 'dia_semana', 'mes', 'segment', 'brand', 'customer_type']
        for col in categorical_cols:
            if col in df.columns:
                # validar y mapear categorias desconocidas
                if col in self.categorical_mappings:
                    valid_categories = self.categorical_mappings[col]
                    current_value = df[col].iloc[0]

                    # si el valor no esta en las categorias conocidas, usar la primera categoria valida
                    if current_value not in valid_categories:
                        print(f"warning: categoria '{current_value}' no conocida en columna '{col}', usando '{valid_categories[0]}'")
                        df.loc[0, col] = valid_categories[0]

                    # convertir a category dtype con las categorias exactas del entrenamiento
                    df[col] = pd.Categorical(df[col], categories=valid_categories)
                else:
                    # si no hay mapping, usar astype normal
                    df[col] = df[col].astype('category')

        # asegurar que las columnas esten en el orden correcto
        df = df[self.feature_names]

        return df

    def predict(self, request_data: dict) -> dict:
        """genera prediccion a partir de request"""

        # calcular features
        X = self._compute_features(request_data)

        # prediccion binaria
        prediction = int(self.model.predict(X)[0])

        # probabilidad
        probability = float(self.model.predict_proba(X)[0, 1])

        # interpretacion
        if probability < 0.3:
            interpretation = "baja"
            confidence = "alta"
            recommendation = "no recomendar este producto"
        elif probability < 0.5:
            interpretation = "media"
            confidence = "media"
            recommendation = "considerar otros productos primero"
        elif probability < 0.7:
            interpretation = "alta"
            confidence = "media"
            recommendation = "producto candidato para recomendacion"
        else:
            interpretation = "muy alta"
            confidence = "alta"
            recommendation = "recomendar este producto"

        # top features importantes
        top_features = sorted(
            [{"feature": k, "importance": v} for k, v in self.feature_importance.items()],
            key=lambda x: x['importance'],
            reverse=True
        )[:5]

        return {
            "prediction": prediction,
            "probability": probability,
            "interpretation": interpretation,
            "confidence_level": confidence,
            "recommendation": recommendation,
            "top_features": top_features,
            "model_metadata": {
                "model_version": self.metrics['mlflow_run_id'][:10],
                "f1_score": self.metrics['f1_score'],
                "roc_auc": self.metrics['roc_auc']
            }
        }
