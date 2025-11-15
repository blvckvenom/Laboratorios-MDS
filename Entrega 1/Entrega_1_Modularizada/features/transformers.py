"""
Funciones para transformaciones de producto y encoders personalizados.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def aplicar_transformaciones_producto(df, calcular_centroide=True):
    """
    Aplica transformaciones a features de producto.

    Args:
        df: DataFrame al que aplicar transformaciones
        calcular_centroide: Si se debe calcular el centroide (solo para train)

    Returns:
        pd.DataFrame: DataFrame con transformaciones aplicadas
    """
    df_fe = df.copy()

    # Transformación logarítmica de size
    df_fe['size_log1p'] = np.log1p(df_fe['size'])

    # Categorización de size
    df_fe['size_categoria'] = pd.cut(
        df_fe['size'],
        bins=[0, 0.33, 0.66, 1.5, 3.0, np.inf],
        labels=['individual', 'personal', 'familiar_pequeno', 'familiar_grande', 'granel']
    )

    # Encoding ordinal para segment
    segment_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'PREMIUM': 3}
    df_fe['segment_ordinal'] = df_fe['segment'].map(segment_order)

    # Distancia geográfica al centro
    if calcular_centroide:
        centroid_x = df_fe['X'].mean()
        centroid_y = df_fe['Y'].mean()
    else:
        # Usar centroide global si está disponible
        centroid_x = df_fe['X'].mean()
        centroid_y = df_fe['Y'].mean()

    df_fe['distancia_al_centro'] = np.sqrt(
        (df_fe['X'] - centroid_x)**2 + (df_fe['Y'] - centroid_y)**2
    )

    return df_fe


def aplicar_transformaciones(df_train_fe, df_val_fe, df_test_fe):
    """
    Aplica transformaciones de producto a los datasets.

    Args:
        df_train_fe: DataFrame de entrenamiento con features
        df_val_fe: DataFrame de validación con features
        df_test_fe: DataFrame de test con features

    Returns:
        tuple: (df_train_fe, df_val_fe, df_test_fe)
    """
    print("\n[5/6] Aplicando transformaciones de producto...")

    # Calcular centroide solo con datos de train
    centroid_x = df_train_fe['X'].mean()
    centroid_y = df_train_fe['Y'].mean()

    # Aplicar transformaciones
    df_train_fe = aplicar_transformaciones_producto(df_train_fe, calcular_centroide=True)
    df_val_fe = aplicar_transformaciones_producto(df_val_fe, calcular_centroide=False)
    df_test_fe = aplicar_transformaciones_producto(df_test_fe, calcular_centroide=False)

    print(f"  Transformaciones aplicadas: 4 (size_log1p, size_categoria, segment_ordinal, distancia_al_centro)")

    return df_train_fe, df_val_fe, df_test_fe


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder de Target con smoothing para prevenir overfitting.
    Codifica variables categóricas usando la media del target por categoría.
    """

    def __init__(self, smoothing=10):
        """
        Inicializa el TargetEncoder.

        Args:
            smoothing: Factor de smoothing para prevenir overfitting
        """
        self.smoothing = smoothing
        self.encoding_dict = {}
        self.global_mean = 0

    def fit(self, X, y):
        """
        Aprende la codificación basada en los datos de entrenamiento.

        Args:
            X: DataFrame de pandas con variables categóricas
            y: Variable target

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas")

        self.global_mean = y.mean()

        for col in X.columns:
            target_by_category = pd.DataFrame({'category': X[col], 'target': y})
            agg = target_by_category.groupby('category')['target'].agg(['mean', 'count'])
            agg['smoothed_mean'] = (
                (agg['count'] * agg['mean'] + self.smoothing * self.global_mean) /
                (agg['count'] + self.smoothing)
            )
            self.encoding_dict[col] = agg['smoothed_mean'].to_dict()

        return self

    def transform(self, X):
        """
        Transforma las variables categóricas usando la codificación aprendida.

        Args:
            X: DataFrame de pandas con variables categóricas

        Returns:
            numpy array con valores codificados
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas")

        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X[col].map(self.encoding_dict[col]).fillna(self.global_mean)

        return X_encoded.values


def crear_pipeline_preprocesamiento(numeric_features, categorical_features_onehot,
                                    categorical_features_target, binary_features):
    """
    Crea pipeline de preprocesamiento para los features.

    Args:
        numeric_features: Lista de features numéricos
        categorical_features_onehot: Lista de features categóricos para one-hot encoding
        categorical_features_target: Lista de features categóricos para target encoding
        binary_features: Lista de features binarios

    Returns:
        ColumnTransformer: Pipeline de preprocesamiento
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat_onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_features_onehot),
            ('cat_target', TargetEncoder(smoothing=10), categorical_features_target),
            ('binary', 'passthrough', binary_features)
        ],
        remainder='drop'
    )

    return preprocessor


def preparar_datasets_para_modelado(df_train_fe, df_val_fe, df_test_fe, feature_columns, target_col='target'):
    """
    Prepara los datasets finales para modelado.

    Args:
        df_train_fe: DataFrame de entrenamiento con features
        df_val_fe: DataFrame de validación con features
        df_test_fe: DataFrame de test con features
        feature_columns: Lista de columnas de features
        target_col: Nombre de la columna target

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = df_train_fe[feature_columns]
    y_train = df_train_fe[target_col]

    X_val = df_val_fe[feature_columns]
    y_val = df_val_fe[target_col]

    X_test = df_test_fe[feature_columns]
    y_test = df_test_fe[target_col]

    print(f"\nDatasets finales para modelado:")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")

    print(f"\nBalance del target:")
    print(f"  - Train: {y_train.mean():.2%}")
    print(f"  - Val: {y_val.mean():.2%}")
    print(f"  - Test: {y_test.mean():.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test
