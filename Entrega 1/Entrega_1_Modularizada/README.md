# Proyecto Entrega 1 - Pipeline de Machine Learning Modularizado

## Descripción

Este proyecto contiene el código modularizado del análisis de datos y modelado predictivo para la empresa SodAI Drinks. El objetivo es predecir la probabilidad de compra semanal de clientes B2B.

## Estructura del Proyecto

```
Entrega_1/
├── config/                          # Configuración del proyecto
│   ├── __init__.py
│   └── paths.py                     # Rutas de archivos y configuraciones globales
│
├── data/                            # Módulos de carga y validación de datos
│   ├── __init__.py
│   ├── loader.py                    # Carga de archivos parquet
│   └── quality.py                   # Validación de calidad de datos
│
├── preprocessing/                   # Preprocesamiento de datos
│   ├── __init__.py
│   ├── cleaner.py                   # Limpieza: duplicados, valores negativos, consolidación
│   └── geographic.py                # Corrección de coordenadas geográficas
│
├── eda/                             # Análisis Exploratorio de Datos
│   ├── __init__.py
│   ├── clientes.py                  # Análisis de clientes
│   ├── productos.py                 # Análisis del catálogo de productos
│   ├── temporal.py                  # Patrones temporales y estacionalidad
│   ├── comportamiento.py            # Patrones de compra y lealtad
│   └── correlaciones.py             # Correlaciones y relaciones entre variables
│
├── features/                        # Feature Engineering
│   ├── __init__.py
│   ├── cliente_features.py          # Features RFM de clientes
│   ├── producto_features.py         # Features de productos
│   ├── interaccion_features.py      # Features de interacción cliente-producto
│   ├── temporal_features.py         # Features temporales y cíclicos
│   └── transformers.py              # Transformaciones y encoders personalizados
│
├── models/                          # Modelado
│   ├── __init__.py
│   ├── baseline.py                  # Modelo baseline (Regresión Logística)
│   ├── comparacion.py               # Comparación de 6 clasificadores
│   ├── optimizer.py                 # Optimización con Optuna
│   └── xgboost_final.py             # Utilidades para modelo final
│
├── evaluation/                      # Evaluación de modelos
│   ├── __init__.py
│   ├── metrics.py                   # Cálculo y visualización de métricas
│   └── importance.py                # Análisis de importancia de features
│
├── interpretability/                # Interpretabilidad
│   ├── __init__.py
│   ├── shap_analysis.py             # Análisis SHAP
│   └── visualizations.py            # Visualizaciones SHAP
│
├── utils/                           # Utilidades generales
│   ├── __init__.py
│   └── plots.py                     # Funciones auxiliares para gráficos
│
├── main.py                          # Script principal que orquesta todo el pipeline
└── README.md                        # Este archivo
```

## Instalación

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna shap
```

### Datos

Coloque los siguientes archivos en el directorio raíz del proyecto:
- `clientes.parquet`
- `productos.parquet`
- `transacciones.parquet`

## Uso

### Ejecución completa del pipeline

```python
python main.py
```

### Uso modular

Puede importar y usar módulos individuales según necesidad:

```python
# Ejemplo 1: Solo carga de datos
from data import loader
df_cliente, df_productos, df_transacciones = loader.cargar_todos_los_datos()

# Ejemplo 2: Solo preprocesamiento
from preprocessing import cleaner, geographic
df_cliente_clean, df_productos_clean, df_trans_agg = cleaner.aplicar_limpieza_completa(
    df_cliente, df_productos, df_transacciones
)

# Ejemplo 3: Solo EDA de clientes
from eda import clientes
resumen = clientes.analisis_clientes_completo(df_cliente_clean, visualizar=True)

# Ejemplo 4: Entrenar modelo baseline
from models import baseline
from features import transformers

preprocessor = transformers.crear_pipeline_preprocesamiento(...)
modelo, metricas, tiempo = baseline.ejecutar_baseline_completo(
    preprocessor, X_train, y_train, X_val, y_val
)
```

## Módulos Principales

### 1. Data (Carga y Calidad)

**data/loader.py**
- `cargar_clientes()`: Carga dataset de clientes
- `cargar_productos()`: Carga catálogo de productos
- `cargar_transacciones()`: Carga historial de transacciones
- `cargar_todos_los_datos()`: Carga los 3 datasets
- `explorar_dataset()`: Muestra información básica

**data/quality.py**
- `verificar_valores_nulos()`: Detecta valores faltantes
- `verificar_integridad_referencial()`: Valida FKs
- `detectar_inconsistencias()`: Items negativos, tamaños inválidos
- `validar_coordenadas_geograficas()`: Valida formato WGS84
- `verificar_duplicados()`: Detecta duplicados
- `analisis_calidad_completo()`: Análisis completo

### 2. Preprocessing (Preprocesamiento)

**preprocessing/cleaner.py**
- `eliminar_duplicados()`: Elimina registros duplicados
- `eliminar_items_negativos()`: Limpia transacciones con items < 0
- `eliminar_clientes_sin_transacciones()`: Elimina clientes sin compras
- `marcar_productos_sin_ventas()`: Flag para productos sin historial
- `consolidar_transacciones_por_dia()`: Agrega transacciones mismo día
- `aplicar_limpieza_completa()`: Pipeline completo

**preprocessing/geographic.py**
- `validar_coordenadas_wgs84()`: Valida coordenadas
- `corregir_coordenadas()`: Aplica swap X↔Y y elimina nulos
- `obtener_coordenadas_invalidas()`: Identifica coordenadas problemáticas
- `calcular_distancia_al_centro()`: Distancia euclidiana al centroide

### 3. EDA (Análisis Exploratorio)

**eda/clientes.py**
- `analizar_clientes_basico()`: Distribución por tipo, estadísticas
- `analizar_distribucion_geografica()`: Rangos, centroide, dispersión
- `visualizar_distribucion_tipos()`: Gráficos de distribución
- `analizar_entregas_por_tipo()`: Entregas semanales por tipo

**eda/productos.py**
- `analizar_productos_basico()`: Categorías, segmentos, marcas
- `analizar_tamaños_productos()`: Distribución de tamaños
- `visualizar_distribucion_segmentos()`: Gráficos de segmentos

**eda/temporal.py**
- `analizar_rango_temporal()`: Periodo cubierto, transacciones diarias
- `analizar_patron_semanal()`: Actividad por día de la semana
- `analizar_patron_mensual()`: Estacionalidad mensual
- `visualizar_serie_temporal_diaria()`: Serie temporal

**eda/comportamiento.py**
- `analizar_metricas_por_cliente()`: Órdenes, productos, items por cliente
- `analizar_frecuencia_recompra()`: Ciclos de recompra
- `clasificar_lealtad_clientes()`: Niveles de lealtad (Alto/Medio/Bajo)
- `analizar_adquisicion_clientes()`: Nuevos clientes por fecha

**eda/correlaciones.py**
- `analizar_correlaciones_clientes()`: Correlaciones entre variables numéricas
- `analizar_relacion_cliente_comportamiento()`: Tipo vs comportamiento
- `analizar_preferencias_por_segmento()`: Cliente × Segmento
- `analizar_productos_mas_vendidos()`: Top productos

### 4. Features (Feature Engineering)

**features/cliente_features.py**
- `crear_features_cliente_rfm()`: Features RFM (Recency, Frequency, Monetary)
- `agregar_features_cliente_temporales()`: Días desde primera/última compra, frecuencia

**features/producto_features.py**
- `crear_features_producto()`: Ventas, clientes únicos, popularidad

**features/interaccion_features.py**
- `crear_features_interaccion()`: Historial cliente-producto
- `agregar_features_interaccion_temporales()`: Veces comprado, días desde última

**features/temporal_features.py**
- `crear_features_temporales()`: Día semana, mes, trimestre, encoding cíclico

**features/transformers.py**
- `TargetEncoder`: Encoder personalizado con smoothing
- `aplicar_transformaciones_producto()`: Log, categorización, distancia al centro
- `crear_pipeline_preprocesamiento()`: ColumnTransformer completo

### 5. Models (Modelado)

**models/baseline.py**
- `crear_modelo_baseline()`: Regresión Logística con class_weight='balanced'
- `entrenar_baseline()`: Entrenamiento
- `evaluar_baseline()`: Evaluación
- `ejecutar_baseline_completo()`: Pipeline completo

**models/comparacion.py**
- `crear_clasificadores()`: 6 modelos (KNN, DT, RF, SVM, XGBoost, LightGBM)
- `entrenar_y_evaluar_modelo()`: Entrena y evalúa un modelo
- `comparar_modelos()`: Compara todos los modelos
- `obtener_mejor_modelo()`: Selecciona el mejor según métrica

**models/optimizer.py**
- `crear_funcion_objetivo()`: Función objetivo para Optuna
- `optimizar_hiperparametros()`: Optimización con TPE Sampler (200 trials)
- `entrenar_modelo_optimizado()`: Entrena con mejores parámetros
- `comparar_modelos_completo()`: Baseline vs Original vs Optimizado

**models/xgboost_final.py**
- `guardar_modelo()`: Serializa modelo
- `cargar_modelo()`: Carga modelo guardado
- `guardar_parametros()`: Guarda parámetros en JSON
- `mostrar_reporte_clasificacion()`: Classification report
- `evaluar_modelo_final()`: Evaluación en test

### 6. Evaluation (Evaluación)

**evaluation/metrics.py**
- `calcular_metricas_completas()`: Accuracy, Precision, Recall, F1, ROC-AUC
- `mostrar_metricas()`: Formato legible
- `visualizar_matriz_confusion()`: Heatmap
- `visualizar_curva_roc()`: Curva ROC
- `comparar_metricas_modelos()`: Gráficos comparativos
- `reporte_evaluacion_completo()`: Reporte completo

**evaluation/importance.py**
- `calcular_importancia_xgboost()`: Importancia con XGBoost Gain
- `calcular_permutation_importance()`: Permutation Importance
- `calcular_mutual_information()`: Mutual Information
- `consolidar_importancias()`: Combina 3 métodos con pesos (40%, 40%, 20%)
- `visualizar_importancias()`: Gráficos de importancia
- `analisis_importancia_completo()`: Pipeline completo

### 7. Interpretability (Interpretabilidad)

**interpretability/shap_analysis.py**
- `crear_explainer()`: TreeExplainer de SHAP
- `calcular_shap_values()`: SHAP values para dataset
- `obtener_importancia_shap()`: Importancia basada en SHAP
- `analizar_prediccion_individual()`: Análisis de predicción específica
- `analisis_shap_completo()`: Pipeline completo

**interpretability/visualizations.py**
- `plot_shap_summary()`: Summary plot (beeswarm)
- `plot_shap_bar()`: Bar plot
- `plot_shap_waterfall()`: Waterfall plot individual
- `plot_shap_force()`: Force plot
- `plot_shap_dependence()`: Dependence plot
- `visualizaciones_shap_completas()`: Todas las visualizaciones

### 8. Utils (Utilidades)

**utils/plots.py**
- `configurar_estilo_plots()`: Configuración global de estilo
- `plot_distribucion_simple()`: Histograma
- `plot_barras_categorias()`: Gráfico de barras
- `plot_series_temporal()`: Serie temporal
- `plot_correlacion_heatmap()`: Heatmap de correlaciones
- `plot_comparacion_modelos()`: Comparativa de modelos
- `guardar_figura()`: Guardar gráfico

## Resultados Esperados

### Métricas del Modelo Final

| Métrica | Baseline | XGBoost Original | XGBoost Optimizado |
|---------|----------|------------------|-------------------|
| F1-Score | 0.5309 | 0.7192 | 0.7405 |
| ROC-AUC | 0.8055 | 0.8211 | 0.8226 |
| Precision | 85.18% | 71.97% | 67.29% |
| Recall | 38.56% | 71.87% | 82.28% |

### Mejoras

- **vs Baseline**: +39.49% en F1-Score
- **vs Original**: +2.96% en F1-Score
- **Recall**: Detecta 82 de cada 100 compras reales (vs 39 del baseline)

## Hallazgos Clave

1. **Feature dominante**: Feature RFM con 75% de importancia SHAP
2. **Hiperparámetro crítico**: `scale_pos_weight` (92% de impacto en optimización)
3. **Patrón temporal**: Lunes/Jueves picos, Diciembre +70% transacciones
4. **Desbalance**: 79.7% clientes tipo ABARROTES
5. **Cold-start**: 88.3% productos sin historial de ventas

## Autores

- Benito Fuentes
- Sebastian Vergara

## Fecha

Septiembre 2024

## Licencia

Proyecto académico - MDS7202 Laboratorio de Programación Científica
