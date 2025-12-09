# Entrega 3 - Informe de Evaluación MLOps

Este directorio contiene el informe completo de la Entrega 3 del proyecto SodAI Drinks.

## Estructura del Directorio

```
entrega_3/
├── informe.ipynb           # Notebook principal con el análisis completo
├── README.md               # Este archivo
├── artifacts/              # Visualizaciones (PNG) de cada semana
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   └── week4/
├── predictions/            # Predicciones del modelo (.parquet)
│   ├── predictions_week1.parquet
│   ├── predictions_week2.parquet
│   ├── predictions_week3.parquet
│   └── predictions_week4.parquet
└── data/                   # Datos de evaluación y ground truth
    ├── evaluation_week*.json
    ├── drift_report_week*.json
    └── batch_t*.parquet
```

## Cómo Ejecutar el Notebook

### Opción 1: Jupyter Notebook
```bash
cd entrega_3
jupyter notebook informe.ipynb
```

### Opción 2: JupyterLab
```bash
cd entrega_3
jupyter lab informe.ipynb
```

### Opción 3: VS Code
1. Abrir VS Code en la carpeta `entrega_3`
2. Instalar extensión "Jupyter" si no está instalada
3. Abrir `informe.ipynb`
4. Seleccionar kernel de Python 3.9+

## Dependencias Requeridas

Instalar todas las dependencias necesarias:

```bash
pip install pandas numpy matplotlib seaborn pyarrow ipython jupyter
```

O usando el archivo requirements del proyecto principal:

```bash
pip install -r ../requirements.txt
```

## Contenido del Informe

### Sección 1: Análisis Individual por Batch [3.0 pts]
- **Batch 1** (01-05 Ene): Análisis del cold start
- **Batch 2** (06-12 Ene): Impacto del primer reentrenamiento
- **Batch 3** (13-19 Ene): Estabilización del modelo
- **Batch 4** (20-26 Ene): Mejor rendimiento alcanzado

### Sección 2: Análisis Comparativo [2.0 pts]
- Gráficos de tendencias (F1, Precision/Recall, ROC-AUC, PSI)
- Tabla consolidada de métricas
- Comparación local vs CodaBench
- Análisis de drift

### Sección 3: Conclusiones y Aprendizajes [1.0 pts]
Respuestas a 9 preguntas clave:
1. Variación de métricas
2. Peor rendimiento y causas
3. Detección de drift
4. Impacto del reentrenamiento
5. Decisión técnica más impactante
6. Hiperparámetro más importante
7. Variable más influyente
8. Aprendizajes de negocio
9. Limitaciones detectadas

## Métricas Clave

| Week | F1 Score | Precision | Recall | ROC-AUC | CodaBench F1 | PSI Drift |
|------|----------|-----------|--------|---------|--------------|-----------|
| 1    | 0.2247   | 0.3786    | 0.1597 | 0.9319  | -            | -         |
| 2    | 0.5263   | 0.4642    | 0.6076 | 0.9294  | 0.4758       | 3.44      |
| 3    | 0.5144   | 0.4794    | 0.5548 | 0.9294  | 0.5244       | 2.96      |
| 4    | 0.5351   | 0.4419    | 0.6783 | 0.9330  | 0.5398       | 3.08      |

## Hallazgos Principales

1. **Mejora de 134%** en F1 Score después del primer reentrenamiento
2. **Drift crítico** detectado en todas las semanas (PSI > 2.0)
3. **ROC-AUC estable** (~0.93) a través de todos los batches
4. **Reentrenamiento semanal** es mandatorio para mantener performance
5. Variable más importante: `cp_dias_desde_ultima_compra`

## Notas Importantes

- **NO ejecutar celdas fuera de orden**: El notebook está diseñado para ejecutarse secuencialmente
- **Rutas relativas**: Todas las rutas son relativas al directorio `entrega_3/`
- **Imágenes**: Las visualizaciones se cargan desde `artifacts/weekN/`
- **Datos**: Los archivos JSON y Parquet están en `data/`

## Troubleshooting

### Error: "FileNotFoundError"
- Verificar que estás ejecutando el notebook desde el directorio `entrega_3/`
- Verificar que los archivos en `data/` y `artifacts/` existen

### Error: "ModuleNotFoundError"
- Instalar las dependencias: `pip install pandas numpy matplotlib seaborn pyarrow`

### Las imágenes no se muestran
- Ejecutar todas las celdas en orden
- Verificar que los archivos PNG existen en `artifacts/weekN/`

