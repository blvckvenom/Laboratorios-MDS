# Conclusiones

En este proyecto, el uso de herramientas de tracking como MLflow y el despliegue mediante contenedores Docker fortaleció la reproducibilidad y el ciclo de desarrollo al permitir versionar experimentos, comparar corridas y recuperar artefactos o modelos concretos para evaluación y producción.

Por otro lado, el trabajo con Gradio y FastAPI facilitó un prototipado rápido de la interfaz y endpoints documentados automáticamente, aunque presentó desafíos prácticos interesantes como:
- Garantizar que el preprocesamiento aplicado en producción (mappings categóricos, tipos category, normalizaciones) sea idéntico al usado en entrenamiento
- Manejar dependencias y optimizar el tamaño de las imágenes Docker
- Validar correctamente las entradas desde la UI para evitar fallos en inferencia

Airflow aportó la orquestación necesaria para ejecutar, programar y auditar todo el flujo (con reintentos, dependencias y XComs), permitiendo además implementar lógica condicional como reentrenamientos sólo cuando se detecta drift y escalar tareas pesadas a workers dedicados.

## Propuestas para futuras versiones

Para futuras versiones podríamos proponer:
- Añadir pruebas automáticas para transformaciones e inferencia
- Instrumentar monitorización en producción (latencia, tasa de errores, distribución de predicciones y métricas de rendimiento para detectar drift)
- Establecer CI/CD que construya y promueva imágenes y modelos entre entornos con gating por métricas
- Mejorar la gestión de versiones y rollbacks de artefactos
- Aumentar la observabilidad de features (feature store o snapshots) para tomar decisiones informadas sobre reentrenamientos (incremental vs full)

Invertir en infraestructura y pruebas reduce sustancialmente el riesgo al llevar modelos a producción y facilita escalar soluciones de ML.
