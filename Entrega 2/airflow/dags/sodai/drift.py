from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import DRIFT_DIR, DRIFT_THRESHOLD_PSI
from .data_io import cargar_dataset_modelado


def calcular_psi(referencia, actual, n_bins=10):
    """
    calcula population stability index (psi) entre dos distribuciones

    psi mide que tan diferente es la distribucion actual vs la de referencia
    valores tipicos:
    - psi < 0.1: sin cambio significativo
    - 0.1 <= psi < 0.2: cambio moderado
    - psi >= 0.2: cambio significativo (drift detectado)

    retorna el valor de psi
    """
    # crear bins basados en la distribucion de referencia
    bins = np.linspace(referencia.min(), referencia.max(), n_bins + 1)
    bins[0] = -np.inf  # incluir valores menores
    bins[-1] = np.inf  # incluir valores mayores

    # calcular frecuencias en cada bin
    ref_counts, _ = np.histogram(referencia, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    # convertir a proporciones
    ref_props = ref_counts / len(referencia)
    act_props = act_counts / len(actual)

    # evitar division por cero agregando un epsilon pequeno
    epsilon = 1e-10
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    act_props = np.where(act_props == 0, epsilon, act_props)

    # calcular psi
    psi = np.sum((act_props - ref_props) * np.log(act_props / ref_props))

    return float(psi)


def calcular_ks_statistic(referencia, actual):
    """
    calcula kolmogorov-smirnov statistic entre dos distribuciones

    ks mide la maxima diferencia entre las funciones de distribucion acumulada
    valores:
    - ks cerca de 0: distribuciones similares
    - ks cerca de 1: distribuciones muy diferentes

    retorna (ks_statistic, p_value)
    """
    ks_stat, p_value = stats.ks_2samp(referencia, actual)
    return float(ks_stat), float(p_value)


def interpretar_drift(psi_score, ks_stat, p_value):
    """
    interpreta los resultados de drift y determina si hay drift significativo

    retorna:
    - drift_detectado: bool indicando si hay drift
    - nivel: str con nivel de drift ("sin_drift", "moderado", "alto")
    - razon: str explicando la decision
    """
    drift_detectado = False
    nivel = "sin_drift"
    razon = ""

    # evaluar psi
    if psi_score >= 0.2:
        drift_detectado = True
        nivel = "alto"
        razon = f"psi={psi_score:.3f} supera umbral de 0.2"
    elif psi_score >= 0.1:
        nivel = "moderado"
        razon = f"psi={psi_score:.3f} indica cambio moderado"
    else:
        razon = f"psi={psi_score:.3f} por debajo de 0.1"

    # complementar con ks test
    # p_value < 0.05 indica que las distribuciones son significativamente diferentes
    if p_value < 0.05 and ks_stat > 0.2:
        if not drift_detectado:
            drift_detectado = True
            nivel = "alto" if ks_stat > 0.4 else "moderado"
            razon += f"; ks_stat={ks_stat:.3f} con p_value={p_value:.4f}"

    return drift_detectado, nivel, razon


def calcular_drift(nombre_actual: str = "df_modelado.parquet",
                   nombre_referencia: str = None) -> Path:
    """
    calcula drift entre dataset actual y uno de referencia

    si no hay dataset de referencia, usa el actual como baseline
    y solo calcula estadisticas descriptivas

    usa metricas estadisticas:
    - psi (population stability index)
    - ks test (kolmogorov-smirnov)

    retorna path del reporte generado
    """

    # cargar dataset actual
    df_actual = cargar_dataset_modelado(nombre_actual)

    if df_actual.empty:
        raise ValueError("el dataframe actual esta vacio")

    # obtener columnas numericas
    numeric_cols = df_actual.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        raise ValueError("no se encontraron columnas numericas para calcular drift")

    print(f"analizando drift en {len(numeric_cols)} columnas numericas")

    # estructura base del reporte
    reporte = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_actual": nombre_actual,
        "n_rows_actual": int(df_actual.shape[0]),
        "n_columns": int(df_actual.shape[1]),
        "numeric_columns": numeric_cols,
    }

    # intentar cargar dataset de referencia
    df_referencia = None
    if nombre_referencia:
        try:
            df_referencia = cargar_dataset_modelado(nombre_referencia)
            reporte["dataset_referencia"] = nombre_referencia
            reporte["n_rows_referencia"] = int(df_referencia.shape[0])
        except Exception as e:
            print(f"no se pudo cargar dataset de referencia: {e}")
            print("usando solo estadisticas del dataset actual")

    # calcular estadisticas y drift
    drift_results = {}
    drift_global = False
    psi_promedio = 0.0

    for col in numeric_cols:
        col_stats = {
            "actual_mean": float(df_actual[col].mean()),
            "actual_std": float(df_actual[col].std()),
            "actual_min": float(df_actual[col].min()),
            "actual_max": float(df_actual[col].max()),
        }

        # si hay dataset de referencia, calcular metricas de drift
        if df_referencia is not None and col in df_referencia.columns:
            ref_values = df_referencia[col].dropna()
            act_values = df_actual[col].dropna()

            # calcular psi
            psi = calcular_psi(ref_values, act_values)

            # calcular ks
            ks_stat, p_value = calcular_ks_statistic(ref_values, act_values)

            # interpretar drift
            drift_detectado, nivel, razon = interpretar_drift(psi, ks_stat, p_value)

            col_stats.update({
                "referencia_mean": float(df_referencia[col].mean()),
                "referencia_std": float(df_referencia[col].std()),
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": p_value,
                "drift_detectado": drift_detectado,
                "nivel_drift": nivel,
                "razon": razon
            })

            psi_promedio += psi

            if drift_detectado:
                drift_global = True
                print(f"  [DRIFT] {col}: {razon}")

        drift_results[col] = col_stats

    # calcular psi promedio si hubo comparacion
    if df_referencia is not None:
        psi_promedio = psi_promedio / len(numeric_cols)

        reporte["psi_promedio"] = float(psi_promedio)
        reporte["drift_detectado"] = drift_global
        reporte["umbral_psi"] = DRIFT_THRESHOLD_PSI

        # decision final de reentrenamiento
        reentrenar = drift_global or psi_promedio >= DRIFT_THRESHOLD_PSI
        reporte["requiere_reentrenamiento"] = reentrenar

        if reentrenar:
            print(f"\n[ALERTA] drift detectado - se recomienda reentrenar modelo")
            print(f"psi promedio: {psi_promedio:.3f}")
        else:
            print(f"\nno se detecto drift significativo")
            print(f"psi promedio: {psi_promedio:.3f}")

    reporte["columnas_drift"] = drift_results

    # guardar reporte
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    salida = DRIFT_DIR / "drift_report.json"

    with open(salida, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"\nreporte de drift guardado en: {salida}")

    return salida


def hay_drift_significativo(drift_report_path: Path = None) -> bool:
    """
    lee el reporte de drift y determina si hay drift significativo
    usado para flujos condicionales en el dag

    retorna true si se requiere reentrenamiento
    """
    if drift_report_path is None:
        drift_report_path = DRIFT_DIR / "drift_report.json"

    if not drift_report_path.exists():
        print("no existe reporte de drift, asumiendo que no hay drift")
        return False

    try:
        with open(drift_report_path, 'r', encoding='utf-8') as f:
            reporte = json.load(f)

        # si no hay dataset de referencia, es primera ejecucion -> forzar reentrenamiento
        if "dataset_referencia" not in reporte:
            print("primera ejecucion detectada - forzando reentrenamiento inicial")
            return True

        # verificar si el reporte tiene la info de reentrenamiento
        requiere_reentrenamiento = reporte.get("requiere_reentrenamiento", False)

        return requiere_reentrenamiento

    except Exception as e:
        print(f"error al leer reporte de drift: {e}")
        return False
