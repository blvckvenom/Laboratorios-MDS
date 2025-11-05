# dags/hiring_functions.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

import json
import tempfile
import os
import socket, time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# ----------------------------
# Helpers de rutas
# ----------------------------

def _base_dir(base_dir: Optional[str | Path] = None) -> Path:
    return Path(base_dir) if base_dir is not None else Path.cwd()

def _run_dir_from_context(ds_nodash: str, base_dir: Optional[str | Path] = None) -> Path:
    return _base_dir(base_dir) / ds_nodash

def _ensure_subdirs(run_dir: Path) -> None:
    tuple(map(lambda s: (run_dir / s).mkdir(parents=True, exist_ok=True), ("raw", "splits", "models")))

def _latest_model_path(base_dir: Optional[str | Path] = None) -> Optional[Path]:
    root = _base_dir(base_dir)
    dirs = list(filter(lambda p: p.is_dir() and re.fullmatch(r"\d{8}", p.name) is not None, root.iterdir()))
    if not dirs:
        return None
    dirs_sorted = sorted(dirs, key=lambda p: p.name, reverse=True)
    candidates = list(map(lambda d: d / "models" / "hiring_rf.joblib", dirs_sorted))
    hits = list(filter(lambda p: p.exists(), candidates))
    return hits[0] if len(hits) > 0 else None


# ----------------------------
# 1) create_folders 
# ----------------------------

def create_folders(ds_nodash: str, base_dir: str = "/opt/airflow") -> None:
    run_dir = Path(base_dir) / ds_nodash
    run_dir.mkdir(parents=True, exist_ok=True)
    # Subcarpetas requeridas
    subdirs = tuple(map(lambda n: run_dir / n, ("raw", "splits", "models")))
    # Crear cada subcarpeta 
    tuple(map(lambda p: p.mkdir(parents=True, exist_ok=True), subdirs))

    print(f"[create_folders] Directorio de ejecución: {run_dir} (raw/splits/models creados)")


# ----------------------------
# 2) split_data 
# ----------------------------

def split_data(ds_nodash: str, base_dir: str = "/opt/airflow", filename: str = "data_1.csv") -> None:
    """
    Busca el CSV primero en base_dir (p.ej., /opt/airflow/<ds>/raw/data_1.csv).
    Si no está, prueba en rutas alternativas típicas de Airflow sin usar for/while.
    Guarda splits en <run_dir>/splits.
    """
    # Candidatas de run_dir
    candidates = (
        Path(base_dir) / ds_nodash,
        Path("/opt/airflow") / ds_nodash,
        Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow")) / ds_nodash,
        Path(tempfile.gettempdir()) / ds_nodash,
        Path.cwd() / ds_nodash,
    )

    run_dir = next(
        (p for p in candidates if (p / "raw" / filename).exists()),
        Path(base_dir) / ds_nodash,
    )

    raw_path = run_dir / "raw" / filename
    if not raw_path.exists():
        raise FileNotFoundError(
            f"[split_data] No se encontró el archivo: {raw_path}. "
            f"Asegúrate de que esté guardado como '{filename}' en '{run_dir / 'raw'}'."
        )

    df = pd.read_csv(raw_path)
    if "HiringDecision" not in df.columns:
        raise KeyError("[split_data] No se encuentra la columna objetivo 'HiringDecision' en el CSV")

    y = df["HiringDecision"]
    X = df.drop(columns=["HiringDecision"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Guardado sin bucles
    (splits_dir / "X_train.csv").write_text(X_train.to_csv(index=False))
    (splits_dir / "X_test.csv").write_text(X_test.to_csv(index=False))
    (splits_dir / "y_train.csv").write_text(y_train.to_csv(index=False))
    (splits_dir / "y_test.csv").write_text(y_test.to_csv(index=False))

    print(f"[split_data] Splits guardados en: {splits_dir}")


# ----------------------------
# 3) preprocess_and_train
# ----------------------------

def preprocess_and_train(ds_nodash: str, base_dir: str = "/opt/airflow") -> None:
    # Resolver run_dir robusto
    candidates = (
        Path(base_dir) / ds_nodash,
        Path("/opt/airflow") / ds_nodash,
        Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow")) / ds_nodash,
        Path(tempfile.gettempdir()) / ds_nodash,
        Path.cwd() / ds_nodash,
    )
    run_dir = next((p for p in candidates if (p / "splits").exists()), Path(base_dir) / ds_nodash)

    splits_dir = run_dir / "splits"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Leer splits
    X_train = pd.read_csv(splits_dir / "X_train.csv")
    X_test  = pd.read_csv(splits_dir / "X_test.csv")
    y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze("columns")
    y_test  = pd.read_csv(splits_dir / "y_test.csv").squeeze("columns")

    # Columnas del dataset
    all_cols = [
        "Age","Gender","EducationLevel","ExperienceYears","PreviousCompanies",
        "DistanceFromCompany","InterviewScore","SkillScore","PersonalityScore","RecruitmentStrategy"
    ]
    # Asegurar orden/consistencia de columnas
    X_train = X_train.loc[:, all_cols]
    X_test  = X_test.loc[:, all_cols]

    # Tratemos como categóricas (one-hot) las de codificación discreta
    cat_cols = ["Gender", "EducationLevel", "RecruitmentStrategy"]
    num_cols = list(filter(lambda c: c not in cat_cols, all_cols))

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=(
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        )
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=(
        ("pre", pre),
        ("clf", clf),
    ))

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1p = f1_score(y_test, y_pred, pos_label=1)

    model_path = models_dir / "hiring_rf.joblib"
    joblib.dump(pipe, model_path)

    print(f"[preprocess_and_train] Modelo guardado en: {model_path}")
    print(f"[preprocess_and_train] Accuracy (test): {acc:.4f}")
    print(f"[preprocess_and_train] F1-score clase positiva (1): {f1p:.4f}")


# ----------------------------
# 4) gradio_interface 
# ----------------------------

def _resolve_run_dir(ds_nodash: str, base_dir: str = "/opt/airflow") -> Path:
    cands = (
        Path(base_dir) / ds_nodash,
        Path("/opt/airflow") / ds_nodash,
        Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow")) / ds_nodash,
        Path(tempfile.gettempdir()) / ds_nodash,
        Path.cwd() / ds_nodash,
    )
    return next((p for p in cands if p.exists()), Path(base_dir) / ds_nodash)

def gradio_interface(ds_nodash: str, base_dir: str, port: int = 7870):
    import os, json, time
    import pandas as pd
    import joblib
    import gradio as gr
    from pathlib import Path

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False" 

    def _resolve_run_dir(ds_nodash, base_dir):
        return Path(base_dir) / "runs" / ds_nodash

    run_dir = _resolve_run_dir(ds_nodash, base_dir)
    model_path = run_dir / "models" / "hiring_rf.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"[gradio_interface] No existe el modelo en {model_path}")

    def _predict(file) -> str:
        with open(file.name, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        pipe = joblib.load(model_path)
        req = getattr(pipe, "feature_names_in_", None)
        df = df.reindex(columns=list(req), fill_value=0) if req is not None else df
        pred = int(pipe.predict(df)[0])
        return "Contratado" if pred == 1 else "No contratado"

    demo = gr.Interface(
        fn=_predict,
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs=gr.Textbox(label="Predicción"),
        title=f"Hiring Decision Prediction — run {ds_nodash}",
        description="Sube un JSON con las características de entrada para predecir si será contratado o no.",
    )


    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,               
        show_error=True,
        show_api=False,            
        prevent_thread_lock=True,  
    )

 # --- Logs útiles y polling ---
    print(f"[gradio] Local URL: http://127.0.0.1:{port}", flush=True)

    import time
    def _wait_share(deadline):
        su = getattr(demo, "share_url", None)
        if su:
            return su
        if time.time() >= deadline:
            return None
        time.sleep(1)
        return _wait_share(deadline)

    su = _wait_share(time.time() + 60)  # espera max 60s
    print(f"[gradio] Public URL: {su or '(no disponible tras 60s)'}", flush=True)


    time.sleep(900)
