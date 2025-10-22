# optimize.py
import os
import json
import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import optuna
import optuna.visualization.matplotlib as ovm

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


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
TARGET = "Potability"

DEFAULT_DATA = "data/water_potability.csv"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_TRIALS = 30


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)
    return df


def make_preprocessor():
    numeric = FEATURES
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric,
            )
        ],
        remainder="drop",
        n_jobs=None,
    )
    return pre


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return (neg / max(pos, 1)) if pos > 0 else 1.0


def save_versions_artifact(out_dir: Path):
    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
    }
    try:
        import pandas as _p

        versions["pandas"] = _p.__version__
    except Exception:
        pass
    try:
        import numpy as _n

        versions["numpy"] = _n.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk

        versions["scikit_learn"] = _sk.__version__
    except Exception:
        pass
    try:
        import mlflow as _ml

        versions["mlflow"] = _ml.__version__
    except Exception:
        pass
    try:
        import optuna as _op

        versions["optuna"] = _op.__version__
    except Exception:
        pass
    try:
        import xgboost as _xgb

        versions["xgboost"] = _xgb.__version__
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def plot_feature_importance(model, feature_names, out_path: Path):
    """
    Intenta primero feature_importances_.
    Si no existe, usa booster.get_score().
    """
    importances = None
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_names):
            values = importances
            names = feature_names
        else:
            raise AttributeError
    except Exception:
        try:
            booster = model.named_steps["clf"].get_booster()
            score_dict = booster.get_score(importance_type="weight")
            pairs = sorted(score_dict.items(), key=lambda kv: int(kv[0][1:]))
            values = [kv[1] for kv in pairs]
            idxs = [int(kv[0][1:]) for kv in pairs]
            names = [feature_names[i] for i in idxs]
        except Exception:
            return

    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(names))
    plt.barh(y_pos, values)
    plt.yticks(y_pos, names)
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    runs_sorted = runs.sort_values("metrics.valid_f1", ascending=False, na_position="last")
    best_model_id = runs_sorted["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model

def optimize_model(
    data_path: str = DEFAULT_DATA,
    n_trials: int = DEFAULT_N_TRIALS,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    warnings.filterwarnings("ignore")
    df = load_dataset(data_path).dropna(subset=[TARGET])
    X = df[FEATURES]
    y = df[TARGET].to_numpy()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pre = make_preprocessor()
    scale_pos_weight = compute_scale_pos_weight(y_tr)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"water-potability-xgb-optuna-{timestamp}"
    exp_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
        }

        clf = XGBClassifier(**params)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        run_name = (
            f"XGB lr={params['learning_rate']:.3f} "
            f"depth={params['max_depth']} subs={params['subsample']:.2f} "
            f"cols={params['colsample_bytree']:.2f} est={params['n_estimators']}"
        )

        with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_va)
            f1 = f1_score(y_va, y_pred)

            mlflow.log_metric("valid_f1", float(f1))
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            return f1

    study = optuna.create_study(direction="maximize", study_name=f"xgb-study-{timestamp}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    with mlflow.start_run(run_name="optuna-summary", experiment_id=exp_id):
        plots_dir = Path("artifacts") / "plots"
        models_dir = Path("artifacts") / "models"
        save_versions_artifact(plots_dir)

        fig1 = ovm.plot_optimization_history(study)
        fig1.figure.set_size_inches(8, 5)
        (plots_dir / "optuna_optimization_history.png").parent.mkdir(parents=True, exist_ok=True)
        fig1.figure.savefig(plots_dir / "optuna_optimization_history.png", dpi=150)
        plt.close(fig1.figure)

        fig2 = ovm.plot_param_importances(study)
        fig2.figure.set_size_inches(8, 5)
        fig2.figure.savefig(plots_dir / "optuna_param_importances.png", dpi=150)
        plt.close(fig2.figure)

        best_model = get_best_model(exp_id)

        models_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = models_dir / "best_model.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)

        try:
            params_json = best_model.get_params()
            (plots_dir / "best_model_params.json").write_text(
                json.dumps(params_json, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass

        try:
            plot_feature_importance(best_model, FEATURES, plots_dir / "feature_importance.png")
        except Exception:
            pass

        mlflow.log_artifact(str(plots_dir / "optuna_optimization_history.png"), artifact_path="plots")
        mlflow.log_artifact(str(plots_dir / "optuna_param_importances.png"), artifact_path="plots")
        if (plots_dir / "feature_importance.png").exists():
            mlflow.log_artifact(str(plots_dir / "feature_importance.png"), artifact_path="plots")
        if (plots_dir / "best_model_params.json").exists():
            mlflow.log_artifact(str(plots_dir / "best_model_params.json"), artifact_path="plots")
        if (plots_dir / "versions.json").exists():
            mlflow.log_artifact(str(plots_dir / "versions.json"), artifact_path="plots")

        mlflow.log_artifact(str(best_model_path), artifact_path="models")

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("best_valid_f1", float(study.best_value))
        mlflow.log_dict(study.best_params, artifact_file="best_params.json")

    print(f"[MLflow] Experiment: {experiment_name} (id={exp_id})")
    print(f"[Optuna] Best F1: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {study.best_params}")
    print(f"Modelo serializado en: artifacts/models/best_model.pkl")
    return exp_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize XGBoost with Optuna + MLflow")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Ruta al CSV")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Número de trials Optuna")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, help="Tamaño de validación")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random state")
    args = parser.parse_args()

    optimize_model(
        data_path=args.data,
        n_trials=args.trials,
        test_size=args.test_size,
        random_state=args.seed,
    )
