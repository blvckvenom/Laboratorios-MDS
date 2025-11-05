# dags/hiring_dynamic_functions.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def _make_estimator(name: str):
    return (
        RandomForestClassifier(n_estimators=300, random_state=42) if name=='rf' else
        (LogisticRegression(max_iter=200) if name=='logreg' else
         (GradientBoostingClassifier(random_state=42) if name=='gboost' else
          RandomForestClassifier(n_estimators=300, random_state=42)))
    )
from pathlib import Path
import pandas as pd

def _run_dir(ds_nodash: str, base_dir: str) -> Path:
    return Path(base_dir) / "runs" / ds_nodash

def create_folders(ds_nodash: str, base_dir: str, **kwargs):
    rd = _run_dir(ds_nodash, base_dir)
    (rd / "raw").mkdir(parents=True, exist_ok=True)
    (rd / "preprocessed").mkdir(parents=True, exist_ok=True)
    (rd / "splits").mkdir(parents=True, exist_ok=True)
    (rd / "models").mkdir(parents=True, exist_ok=True)
    print(f"[create_folders] ready at {rd}")

def load_and_merge(ds_nodash: str, base_dir: str, **kwargs):
    rd = _run_dir(ds_nodash, base_dir)
    raw = rd / "raw"
    pre = rd / "preprocessed"
    pre.mkdir(parents=True, exist_ok=True)

    p1 = raw / "data_1.csv"
    p2 = raw / "data_2.csv"

    if p1.exists() and p2.exists():
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        df = pd.concat((df1, df2), axis=0, ignore_index=True)
    elif p1.exists():
        df = pd.read_csv(p1)
    elif p2.exists():
        df = pd.read_csv(p2)
    else:
        raise FileNotFoundError("[load_and_merge] No hay data en raw/ (data_1.csv o data_2.csv)")

    out = pre / "data_merged.csv"
    df.to_csv(out, index=False)
    print(f"[load_and_merge] merged -> {out} (rows={len(df)})")
# ===== split_data =====
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(ds_nodash: str, base_dir: str, test_size: float = 0.2, seed: int = 42):
    run_dir = Path(base_dir) / 'runs' / ds_nodash
    merged = run_dir / 'preprocessed' / 'data_merged.csv'
    if not merged.exists():
        raise FileNotFoundError(f'[split_data] No existe {merged}')

    df = pd.read_csv(merged)

    # Detecta target
    target_col = (
        'HiringDecision' if 'HiringDecision' in df.columns else
        ('Hired' if 'Hired' in df.columns else
         ('target' if 'target' in df.columns else df.columns[-1]))
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    out = run_dir / 'splits'
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([X_tr, y_tr], axis=1).to_csv(out / 'train.csv', index=False)
    pd.concat([X_te, y_te], axis=1).to_csv(out / 'test.csv', index=False)

    print(f'[split_data] target={target_col} train={len(X_tr)} test={len(X_te)} -> {out}')
# ===== train_model =====
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def train_model(ds_nodash: str, base_dir: str, model_name: str, model=None):
    run_dir = Path(base_dir) / 'runs' / ds_nodash
    train_csv = run_dir / 'splits' / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(f'[train_model] No existe {train_csv}')

    df = pd.read_csv(train_csv)

    # Detecta target
    target_col = (
        'HiringDecision' if 'HiringDecision' in df.columns else
        ('Hired' if 'Hired' in df.columns else
         ('target' if 'target' in df.columns else df.columns[-1]))
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Separa columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    pipe = Pipeline(steps=[('pre', pre), ('clf', _make_estimator(model_name))])
    pipe.fit(X, y)

    out_dir = run_dir / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{model_name}.joblib'
    joblib.dump(pipe, out_path)

    print(f'[train_model] saved: {out_path}')
# ===== evaluate_models =====
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_models(ds_nodash: str, base_dir: str):
    run_dir = Path(base_dir) / 'runs' / ds_nodash
    test_csv = run_dir / 'splits' / 'test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f'[evaluate_models] No existe {test_csv}')

    df = pd.read_csv(test_csv)

    # Detecta target
    target_col = (
        'HiringDecision' if 'HiringDecision' in df.columns else
        ('Hired' if 'Hired' in df.columns else
         ('target' if 'target' in df.columns else df.columns[-1]))
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    models_dir = run_dir / 'models'
    rf_path     = models_dir / 'rf.joblib'
    logreg_path = models_dir / 'logreg.joblib'
    gboost_path = models_dir / 'gboost.joblib'

    # Carga solo los 3 modelos esperados
    rf     = joblib.load(rf_path)     if rf_path.exists()     else None
    logreg = joblib.load(logreg_path) if logreg_path.exists() else None
    gboost = joblib.load(gboost_path) if gboost_path.exists() else None

    # Calcula accuracies individualmente
    rf_acc     = rf.score(X, y)     if rf is not None else -1
    logreg_acc = logreg.score(X, y) if logreg is not None else -1
    gboost_acc = gboost.score(X, y) if gboost is not None else -1

    # Selección del mejor 
    best_score, best_name, best_model = max(
        (rf_acc,     'rf',     rf),
        (logreg_acc, 'logreg', logreg),
        (gboost_acc, 'gboost', gboost),
        key=lambda t: t[0]
    )

    # Guarda el mejor como best.joblib
    if best_model is None:
        raise ValueError('[evaluate_models] No se encontró ningún modelo válido')
    best_out = models_dir / 'best.joblib'
    joblib.dump(best_model, best_out)

    print(f'[evaluate_models] rf={rf_acc:.4f}  logreg={logreg_acc:.4f}  gboost={gboost_acc:.4f}')
    print(f'[evaluate_models] best={best_name}  acc={best_score:.4f} -> {best_out}')
