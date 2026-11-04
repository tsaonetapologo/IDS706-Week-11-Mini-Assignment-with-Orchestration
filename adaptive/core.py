"""Shared pieces: class-repo imports, fault injection, and a fit/eval function that mirrors the class
`loop_runtime._fit_score` (same split convention, preprocessing and model families) but also returns the
train score so an agent can diagnose over/underfitting."""
import contextlib
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CLASS_REPO = Path(__file__).resolve().parents[1] / "third_party" / "automl-research-lab"
sys.path.insert(0, str(CLASS_REPO))

import loop_runtime  # noqa: E402  (class repo)
from loop_runtime import _build_model, _heuristic_profile  # noqa: E402


class TransientError(Exception):
    pass


@contextlib.contextmanager
def inject_faults(spec):
    """Fail the first N pd.read_csv calls ('db_fail') and the first N Pipeline.fit calls ('train_fail').
    Patching library boundaries lets the *unmodified* class code be tested under the same faults."""
    left = dict(spec or {})
    orig_read, orig_fit = pd.read_csv, Pipeline.fit

    def read_csv(*a, **k):
        if left.get("db_fail", 0) > 0:
            left["db_fail"] -= 1
            raise TransientError("injected read failure")
        return orig_read(*a, **k)

    def fit(self, *a, **k):
        if left.get("train_fail", 0) > 0:
            left["train_fail"] -= 1
            raise TransientError("injected training failure")
        return orig_fit(self, *a, **k)

    pd.read_csv, Pipeline.fit = read_csv, fit
    try:
        yield
    finally:
        pd.read_csv, Pipeline.fit = orig_read, orig_fit


# name -> (spec, complexity rank). Same three families as the class loop.
POOL = {
    "LogReg": ({"model_family": "logistic_regression", "hyperparameters": {"C": 1.0, "max_iter": 500}}, 1),
    "RF(depth=3)": ({"model_family": "random_forest", "hyperparameters": {"n_estimators": 100, "max_depth": 3}}, 2),
    "RF(depth=6)": ({"model_family": "random_forest", "hyperparameters": {"n_estimators": 100, "max_depth": 6}}, 3),
    "HGB": ({"model_family": "hist_gradient_boosting", "hyperparameters": {"learning_rate": 0.1, "max_iter": 100}}, 4),
    "HGB(slow,200)": ({"model_family": "hist_gradient_boosting", "hyperparameters": {"learning_rate": 0.05, "max_iter": 200}}, 5),
}


def fit_eval(df, target, spec, seed=42, holdout_frac=0.2, on_holdout=False, drops=()):
    y = df[target]
    X = df.drop(columns=[target, *[c for c in drops if c in df.columns]])
    strat = y if y.value_counts().min() >= 2 else None
    X_t, X_h, y_t, y_h = train_test_split(X, y, test_size=holdout_frac, random_state=seed, stratify=strat)
    if on_holdout:
        Xtr, ytr, Xev, yev = X_t, y_t, X_h, y_h
    else:
        strat = y_t if y_t.value_counts().min() >= 2 else None
        Xtr, Xev, ytr, yev = train_test_split(X_t, y_t, test_size=0.25, random_state=seed, stratify=strat)
    num = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat = [c for c in Xtr.columns if c not in num]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    pipe = Pipeline([("pre", pre), ("clf", _build_model(spec["model_family"], spec["hyperparameters"]))])
    t = time.time()
    pipe.fit(Xtr, ytr)
    return {"f1": float(f1_score(yev, pipe.predict(Xev), zero_division=0)),
            "train_f1": float(f1_score(ytr, pipe.predict(Xtr), zero_division=0)),
            "seconds": time.time() - t}


def holdout_f1(path, target, spec, drops=()):
    """Common yardstick for every system: sealed-holdout F1 using the same split/seed."""
    df = pd.read_csv(path)
    df = df[df[target].notna()]
    return fit_eval(df, target, spec, on_holdout=True, drops=drops)["f1"]
