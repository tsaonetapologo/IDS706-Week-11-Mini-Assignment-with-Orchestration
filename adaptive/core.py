"""Shared data access, features and model pool used by all three systems (fair comparison)."""
import copy
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


class TransientError(Exception):
    pass


def _scale(X):
    X = X.copy()
    X["n_ratings"], X["n_tags"] = np.log1p(X.n_ratings), np.log1p(X.n_tags)
    X["year"] = (X.year - 1990) / 30
    return X


# name -> (factory, complexity rank, scaler). Higher rank = more flexible model.
POOL = {
    "LogisticRegression": (lambda w: LogisticRegression(max_iter=2000, class_weight=w), 1, _scale),
    "DecisionTree(depth=3)": (lambda w: DecisionTreeClassifier(max_depth=3, class_weight=w, random_state=0), 2, None),
    "DecisionTree(depth=6)": (lambda w: DecisionTreeClassifier(max_depth=6, class_weight=w, random_state=0), 3, None),
    "RandomForest(300)": (lambda w: RandomForestClassifier(n_estimators=300, class_weight=w, random_state=0, n_jobs=-1), 4, None),
    "GradientBoosting": (lambda w: GradientBoostingClassifier(random_state=0), 5, None),
    "GradientBoosting(deep)": (lambda w: GradientBoostingClassifier(max_depth=5, n_estimators=200, random_state=0), 6, None),
}


def make_model(name, class_weight=None):
    factory, _, scaler = POOL[name]
    return factory(class_weight), (scaler or (lambda X: X))


class Faults:
    """Per-run fault injection: fail the first N DB loads / N model fits."""

    def __init__(self, spec):
        self.spec = copy.deepcopy(spec or {})

    def maybe_fail(self, kind):
        left = self.spec.get(kind, 0)
        if left > 0:
            self.spec[kind] = left - 1
            raise TransientError(f"injected {kind} failure")


def load_raw(cfg, faults):
    faults.maybe_fail("db_fail")
    url = os.environ.get("WAREHOUSE_URL", "postgresql+psycopg2://airflow:airflow@localhost:5433/warehouse")
    df = pd.read_sql("SELECT * FROM movie_summary", create_engine(url))
    if cfg.get("n_rows"):
        df = df.sample(cfg["n_rows"], random_state=0)
    if cfg.get("corrupt_frac"):
        rng = np.random.default_rng(0)
        for col in ("year", "n_ratings"):
            df.loc[rng.random(len(df)) < cfg["corrupt_frac"], col] = np.nan
    return df.reset_index(drop=True)


def build_xy(df, label_threshold):
    y = (df["avg_rating"] >= label_threshold).astype(int)
    X = pd.concat([df[["year", "n_ratings", "n_tags"]], df["genres"].str.get_dummies("|")], axis=1)
    return X, y


def split3(X, y):
    strat = y if y.value_counts().min() >= 3 else None
    X_t, Xte, y_t, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    strat = y_t if y_t.value_counts().min() >= 3 else None
    Xtr, Xva, ytr, yva = train_test_split(X_t, y_t, test_size=0.25, random_state=42, stratify=strat)
    return Xtr, Xva, Xte, ytr, yva, yte


def score(y, pred, metric):
    from sklearn.metrics import accuracy_score, f1_score
    return f1_score(y, pred, zero_division=0) if metric == "f1" else accuracy_score(y, pred)
