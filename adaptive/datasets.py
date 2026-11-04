"""Builds the dataset variants used by the test cases (all derived from the class repo's CSVs)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.core import CLASS_REPO

D = CLASS_REPO / "datasets"
OUT = Path(os.environ.get("OUTPUT_DIR", "output")) / "cases"


def prepare(kind):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if kind == "churn":
        return str(D / "churn_sample.csv"), "churned"
    df = pd.read_csv(D / "Titanic-Dataset.csv")
    if kind == "titanic":
        return str(D / "Titanic-Dataset.csv"), "Survived"
    if kind == "titanic_nan_labels":
        df.loc[rng.random(len(df)) < 0.06, "Survived"] = np.nan
    elif kind == "titanic_nan_features":
        for c in ("Age", "Fare", "Pclass"):
            df.loc[rng.random(len(df)) < 0.15, c] = np.nan
    elif kind == "titanic_tiny":
        df = df.sample(60, random_state=0)
    p = OUT / f"{kind}.csv"
    df.to_csv(p, index=False)
    return str(p), "Survived"
