"""Agent workflow with a train -> evaluate -> tune loop.

DataAgent -> FeatureAgent -> TrainerAgent -> EvaluatorAgent --(acc < target)--> TunerAgent -> TrainerAgent ...
                                                  \\--(target met / budget used)--> ReporterAgent
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

from agents import graph_viz

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
TARGET_ACC = 0.80
CANDIDATES = [
    ("LogisticRegression", lambda: LogisticRegression(max_iter=2000)),
    ("DecisionTree(depth=5)", lambda: DecisionTreeClassifier(max_depth=5, random_state=0)),
    ("RandomForest(50)", lambda: RandomForestClassifier(n_estimators=50, random_state=0)),
    ("RandomForest(300)", lambda: RandomForestClassifier(n_estimators=300, random_state=0)),
    ("GradientBoosting", lambda: GradientBoostingClassifier(random_state=0)),
]


def _scale(X):
    return X.assign(n_ratings=np.log1p(X.n_ratings), n_tags=np.log1p(X.n_tags), year=(X.year - 1990) / 30)


class State(dict):
    pass


def data_agent(s):
    url = os.environ.get("WAREHOUSE_URL", "postgresql+psycopg2://airflow:airflow@localhost:5433/warehouse")
    s["df"] = pd.read_sql("SELECT * FROM movie_summary", create_engine(url))
    print(f"[DataAgent] loaded {len(s['df'])} rows from Postgres")
    return "FeatureAgent"


def feature_agent(s):
    df = s["df"]
    y = (df["avg_rating"] >= 3.5).astype(int)
    X = pd.concat([df[["year", "n_ratings", "n_tags"]], df["genres"].str.get_dummies("|")], axis=1)
    X_tmp, s["Xte"], y_tmp, s["yte"] = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    s["Xtr"], s["Xva"], s["ytr"], s["yva"] = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
    s["i"], s["history"], s["models"] = 0, [], {}
    print(f"[FeatureAgent] {X.shape[1]} features; train/val/test = {len(s['Xtr'])}/{len(s['Xva'])}/{len(s['Xte'])}")
    return "TrainerAgent"


def trainer_agent(s):
    name, make = CANDIDATES[s["i"]]
    # Logistic regression needs scaled inputs; tree models do not.
    prep = _scale if name == "LogisticRegression" else (lambda X: X)
    model = make().fit(prep(s["Xtr"]), s["ytr"])
    s["models"][name] = (model, prep)
    s["current"] = name
    print(f"[TrainerAgent] trained {name}")
    return "EvaluatorAgent"


def evaluator_agent(s):
    name = s["current"]
    model, prep = s["models"][name]
    pred = model.predict(prep(s["Xva"]))
    acc, f1 = accuracy_score(s["yva"], pred), f1_score(s["yva"], pred)
    s["history"].append({"model": name, "val_accuracy": acc, "val_f1": f1})
    print(f"[EvaluatorAgent] {name}: val_acc={acc:.4f} val_f1={f1:.4f}")
    if acc >= TARGET_ACC or s["i"] + 1 >= len(CANDIDATES):
        return "ReporterAgent"
    return "TunerAgent"


def tuner_agent(s):
    s["i"] += 1
    print(f"[TunerAgent] below target {TARGET_ACC}; next candidate: {CANDIDATES[s['i']][0]}")
    return "TrainerAgent"


def reporter_agent(s):
    best = max(s["history"], key=lambda h: h["val_accuracy"])
    model, prep = s["models"][best["model"]]
    pred = model.predict(prep(s["Xte"]))
    result = {
        "best_model": best["model"],
        "test_accuracy": accuracy_score(s["yte"], pred),
        "test_f1": f1_score(s["yte"], pred),
        "target_accuracy": TARGET_ACC,
        "iterations": len(s["history"]),
        "history": s["history"],
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "agent_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ReporterAgent] best={best['model']} TEST accuracy={result['test_accuracy']:.4f} f1={result['test_f1']:.4f}")
    return None


AGENTS = {
    "DataAgent": data_agent, "FeatureAgent": feature_agent, "TrainerAgent": trainer_agent,
    "EvaluatorAgent": evaluator_agent, "TunerAgent": tuner_agent, "ReporterAgent": reporter_agent,
}


def run():
    s, node, visited = State(), "DataAgent", set()
    while node:
        graph_viz.draw(active=node, visited=visited)
        nxt = AGENTS[node](s)
        visited.add(node)
        node = nxt
    graph_viz.draw(active=None, visited=visited)


if __name__ == "__main__":
    run()
