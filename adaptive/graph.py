"""Adaptive multi-agent workflow built as a LangGraph StateGraph.

Agents: Loader, Validator, Cleaner, Planner, Trainer, Evaluator, Diagnostician, Reporter.
Adaptive behaviour (all via conditional edges on intermediate state):
  * Loader/Trainer failures  -> retry (bounded), then degrade gracefully to Reporter
  * Validator finds NaNs      -> route through Cleaner
  * Planner picks the model queue from the data profile (size, class imbalance)
  * Evaluator stops early on target / budget / plateau
  * Diagnostician reads the train-val gap and re-orders the queue (overfit -> simpler, underfit -> stronger)
"""
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from adaptive import core

MAX_RETRIES = 3


class S(TypedDict, total=False):
    cfg: dict
    faults: Any
    df: Any
    split: tuple
    profile: dict
    queue: list
    current: str
    model: Any
    prep: Any
    class_weight: Any
    metric: str
    history: list
    best: dict
    error: str
    retries: dict
    fits: int
    path: list
    status: str
    result: dict


def _visit(s, name):
    return (s.get("path") or []) + [name]


def loader(s):
    r = dict(s.get("retries") or {})
    try:
        df = core.load_raw(s["cfg"], s["faults"])
        return {"df": df, "error": "", "path": _visit(s, "loader")}
    except core.TransientError as e:
        r["loader"] = r.get("loader", 0) + 1
        return {"error": str(e), "retries": r, "path": _visit(s, "loader")}


def after_loader(s):
    if not s.get("error"):
        return "validator"
    return "loader" if s["retries"]["loader"] <= MAX_RETRIES else "reporter"


def validator(s):
    df = s["df"]
    X, y = core.build_xy(df, s["cfg"].get("label_threshold", 3.5))
    prof = {
        "n": len(df),
        "nan_cells": int(X.isna().sum().sum()),
        "minority": float(min(y.mean(), 1 - y.mean())),
    }
    return {"profile": prof, "path": _visit(s, "validator")}


def after_validator(s):
    return "cleaner" if s["profile"]["nan_cells"] > 0 else "planner"


def cleaner(s):
    df = s["df"].copy()
    for col in ("year", "n_ratings", "n_tags"):
        df[col] = df[col].fillna(df[col].median())
    prof = dict(s["profile"], nan_cells=0)
    return {"df": df, "profile": prof, "path": _visit(s, "cleaner")}


def planner(s):
    p, cfg = s["profile"], s["cfg"]
    X, y = core.build_xy(s["df"], cfg.get("label_threshold", 3.5))
    imbalanced = p["minority"] < 0.25
    small = p["n"] < 1000
    if small:  # little data: start simple to avoid overfitting
        queue = ["LogisticRegression", "DecisionTree(depth=3)", "DecisionTree(depth=6)", "RandomForest(300)"]
    elif imbalanced:  # models that accept class_weight
        queue = ["RandomForest(300)", "DecisionTree(depth=6)", "LogisticRegression"]
    else:  # plenty of balanced data: start with the strongest general model
        queue = ["GradientBoosting", "RandomForest(300)", "GradientBoosting(deep)", "LogisticRegression"]
    return {
        "split": core.split3(X, y),
        "queue": queue,
        "class_weight": "balanced" if imbalanced else None,
        "metric": "f1" if imbalanced else "accuracy",
        "history": [], "fits": 0, "retries": s.get("retries") or {},
        "path": _visit(s, "planner"),
    }


def trainer(s):
    name = s["queue"][0]
    Xtr, _, _, ytr, _, _ = s["split"]
    r = dict(s["retries"])
    try:
        s["faults"].maybe_fail("train_fail")
        model, prep = core.make_model(name, s["class_weight"])
        model.fit(prep(Xtr), ytr)
        return {"current": name, "model": model, "prep": prep, "fits": s["fits"] + 1,
                "error": "", "path": _visit(s, "trainer")}
    except core.TransientError as e:
        r["trainer"] = r.get("trainer", 0) + 1
        return {"error": str(e), "retries": r, "path": _visit(s, "trainer")}


def after_trainer(s):
    if not s.get("error"):
        return "evaluator"
    return "trainer" if s["retries"]["trainer"] <= MAX_RETRIES else "reporter"


def evaluator(s):
    Xtr, Xva, _, ytr, yva, _ = s["split"]
    m, prep, metric = s["model"], s["prep"], s["metric"]
    val = core.score(yva, m.predict(prep(Xva)), metric)
    tr = core.score(ytr, m.predict(prep(Xtr)), metric)
    rec = {"model": s["current"], "val": val, "train": tr, "gap": tr - val}
    hist = s["history"] + [rec]
    best = max(hist, key=lambda h: h["val"])
    best = dict(best, model_obj=(m if best is rec else s["best"]["model_obj"]),
                prep=(prep if best is rec else s["best"]["prep"]))
    return {"history": hist, "best": best, "queue": s["queue"][1:], "path": _visit(s, "evaluator")}


def after_evaluator(s):
    cfg = s["cfg"]
    last = s["history"][-1]
    if last["val"] >= cfg.get("target", 0.66):
        return "reporter"
    if not s["queue"] or s["fits"] >= cfg.get("max_fits", 5):
        return "reporter"
    if len(s["history"]) >= 3 and s["best"]["val"] <= s["history"][-3]["val"] + 1e-9:
        return "reporter"  # plateau: no improvement over the last 2 attempts
    return "diagnostician"


def diagnostician(s):
    last, q = s["history"][-1], list(s["queue"])
    rank = lambda n: core.POOL[n][1]
    # Overfit -> try the simplest remaining model; underfit -> the most flexible remaining one.
    q.sort(key=rank, reverse=(last["gap"] <= 0.10))
    return {"queue": q, "path": _visit(s, "diagnostician")}


def reporter(s):
    path = _visit(s, "reporter")
    if "best" not in s or not s.get("best"):
        return {"status": "failed", "path": path,
                "result": {"completed": False, "fits": s.get("fits", 0)}}
    b = s["best"]
    _, _, Xte, _, _, yte = s["split"]
    test = core.score(yte, b["model_obj"].predict(b["prep"](Xte)), s["metric"])
    return {"status": "ok", "path": path, "result": {
        "completed": True, "model": b["model"], "metric_name": s["metric"], "test_metric": test,
        "val_metric": b["val"], "gap": b["gap"], "fits": s["fits"]}}


def build():
    g = StateGraph(S)
    for n, f in [("loader", loader), ("validator", validator), ("cleaner", cleaner), ("planner", planner),
                 ("trainer", trainer), ("evaluator", evaluator), ("diagnostician", diagnostician),
                 ("reporter", reporter)]:
        g.add_node(n, f)
    g.add_edge(START, "loader")
    g.add_conditional_edges("loader", after_loader, ["loader", "validator", "reporter"])
    g.add_conditional_edges("validator", after_validator, ["cleaner", "planner"])
    g.add_edge("cleaner", "planner")
    g.add_edge("planner", "trainer")
    g.add_conditional_edges("trainer", after_trainer, ["trainer", "evaluator", "reporter"])
    g.add_conditional_edges("evaluator", after_evaluator, ["reporter", "diagnostician"])
    g.add_edge("diagnostician", "trainer")
    g.add_edge("reporter", END)
    return g.compile()


def run_adaptive(cfg):
    t = time.time()
    out = build().invoke({"cfg": cfg, "faults": core.Faults(cfg.get("faults")), "path": [], "retries": {}},
                         {"recursion_limit": 60})
    res = dict(out["result"], path=out["path"], retries=out.get("retries", {}), seconds=time.time() - t,
               system="adaptive_langgraph")
    return res
