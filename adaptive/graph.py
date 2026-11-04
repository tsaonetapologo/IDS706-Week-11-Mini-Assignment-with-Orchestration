"""Adaptive multi-agent workflow as a LangGraph StateGraph (CSV in, holdout F1 out).

Agents: Loader, Validator, Cleaner, Planner, Trainer, Evaluator, Diagnostician, Reporter.
Adaptive behaviour (conditional edges on intermediate results):
  * Loader / Trainer failures -> bounded retry, then graceful exit to Reporter
  * Validator finds bad labels -> routed through Cleaner (skipped otherwise; feature NaNs are handled by the imputer)
  * Planner picks a model queue from the data profile (tiny data -> simple models, imbalance -> tree ensembles)
  * Evaluator stops early on target / fit budget / plateau
  * Diagnostician reads the train-val gap and re-orders the queue (overfit -> simpler, underfit -> stronger)
"""
import time
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from adaptive import core

MAX_RETRIES = 3


class S(TypedDict, total=False):
    cfg: dict
    df: Any
    profile: dict
    queue: list
    current: str
    history: list
    best: dict
    error: str
    retries: dict
    fits: int
    path: list
    result: dict


def _p(s, n):
    return (s.get("path") or []) + [n]


def loader(s):
    r = dict(s.get("retries") or {})
    try:
        return {"df": pd.read_csv(s["cfg"]["dataset"]), "error": "", "path": _p(s, "loader")}
    except Exception as e:
        r["loader"] = r.get("loader", 0) + 1
        return {"error": str(e), "retries": r, "path": _p(s, "loader")}


def after_loader(s):
    if not s["error"]:
        return "validator"
    return "loader" if s["retries"]["loader"] <= MAX_RETRIES else "reporter"


def validator(s):
    df, t = s["df"], s["cfg"]["target"]
    if t not in df.columns:
        return {"error": f"target {t} missing", "profile": {"bad": True}, "path": _p(s, "validator")}
    y = df[t].dropna()
    rate = float(y.mean()) if len(y) else 0.0
    prof = {"n": int(len(y)), "nan_target": int(df[t].isna().sum()),
            "minority": min(rate, 1 - rate), "drops": _drops(df, t)}
    return {"profile": prof, "error": "", "path": _p(s, "validator")}


def _drops(df, t):
    return list(_safe_profile(df, t).get("recommended_drops") or [])


def _safe_profile(df, t):
    try:
        return core._heuristic_profile(df.dropna(subset=[t]), t)
    except Exception:
        return {}


def after_validator(s):
    if s.get("error"):
        return "reporter"
    return "cleaner" if s["profile"]["nan_target"] > 0 else "planner"


def cleaner(s):
    df = s["df"].dropna(subset=[s["cfg"]["target"]]).reset_index(drop=True)
    return {"df": df, "path": _p(s, "cleaner")}


def planner(s):
    p = s["profile"]
    if p["n"] < 200:
        q = ["LogReg", "RF(depth=3)", "RF(depth=6)", "HGB"]
    elif p["minority"] < 0.20:
        q = ["LogReg", "RF(depth=3)", "RF(depth=6)", "HGB"]  # rare positives: balanced LR before overfit-prone trees
    else:
        q = ["HGB", "RF(depth=6)", "LogReg", "HGB(slow,200)"]
    return {"queue": q, "history": [], "fits": 0, "retries": s.get("retries") or {}, "path": _p(s, "planner")}


def trainer(s):
    name, r = s["queue"][0], dict(s["retries"])
    try:
        res = core.fit_eval(s["df"], s["cfg"]["target"], core.POOL[name][0], drops=s["profile"]["drops"])
        rec = {"model": name, "val": res["f1"], "gap": res["train_f1"] - res["f1"]}
        hist = s["history"] + [rec]
        best = max(hist, key=lambda h: h["val"])
        return {"current": name, "history": hist, "best": best, "fits": s["fits"] + 1,
                "queue": s["queue"][1:], "error": "", "path": _p(s, "trainer")}
    except core.TransientError as e:
        r["trainer"] = r.get("trainer", 0) + 1
        return {"error": str(e), "retries": r, "path": _p(s, "trainer")}


def after_trainer(s):
    if s["error"]:
        return "trainer" if s["retries"]["trainer"] <= MAX_RETRIES else "reporter"
    return "evaluator"


def evaluator(s):
    return {"path": _p(s, "evaluator")}


def after_evaluator(s):
    cfg, h = s["cfg"], s["history"]
    if h[-1]["val"] >= cfg.get("target_score", 0.7) or not s["queue"] or s["fits"] >= cfg.get("budget", 3):
        return "reporter"
    if len(h) >= 3 and s["best"]["val"] <= h[-3]["val"] + 1e-9:
        return "reporter"  # plateau
    return "diagnostician"


def diagnostician(s):
    q = sorted(s["queue"], key=lambda n: core.POOL[n][1], reverse=(s["history"][-1]["gap"] <= 0.10))
    return {"queue": q, "path": _p(s, "diagnostician")}


def reporter(s):
    path = _p(s, "reporter")
    if s.get("error") or not s.get("best"):
        return {"path": path, "result": {"completed": False, "fits": s.get("fits", 0)}}
    b, cfg = s["best"], s["cfg"]
    f1 = core.holdout_f1(cfg["dataset"], cfg["target"], core.POOL[b["model"]][0], s["profile"]["drops"])
    return {"path": path, "result": {"completed": True, "model": b["model"], "test_metric": f1,
                                     "val_metric": b["val"], "fits": s["fits"]}}


def build():
    g = StateGraph(S)
    for n, f in [("loader", loader), ("validator", validator), ("cleaner", cleaner), ("planner", planner),
                 ("trainer", trainer), ("evaluator", evaluator), ("diagnostician", diagnostician),
                 ("reporter", reporter)]:
        g.add_node(n, f)
    g.add_edge(START, "loader")
    g.add_conditional_edges("loader", after_loader, ["loader", "validator", "reporter"])
    g.add_conditional_edges("validator", after_validator, ["cleaner", "planner", "reporter"])
    g.add_edge("cleaner", "planner")
    g.add_edge("planner", "trainer")
    g.add_conditional_edges("trainer", after_trainer, ["trainer", "evaluator", "reporter"])
    g.add_conditional_edges("evaluator", after_evaluator, ["reporter", "diagnostician"])
    g.add_edge("diagnostician", "trainer")
    g.add_edge("reporter", END)
    return g.compile()


def run_adaptive(cfg):
    t = time.time()
    try:
        out = build().invoke({"cfg": cfg, "path": [], "retries": {}}, {"recursion_limit": 60})
    except Exception as e:
        return {"completed": False, "fits": 0, "path": ["crash"], "retries": {}, "error": str(e),
                "seconds": time.time() - t}
    return dict(out["result"], path=out["path"], retries=out.get("retries", {}), seconds=time.time() - t)
