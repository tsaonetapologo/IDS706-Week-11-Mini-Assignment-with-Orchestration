"""The two baselines from class, run as-is from the pinned class repo (third_party/automl-research-lab).

* airflow_fixed: the class Airflow DAG's steps (`mini_loop.baseline`): validate_csv -> profile_light -> train_baseline
  (one fixed logistic regression). We then score its model spec on the sealed holdout so all systems share a metric.
* class_loop: the class LangGraph simple loop (`mini_loop.graph.run_simple_loop`:
  data_scientist -> hypothesis -> ml_engineer -> judge -> ... -> holdout_eval), no API key => its heuristic fallbacks.
"""
import time

from adaptive import core
from mini_loop import baseline as cb
from mini_loop.graph import run_simple_loop


def run_airflow_fixed(cfg):
    t, path = time.time(), []
    try:
        path.append("validate_csv")
        cb.validate_csv(cfg["dataset"], require_columns=[cfg["target"]])
        path.append("profile_light")
        prof = cb.profile_light(cfg["dataset"], target_column=cfg["target"])
        path.append("train_baseline")
        b = cb.train_baseline(cfg["dataset"], target_column=cfg["target"],
                              drop_columns=list(prof.get("recommended_drops") or []),
                              model_family="logistic_regression",
                              hyperparameters={"C": 1.0, "max_iter": 500, "class_weight": "balanced"})
        path.append("publish")
        f1 = core.holdout_f1(cfg["dataset"], cfg["target"], core.POOL["LogReg"][0], prof.get("recommended_drops") or [])
        return {"completed": True, "model": "logistic_regression", "test_metric": f1, "fits": 1,
                "path": path, "retries": {}, "seconds": time.time() - t}
    except Exception as e:
        return {"completed": False, "fits": 0, "path": path, "retries": {}, "error": str(e)[:120],
                "seconds": time.time() - t}


def run_class_loop(cfg):
    t = time.time()
    try:
        r = run_simple_loop(dataset_path=cfg["dataset"], target_column=cfg["target"],
                            target_score=cfg.get("target_score", 0.7), max_cycles=cfg.get("budget", 3))
        steps = r.get("steps") or []
        hm = r.get("holdout_metrics") or {}
        ok = "f1" in hm
        return {"completed": ok, "model": (r.get("best_spec") or {}).get("model_family"),
                "test_metric": hm.get("f1", float("nan")),
                "fits": sum(1 for s in steps if s["node"] == "ml_engineer" and s.get("status") == "succeeded"),
                "path": [s["node"] for s in steps], "retries": {}, "seconds": time.time() - t}
    except Exception as e:
        return {"completed": False, "fits": 0, "path": ["crash"], "retries": {}, "error": str(e)[:120],
                "seconds": time.time() - t}
