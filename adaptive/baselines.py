"""The two baselines, re-implemented behind the same interface and the same data/model code."""
import time

from adaptive import core


def run_airflow_fixed(cfg):
    """Fixed DAG: load -> features -> fit RandomForest -> evaluate. No validation, routing or retries."""
    t, path, faults = time.time(), [], core.Faults(cfg.get("faults"))
    try:
        path.append("ingest")
        df = core.load_raw(cfg, faults)
        X, y = core.build_xy(df, cfg.get("label_threshold", 3.5))
        Xtr, _, Xte, ytr, _, yte = core.split3(X, y)
        path.append("train")
        faults.maybe_fail("train_fail")
        m, prep = core.make_model("RandomForest(300)")
        m.fit(prep(Xtr), ytr)
        path.append("evaluate")
        test = core.score(yte, m.predict(prep(Xte)), cfg.get("eval_metric", "accuracy"))
        return {"completed": True, "model": "RandomForest(300)", "metric_name": cfg.get("eval_metric", "accuracy"), "test_metric": test,
                "fits": 1, "path": path, "retries": {}, "seconds": time.time() - t, "system": "airflow_fixed"}
    except Exception as e:
        return {"completed": False, "fits": 0, "path": path, "retries": {}, "error": str(e),
                "seconds": time.time() - t, "system": "airflow_fixed"}


ORDER = ["LogisticRegression", "DecisionTree(depth=6)", "RandomForest(300)", "GradientBoosting", "GradientBoosting(deep)"]


def run_simple_loop(cfg, target=0.66):
    """Part 2 loop: fixed candidate order, train/evaluate until accuracy target or list exhausted.
    No retries, no data validation, ignores budget and class imbalance."""
    t, path, faults = time.time(), [], core.Faults(cfg.get("faults"))
    try:
        path.append("DataAgent")
        df = core.load_raw(cfg, faults)
        path.append("FeatureAgent")
        X, y = core.build_xy(df, cfg.get("label_threshold", 3.5))
        Xtr, Xva, Xte, ytr, yva, yte = core.split3(X, y)
        best, fits = None, 0
        for name in ORDER:
            path += ["TrainerAgent", "EvaluatorAgent"]
            faults.maybe_fail("train_fail")
            m, prep = core.make_model(name)
            m.fit(prep(Xtr), ytr)
            fits += 1
            val = core.score(yva, m.predict(prep(Xva)), "accuracy")
            if best is None or val > best[0]:
                best = (val, name, m, prep)
            if val >= target:
                break
            path.append("TunerAgent")
        path.append("ReporterAgent")
        test = core.score(yte, best[2].predict(best[3](Xte)), cfg.get("eval_metric", "accuracy"))
        return {"completed": True, "model": best[1], "metric_name": cfg.get("eval_metric", "accuracy"), "test_metric": test,
                "fits": fits, "path": path, "retries": {}, "seconds": time.time() - t, "system": "simple_loop"}
    except Exception as e:
        return {"completed": False, "fits": 0, "path": path, "retries": {}, "error": str(e),
                "seconds": time.time() - t, "system": "simple_loop"}
