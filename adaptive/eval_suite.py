"""Evaluation suite. Every case states: input (cfg), expected behaviour, weighted scoring checks, pass threshold.

Final-output checks look at the result (completion, metric, efficiency).
Intermediate/coordination checks look at the execution trace (path, retries, fits).
Case score = sum(weight * check_passed) / sum(weight); case passes if score >= pass_threshold.
"""
from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    kind: str  # "final" or "coordination"
    weight: float
    fn: callable


@dataclass
class Case:
    id: str
    description: str
    cfg: dict
    expected: str
    checks: list
    pass_threshold: float = 0.8
    scoring: str = ""


completed = Check("completed", "final", 3, lambda r: r["completed"])
def metric_ge(v):
    return Check(f"test_metric>={v}", "final", 3, lambda r: r["completed"] and r["test_metric"] >= v)
def fits_le(n):
    return Check(f"fits<={n}", "coordination", 2, lambda r: r["fits"] <= n)
def visited(node):
    return Check(f"visited:{node}", "coordination", 2, lambda r: node in r["path"])
def retried(kind, n):
    return Check(f"retried_{kind}>={n}", "coordination", 2, lambda r: r["retries"].get(kind, 0) >= n)


CASES = [
    Case("T1_nominal", "Clean data, balanced label (rating>=3.5).",
         {"label_threshold": 3.5, "target": 0.66},
         "Completes with accuracy >= 0.65 using <= 3 model fits (efficient routing).",
         [completed, metric_ge(0.65), fits_le(3)]),
    Case("T2_db_outage", "Warehouse read fails twice before succeeding.",
         {"label_threshold": 3.5, "target": 0.66, "faults": {"db_fail": 2}},
         "Retries the load >=2 times, then completes with accuracy >= 0.65.",
         [completed, metric_ge(0.65), retried("loader", 2)]),
    Case("T3_trainer_crash", "First model fit crashes (transient).",
         {"label_threshold": 3.5, "target": 0.66, "faults": {"train_fail": 1}},
         "Retries the failed training step and completes with accuracy >= 0.65.",
         [completed, metric_ge(0.65), retried("trainer", 1)]),
    Case("T4_corrupt_features", "8% of year/n_ratings values are missing (NaN).",
         {"label_threshold": 3.5, "target": 0.66, "corrupt_frac": 0.08},
         "Detects NaNs, routes through a cleaning agent, completes with accuracy >= 0.62.",
         [completed, metric_ge(0.62), visited("cleaner")]),
    Case("T5_imbalanced", "Label is rating>=4.0 (minority class ~10%); judged on F1 of the positive class.",
         {"label_threshold": 4.0, "target": 0.60, "eval_metric": "f1"},
         "Detects imbalance, switches to class-weighted models; F1 >= 0.40.",
         [completed, metric_ge(0.40), Check("planner_picked_f1", "coordination", 2, lambda r: r.get("metric_name") == "f1" and "planner" in r["path"])]),
    Case("T6_budget", "Compute budget of at most 2 model fits.",
         {"label_threshold": 3.5, "target": 0.70, "max_fits": 2},
         "Respects the budget (<=2 fits) and still reaches accuracy >= 0.65.",
         [completed, metric_ge(0.65), fits_le(2)]),
    Case("T7_small_data", "Only 300 rows available (overfitting risk).",
         {"label_threshold": 3.5, "target": 0.62, "n_rows": 300},
         "Starts with simple models; completes with accuracy >= 0.55 and <=3 fits.",
         [completed, metric_ge(0.55), fits_le(3)]),
]


def score_case(case, res):
    got = {c.name: bool(c.fn(res)) for c in case.checks}
    tot = sum(c.weight for c in case.checks)
    sc = sum(c.weight for c in case.checks if got[c.name]) / tot
    fin = [c for c in case.checks if c.kind == "final"]
    co = [c for c in case.checks if c.kind == "coordination"]
    part = lambda cs: (sum(c.weight for c in cs if got[c.name]) / sum(c.weight for c in cs)) if cs else None
    return {"score": sc, "passed": sc >= case.pass_threshold, "checks": got,
            "final_score": part(fin), "coordination_score": part(co)}
