"""Evaluation suite. Every case states: input (dataset variant + faults + budget), expected behaviour,
weighted scoring checks, and a pass threshold.

* final checks       -> quality of the final output (completed, sealed-holdout F1)
* coordination checks -> quality of the execution (retries, fit budget, no wasted steps)
Case score = sum(weight of passed checks) / sum(weights); the case passes if score >= pass_threshold.
All three systems get the identical case, dataset, metric (holdout F1) and budget.
"""
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    kind: str
    weight: float
    fn: callable


@dataclass
class Case:
    id: str
    description: str
    dataset: str
    cfg: dict
    expected: str
    checks: list
    pass_threshold: float = 0.8


completed = Check("completed", "final", 3, lambda r: r["completed"])
def f1_ge(v):
    return Check(f"holdout_f1>={v}", "final", 3, lambda r: bool(r["completed"] and r["test_metric"] >= v))
def fits_le(n):
    return Check(f"fits<={n}", "coordination", 2, lambda r: r["fits"] <= n)
no_cleaner = Check("no_unneeded_cleaning", "coordination", 2, lambda r: "cleaner" not in r["path"])
no_crash = Check("no_crash", "coordination", 2, lambda r: "crash" not in r["path"])


CASES = [
    Case("T1_nominal", "Clean Titanic data (class dataset).", "titanic", {},
         "Completes with holdout F1 >= 0.70 using <= 3 fits.", [completed, f1_ge(0.70), fits_le(3)]),
    Case("T2_read_outage", "Data read fails on the first 2 attempts.", "titanic", {"faults": {"db_fail": 2}},
         "Recovers from the transient outage (any mechanism) and completes with F1 >= 0.70 within 3 fits.",
         [completed, f1_ge(0.70), fits_le(3)]),
    Case("T3_flaky_trainer", "The first 3 model fits crash (persistent flakiness).", "titanic",
         {"faults": {"train_fail": 3}},
         "Recovers from repeated crashes (any mechanism) and completes with F1 >= 0.70 within 3 fits.",
         [completed, f1_ge(0.70), fits_le(3)]),
    Case("T4_bad_labels", "6% of target labels are missing.", "titanic_nan_labels", {},
         "Detects and drops unlabeled rows, completes with F1 >= 0.65.", [completed, f1_ge(0.65), no_crash]),
    Case("T5_missing_features", "15% missing values in Age/Fare/Pclass (control: imputer handles it).",
         "titanic_nan_features", {},
         "Completes with F1 >= 0.65 and does NOT waste a cleaning step.", [completed, f1_ge(0.65), no_cleaner]),
    Case("T6_imbalanced", "Churn data, ~10% positives; judged on positive-class F1.", "churn", {},
         "Completes with holdout F1 >= 0.25 within 3 fits.", [completed, f1_ge(0.25), fits_le(3)]),
    Case("T7_tiny_data", "Only 60 labeled rows.", "titanic_tiny", {},
         "Completes with F1 >= 0.70 within 3 fits.", [completed, f1_ge(0.70), fits_le(3)]),
    Case("T8_tight_budget", "Budget of exactly 1 fit with an unreachable target (0.95).", "titanic",
         {"budget": 1, "target_score": 0.95},
         "Respects the budget (<=1 fit) and still gets F1 >= 0.70.", [completed, f1_ge(0.70), fits_le(1)]),
]


def score_case(case, res):
    got = {c.name: bool(c.fn(res)) for c in case.checks}
    part = lambda cs: (sum(c.weight for c in cs if got[c.name]) / sum(c.weight for c in cs)) if cs else None
    sc = sum(c.weight for c in case.checks if got[c.name]) / sum(c.weight for c in case.checks)
    return {"score": sc, "passed": sc >= case.pass_threshold, "checks": got,
            "final_score": part([c for c in case.checks if c.kind == "final"]),
            "coordination_score": part([c for c in case.checks if c.kind == "coordination"])}
