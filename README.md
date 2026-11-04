# IDS706 Week 11 – Airflow Pipeline + Agent-Based ML Workflow

End-to-end pipeline on the **MovieLens (ml-latest-small)** data: Airflow ingests, cleans, merges and loads into PostgreSQL, analyzes the result, and then an agent workflow with a train → evaluate → tune loop trains models on the cleaned table.

## Run it

Requirements: Docker + Docker Compose (or open in VS Code with the Dev Containers extension using [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)).

```bash
docker compose up -d --build          # Postgres + Airflow (webserver, scheduler)
# UI: http://localhost:8080  (admin / admin); DAG "movielens_pipeline" is unpaused and scheduled @daily
docker compose exec airflow-scheduler airflow dags trigger movielens_pipeline   # or use the UI

# Part 2 (after the DAG has loaded the table):
docker compose run --rm agents        # writes output/agent_graph.png and output/agent_results.json
docker compose down                   # stop (add -v to wipe the database)
```

The warehouse DB (`warehouse`) is also exposed on `localhost:5433` (user/password `airflow`).

## Part 1 – Airflow pipeline

**Data sources** (three related tables from GroupLens, downloaded in the `ingest` task): `movies.csv`, `ratings.csv` (100k ratings), `tags.csv`.

**Transformations** ([dags/etl.py](dags/etl.py)):
- movies: drop duplicates/nulls, extract `year` from the title, normalise genres, impute missing year.
- ratings: drop nulls/duplicates/out-of-range, aggregate per movie → `n_ratings`, `avg_rating`, `rating_std`.
- tags: lowercase/trim/dedupe, aggregate per movie → `n_tags`, `n_unique_tags`.
- Merge: movies ⨝ ratings_agg ⨝ tags_agg (left, missing tag counts = 0) → table `movie_summary` (9,724 rows) in PostgreSQL.

**Structure** ([dags/movielens_pipeline.py](dags/movielens_pipeline.py)):

```
ingest → [process_movies | process_ratings | process_tags] → merge_and_load_postgres → [analyze_visualization | analyze_model] → done
```
Each `process_*` is a TaskGroup (`clean_*` → `transform_*`).

**Scheduling & parallelism:** `schedule="@daily"`, `catchup=False`, LocalExecutor. The three TaskGroups are independent so they run concurrently after `ingest` (see overlapping start times in the run), and the two analysis tasks run in parallel after the load.

**Analysis** (reads from Postgres): bar chart of mean rating per genre ([docs/genre_avg_rating.png](docs/genre_avg_rating.png)) and a RandomForest predicting "highly rated" (`avg_rating ≥ 3.5`): **accuracy 0.641**; `year` and `n_ratings` are the most important features ([docs/airflow_model_results.txt](docs/airflow_model_results.txt)).

Screenshots: [DAG runs/grid](docs/airflow_dag_runs_grid.png), [DAG graph (all tasks successful)](docs/airflow_dag_graph.png).

![graph](docs/airflow_dag_graph.png)

## Part 2 – Agent workflow with a loop

Code: [agents/workflow.py](agents/workflow.py), live graph rendering in [agents/graph_viz.py](agents/graph_viz.py). Each agent is a node; the return value picks the next agent. The graph PNG (`output/agent_graph.png`) is re-rendered every time the active agent changes (orange = active, green = done).

| Agent | Role |
|---|---|
| DataAgent | Reads `movie_summary` from Postgres |
| FeatureAgent | Builds features (year, counts, genre one-hot), train/val/test split |
| TrainerAgent | Trains the current candidate model |
| EvaluatorAgent | Validation accuracy/F1; loops back or finishes |
| TunerAgent | **Added** – picks next candidate when val accuracy < 0.80 |
| ReporterAgent | Test-set evaluation of the best model, writes results |

![agent graph](docs/agent_graph.png)

**Results** (target label: avg rating ≥ 3.5; full log in [docs/agent_run_log.txt](docs/agent_run_log.txt), [docs/agent_results.json](docs/agent_results.json)):

| Model | Val accuracy | Val F1 |
|---|---|---|
| LogisticRegression | 0.649 | 0.619 |
| DecisionTree(depth=5) | 0.639 | 0.636 |
| RandomForest(50) | 0.644 | 0.620 |
| RandomForest(300) | 0.645 | 0.623 |
| GradientBoosting | **0.676** | 0.655 |

The 0.80 target was never reached, so the loop exhausted all candidates; the best model (GradientBoosting) scores **test accuracy 0.671, F1 0.655**.

## Key findings
- Predicting movie quality from metadata alone (year, popularity, genres) is hard: ~65–68% accuracy. Popularity (`n_ratings`) and release year carry most of the signal.
- Gradient boosting beat the other models modestly; the agent loop makes comparing them automatic.

---

## Part 3 (bonus) – Adaptive multi-agent system (LangGraph) + evaluation suite

The baselines are the **actual class code** ([Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab), pinned as a git submodule in `third_party/`), run unmodified:
- **Fixed Airflow workflow** = the class DAG's steps from `mini_loop/baseline.py` (`validate_csv → profile_light → train_baseline`, one fixed logistic regression).
- **Class simple agentic loop** = `mini_loop.graph.run_simple_loop` (data_scientist → hypothesis → ml_engineer → judge → … → holdout_eval). No API key was set, so its LLM nodes use the repo's heuristic fallbacks.

```bash
git clone --recurse-submodules <this repo>     # or: git submodule update --init
docker compose build
docker compose run --rm adaptive               # graph render + 8-case eval x 3 systems -> output/dashboard.html
# optional LangSmith tracing: LANGSMITH_TRACING=true LANGSMITH_API_KEY=... docker compose run --rm adaptive
```
Code: [adaptive/graph.py](adaptive/graph.py) (LangGraph `StateGraph`), [adaptive/baselines.py](adaptive/baselines.py) (adapters around the class code), [adaptive/eval_suite.py](adaptive/eval_suite.py), [adaptive/run_all.py](adaptive/run_all.py). Dashboard: [docs/dashboard.html](docs/dashboard.html) ([screenshot](docs/part3_dashboard.png)); Mermaid: [docs/adaptive_graph.mmd](docs/adaptive_graph.mmd).

![adaptive graph](docs/adaptive_graph.png)

**Agents & adaptivity.** Loader, Validator, Cleaner, Planner, Trainer, Evaluator, Diagnostician, Reporter. Conditional edges: bounded **retries** of Loader/Trainer (then graceful exit); Validator **routes** through Cleaner only if labels are missing; Planner **selects** the model queue from the data profile (tiny data → simple models; rare positives → balanced logistic regression first); Evaluator stops on target / fit budget / plateau; Diagnostician reads the train–val gap and re-orders the queue (overfit → simpler, underfit → stronger).

**Evaluation suite (8 cases).** Each case gives the input (dataset variant, injected faults, budget), expected behaviour, weighted checks and a pass threshold of 0.8 (full table in the dashboard). *Final* checks: completed, sealed-holdout F1. *Coordination* checks: fit budget, no crash, no unneeded cleaning step. All systems get the same CSV, the same model families/preprocessing (class `loop_runtime`) and the same holdout-F1 yardstick. Faults are injected at the `pandas.read_csv` / `sklearn Pipeline.fit` boundaries so the class code runs unmodified.

| Case | Airflow fixed | Class simple loop | Adaptive LangGraph |
|---|---|---|---|
| T1 nominal (Titanic) F1 | 0.738 ✅ | 0.761 ✅ | 0.767 ✅ |
| T2 read outage (2 failures) | ❌ crash | ❌ crash | 0.767 ✅ |
| T3 flaky trainer (3 crashes) | ❌ | 0.735 ✅ (own repair step) | 0.767 ✅ |
| T4 6% missing labels | ❌ | ❌ crash | 0.730 ✅ |
| T5 missing features (control) | 0.712 ✅ | 0.709 ✅ | 0.710 ✅ |
| T6 imbalanced (churn) | 0.263 ✅ | 0.263 ✅ | 0.263 ✅ |
| T7 tiny data (60 rows) | 0.800 ✅ | 0.727 ✅ | 0.800 ✅ |
| T8 budget = 1 fit | 0.738 ✅ | 0.761 ✅ | 0.767 ✅ |
| **Cases passed** | 5/8 | 6/8 | **8/8** |
| Mean case score | 0.72 | 0.78 | 1.00 |

**Comparison summary**
- *Fixed Airflow workflow:* cheapest and fully deterministic (always 1 fit) and fine on clean/small data, but any fault or bad label kills the run; no model search or recovery.
- *Class simple loop:* adds model search and a judge, and it self-repairs a crashing fit (T3), but it has no data validation or load retry, so it crashed on a read failure and on missing labels. It is the most expensive on small data (3 fits for a worse F1 than the fixed baseline on T7).
- *Adaptive graph:* the only system to pass all cases; the gains come from robustness (retry, label cleaning) rather than better models – on clean data (T1, T5, T6, T8) all three are within ~0.03 F1 of each other, and T6 is a three-way tie. It uses more fits than the fixed workflow on T6/T7 (3 vs 1).

**Caveats.** (1) The agents are deterministic rule-based policies, not LLM-driven; LangSmith tracing is wired through env vars but I did not run it. (2) The test cases and thresholds are mine, and the faults are ones the adaptive system was built to handle, so the 8/8 is partly by construction; the clean-data cases show the honest parity. (3) After a first run showed my planner scored 0.0 F1 on the imbalanced data (trees first), I changed it to try balanced logistic regression first – i.e. one planner rule was adjusted after seeing a test result. (4) Single seed/split, small datasets (≤891 rows): differences of a few F1 points are noise. (5) Airflow's "coordination" score is inflated on failed runs because 0 fits trivially satisfies the fit-budget check; the *final* checks capture those failures. (6) The Airflow baseline is the class DAG's task functions run directly (not a live Airflow scheduler) so faults can be injected identically.
