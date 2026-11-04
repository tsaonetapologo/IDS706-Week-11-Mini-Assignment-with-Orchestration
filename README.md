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

```bash
docker compose run --rm adaptive      # runs graph render + 7-case eval x 3 systems; writes output/dashboard.html
# optional LangSmith tracing: LANGSMITH_TRACING=true LANGSMITH_API_KEY=... docker compose run --rm adaptive
```
Code: [adaptive/graph.py](adaptive/graph.py) (LangGraph `StateGraph`), [adaptive/baselines.py](adaptive/baselines.py), [adaptive/eval_suite.py](adaptive/eval_suite.py), [adaptive/run_all.py](adaptive/run_all.py). Dashboard: [docs/dashboard.html](docs/dashboard.html) ([screenshot](docs/part3_dashboard.png)); Mermaid source: [docs/adaptive_graph.mmd](docs/adaptive_graph.mmd).

![adaptive graph](docs/adaptive_graph.png)

**Agents & adaptivity:** Loader, Validator, Cleaner, Planner, Trainer, Evaluator, Diagnostician, Reporter. Conditional edges: bounded **retries** on Loader/Trainer failures (then graceful exit); Validator **routes** through Cleaner only when NaNs exist; Planner **selects** the model queue from the data profile (small data → simple models, imbalance → class-weighted models + F1); Evaluator stops on target / fit budget / plateau; Diagnostician reads the train–val gap and re-orders the queue (overfit → simpler, underfit → stronger).

**Evaluation suite:** 7 cases (T1–T7), each with input config, expected behaviour, weighted checks (*final* = output quality, *coordination* = trace/retries/fits) and a pass threshold of 0.8 – see the table in the dashboard. Fault injection (DB outage, trainer crash), NaN corruption, class imbalance, fit budget and tiny data are simulated identically for all three systems.

| System | Cases passed | Mean score | Mean coordination score | Total fits |
|---|---|---|---|---|
| Fixed Airflow-style (load → RF → evaluate) | 1/7 | 0.48 | 0.43 | 5 |
| Simple in-class loop (Part 2) | 1/7 | 0.41 | 0.14 | 12 |
| **Adaptive LangGraph** | **7/7** | **1.00** | **1.00** | 12 |

**Comparison summary**
- *Fixed workflow:* cheap and deterministic (always 1 fit) but any fault kills the run, it can't react to imbalance/NaNs, and its single model is weaker (0.643 vs 0.671 accuracy nominal).
- *Simple loop:* adds model search, but order is fixed, so it wastes fits (4 on the nominal case vs 1), ignores budgets, and has no retry/validation – it crashed on the same faults as the fixed workflow and on NaNs.
- *Adaptive graph:* recovers from transient failures, handles dirty/imbalanced/small data, respects budgets, and reached the nominal target in 1 fit. It is not cheaper on every case (T5/T7 use 3 fits – the price of diagnosing and adapting), and on T7 all three systems perform identically.

**Caveats (honest limits):** the agents are deterministic rule-based policies, not LLM-driven (no API key needed; LangSmith tracing is wired via env vars but I did not run it). The baselines are faithful re-implementations behind the same interface (not the Airflow/Part 2 code executed directly) so faults can be injected; the simple loop uses the same 0.66 target for comparability (Part 2 used 0.80). Test thresholds were set by me and the suite is small (single seed/split), so treat differences as illustrative.
