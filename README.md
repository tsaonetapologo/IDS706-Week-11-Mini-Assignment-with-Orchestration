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
