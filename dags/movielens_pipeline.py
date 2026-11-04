from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

import etl

with DAG(
    dag_id="movielens_pipeline",
    description="MovieLens ingest -> parallel clean/transform -> merge -> Postgres -> analysis",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["ids706"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=etl.ingest)

    # The three groups are independent, so they run in parallel.
    groups = []
    for name, clean, transform in [
        ("movies", etl.clean_movies, etl.transform_movies),
        ("ratings", etl.clean_ratings, etl.transform_ratings),
        ("tags", etl.clean_tags, etl.transform_tags),
    ]:
        with TaskGroup(group_id=f"process_{name}") as tg:
            c = PythonOperator(task_id=f"clean_{name}", python_callable=clean)
            t = PythonOperator(task_id=f"transform_{name}", python_callable=transform)
            c >> t
        groups.append(tg)

    merge_load = PythonOperator(task_id="merge_and_load_postgres", python_callable=etl.merge_and_load)
    visual = PythonOperator(task_id="analyze_visualization", python_callable=etl.analyze_visual)
    model = PythonOperator(task_id="analyze_model", python_callable=etl.analyze_model)
    done = EmptyOperator(task_id="done")

    ingest >> groups >> merge_load >> [visual, model] >> done
