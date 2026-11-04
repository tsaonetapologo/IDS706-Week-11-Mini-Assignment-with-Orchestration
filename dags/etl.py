"""ETL helpers for the MovieLens pipeline (movies + ratings + tags)."""
import io
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sqlalchemy import create_engine

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.environ.get("DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
RAW = os.path.join(DATA_DIR, "raw")
PROC = os.path.join(DATA_DIR, "processed")
TABLE = "movie_summary"


def engine():
    return create_engine(os.environ["WAREHOUSE_URL"])


def ingest():
    os.makedirs(RAW, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in ("movies.csv", "ratings.csv", "tags.csv"):
            with z.open(f"ml-latest-small/{name}") as src, open(os.path.join(RAW, name), "wb") as dst:
                dst.write(src.read())


def _read(name):
    return pd.read_csv(os.path.join(RAW, name))


def _save(df, name):
    os.makedirs(PROC, exist_ok=True)
    df.to_csv(os.path.join(PROC, name), index=False)


# ---- movies group ----
def clean_movies():
    df = _read("movies.csv").drop_duplicates("movieId").dropna(subset=["title"])
    df["title"] = df["title"].str.strip()
    _save(df, "movies_stage.csv")


def transform_movies():
    df = pd.read_csv(os.path.join(PROC, "movies_stage.csv"))
    df["year"] = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    df["genres"] = df["genres"].replace("(no genres listed)", "Unknown")
    df["year"] = df["year"].fillna(df["year"].median())
    _save(df, "movies_clean.csv")


# ---- ratings group ----
def clean_ratings():
    df = _read("ratings.csv").dropna().drop_duplicates(["userId", "movieId"])
    df = df[df["rating"].between(0.5, 5)]
    _save(df, "ratings_stage.csv")


def transform_ratings():
    df = pd.read_csv(os.path.join(PROC, "ratings_stage.csv"))
    agg = df.groupby("movieId").agg(
        n_ratings=("rating", "size"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
    ).reset_index()
    agg["rating_std"] = agg["rating_std"].fillna(0)
    _save(agg, "ratings_agg.csv")


# ---- tags group ----
def clean_tags():
    df = _read("tags.csv").dropna(subset=["tag"])
    df["tag"] = df["tag"].str.lower().str.strip()
    _save(df.drop_duplicates(["userId", "movieId", "tag"]), "tags_stage.csv")


def transform_tags():
    df = pd.read_csv(os.path.join(PROC, "tags_stage.csv"))
    agg = df.groupby("movieId").agg(n_tags=("tag", "size"), n_unique_tags=("tag", "nunique")).reset_index()
    _save(agg, "tags_agg.csv")


def merge_and_load():
    p = lambda n: pd.read_csv(os.path.join(PROC, n))
    df = p("movies_clean.csv").merge(p("ratings_agg.csv"), on="movieId", how="inner")
    df = df.merge(p("tags_agg.csv"), on="movieId", how="left")
    df[["n_tags", "n_unique_tags"]] = df[["n_tags", "n_unique_tags"]].fillna(0).astype(int)
    df.to_sql(TABLE, engine(), if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into {TABLE}")


# ---- analysis ----
def load_table():
    return pd.read_sql(f"SELECT * FROM {TABLE}", engine())


def analyze_visual():
    df = load_table()
    df = df.assign(genre=df["genres"].str.split("|")).explode("genre")
    g = df.groupby("genre").agg(avg=("avg_rating", "mean"), n=("movieId", "nunique"))
    g = g[g["n"] >= 20].sort_values("avg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ax = g["avg"].plot.barh(figsize=(8, 7), color="steelblue")
    ax.set_xlabel("Mean of movie average rating")
    ax.set_title("Average rating by genre")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "genre_avg_rating.png"), dpi=120)
    g.to_csv(os.path.join(OUTPUT_DIR, "genre_stats.csv"))


def analyze_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    df = load_table()
    y = (df["avg_rating"] >= 3.5).astype(int)
    X = pd.concat([df[["year", "n_ratings", "n_tags"]], df["genres"].str.get_dummies("|")], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False)
    with open(os.path.join(OUTPUT_DIR, "airflow_model_results.txt"), "w") as f:
        f.write(f"RandomForest accuracy (avg_rating>=3.5): {acc:.4f}\nTop features:\n{imp.head(10)}\n")
    print(f"accuracy={acc:.4f}")
