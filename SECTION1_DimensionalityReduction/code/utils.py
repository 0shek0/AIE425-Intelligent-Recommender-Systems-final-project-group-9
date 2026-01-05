from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
import numpy as np
import pandas as pd


@dataclass
class Config:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = ROOT / "data" / "ml-20m"
    RESULTS_DIR: Path = ROOT / "results"
    TABLES_DIR: Path = RESULTS_DIR / "tables"
    PLOTS_DIR: Path = RESULTS_DIR / "plots"

    RATINGS_CSV: str = "ratings.csv"
    MOVIES_CSV: str = "movies.csv"

    # Light filtering (keeps stability but still large)
    MIN_USER_RATINGS: int = 5
    MIN_ITEM_RATINGS: int = 5

    # For building a manageable matrix (SVD on full ML20M is too big in RAM)
    # You can increase these if your machine can handle it.
    MAX_USERS: int = 800
    MAX_ITEMS: int = 800

    TEST_RATIO: float = 0.2
    SEED: int = 42


CFG = Config()


def ensure_dirs() -> None:
    CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ratings() -> pd.DataFrame:
    path = CFG.DATA_DIR / CFG.RATINGS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put ratings.csv in data/ml-20m/")

    df = pd.read_csv(path, usecols=["userId", "movieId", "rating", "timestamp"])
    df["rating"] = df["rating"].astype(float)
    return df


def load_movies() -> pd.DataFrame | None:
    path = CFG.DATA_DIR / CFG.MOVIES_CSV
    if not path.exists():
        return None
    return pd.read_csv(path)


def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    user_counts = df.groupby("userId")["movieId"].count()
    keep_users = user_counts[user_counts >= CFG.MIN_USER_RATINGS].index
    df = df[df["userId"].isin(keep_users)]

    item_counts = df.groupby("movieId")["userId"].count()
    keep_items = item_counts[item_counts >= CFG.MIN_ITEM_RATINGS].index
    df = df[df["movieId"].isin(keep_items)]

    return df.reset_index(drop=True)


def compute_basic_stats(df: pd.DataFrame) -> dict:
    n_ratings = len(df)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()
    density = n_ratings / (n_users * n_items)
    sparsity = 1.0 - density
    return {
        "num_ratings": n_ratings,
        "num_users": n_users,
        "num_items": n_items,
        "density": density,
        "sparsity": sparsity,
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean()),
    }


def item_popularity(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    pop = (
        df.groupby("movieId")["rating"]
        .agg(num_ratings="count", avg_rating="mean")
        .reset_index()
    )
    pop["pct_of_all_ratings"] = (pop["num_ratings"] / total) * 100.0
    return pop.sort_values("num_ratings", ascending=False).reset_index(drop=True)


def choose_target_items(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """
    Select two target items:
    - I1: medium popularity
    - I2: high popularity

    Robust to filtering/sampling.
    """
    pop = item_popularity(df)

    # ---- I2: HIGH popularity ----
    high = pop[pop["pct_of_all_ratings"] > 10.0]

    if len(high) > 0:
        I2 = int(high.iloc[0]["movieId"])
    else:
        # Fallback: top-ranked item
        I2 = int(pop.iloc[0]["movieId"])

    # ---- I1: MEDIUM popularity ----
    medium = pop[
        (pop["pct_of_all_ratings"] >= 2.0) &
        (pop["pct_of_all_ratings"] <= 5.0)
    ]

    if len(medium) > 0:
        I1 = int(medium.iloc[0]["movieId"])
    else:
        # Fallback: middle of popularity distribution
        mid_idx = len(pop) // 2
        I1 = int(pop.iloc[mid_idx]["movieId"])

    return I1, I2, pop



def build_small_user_item_matrix(df: pd.DataFrame, pop: pd.DataFrame, forced_items=None) -> pd.DataFrame:
    """
    Build a manageable user-item matrix and FORCE-INCLUDE target items.
    """
    if forced_items is None:
        forced_items = []

    # Top popular items
    top_items = pop.head(CFG.MAX_ITEMS)["movieId"].astype(int).tolist()

    # Force include targets
    for it in forced_items:
        if it not in top_items:
            top_items.append(it)

    sub = df[df["movieId"].isin(top_items)].copy()

    # Select most active users
    ucnt = sub.groupby("userId")["movieId"].count().sort_values(ascending=False)
    top_users = ucnt.head(CFG.MAX_USERS).index
    sub = sub[sub["userId"].isin(top_users)]

    mat = sub.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    return mat.sort_index(axis=0).sort_index(axis=1)



def mean_fill_matrix(mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    item_means = mat.mean(axis=0, skipna=True)
    filled = mat.apply(lambda col: col.fillna(item_means[col.name]), axis=0)
    return filled, item_means


def train_test_holdout_for_item(mat: pd.DataFrame, item_id: int, test_ratio: float, seed: int):
    """
    Hold out a subset of observed ratings for a target item.
    Returns (masked_matrix, ground_truth_series).
    """
    if item_id not in mat.columns:
        raise ValueError(f"Target item {item_id} not found in matrix columns.")

    rng = np.random.default_rng(seed)
    rated_users = mat[item_id].dropna().index.to_numpy()
    if len(rated_users) < 5:
        raise ValueError(f"Not enough ratings for item {item_id} after sampling.")

    n_test = max(1, int(len(rated_users) * test_ratio))
    test_users = rng.choice(rated_users, size=n_test, replace=False)

    gt = mat.loc[test_users, item_id].copy()
    masked = mat.copy()
    masked.loc[test_users, item_id] = np.nan
    return masked, gt


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


class Profiler:
    """
    Lightweight runtime + memory profiler.
    """
    def __init__(self):
        self.t0 = None
        self.mem0 = None

    def __enter__(self):
        self.t0 = time.time()
        tracemalloc.start()
        self.mem0 = tracemalloc.get_traced_memory()[0]
        return self

    def __exit__(self, exc_type, exc, tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.runtime_s = time.time() - self.t0
        self.mem_peak_mb = peak / (1024 * 1024)
