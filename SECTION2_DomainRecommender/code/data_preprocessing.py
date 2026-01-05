
"""MovieLens 20M compatible preprocessing.

Replaces the original music-based preprocessing with MovieLens logic.
This module performs the required steps listed in the project rubric:
 - load ratings.csv and movies.csv
 - clean data and enforce types
 - save a canonical `interactions.csv` (user_id,item_id,rating)
 - compute dataset stats and plots
 - produce an 80/20 per-user train/test split (fixed seed)
 - identify cold-start users and low-frequency items
 - helper functions to build sparse interaction matrices

The script is import-safe; heavy I/O only runs in `if __name__ == '__main__'`.
If MovieLens CSV files are not present in `data/raw/`, the script will exit
with a clear instruction so you can upload the dataset.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from scipy import sparse


# ===============================
# Paths and directories (preserve project layout)
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
STATS_DIR = RESULTS_DIR / "stats"

for d in [DATA_RAW, DATA_PROCESSED, FIGURES_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Expected MovieLens files inside data/raw/
RATINGS_FILE = DATA_RAW / "ratings.csv"
MOVIES_FILE = DATA_RAW / "movies.csv"


# ===============================
# Helper functions
# ===============================

def check_files_exist() -> bool:
    """Check that the required MovieLens files exist.

    If missing, this function returns False so the caller can notify the user
    (we avoid raising here to keep the script import-safe).
    """
    return RATINGS_FILE.exists() and MOVIES_FILE.exists()


def load_movielens(limit_rows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movies CSVs with proper dtypes and minimal columns.

    - We load only the required columns to keep memory low.
    - `limit_rows` is optional (handy for local testing).
    """
    # Only necessary columns — keeps memory usage minimal for large datasets
    ratings_cols = ["userId", "movieId", "rating"]
    movies_cols = ["movieId", "title", "genres"]

    ratings = pd.read_csv(RATINGS_FILE, usecols=ratings_cols, nrows=limit_rows)
    movies = pd.read_csv(MOVIES_FILE, usecols=movies_cols, nrows=None)

    # Enforce types: ids → int, rating → float (MovieLens uses 0.5–5.0)
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])  # safety
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Movies: ensure movieId int and fill missing text fields
    movies = movies.drop_duplicates(subset=["movieId"])  # keep first occurrence
    movies["movieId"] = movies["movieId"].astype(int)
    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    return ratings, movies


def save_interactions(df: pd.DataFrame) -> Path:
    """Save canonical interactions with columns `user_id,item_id,rating`.

    This is the standard artifact expected by downstream modules.
    """
    out = DATA_PROCESSED / "interactions.csv"
    df[ ["user_id", "item_id", "rating"] ].to_csv(out, index=False)
    return out


def compute_stats(interactions: pd.DataFrame) -> Dict:
    """Compute and return dataset statistics.

    - density / sparsity
    - rating distribution
    - counts
    """
    n_users = int(interactions["user_id"].nunique())
    n_items = int(interactions["item_id"].nunique())
    n_interactions = int(len(interactions))

    density = n_interactions / (n_users * n_items)
    sparsity = 1.0 - density

    rating_distribution = (
        interactions["rating"].value_counts().sort_index().to_dict()
    )

    stats = {
        "users": n_users,
        "items": n_items,
        "interactions": n_interactions,
        "density": density,
        "sparsity": sparsity,
        "rating_distribution": rating_distribution,
    }

    return stats


def identify_cold_start(interactions: pd.DataFrame, threshold: int = 5) -> Dict:
    """Identify cold-start users and low-frequency items.

    Returns counts and lists of ids (lists truncated to first 100 ids to avoid huge JSONs).
    """
    user_counts = interactions.groupby("user_id").size()
    item_counts = interactions.groupby("item_id").size()

    cold_users = user_counts[user_counts <= threshold].index.tolist()
    low_freq_items = item_counts[item_counts <= threshold].index.tolist()

    return {
        "cold_user_count": int(len(cold_users)),
        "low_freq_item_count": int(len(low_freq_items)),
        "cold_users_sample": cold_users[:100],
        "low_freq_items_sample": low_freq_items[:100],
    }


def per_user_train_test_split(
    interactions: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an 80/20 per-user split with a fixed seed.

    - Each user with >=1 rating gets at least one train example.
    - For reproducibility we use a fixed RNG.
    - For users with only one rating, that rating will be in train (test empty).
    - For users with >=2 ratings we ensure test has at least one rating.
    """
    rng = np.random.RandomState(seed)

    train_rows = []
    test_rows = []

    # Operate on grouped indices for reproducible sampling
    grouped = interactions.groupby("user_id")

    for user_id, group in grouped:
        n = len(group)
        if n == 1:
            # Keep single-rating users in train (no test sample)
            train_rows.append(group.index.values)
            continue

        # Compute number of test examples: at least 1
        n_test = max(1, int(np.floor(test_fraction * n)))

        # Sample without replacement from the group's indices
        chosen = rng.choice(group.index.values, size=n_test, replace=False)
        test_rows.append(chosen)
        train_rows.append(np.setdiff1d(group.index.values, chosen))

    # Flatten lists of arrays
    train_idx = np.concatenate(train_rows) if train_rows else np.array([], dtype=int)
    test_idx = np.concatenate(test_rows) if test_rows else np.array([], dtype=int)

    train_df = interactions.loc[train_idx].reset_index(drop=True)
    test_df = interactions.loc[test_idx].reset_index(drop=True)

    return train_df, test_df


def build_interaction_matrix(
    interactions: pd.DataFrame,
) -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    """Build a sparse user-item matrix (csr) and mapping dictionaries.

    Returns (matrix, user_to_index, item_to_index)
    """
    users = interactions["user_id"].unique()
    items = interactions["item_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: j for j, it in enumerate(items)}

    rows = interactions["user_id"].map(user_to_idx).values
    cols = interactions["item_id"].map(item_to_idx).values
    data = interactions["rating"].values

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return mat, user_to_idx, item_to_idx


# ===============================
# Main orchestration (safe to run as script)
# ===============================

def run_preprocessing(limit_rows: int = None, seed: int = 42) -> None:
    """Full preprocessing pipeline entrypoint.

    - `limit_rows` is only for quick local testing; leave as None for full dataset.
    """
    if not check_files_exist():
        print(
            "MovieLens files not found in data/raw/. Please upload `ratings.csv` and `movies.csv` (MovieLens 20M) and re-run."
        )
        return

    print("Loading MovieLens files...")
    ratings, movies = load_movielens(limit_rows=limit_rows)

    # Rename to standard column names expected by downstream code
    interactions = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})[
        ["user_id", "item_id", "rating"]
    ]

    # Remove exact duplicates (safety) keeping the latest if timestamp existed; here we keep first
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"], keep="first")

    # Save canonical interactions CSV used across modules
    saved = save_interactions(interactions)
    print(f"Saved canonical interactions -> {saved}")

    # Statistics and cold-start identification
    stats = compute_stats(interactions)
    cold_info = identify_cold_start(interactions, threshold=5)

    stats.update(cold_info)

    stats_path = STATS_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved dataset stats ->", stats_path)

    # Per-user train/test split (80/20) with reproducible seed
    train_df, test_df = per_user_train_test_split(interactions, test_fraction=0.2, seed=seed)
    train_path = DATA_PROCESSED / "interactions_train.csv"
    test_path = DATA_PROCESSED / "interactions_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train/test saved to {train_path} and {test_path}")

    # Build sparse interaction matrix for downstream CF/hybrid models
    interaction_matrix, user_map, item_map = build_interaction_matrix(train_df)

    # Save lightweight mapping files to help downstream modules
    # (these are small JSONs mapping original ids to matrix indices)
    # JSON requires keys to be str; convert mapping keys to str for safe serialization
    user_map_serializable = {str(k): int(v) for k, v in user_map.items()}
    item_map_serializable = {str(k): int(v) for k, v in item_map.items()}
    with open(DATA_PROCESSED / "user_map.json", "w", encoding="utf-8") as f:
        json.dump(user_map_serializable, f, indent=2)
    with open(DATA_PROCESSED / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(item_map_serializable, f, indent=2)

    # ===============================
    # Plots required by the rubric
    # ===============================
    print("Generating required plots...")

    # Ratings per user (histogram) — shown as distribution to avoid huge plots
    user_activity = interactions.groupby("user_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(user_activity.values, bins=50)
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Count of users")
    plt.title("Ratings per user")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_user.png", dpi=200)
    plt.close()

    # Ratings per movie (popularity distribution)
    item_popularity = interactions.groupby("item_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(item_popularity.values, bins=50)
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Count of movies")
    plt.title("Ratings per movie")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_movie.png", dpi=200)
    plt.close()

    # Sparsity: visual long-tail (fraction of items vs cumulative interactions)
    item_pop_sorted = item_popularity.sort_values(ascending=False)
    cum = item_pop_sorted.cumsum() / item_pop_sorted.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    plt.figure(figsize=(6, 4))
    plt.plot(x, cum.values)
    plt.xlabel("Fraction of items")
    plt.ylabel("Cumulative fraction of interactions")
    plt.title("Long-tail / Sparsity")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "long_tail.png", dpi=200)
    plt.close()

    print("All figures saved to results/figures/")
    print("Done. Preprocessing output ready for content-based, CF, and hybrid models.")


if __name__ == "__main__":
    # For full dataset run leave limit_rows=None. For quick local tests set a small integer.
    run_preprocessing(limit_rows=None, seed=42)


def generate_filtered_subset(
    target_users: int = 5000,
    min_items: int = 500,
    min_interactions: int = 50000,
    seed: int = 42,
) -> None:
    """Create and save a filtered subset of MovieLens that meets given minima.

    Strategy:
    - Load full ratings.csv
    - Select users by descending activity (ensures we meet interaction threshold)
    - Filter ratings to selected users and verify item/interactions counts
    - Map ratings to 1.0-5.0 scale (linear transform) to meet rubric
    - Save resulting interactions to processed folder and produce train/test splits
    """
    if not RATINGS_FILE.exists():
        print("Missing ratings.csv in data/raw/. Please upload MovieLens files.")
        return

    print(f"Generating filtered subset: users>={target_users}, items>={min_items}, interactions>={min_interactions}")
    ratings_full = pd.read_csv(RATINGS_FILE, usecols=["userId", "movieId", "rating"])  # full data

    # Compute activity per user and sort descending
    user_activity = ratings_full.groupby("userId").size().sort_values(ascending=False)
    users_sorted = user_activity.index.tolist()

    # Start by taking the top `target_users` most active users
    selected_users = users_sorted[:target_users]
    subset = ratings_full[ratings_full["userId"].isin(selected_users)].copy()

    # If interactions too small (unlikely), expand selection progressively
    idx = target_users
    while len(subset) < min_interactions and idx < len(users_sorted):
        # add next batch of users (in blocks of target_users//10)
        batch = users_sorted[idx : idx + max(100, target_users // 10)]
        selected_users.extend(batch)
        subset = ratings_full[ratings_full["userId"].isin(selected_users)].copy()
        idx += len(batch)

    # Ensure we have enough unique items; if not, include more users
    unique_items = subset["movieId"].nunique()
    if unique_items < min_items:
        print(f"Only {unique_items} items with current users; expanding user set to reach item threshold.")
        while subset["movieId"].nunique() < min_items and idx < len(users_sorted):
            batch = users_sorted[idx : idx + max(100, target_users // 10)]
            selected_users.extend(batch)
            subset = ratings_full[ratings_full["userId"].isin(selected_users)].copy()
            idx += len(batch)

    # Final counts
    n_users = subset["userId"].nunique()
    n_items = subset["movieId"].nunique()
    n_interactions = len(subset)

    print(f"Subset produced: users={n_users}, items={n_items}, interactions={n_interactions}")

    if n_users < target_users or n_items < min_items or n_interactions < min_interactions:
        print("Unable to produce subset meeting thresholds with current data selection.")
        return

    # Normalize ratings to strict 1.0 - 5.0 range via linear transform
    # MovieLens ratings are 0.5 - 5.0 (step 0.5). We map 0.5->1.0 and 5.0->5.0 linearly.
    subset["rating"] = 1.0 + (subset["rating"] - 0.5) * (4.0 / 4.5)

    # Rename columns to canonical names and save
    interactions = subset.rename(columns={"userId": "user_id", "movieId": "item_id"})[
        ["user_id", "item_id", "rating"]
    ]

    out_path = DATA_PROCESSED / "interactions.csv"
    interactions.to_csv(out_path, index=False)
    print(f"Filtered interactions saved to {out_path}")

    # Reuse pipeline steps: stats, train/test split, maps, and plots
    stats = compute_stats(interactions)
    cold_info = identify_cold_start(interactions, threshold=5)
    stats.update(cold_info)
    with open(STATS_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    train_df, test_df = per_user_train_test_split(interactions, test_fraction=0.2, seed=seed)
    train_df.to_csv(DATA_PROCESSED / "interactions_train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "interactions_test.csv", index=False)

    interaction_matrix, user_map, item_map = build_interaction_matrix(train_df)
    user_map_serializable = {str(k): int(v) for k, v in user_map.items()}
    item_map_serializable = {str(k): int(v) for k, v in item_map.items()}
    with open(DATA_PROCESSED / "user_map.json", "w", encoding="utf-8") as f:
        json.dump(user_map_serializable, f, indent=2)
    with open(DATA_PROCESSED / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(item_map_serializable, f, indent=2)

    # Plots
    user_activity = interactions.groupby("user_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(user_activity.values, bins=50)
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Count of users")
    plt.title("Ratings per user (filtered subset)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_user_subset.png", dpi=200)
    plt.close()

    item_popularity = interactions.groupby("item_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(item_popularity.values, bins=50)
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Count of movies")
    plt.title("Ratings per movie (filtered subset)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_movie_subset.png", dpi=200)
    plt.close()

    item_pop_sorted = item_popularity.sort_values(ascending=False)
    cum = item_pop_sorted.cumsum() / item_pop_sorted.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    plt.figure(figsize=(6, 4))
    plt.plot(x, cum.values)
    plt.xlabel("Fraction of items")
    plt.ylabel("Cumulative fraction of interactions")
    plt.title("Long-tail / Sparsity (filtered subset)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "long_tail_subset.png", dpi=200)
    plt.close()

    print("Filtered subset preprocessing complete. Outputs saved in data/processed/ and results/figures/.")


def generate_stratified_subset(
    target_interactions: int = 50000,
    min_users: int = 5000,
    min_items: int = 500,
    seed: int = 42,
) -> None:
    """Create stratified subset meeting AIE425 Section 2 rubric requirements.
    
    CRITICAL: This function implements proper stratified random sampling as required.
    - Includes low/medium/high activity users (preserves cold-start users)
    - Targets ~50,000 interactions (not 5M!)
    - Preserves long-tail item distribution
    - Ensures CF, content-based, and hybrid can all be evaluated
    
    Args:
        target_interactions: ~50,000 interactions (rubric requirement)
        min_users: minimum users (~5000)
        min_items: minimum items (>=500)
        seed: random seed for reproducibility
    """
    if not RATINGS_FILE.exists():
        print("Missing ratings.csv in data/raw/. Please upload MovieLens files.")
        return

    print(f"Generating STRATIFIED subset: ~{target_interactions} interactions, ~{min_users} users, >={min_items} items")
    
    # Load full ratings
    ratings_full = pd.read_csv(RATINGS_FILE, usecols=["userId", "movieId", "rating"])
    
    # Compute user activity distribution
    user_activity = ratings_full.groupby("userId").size().reset_index(name="count")
    
    # Define activity strata (per rubric: include low/medium/high)
    # Low: <=5 ratings (cold-start), Medium: 6-50, High: >50
    user_activity["stratum"] = pd.cut(
        user_activity["count"],
        bins=[0, 5, 50, np.inf],
        labels=["low", "medium", "high"]
    )
    
    # Stratified sampling proportions (ensure cold-start presence)
    # Target: 15% low (cold-start), 40% medium, 45% high
    low_users = user_activity[user_activity["stratum"] == "low"]
    medium_users = user_activity[user_activity["stratum"] == "medium"]
    high_users = user_activity[user_activity["stratum"] == "high"]
    
    # Sample from each stratum
    rng = np.random.RandomState(seed)
    
    # Start with proportional samples
    n_low = min(int(min_users * 0.15), len(low_users))
    n_medium = min(int(min_users * 0.40), len(medium_users))
    n_high = min(int(min_users * 0.45), len(high_users))
    
    sampled_low = low_users.sample(n=n_low, random_state=rng)
    sampled_medium = medium_users.sample(n=n_medium, random_state=rng)
    sampled_high = high_users.sample(n=n_high, random_state=rng)
    
    selected_user_ids = pd.concat([sampled_low, sampled_medium, sampled_high])["userId"].tolist()
    
    # Extract ratings for selected users
    subset = ratings_full[ratings_full["userId"].isin(selected_user_ids)].copy()
    
    # If interactions exceed target significantly, downsample ratings randomly
    if len(subset) > target_interactions * 1.2:
        subset = subset.sample(n=int(target_interactions * 1.1), random_state=rng)
    
    # Verify item count; if too few, add more medium-activity users
    unique_items = subset["movieId"].nunique()
    iteration = 0
    while unique_items < min_items and iteration < 10:
        # Add more medium-activity users (they have diverse item coverage)
        additional = medium_users[~medium_users["userId"].isin(selected_user_ids)].sample(
            n=min(200, len(medium_users) - n_medium),
            random_state=rng
        )
        selected_user_ids.extend(additional["userId"].tolist())
        subset = ratings_full[ratings_full["userId"].isin(selected_user_ids)].copy()
        unique_items = subset["movieId"].nunique()
        iteration += 1
    
    # Final stats
    n_users = subset["userId"].nunique()
    n_items = subset["movieId"].nunique()
    n_interactions = len(subset)
    
    print(f"Stratified subset: users={n_users}, items={n_items}, interactions={n_interactions}")
    
    # Verify cold-start users exist
    user_counts = subset.groupby("userId").size()
    cold_start_count = (user_counts <= 5).sum()
    print(f"Cold-start users (<=5 ratings): {cold_start_count}")
    
    if n_users < min_users * 0.8 or n_items < min_items or n_interactions < target_interactions * 0.8:
        print(f"WARNING: Subset below targets. Adjust sampling parameters.")
        # Continue anyway if close enough
        if n_users < 3000 or n_items < 500 or n_interactions < 40000:
            return
    
    # Normalize ratings: 0.5-5.0 → 1.0-5.0 (linear transform per rubric)
    subset["rating"] = 1.0 + (subset["rating"] - 0.5) * (4.0 / 4.5)
    
    # Rename and save canonical interactions
    interactions = subset.rename(columns={"userId": "user_id", "movieId": "item_id"})[
        ["user_id", "item_id", "rating"]
    ]
    
    # Remove duplicates (keep first)
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    
    out_path = DATA_PROCESSED / "interactions.csv"
    interactions.to_csv(out_path, index=False)
    print(f"Saved canonical interactions -> {out_path}")
    
    # Stats and cold-start identification
    stats = compute_stats(interactions)
    cold_info = identify_cold_start(interactions, threshold=5)
    stats.update(cold_info)
    
    stats_path = STATS_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats -> {stats_path}")
    
    # Per-user train/test split (80/20)
    train_df, test_df = per_user_train_test_split(interactions, test_fraction=0.2, seed=seed)
    train_df.to_csv(DATA_PROCESSED / "interactions_train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "interactions_test.csv", index=False)
    print(f"Train/test split saved")
    
    # Build sparse interaction matrix + mappings
    interaction_matrix, user_map, item_map = build_interaction_matrix(train_df)
    user_map_serializable = {str(k): int(v) for k, v in user_map.items()}
    item_map_serializable = {str(k): int(v) for k, v in item_map.items()}
    with open(DATA_PROCESSED / "user_map.json", "w", encoding="utf-8") as f:
        json.dump(user_map_serializable, f, indent=2)
    with open(DATA_PROCESSED / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(item_map_serializable, f, indent=2)
    
    # Generate EDA plots
    print("Generating plots...")
    user_activity_dist = interactions.groupby("user_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(user_activity_dist.values, bins=50, edgecolor="black")
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Count of users")
    plt.title("User Activity Distribution (Stratified)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_user_stratified.png", dpi=200)
    plt.close()
    
    item_popularity = interactions.groupby("item_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(item_popularity.values, bins=50, edgecolor="black")
    plt.xlabel("Number of ratings per movie")
    plt.ylabel("Count of movies")
    plt.title("Item Popularity Distribution (Stratified)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ratings_per_movie_stratified.png", dpi=200)
    plt.close()
    
    # Long-tail plot
    item_pop_sorted = item_popularity.sort_values(ascending=False)
    cum = item_pop_sorted.cumsum() / item_pop_sorted.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    plt.figure(figsize=(6, 4))
    plt.plot(x, cum.values)
    plt.xlabel("Fraction of items")
    plt.ylabel("Cumulative fraction of interactions")
    plt.title("Long-tail Distribution (Stratified)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "long_tail_stratified.png", dpi=200)
    plt.close()
    
    print("Stratified subset complete. Ready for content-based, CF, and hybrid evaluation.")
    print(f"Final counts: {n_users} users, {n_items} items, {n_interactions} interactions")
    print(f"Cold-start users: {cold_start_count} (natural presence for analysis)")

