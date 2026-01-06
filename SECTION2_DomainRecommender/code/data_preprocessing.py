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

# Expected crypto dataset file
CRYPTO_DATASET_FILE = DATA_RAW / "crypto_transactions.jsonl"


# ===============================
# Helper functions
# ===============================

def check_files_exist() -> bool:
    """Check that the required crypto dataset file exists."""
    return CRYPTO_DATASET_FILE.exists()


def load_crypto_transactions(limit_rows: int = None) -> pd.DataFrame:
    """Load CryptoPunk transaction history from JSONL.

    - Extracts user (from wallet), item (punk_id), and rating (normalized ETH price)
    - Filters out transactions with missing critical fields
    """
    transactions = []
    
    with open(CRYPTO_DATASET_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit_rows and i >= limit_rows:
                break
            try:
                txn = json.loads(line)
                
                # Skip if missing essential fields
                if not txn.get('from') or txn.get('punk_id') is None or txn.get('eth') is None:
                    continue
                
                # Extract fields
                user_id = txn['from']
                item_id = int(txn['punk_id'])
                eth_value = float(txn['eth'])
                
                # Convert ETH to rating scale (1-5)
                # Strategy: use percentile-based normalization
                # Will normalize after loading all data
                transactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'eth': eth_value,
                    'txn_type': txn.get('txn_type', 'Unknown'),
                    'timestamp': txn.get('timestamp', '')
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    
    if not transactions:
        return pd.DataFrame()
    
    df = pd.DataFrame(transactions)
    return df


def normalize_ratings(df: pd.DataFrame, min_rating: float = 1.0, max_rating: float = 5.0) -> pd.DataFrame:
    """Normalize ETH values to rating scale using percentile-based approach.

    Args:
        df: DataFrame with 'eth' column
        min_rating: Minimum rating (default 1.0)
        max_rating: Maximum rating (default 5.0)
    
    Returns:
        DataFrame with new 'rating' column
    """
    df = df.copy()
    
    # Get percentiles from ETH values
    p1, p99 = df['eth'].quantile([0.01, 0.99])
    
    # Clip to percentile range to handle outliers
    eth_clipped = df['eth'].clip(p1, p99)
    
    # Normalize to [0, 1]
    eth_normalized = (eth_clipped - eth_clipped.min()) / (eth_clipped.max() - eth_clipped.min() + 1e-8)
    
    # Scale to [min_rating, max_rating]
    df['rating'] = min_rating + eth_normalized * (max_rating - min_rating)
    
    return df


def save_interactions(df: pd.DataFrame) -> Path:
    """Save canonical interactions with columns user_id, item_id, rating."""
    out = DATA_PROCESSED / "interactions.csv"
    df[["user_id", "item_id", "rating"]].to_csv(out, index=False)
    return out


def compute_stats(interactions: pd.DataFrame) -> Dict:
    """Compute and return dataset statistics."""
    n_users = int(interactions["user_id"].nunique())
    n_items = int(interactions["item_id"].nunique())
    n_interactions = int(len(interactions))

    density = n_interactions / (n_users * n_items) if (n_users * n_items) > 0 else 0
    sparsity = 1.0 - density

    rating_distribution = (
        interactions["rating"].value_counts(bins=5).sort_index().to_dict()
    )

    stats = {
        "users": n_users,
        "items": n_items,
        "interactions": n_interactions,
        "density": density,
        "sparsity": sparsity,
        "rating_distribution": {str(k): v for k, v in rating_distribution.items()},
    }

    return stats


def identify_cold_start(interactions: pd.DataFrame, threshold: int = 5) -> Dict:
    """Identify cold-start users and low-frequency items."""
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
    """Perform 80/20 per-user split with fixed seed."""
    rng = np.random.RandomState(seed)

    train_rows = []
    test_rows = []

    grouped = interactions.groupby("user_id")

    for user_id, group in grouped:
        n = len(group)
        if n == 1:
            train_rows.append(group.index.values)
            continue

        n_test = max(1, int(np.floor(test_fraction * n)))
        chosen = rng.choice(group.index.values, size=n_test, replace=False)
        test_rows.append(chosen)
        train_rows.append(np.setdiff1d(group.index.values, chosen))

    train_idx = np.concatenate(train_rows) if train_rows else np.array([], dtype=int)
    test_idx = np.concatenate(test_rows) if test_rows else np.array([], dtype=int)

    train_df = interactions.loc[train_idx].reset_index(drop=True)
    test_df = interactions.loc[test_idx].reset_index(drop=True)

    return train_df, test_df


def build_interaction_matrix(
    interactions: pd.DataFrame,
) -> Tuple[sparse.csr_matrix, Dict, Dict]:
    """Build sparse user-item matrix and mapping dictionaries."""
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
# Main orchestration
# ===============================

def run_preprocessing(limit_rows: int = None, seed: int = 42) -> None:
    """Full preprocessing pipeline entrypoint for crypto data."""
    if not check_files_exist():
        print(
            f"Crypto dataset not found at {CRYPTO_DATASET_FILE}. "
            "Please ensure crypto_transactions.jsonl exists in data/raw/"
        )
        return

    print("Loading CryptoPunk transaction data...")
    transactions = load_crypto_transactions(limit_rows=limit_rows)
    
    if transactions.empty:
        print("No valid transactions found. Please check the dataset format.")
        return
    
    print(f"Loaded {len(transactions)} transactions")

    # Normalize ETH prices to rating scale
    transactions = normalize_ratings(transactions)
    
    # Create canonical interactions
    interactions = transactions[["user_id", "item_id", "rating"]].copy()
    
    # Remove exact duplicates (keep first occurrence)
    interactions = interactions.drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    
    print(f"After deduplication: {len(interactions)} interactions")

    # Save canonical interactions
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

    # Per-user train/test split
    train_df, test_df = per_user_train_test_split(interactions, test_fraction=0.2, seed=seed)
    train_path = DATA_PROCESSED / "interactions_train.csv"
    test_path = DATA_PROCESSED / "interactions_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train/test saved to {train_path} and {test_path}")

    # Build sparse interaction matrix for downstream models
    interaction_matrix, user_map, item_map = build_interaction_matrix(train_df)

    # Save lightweight mapping files
    # Important: Save as {index: id} for easier loading downstream
    user_id_to_map = {v: k for k, v in user_map.items()}  # Reverse: index -> user_id
    item_id_to_map = {v: k for k, v in item_map.items()}  # Reverse: index -> item_id
    
    user_map_serializable = {str(k): str(v) for k, v in user_id_to_map.items()}
    item_map_serializable = {str(k): int(v) for k, v in item_id_to_map.items()}
    with open(DATA_PROCESSED / "user_map.json", "w", encoding="utf-8") as f:
        json.dump(user_map_serializable, f, indent=2)
    with open(DATA_PROCESSED / "item_map.json", "w", encoding="utf-8") as f:
        json.dump(item_map_serializable, f, indent=2)

    # ===============================
    # Plots
    # ===============================
    print("Generating required plots...")

    # User activity distribution
    user_activity = interactions.groupby("user_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(user_activity.values, bins=50, edgecolor='black')
    plt.xlabel("Number of transactions per user")
    plt.ylabel("Count of users")
    plt.title("User Activity Distribution (CryptoPunk)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "user_activity_distribution.png", dpi=200)
    plt.close()

    # Item popularity distribution
    item_popularity = interactions.groupby("item_id").size()
    plt.figure(figsize=(6, 4))
    plt.hist(item_popularity.values, bins=50, edgecolor='black')
    plt.xlabel("Number of transactions per punk")
    plt.ylabel("Count of punks")
    plt.title("Punk Popularity Distribution (CryptoPunk)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "punk_popularity_distribution.png", dpi=200)
    plt.close()

    # Long-tail distribution
    item_pop_sorted = item_popularity.sort_values(ascending=False)
    cum = item_pop_sorted.cumsum() / item_pop_sorted.sum()
    x = np.arange(1, len(cum) + 1) / len(cum)
    plt.figure(figsize=(6, 4))
    plt.plot(x, cum.values, linewidth=2)
    plt.xlabel("Fraction of punks")
    plt.ylabel("Cumulative fraction of transactions")
    plt.title("Long-tail Distribution (CryptoPunk)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "long_tail_distribution.png", dpi=200)
    plt.close()

    # Rating distribution
    plt.figure(figsize=(6, 4))
    plt.hist(interactions["rating"].values, bins=30, edgecolor='black')
    plt.xlabel("Rating (normalized ETH)")
    plt.ylabel("Count")
    plt.title("Rating Distribution (CryptoPunk)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_distribution.png", dpi=200)
    plt.close()

    print("All figures saved to results/figures/")
    print("\nPreprocessing complete!")
    print(f"Summary: {stats['users']} users, {stats['items']} items, {stats['interactions']} interactions")
    print(f"Density: {stats['density']:.4f}, Sparsity: {stats['sparsity']:.4f}")


if __name__ == "__main__":
    run_preprocessing(limit_rows=None, seed=42)
