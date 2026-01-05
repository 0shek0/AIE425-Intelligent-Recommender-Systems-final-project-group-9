"""
Hybrid Recommender System

Combines content-based and collaborative filtering using weighted combination:
    Score = α × Content + (1 - α) × CF

Supports:
- Weighted hybrid with tunable alpha
- Switching hybrid (content-based for cold-start, CF otherwise)
- Score normalization for fair combination
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Import models
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from content_based import (
    load_data as load_data_cb,
    build_item_features,
    build_user_profile,
    cold_start_profile,
)
from collaborative_filtering import (
    load_data as load_data_cf,
    build_interaction_matrix,
    compute_item_similarity,
    predict_cf,
)

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

# ===============================
# Data Loading
# ===============================

def load_all_data():
    """Load all data needed for hybrid model."""
    # Load train/test/movies
    train_df = pd.read_csv(DATA_PROCESSED / "interactions_train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "interactions_test.csv")
    
    # Find movies file
    movies_file = DATA_PROCESSED.parent / "raw" / "movies.csv"
    if not movies_file.exists():
        # Try symlink target
        import os
        if movies_file.is_symlink():
            movies_file = Path(os.readlink(movies_file))
    
    movies_df = pd.read_csv(movies_file)
    
    # Load mappings
    with open(DATA_PROCESSED / "user_map.json", encoding="utf-8") as f:
        user_map = {int(k): v for k, v in json.load(f).items()}
    
    with open(DATA_PROCESSED / "item_map.json", encoding="utf-8") as f:
        item_map = {int(k): v for k, v in json.load(f).items()}
    
    return train_df, test_df, movies_df, user_map, item_map

# ===============================
# Score Normalization
# ===============================

def normalize_scores(scores_dict, method="minmax"):
    """
    Normalize scores to [0, 1] range for fair combination.
    
    Args:
        scores_dict: {item_id: score} mapping
        method: "minmax" or "zscore"
    
    Returns:
        normalized {item_id: score} mapping
    """
    if not scores_dict:
        return {}
    
    items = list(scores_dict.keys())
    scores = np.array([scores_dict[i] for i in items]).reshape(-1, 1)
    
    if method == "minmax":
        if scores.max() == scores.min():
            # All same score
            normalized = np.ones_like(scores) * 0.5
        else:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(scores).flatten()
    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            normalized = np.zeros_like(scores).flatten()
        else:
            normalized = ((scores - mean) / std).flatten()
            # Map to [0, 1]
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-9)
    else:
        normalized = scores.flatten()
    
    return {item: score for item, score in zip(items, normalized)}

# ===============================
# Weighted Hybrid
# ===============================

def recommend_hybrid_weighted(
    user_id,
    train_df,
    movies_df,
    tfidf_matrix,
    item_id_to_idx_cb,
    item_ids_cb,
    user_map,
    item_map,
    interaction_matrix,
    item_similarity,
    alpha=0.5,
    k_cf=20,
    top_n=10,
):
    """
    Weighted hybrid: Score = α × Content + (1 - α) × CF
    
    Args:
        user_id: user ID
        train_df: training data
        movies_df: movie metadata
        tfidf_matrix: content-based feature matrix
        item_id_to_idx_cb: item_id -> index in tfidf_matrix
        item_ids_cb: list of item_ids corresponding to tfidf_matrix rows
        user_map: user_id -> matrix index
        item_map: item_id -> matrix index
        interaction_matrix: sparse user-item matrix
        item_similarity: sparse item-item similarity
        alpha: weight for content-based (0-1)
        k_cf: number of CF neighbors
        top_n: number of recommendations
    
    Returns:
        DataFrame with item_id and hybrid_score
    """
    # Get content-based scores
    user_profile = build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx_cb)
    
    if user_profile is None:
        user_profile = cold_start_profile(train_df, tfidf_matrix, item_id_to_idx_cb)
    
    # Get already-rated items to filter
    rated_items = train_df[train_df["user_id"] == user_id]["item_id"].values
    
    # Content-based scores
    cb_scores = {}
    if user_profile is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
        for item_id, score in zip(item_ids_cb, sim_scores):
            if item_id not in rated_items:
                cb_scores[item_id] = score
    
    # Normalize content-based scores
    cb_scores_norm = normalize_scores(cb_scores)
    
    # CF scores
    cf_scores = {}
    if user_id in user_map:
        user_idx = user_map[user_id]
        
        # Get candidate items
        candidate_items = [i for i in item_map.keys() if i not in rated_items]
        
        for item_id in candidate_items:
            if item_id in item_map:
                item_idx = item_map[item_id]
                pred = predict_cf(user_idx, item_idx, interaction_matrix, item_similarity, k=k_cf)
                if pred is not None:
                    cf_scores[item_id] = pred
    
    # Normalize CF scores (ratings are 1-5, map to 0-1)
    cf_scores_norm = {item: (score - 1.0) / 4.0 for item, score in cf_scores.items()}
    
    # Combine scores
    all_items = set(cb_scores_norm.keys()) | set(cf_scores_norm.keys())
    hybrid_scores = {}
    
    for item_id in all_items:
        cb_score = cb_scores_norm.get(item_id, 0.0)
        cf_score = cf_scores_norm.get(item_id, 0.0)
        hybrid_scores[item_id] = alpha * cb_score + (1 - alpha) * cf_score
    
    # Sort and return top-N
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    recs_df = pd.DataFrame(sorted_items, columns=["item_id", "hybrid_score"])
    
    return recs_df

# ===============================
# Switching Hybrid
# ===============================

def recommend_hybrid_switching(
    user_id,
    train_df,
    movies_df,
    tfidf_matrix,
    item_id_to_idx_cb,
    item_ids_cb,
    user_map,
    item_map,
    interaction_matrix,
    item_similarity,
    cold_start_threshold=5,
    k_cf=20,
    top_n=10,
):
    """
    Switching hybrid: use content-based for cold-start users, CF otherwise.
    
    Args:
        user_id: user ID
        train_df: training data
        cold_start_threshold: users with ≤ this many ratings are cold-start
        ... (other args same as weighted hybrid)
    
    Returns:
        DataFrame with item_id and score
    """
    # Check if cold-start user
    user_ratings = train_df[train_df["user_id"] == user_id]
    is_cold_start = len(user_ratings) <= cold_start_threshold
    
    if is_cold_start:
        # Use content-based
        from content_based import recommend_content_based
        recs = recommend_content_based(user_id, train_df, tfidf_matrix, item_id_to_idx_cb, item_ids_cb, top_n=top_n)
        recs = recs.rename(columns={"score": "hybrid_score"})
    else:
        # Use CF
        from collaborative_filtering import recommend_cf
        recs = recommend_cf(user_id, train_df, user_map, item_map, interaction_matrix, item_similarity, k=k_cf, top_n=top_n)
        recs = recs.rename(columns={"predicted_rating": "hybrid_score"})
    
    return recs

# ===============================
# Batch Prediction
# ===============================

def predict_hybrid_batch(
    test_df,
    train_df,
    movies_df,
    tfidf_matrix,
    item_id_to_idx_cb,
    item_ids_cb,
    user_map,
    item_map,
    interaction_matrix,
    item_similarity,
    alpha=0.5,
    k_cf=20,
):
    """Predict ratings for test set using weighted hybrid."""
    test_df = test_df.copy()
    predictions = []
    
    print(f"Predicting {len(test_df)} test ratings with hybrid (alpha={alpha})...")
    
    for idx, row in test_df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        
        # Content-based score
        cb_score = 0.0
        user_profile = build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx_cb)
        if user_profile is not None and item_id in item_id_to_idx_cb:
            from sklearn.metrics.pairwise import cosine_similarity
            item_idx_cb = item_id_to_idx_cb[item_id]
            item_vec = tfidf_matrix[item_idx_cb]
            cb_score = cosine_similarity(user_profile, item_vec).flatten()[0]
            # Normalize to 0-1 (assume max similarity is 1.0)
            cb_score = max(0.0, min(1.0, cb_score))
        
        # CF score
        cf_score = 0.0
        if user_id in user_map and item_id in item_map:
            user_idx = user_map[user_id]
            item_idx = item_map[item_id]
            pred = predict_cf(user_idx, item_idx, interaction_matrix, item_similarity, k=k_cf)
            if pred is not None:
                # Normalize to 0-1 (ratings are 1-5)
                cf_score = (pred - 1.0) / 4.0
        
        # Hybrid score (in 0-1 range)
        hybrid_score_norm = alpha * cb_score + (1 - alpha) * cf_score
        
        # Map back to rating scale (1-5)
        hybrid_rating = 1.0 + hybrid_score_norm * 4.0
        
        predictions.append(hybrid_rating)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(test_df)} predictions done")
    
    test_df["predicted_rating"] = predictions
    
    return test_df

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    print("Loading all data...")
    train_df, test_df, movies_df, user_map, item_map = load_all_data()
    
    print(f"Train: {len(train_df)} ratings")
    print(f"Test: {len(test_df)} ratings")
    
    print("\nBuilding content-based features...")
    tfidf_matrix, item_id_to_idx_cb, item_ids_cb = build_item_features(movies_df, use_title=False)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")
    
    print("\nBuilding CF components...")
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    print("\nTesting weighted hybrid (alpha=0.5)...")
    sample_user = train_df["user_id"].iloc[100]
    recs = recommend_hybrid_weighted(
        sample_user,
        train_df,
        movies_df,
        tfidf_matrix,
        item_id_to_idx_cb,
        item_ids_cb,
        user_map,
        item_map,
        interaction_matrix,
        item_similarity,
        alpha=0.5,
        top_n=10,
    )
    print(f"\nTop-10 hybrid recommendations for user {sample_user}:")
    print(recs)
    
    print("\nHybrid model ready for evaluation.")
