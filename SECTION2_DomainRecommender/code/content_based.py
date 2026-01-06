import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

INTERACTIONS_TRAIN_FILE = DATA_PROCESSED / "interactions_train.csv"
INTERACTIONS_TEST_FILE = DATA_PROCESSED / "interactions_test.csv"
PUNK_METADATA_FILE = DATA_RAW / "punk_metadata.json"

# ===============================
# Feature Extraction (TF-IDF on attributes)
# ===============================

def load_data():
    """Load train, test interactions and punk metadata."""
    train = pd.read_csv(INTERACTIONS_TRAIN_FILE)
    test = pd.read_csv(INTERACTIONS_TEST_FILE)
    
    # Try to load metadata if it exists, otherwise create dummy metadata
    if PUNK_METADATA_FILE.exists():
        with open(PUNK_METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        punks = pd.DataFrame([
            {'item_id': int(punk_id), 'attributes': ' '.join(attrs)}
            for punk_id, attrs in metadata_dict.items()
        ])
    else:
        # Fallback: create placeholder metadata from items in interactions
        all_items = pd.concat([train, test])['item_id'].unique()
        punks = pd.DataFrame({
            'item_id': all_items,
            'attributes': [f'punk_{i}' for i in all_items]
        })
    
    return train, test, punks


def build_item_features(punks, use_id=True):
    """Build sparse TF-IDF feature matrix for punk items.
    
    Args:
        punks: DataFrame with item_id and attributes
        use_id: if True, include punk ID in feature vector
    
    Returns:
        tfidf_matrix: sparse CSR matrix (n_items x n_features)
        item_id_to_idx: dict mapping item_id -> row index
        item_ids: list of item_ids corresponding to rows
    """
    punks = punks.copy()
    
    # Ensure we have item_id column
    if "item_id" not in punks.columns:
        raise ValueError("punks DataFrame must have 'item_id' column")
    
    # Clean and prepare text features
    punks["attributes"] = punks["attributes"].fillna("crypto punk")
    
    if use_id:
        punks["text"] = "punk_" + punks["item_id"].astype(str) + " " + punks["attributes"]
    else:
        punks["text"] = punks["attributes"]
    
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(punks["text"])
    item_ids = punks["item_id"].tolist()
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    print(f"Item feature matrix: {tfidf_matrix.shape} (sparse)")
    return tfidf_matrix, item_id_to_idx, item_ids

# ===============================
# User Profile Construction (sparse-aware)
# ===============================

def build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx):
    """Build weighted user profile from rated items (sparse).
    
    Args:
        user_id: user ID
        train_df: DataFrame with user_id, item_id, rating
        tfidf_matrix: sparse CSR matrix of item features
        item_id_to_idx: dict mapping item_id -> row index in tfidf_matrix
    
    Returns:
        user_profile: sparse row vector (1 x n_features) or None if no ratings
    """
    user_data = train_df[train_df["user_id"] == user_id]
    if user_data.empty:
        return None
    
    # Map item_ids to indices and filter out items not in feature matrix
    user_data = user_data[user_data["item_id"].isin(item_id_to_idx.keys())].copy()
    if user_data.empty:
        return None
    
    user_data["idx"] = user_data["item_id"].map(item_id_to_idx)
    indices = user_data["idx"].values
    weights = user_data["rating"].values
    
    # Weighted sum: profile = sum(rating_i * feature_vector_i) / sum(ratings)
    item_vectors = tfidf_matrix[indices]
    weighted_sum = (item_vectors.T.multiply(weights)).T.sum(axis=0)
    user_profile = weighted_sum / weights.sum()
    
    # Convert to array if matrix (numpy 2.0 compatibility)
    if hasattr(user_profile, 'A'):
        user_profile = user_profile.A
    
    return user_profile

def cold_start_profile(train_df, tfidf_matrix, item_id_to_idx, top_k=20):
    """Build profile from top popular items (cold-start fallback)."""
    popular_items = train_df.groupby("item_id").size().sort_values(ascending=False).head(top_k)
    popular_ids = popular_items.index.tolist()
    popular_ids = [i for i in popular_ids if i in item_id_to_idx]
    if not popular_ids:
        return None
    indices = [item_id_to_idx[i] for i in popular_ids]
    profile = tfidf_matrix[indices].mean(axis=0)
    
    # Convert to array if matrix (numpy 2.0 compatibility)
    if hasattr(profile, 'A'):
        profile = profile.A
    
    return profile

# ===============================
# Recommendation Generation
# ===============================

def recommend_content_based(user_id, train_df, tfidf_matrix, item_id_to_idx, item_ids, top_n=10):
    """Generate top-N recommendations for a user via content-based filtering.
    
    Args:
        user_id: target user
        train_df: training interactions
        tfidf_matrix: sparse item feature matrix
        item_id_to_idx: dict mapping item_id -> row index
        item_ids: array of item IDs corresponding to tfidf rows
        top_n: number of recommendations
    
    Returns:
        DataFrame with item_id and similarity score
    """
    user_profile = build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx)
    
    if user_profile is None:
        print(f"User {user_id} cold-start → using popular profile")
        user_profile = cold_start_profile(train_df, tfidf_matrix, item_id_to_idx)
        if user_profile is None:
            return pd.DataFrame(columns=["item_id", "score"])
    
    # Compute similarity: user_profile (1 x n_features) vs tfidf_matrix (n_items x n_features)
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    sim_df = pd.DataFrame({"item_id": item_ids, "score": sim_scores})
    
    # Remove already-rated items
    rated_items = train_df[train_df["user_id"] == user_id]["item_id"].values
    sim_df = sim_df[~sim_df["item_id"].isin(rated_items)]
    
    return sim_df.sort_values("score", ascending=False).head(top_n)


# ===============================
# Main Execution (safe import)
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    train_df, test_df, punks = load_data()
    
    print("Building item features (TF-IDF on attributes)...")
    tfidf_matrix, item_id_to_idx, item_ids = build_item_features(punks, use_id=True)
    
    # Sample a user from train set
    if len(train_df) > 0:
        sample_user = train_df["user_id"].iloc[0]
        
        print(f"\nGenerating top-10 recommendations for user {sample_user}...")
        recs = recommend_content_based(sample_user, train_df, tfidf_matrix, item_id_to_idx, item_ids, top_n=10)
        print(recs)
    
    print("\nContent-based model ready for evaluation.")
