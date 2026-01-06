import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

# ===============================
# Data Loading
# ===============================

def load_data():
    """Load train/test splits and mappings."""
    train_df = pd.read_csv(DATA_PROCESSED / "interactions_train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "interactions_test.csv")
    
    # Load maps: {index: id}
    with open(DATA_PROCESSED / "user_map.json", encoding="utf-8") as f:
        user_map_data = json.load(f)
        user_map = {int(k): v for k, v in user_map_data.items()}  # {index: user_id}
    
    with open(DATA_PROCESSED / "item_map.json", encoding="utf-8") as f:
        item_map_data = json.load(f)
        item_map = {int(k): v for k, v in item_map_data.items()}  # {index: item_id}
    
    return train_df, test_df, user_map, item_map

def build_interaction_matrix(train_df, user_map, item_map):
    """Build sparse user-item rating matrix.
    
    Args:
        train_df: DataFrame with user_id, item_id, rating
        user_map: {index: user_id} mapping
        item_map: {index: item_id} mapping
    
    Returns:
        Sparse CSR matrix
    """
    # Create reverse maps: user_id/item_id -> index
    user_id_to_idx = {v: k for k, v in user_map.items()}
    item_id_to_idx = {v: k for k, v in item_map.items()}
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    train_df = train_df.copy()
    train_df["user_idx"] = train_df["user_id"].map(user_id_to_idx)
    train_df["item_idx"] = train_df["item_id"].map(item_id_to_idx)
    
    # Remove unmapped entries
    train_df = train_df.dropna(subset=["user_idx", "item_idx"])
    train_df["user_idx"] = train_df["user_idx"].astype(int)
    train_df["item_idx"] = train_df["item_idx"].astype(int)
    
    row_ind = train_df["user_idx"].values
    col_ind = train_df["item_idx"].values
    data = train_df["rating"].values
    
    matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_users, n_items))
    return matrix

# ===============================
# Item-Item Similarity
# ===============================

def compute_item_similarity(interaction_matrix, top_k=50):
    """
    Compute item-item cosine similarity on user-item matrix.
    
    Args:
        interaction_matrix: sparse CSR matrix (users x items)
        top_k: keep only top-k most similar items per item (memory efficiency)
    
    Returns:
        item_similarity: sparse CSR matrix (items x items)
    """
    # Transpose: items x users
    item_user_matrix = interaction_matrix.T.tocsr()
    
    # Compute cosine similarity between items
    print(f"Computing item-item similarity matrix...")
    similarity = cosine_similarity(item_user_matrix, dense_output=False)
    
    # Sparse version: avoid slow CSR item assignment
    # Instead, build csr directly from filtered data
    print(f"Keeping top-{top_k} neighbors per item...")
    
    # Convert to lil format for efficient row operations
    similarity_lil = similarity.tolil()
    
    for i in range(similarity_lil.shape[0]):
        row_data = similarity_lil.data[i]
        row_rows = similarity_lil.rows[i]
        
        if len(row_data) > top_k + 1:
            # Get indices of top-k values
            indices = np.argsort(row_data)[-top_k-1:]
            # Keep only top-k + self
            keep_idx = set(indices)
            new_data = [row_data[j] for j in range(len(row_data)) if j in keep_idx]
            new_rows = [row_rows[j] for j in range(len(row_rows)) if j in keep_idx]
            similarity_lil.data[i] = new_data
            similarity_lil.rows[i] = new_rows
    
    # Convert back to CSR
    similarity = similarity_lil.tocsr()
    similarity.eliminate_zeros()
    
    print(f"Similarity matrix: {similarity.shape}, {similarity.nnz} non-zeros")
    
    return similarity

# ===============================
# Prediction
# ===============================

def predict_cf(user_idx, item_idx, interaction_matrix, item_similarity, k=20):
    """
    Predict rating for user-item pair using item-based CF.
    
    Args:
        user_idx: user matrix index
        item_idx: item matrix index
        interaction_matrix: sparse CSR (users x items)
        item_similarity: sparse CSR (items x items)
        k: number of neighbors to use
    
    Returns:
        predicted rating (float) or None if cannot predict
    """
    # Get user's ratings
    user_ratings = interaction_matrix.getrow(user_idx).toarray().flatten()
    rated_items = np.nonzero(user_ratings)[0]
    
    if len(rated_items) == 0:
        return None
    
    # Get similarity scores between target item and rated items
    sim_scores = item_similarity.getrow(item_idx).toarray().flatten()
    
    # Filter to only rated items
    neighbor_sims = sim_scores[rated_items]
    neighbor_ratings = user_ratings[rated_items]
    
    # Remove zero similarities
    nonzero_mask = neighbor_sims > 0
    neighbor_sims = neighbor_sims[nonzero_mask]
    neighbor_ratings = neighbor_ratings[nonzero_mask]
    
    if len(neighbor_sims) == 0:
        return None
    
    # Select top-k neighbors
    if len(neighbor_sims) > k:
        top_k_idx = np.argsort(neighbor_sims)[-k:]
        neighbor_sims = neighbor_sims[top_k_idx]
        neighbor_ratings = neighbor_ratings[top_k_idx]
    
    # Weighted average
    if neighbor_sims.sum() == 0:
        return None
    
    prediction = np.dot(neighbor_sims, neighbor_ratings) / neighbor_sims.sum()
    
    # Clip to rating scale
    prediction = np.clip(prediction, 1.0, 5.0)
    
    return prediction

# ===============================
# Batch Prediction
# ===============================

def predict_cf_batch(test_df, user_map, item_map, interaction_matrix, item_similarity, k=20):
    """Predict ratings for all test samples."""
    test_df = test_df.copy()
    test_df["user_idx"] = test_df["user_id"].map(user_map)
    test_df["item_idx"] = test_df["item_id"].map(item_map)
    
    # Remove unmapped
    test_df = test_df.dropna(subset=["user_idx", "item_idx"])
    test_df["user_idx"] = test_df["user_idx"].astype(int)
    test_df["item_idx"] = test_df["item_idx"].astype(int)
    
    predictions = []
    print(f"Predicting {len(test_df)} test ratings...")
    
    for idx, row in test_df.iterrows():
        pred = predict_cf(
            row["user_idx"],
            row["item_idx"],
            interaction_matrix,
            item_similarity,
            k=k
        )
        predictions.append(pred)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(test_df)} predictions done")
    
    test_df["predicted_rating"] = predictions
    
    # Remove unpredictable samples
    valid_df = test_df.dropna(subset=["predicted_rating"])
    print(f"Coverage: {len(valid_df)}/{len(test_df)} ({100*len(valid_df)/len(test_df):.1f}%)")
    
    return valid_df

# ===============================
# Recommendation Generation
# ===============================

def recommend_cf(user_id, train_df, user_map, item_map, interaction_matrix, item_similarity, k=20, top_n=10):
    """
    Generate top-N recommendations for a user using CF.
    
    Args:
        user_id: original user ID
        train_df: training data (to filter already-rated items)
        user_map: {index: user_id} mapping
        item_map: {index: item_id} mapping
        interaction_matrix: sparse CSR (users x items)
        item_similarity: sparse CSR (items x items)
        k: number of neighbors for prediction
        top_n: number of recommendations
    
    Returns:
        DataFrame with item_id and predicted_rating
    """
    # Create reverse maps
    user_id_to_idx = {v: k for k, v in user_map.items()}
    item_id_to_idx = {v: k for k, v in item_map.items()}
    
    if user_id not in user_id_to_idx:
        return pd.DataFrame(columns=["item_id", "predicted_rating"])
    
    user_idx = user_id_to_idx[user_id]
    
    # Get all items from item_map
    all_items = list(item_map.values())
    item_indices = list(item_map.keys())
    
    # Filter out already-rated items
    rated_items = train_df[train_df["user_id"] == user_id]["item_id"].values
    candidate_items = [i for i in all_items if i not in rated_items]
    candidate_indices = [item_id_to_idx[i] for i in candidate_items]
    
    # Predict for all candidates
    predictions = []
    for item_id, item_idx in zip(candidate_items, candidate_indices):
        pred = predict_cf(user_idx, item_idx, interaction_matrix, item_similarity, k=k)
        if pred is not None:
            predictions.append({"item_id": item_id, "predicted_rating": pred})
    
    if not predictions:
        return pd.DataFrame(columns=["item_id", "predicted_rating"])
    
    recs_df = pd.DataFrame(predictions)
    recs_df = recs_df.sort_values("predicted_rating", ascending=False).head(top_n)
    
    return recs_df

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    train_df, test_df, user_map, item_map = load_data()
    
    print(f"Train: {len(train_df)} ratings")
    print(f"Test: {len(test_df)} ratings")
    print(f"Users: {len(user_map)}, Items: {len(item_map)}")
    
    print("\nBuilding interaction matrix...")
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    print(f"Matrix shape: {interaction_matrix.shape}")
    print(f"Sparsity: {100 * (1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])):.2f}%")
    
    print("\nComputing item-item similarity...")
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    print("\nTesting CF prediction on sample user...")
    if len(train_df) > 0:
        sample_user = train_df["user_id"].iloc[0]
        recs = recommend_cf(sample_user, train_df, user_map, item_map, interaction_matrix, item_similarity, k=20, top_n=10)
        print(f"\nTop-10 CF recommendations for user {sample_user}:")
        print(recs)
    
    print("\nCollaborative filtering model ready for evaluation.")
