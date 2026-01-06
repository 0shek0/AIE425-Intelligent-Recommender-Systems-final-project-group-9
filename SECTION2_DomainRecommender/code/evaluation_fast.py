import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
from hybrid import (
    load_all_data,
    predict_hybrid_batch,
)

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
STATS_DIR = RESULTS_DIR / "stats"

# ===============================
# Metrics
# ===============================

def evaluate_mae(predictions, actuals):
    """Mean Absolute Error."""
    return mean_absolute_error(actuals, predictions)

def evaluate_rmse(predictions, actuals):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(actuals, predictions))

# ===============================
# Evaluation (Fast Version)
# ===============================

def evaluate_content_based_fast(train_df, test_df, punks_df):
    """Fast content-based evaluation with limited samples."""
    print("Evaluating Content-Based Model...")
    
    # Build features
    tfidf_matrix, item_id_to_idx, item_ids = build_item_features(punks_df, use_id=True)
    
    predictions = []
    actuals = []
    
    # Limit to first 1000 test samples for speed
    test_sample = test_df.head(1000)
    print(f"Predicting {len(test_sample)} test samples (limited for speed)...")
    
    for idx, row in test_sample.iterrows():
        user_id = row["user_id"]
        item_id = int(row["item_id"])
        actual_rating = row["rating"]
        
        # Build user profile
        user_profile = build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx)
        
        if user_profile is None:
            user_profile = cold_start_profile(train_df, tfidf_matrix, item_id_to_idx)
        
        # Predict if item in features
        if user_profile is not None and item_id in item_id_to_idx:
            from sklearn.metrics.pairwise import cosine_similarity
            item_idx_cb = item_id_to_idx[item_id]
            item_vec = tfidf_matrix[item_idx_cb]
            sim_score = cosine_similarity(user_profile, item_vec).flatten()[0]
            predicted_rating = 1.0 + sim_score * 4.0
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(predictions, actuals)
    rmse = evaluate_rmse(predictions, actuals)
    coverage = 100 * len(predictions) / len(test_df)
    
    print(f"Content-Based: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": "Content-Based",
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage": float(coverage),
        "n_predictions": len(predictions),
    }

def evaluate_cf_fast(train_df, test_df, user_map, item_map, k=20):
    """Fast CF evaluation with limited samples."""
    print("\nEvaluating Collaborative Filtering Model...")
    
    # Create reverse maps
    user_id_to_idx = {v: k for k, v in user_map.items()}
    item_id_to_idx = {v: k for k, v in item_map.items()}
    
    # Build interaction matrix and similarity
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    test_df_copy = test_df.copy()
    test_df_copy["user_idx"] = test_df_copy["user_id"].map(user_id_to_idx)
    test_df_copy["item_idx"] = test_df_copy["item_id"].map(item_id_to_idx)
    
    # Remove unmapped
    test_df_copy = test_df_copy.dropna(subset=["user_idx", "item_idx"])
    test_df_copy["user_idx"] = test_df_copy["user_idx"].astype(int)
    test_df_copy["item_idx"] = test_df_copy["item_idx"].astype(int)
    
    # Limit to first 1000 for speed
    test_df_copy = test_df_copy.head(1000)
    
    predictions = []
    actuals = []
    
    print(f"Predicting {len(test_df_copy)} test samples...")
    
    for idx, row in test_df_copy.iterrows():
        pred = predict_cf(
            int(row["user_idx"]),
            int(row["item_idx"]),
            interaction_matrix,
            item_similarity,
            k=k
        )
        
        if pred is not None:
            predictions.append(pred)
            actuals.append(row["rating"])
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(predictions, actuals)
    rmse = evaluate_rmse(predictions, actuals)
    coverage = 100 * len(predictions) / len(test_df)
    
    print(f"CF: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": "Collaborative Filtering",
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage": float(coverage),
        "n_predictions": len(predictions),
    }

def evaluate_hybrid_fast(train_df, test_df, punks_df, user_map, item_map, alpha=0.5, k_cf=20):
    """Fast hybrid evaluation with limited samples."""
    print(f"\nEvaluating Hybrid Model (alpha={alpha})...")
    
    # Build components
    tfidf_matrix, item_id_to_idx, item_ids = build_item_features(punks_df, use_id=True)
    
    # Create reverse maps
    user_id_to_idx = {v: k for k, v in user_map.items()}
    item_id_to_idx_cf = {v: k for k, v in item_map.items()}
    
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    # Limit to first 1000 test samples
    test_sample = test_df.head(1000)
    
    predictions = []
    actuals = []
    
    from sklearn.metrics.pairwise import cosine_similarity
    print(f"Predicting {len(test_sample)} test samples...")
    
    for idx, row in test_sample.iterrows():
        user_id = row["user_id"]
        item_id = int(row["item_id"])
        actual_rating = row["rating"]
        
        # Content-based score
        cb_score = 0.0
        user_profile = build_user_profile(user_id, train_df, tfidf_matrix, item_id_to_idx)
        if user_profile is not None and item_id in item_id_to_idx:
            item_idx_cb = item_id_to_idx[item_id]
            item_vec = tfidf_matrix[item_idx_cb]
            cb_score = float(cosine_similarity(user_profile, item_vec).flatten()[0])
            cb_score = max(0.0, min(1.0, cb_score))
        
        # CF score
        cf_score = 0.0
        if user_id in user_id_to_idx and item_id in item_id_to_idx_cf:
            user_idx = user_id_to_idx[user_id]
            item_idx = item_id_to_idx_cf[item_id]
            pred = predict_cf(user_idx, item_idx, interaction_matrix, item_similarity, k=k_cf)
            if pred is not None:
                cf_score = (pred - 1.0) / 4.0
        
        # Hybrid score
        hybrid_score_norm = alpha * cb_score + (1 - alpha) * cf_score
        hybrid_rating = 1.0 + hybrid_score_norm * 4.0
        
        predictions.append(hybrid_rating)
        actuals.append(actual_rating)
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(predictions, actuals)
    rmse = evaluate_rmse(predictions, actuals)
    coverage = 100 * len(predictions) / len(test_df)
    
    print(f"Hybrid: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": f"Hybrid (alpha={alpha})",
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage": float(coverage),
        "n_predictions": len(predictions),
    }

def compare_models_fast(train_df, test_df, punks_df, user_map, item_map):
    """Compare all models and generate plots."""
    print("=" * 60)
    print("FAST MODEL EVALUATION (CryptoPunk)")
    print("=" * 60)
    
    results = []
    
    # Content-based
    cb_result = evaluate_content_based_fast(train_df, test_df, punks_df)
    if cb_result:
        results.append(cb_result)
    
    # CF
    cf_result = evaluate_cf_fast(train_df, test_df, user_map, item_map, k=20)
    if cf_result:
        results.append(cf_result)
    
    # Hybrid (multiple alpha values)
    for alpha in [0.3, 0.5, 0.7]:
        hybrid_result = evaluate_hybrid_fast(train_df, test_df, punks_df, user_map, item_map, alpha=alpha, k_cf=20)
        if hybrid_result:
            results.append(hybrid_result)
    
    if not results:
        print("No results generated!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(STATS_DIR / "model_comparison.csv", index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Create plot with simpler backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE comparison
        axes[0].bar(range(len(results_df)), results_df["mae"], color="skyblue", edgecolor="black")
        axes[0].set_xticks(range(len(results_df)))
        axes[0].set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("MAE")
        axes[0].set_title("Mean Absolute Error (CryptoPunk)")
        axes[0].grid(axis="y", alpha=0.3)
        
        # RMSE comparison
        axes[1].bar(range(len(results_df)), results_df["rmse"], color="salmon", edgecolor="black")
        axes[1].set_xticks(range(len(results_df)))
        axes[1].set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Root Mean Squared Error (CryptoPunk)")
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\nComparison plot saved: {FIGURES_DIR / 'model_comparison.png'}")
    except Exception as e:
        print(f"\nWarning: Could not save plot: {e}")
        print("Results CSV saved successfully.")
    
    return results_df

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    train_df, test_df, punks_df, user_map, item_map = load_all_data()
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Full comparison (limited to 1000 samples per model)
    results_df = compare_models_fast(train_df, test_df, punks_df, user_map, item_map)
    
    print("\n✅ Evaluation complete!")
