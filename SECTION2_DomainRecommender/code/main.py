"""
Evaluation Module for Recommender Systems

Implements evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Coverage
- Cold-start analysis

Supports comparison between content-based, CF, and hybrid models.
"""

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

def evaluate_coverage(predictions_df, test_df):
    """Percentage of test samples with predictions."""
    return 100 * len(predictions_df) / len(test_df)

# ===============================
# Content-Based Evaluation
# ===============================

def evaluate_content_based(train_df, test_df, movies_df):
    """Evaluate content-based model on test set."""
    print("Evaluating Content-Based Model...")
    
    # Build features
    tfidf_matrix, item_id_to_idx, item_ids = build_item_features(movies_df, use_title=False)
    
    predictions = []
    actuals = []
    
    print(f"Predicting {len(test_df)} test ratings...")
    
    for idx, row in test_df.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
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
            
            # Map similarity [0, 1] to rating [1, 5]
            predicted_rating = 1.0 + sim_score * 4.0
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(test_df)} predictions done")
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(predictions, actuals)
    rmse = evaluate_rmse(predictions, actuals)
    coverage = 100 * len(predictions) / len(test_df)
    
    print(f"Content-Based: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": "Content-Based",
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
        "n_predictions": len(predictions),
    }

# ===============================
# CF Evaluation
# ===============================

def evaluate_cf(train_df, test_df, user_map, item_map, k=20):
    """Evaluate collaborative filtering on test set."""
    print("\nEvaluating Collaborative Filtering Model...")
    
    # Build interaction matrix and similarity
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    test_df = test_df.copy()
    test_df["user_idx"] = test_df["user_id"].map(user_map)
    test_df["item_idx"] = test_df["item_id"].map(item_map)
    
    # Remove unmapped
    test_df = test_df.dropna(subset=["user_idx", "item_idx"])
    test_df["user_idx"] = test_df["user_idx"].astype(int)
    test_df["item_idx"] = test_df["item_idx"].astype(int)
    
    predictions = []
    actuals = []
    
    print(f"Predicting {len(test_df)} test ratings...")
    
    for idx, row in test_df.iterrows():
        pred = predict_cf(
            row["user_idx"],
            row["item_idx"],
            interaction_matrix,
            item_similarity,
            k=k
        )
        
        if pred is not None:
            predictions.append(pred)
            actuals.append(row["rating"])
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(test_df)} predictions done")
    
    if len(predictions) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(predictions, actuals)
    rmse = evaluate_rmse(predictions, actuals)
    coverage = 100 * len(predictions) / len(test_df)
    
    print(f"CF: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": "Collaborative Filtering",
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
        "n_predictions": len(predictions),
    }

# ===============================
# Hybrid Evaluation
# ===============================

def evaluate_hybrid(train_df, test_df, movies_df, user_map, item_map, alpha=0.5, k_cf=20):
    """Evaluate hybrid model on test set."""
    print(f"\nEvaluating Hybrid Model (alpha={alpha})...")
    
    # Build components
    tfidf_matrix, item_id_to_idx, item_ids = build_item_features(movies_df, use_title=False)
    interaction_matrix = build_interaction_matrix(train_df, user_map, item_map)
    item_similarity = compute_item_similarity(interaction_matrix, top_k=50)
    
    test_df_pred = predict_hybrid_batch(
        test_df,
        train_df,
        movies_df,
        tfidf_matrix,
        item_id_to_idx,
        item_ids,
        user_map,
        item_map,
        interaction_matrix,
        item_similarity,
        alpha=alpha,
        k_cf=k_cf,
    )
    
    # Remove NaN predictions
    valid_df = test_df_pred[test_df_pred["predicted_rating"].notna()]
    
    if len(valid_df) == 0:
        print("No predictions generated!")
        return None
    
    mae = evaluate_mae(valid_df["predicted_rating"], valid_df["rating"])
    rmse = evaluate_rmse(valid_df["predicted_rating"], valid_df["rating"])
    coverage = 100 * len(valid_df) / len(test_df)
    
    print(f"Hybrid: MAE={mae:.4f}, RMSE={rmse:.4f}, Coverage={coverage:.1f}%")
    
    return {
        "model": f"Hybrid (alpha={alpha})",
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
        "n_predictions": len(valid_df),
    }

# ===============================
# Cold-Start Analysis
# ===============================

def cold_start_analysis(train_df, test_df, movies_df, user_map, item_map, threshold=5):
    """Evaluate models separately on cold-start vs non-cold-start users."""
    print(f"\n=== COLD-START ANALYSIS (threshold={threshold} ratings) ===")
    
    # Identify cold-start users in test set
    user_train_counts = train_df.groupby("user_id").size()
    test_df = test_df.copy()
    test_df["train_count"] = test_df["user_id"].map(user_train_counts).fillna(0)
    
    cold_start_df = test_df[test_df["train_count"] <= threshold]
    non_cold_start_df = test_df[test_df["train_count"] > threshold]
    
    print(f"Cold-start users: {cold_start_df['user_id'].nunique()}")
    print(f"Non-cold-start users: {non_cold_start_df['user_id'].nunique()}")
    print(f"Cold-start test samples: {len(cold_start_df)}")
    print(f"Non-cold-start test samples: {len(non_cold_start_df)}")
    
    results = []
    
    # Evaluate on cold-start
    if len(cold_start_df) > 0:
        print("\n--- Cold-Start Users ---")
        cb_result = evaluate_content_based(train_df, cold_start_df, movies_df)
        if cb_result:
            cb_result["segment"] = "Cold-Start"
            results.append(cb_result)
        
        cf_result = evaluate_cf(train_df, cold_start_df, user_map, item_map, k=20)
        if cf_result:
            cf_result["segment"] = "Cold-Start"
            results.append(cf_result)
        
        hybrid_result = evaluate_hybrid(train_df, cold_start_df, movies_df, user_map, item_map, alpha=0.7, k_cf=20)
        if hybrid_result:
            hybrid_result["segment"] = "Cold-Start"
            results.append(hybrid_result)
    
    # Evaluate on non-cold-start
    if len(non_cold_start_df) > 0:
        print("\n--- Non-Cold-Start Users ---")
        cb_result = evaluate_content_based(train_df, non_cold_start_df, movies_df)
        if cb_result:
            cb_result["segment"] = "Non-Cold-Start"
            results.append(cb_result)
        
        cf_result = evaluate_cf(train_df, non_cold_start_df, user_map, item_map, k=20)
        if cf_result:
            cf_result["segment"] = "Non-Cold-Start"
            results.append(cf_result)
        
        hybrid_result = evaluate_hybrid(train_df, non_cold_start_df, movies_df, user_map, item_map, alpha=0.3, k_cf=20)
        if hybrid_result:
            hybrid_result["segment"] = "Non-Cold-Start"
            results.append(hybrid_result)
    
    return pd.DataFrame(results)

# ===============================
# Comparison & Visualization
# ===============================

def compare_models(train_df, test_df, movies_df, user_map, item_map):
    """Compare all models and generate plots."""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    results = []
    
    # Content-based
    cb_result = evaluate_content_based(train_df, test_df, movies_df)
    if cb_result:
        results.append(cb_result)
    
    # CF
    cf_result = evaluate_cf(train_df, test_df, user_map, item_map, k=20)
    if cf_result:
        results.append(cf_result)
    
    # Hybrid (multiple alpha values)
    for alpha in [0.3, 0.5, 0.7]:
        hybrid_result = evaluate_hybrid(train_df, test_df, movies_df, user_map, item_map, alpha=alpha, k_cf=20)
        if hybrid_result:
            results.append(hybrid_result)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(STATS_DIR / "model_comparison.csv", index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MAE comparison
    axes[0].bar(range(len(results_df)), results_df["mae"], color="skyblue", edgecolor="black")
    axes[0].set_xticks(range(len(results_df)))
    axes[0].set_xticklabels(results_df["model"], rotation=45, ha="right")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Mean Absolute Error")
    axes[0].grid(axis="y", alpha=0.3)
    
    # RMSE comparison
    axes[1].bar(range(len(results_df)), results_df["rmse"], color="salmon", edgecolor="black")
    axes[1].set_xticks(range(len(results_df)))
    axes[1].set_xticklabels(results_df["model"], rotation=45, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Root Mean Squared Error")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved: {FIGURES_DIR / 'model_comparison.png'}")
    
    return results_df

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    train_df, test_df, movies_df, user_map, item_map = load_all_data()
    
    # Full comparison
    results_df = compare_models(train_df, test_df, movies_df, user_map, item_map)
    
    # Cold-start analysis
    cold_start_results = cold_start_analysis(train_df, test_df, movies_df, user_map, item_map, threshold=5)
    
    if cold_start_results is not None and len(cold_start_results) > 0:
        cold_start_results.to_csv(STATS_DIR / "cold_start_analysis.csv", index=False)
        print("\n" + "=" * 60)
        print("COLD-START ANALYSIS RESULTS")
        print("=" * 60)
        print(cold_start_results.to_string(index=False))
        print(f"\nCold-start results saved: {STATS_DIR / 'cold_start_analysis.csv'}")
    
    print("\n✅ Evaluation complete!")
