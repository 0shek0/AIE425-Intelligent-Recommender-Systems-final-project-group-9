from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    CFG,
    ensure_dirs,
    load_ratings,
    basic_filter,
    choose_target_items,
    build_small_user_item_matrix,
    mean_fill_matrix,
    train_test_holdout_for_item,
    mae_rmse,
    Profiler,
)


# -------------------------------------------------
# SVD FROM SCRATCH (using eigen-decomposition)
# -------------------------------------------------
def svd_from_scratch(A: np.ndarray, eps: float = 1e-10):
    """
    Compute SVD using eigen-decomposition of A^T A:
    A = U Σ V^T
    """
    m, n = A.shape

    # Gram matrix
    G = A.T @ A  # (n x n)

    # Eigen-decomposition
    eigvals, V = np.linalg.eigh(G)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Singular values
    sing_vals = np.sqrt(np.clip(eigvals, 0, None))

    # Compute U
    U = np.zeros((m, n))
    for i in range(n):
        if sing_vals[i] > eps:
            U[:, i] = (A @ V[:, i]) / sing_vals[i]

    # Orthonormalize U
    U, _ = np.linalg.qr(U)

    # Sigma matrix
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = sing_vals[i]

    return U, Sigma, V.T, sing_vals


def truncated_reconstruct(U, Sigma, Vt, k: int):
    k = min(k, U.shape[1], Vt.shape[0])
    return U[:, :k] @ Sigma[:k, :k] @ Vt[:k, :]


def reconstruction_metrics(A_true, A_hat):
    return mae_rmse(A_true.flatten(), A_hat.flatten())


def predict_target(mat: pd.DataFrame, R_hat: np.ndarray, item_id: int, gt: pd.Series):
    col_idx = list(mat.columns).index(item_id)
    user_to_row = {u: i for i, u in enumerate(mat.index)}

    y_true, y_pred = [], []
    for user, rating in gt.items():
        y_true.append(rating)
        y_pred.append(R_hat[user_to_row[user], col_idx])

    return mae_rmse(np.array(y_true), np.array(y_pred))


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    ensure_dirs()

    # Load & preprocess
    df = basic_filter(load_ratings())
    I1, I2, pop = choose_target_items(df)

    # Force include target items
    mat = build_small_user_item_matrix(df, pop, forced_items=[I1, I2])

    # Mean-fill
    mat_filled, _ = mean_fill_matrix(mat)
    A = mat_filled.values.astype(float)

    # -------------------------
    # FULL SVD
    # -------------------------
    with Profiler() as prof:
        U, Sigma, Vt, svals = svd_from_scratch(A)

    pd.DataFrame({"singular_value": svals}).to_csv(
        CFG.TABLES_DIR / "svd_singular_values.csv", index=False
    )

    # -------------------------
    # Prepare target holdouts (safe)
    # -------------------------
    gt_I1 = None
    gt_I2 = None

    if I1 in mat.columns and mat[I1].count() >= 5:
        _, gt_I1 = train_test_holdout_for_item(
            mat, I1, CFG.TEST_RATIO, CFG.SEED + 111
        )

    if I2 in mat.columns and mat[I2].count() >= 5:
        _, gt_I2 = train_test_holdout_for_item(
            mat, I2, CFG.TEST_RATIO, CFG.SEED + 222
        )

    # -------------------------
    # TRUNCATED SVD EXPERIMENTS
    # -------------------------
    ks = [5, 20, 50, 100]
    ks = [k for k in ks if k <= min(A.shape)]

    results = []

    for k in ks:
        R_hat = truncated_reconstruct(U, Sigma, Vt, k)

        rec_mae, rec_rmse = reconstruction_metrics(A, R_hat)

        mae_I1, rmse_I1 = (None, None)
        mae_I2, rmse_I2 = (None, None)

        if gt_I1 is not None:
            mae_I1, rmse_I1 = predict_target(mat, R_hat, I1, gt_I1)

        if gt_I2 is not None:
            mae_I2, rmse_I2 = predict_target(mat, R_hat, I2, gt_I2)

        results.append({
            "method": "SVD_Truncated",
            "k": k,
            "reconstruction_MAE": rec_mae,
            "reconstruction_RMSE": rec_rmse,
            "target_I1_MAE": mae_I1,
            "target_I1_RMSE": rmse_I1,
            "target_I2_MAE": mae_I2,
            "target_I2_RMSE": rmse_I2,
            "svd_runtime_s": prof.runtime_s,
            "svd_peak_mem_mb": prof.mem_peak_mb,
        })

    out = pd.DataFrame(results)
    out.to_csv(CFG.TABLES_DIR / "svd_metrics.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
