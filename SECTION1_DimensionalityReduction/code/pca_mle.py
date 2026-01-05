from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    CFG, ensure_dirs, load_ratings, basic_filter, choose_target_items,
    build_small_user_item_matrix, mean_fill_matrix,
    train_test_holdout_for_item, mae_rmse, Profiler
)


def pairwise_mle_cov(mat: pd.DataFrame) -> pd.DataFrame:
    """
    MLE covariance: cov(i,j) computed using only users who rated both i and j.
    If overlap < 2 => 0.
    """
    cols = list(mat.columns)
    cov = pd.DataFrame(0.0, index=cols, columns=cols)

    series = {c: mat[c] for c in cols}
    for a_i, i in enumerate(cols):
        xi = series[i]
        for j in cols[a_i:]:
            xj = series[j]
            common = xi.notna() & xj.notna()
            if common.sum() < 2:
                val = 0.0
            else:
                u = xi[common].astype(float).values
                v = xj[common].astype(float).values
                val = float(np.cov(u, v, bias=False)[0, 1])
            cov.loc[i, j] = val
            cov.loc[j, i] = val
    return cov


def top_k_peers(cov: pd.DataFrame, target: int, k: int) -> list[int]:
    s = cov[target].drop(index=target, errors="ignore").sort_values(ascending=False)
    return [int(x) for x in s.head(k).index]


def pca_reconstruct(X: np.ndarray, n_components: int) -> np.ndarray:
    C = np.cov(X, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    n_components = max(1, min(n_components, eigvecs.shape[1]))

    W = eigvecs[:, :n_components]
    Z = X @ W
    return Z @ W.T


def predict_target(mat: pd.DataFrame, target: int, peers: list[int], n_components: int):
    cols = [target] + peers
    sub = mat[cols].copy()

    masked, gt = train_test_holdout_for_item(sub, target, CFG.TEST_RATIO, CFG.SEED + 999 + target + len(peers))
    filled, _ = mean_fill_matrix(masked)

    X = filled.values
    X_centered = X - X.mean(axis=0, keepdims=True)
    X_hat = pca_reconstruct(X_centered, n_components) + X.mean(axis=0, keepdims=True)

    pred_series = pd.Series(X_hat[:, 0], index=filled.index)
    y_true = gt.values
    y_pred = pred_series.loc[gt.index].values
    return mae_rmse(y_true, y_pred)


def main():
    ensure_dirs()

    df = basic_filter(load_ratings())
    I1, I2, pop = choose_target_items(df)

    mat = build_small_user_item_matrix(df, pop)

    with Profiler() as p:
        cov = pairwise_mle_cov(mat)

    cov.to_csv(CFG.TABLES_DIR / "cov_mle.csv")

    results = []
    for target in [I1, I2]:
        if target not in cov.columns:
            continue
        for k in [5, 10]:
            peers = top_k_peers(cov, target, k)
            n_components = min(3, len(peers))
            mae, rmse = predict_target(mat, target, peers, n_components)
            results.append({
                "method": "PCA_MLE",
                "target_item": target,
                "num_peers": k,
                "n_components": n_components,
                "MAE": mae,
                "RMSE": rmse,
                "cov_runtime_s": getattr(p, "runtime_s", None),
                "cov_peak_mem_mb": getattr(p, "mem_peak_mb", None),
            })

            pd.DataFrame({"target_item": [target]*len(peers), "peer_item": peers}).to_csv(
                CFG.TABLES_DIR / f"peers_mle_target{target}_k{k}.csv", index=False
            )

    out = pd.DataFrame(results)
    out.to_csv(CFG.TABLES_DIR / "pca_mle_metrics.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
