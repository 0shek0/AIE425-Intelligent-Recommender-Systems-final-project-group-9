import pandas as pd
from utils import CFG, ensure_dirs

def main():
    ensure_dirs()

    # Load results
    pca_mean = pd.read_csv(CFG.TABLES_DIR / "pca_meanfill_metrics.csv")
    pca_mle  = pd.read_csv(CFG.TABLES_DIR / "pca_mle_metrics.csv")
    svd      = pd.read_csv(CFG.TABLES_DIR / "svd_metrics.csv")

    # ---- PCA Mean ----
    pca_mean["Method"] = "PCA Mean-Filling"
    pca_mean["Config"] = pca_mean["num_peers"].apply(lambda x: f"{x} peers")
    pca_mean = pca_mean[["Method", "Config", "MAE", "RMSE", "cov_runtime_s", "cov_peak_mem_mb"]]

    # ---- PCA MLE ----
    pca_mle["Method"] = "PCA with MLE"
    pca_mle["Config"] = pca_mle["num_peers"].apply(lambda x: f"{x} peers")
    pca_mle = pca_mle[["Method", "Config", "MAE", "RMSE", "cov_runtime_s", "cov_peak_mem_mb"]]

    # ---- SVD ----
    svd["Method"] = "Truncated SVD"
    svd["Config"] = svd["k"].apply(lambda x: f"k={x}")
    svd = svd.rename(columns={
        "reconstruction_MAE": "MAE",
        "reconstruction_RMSE": "RMSE",
        "svd_runtime_s": "cov_runtime_s",
        "svd_peak_mem_mb": "cov_peak_mem_mb"
    })
    svd = svd[["Method", "Config", "MAE", "RMSE", "cov_runtime_s", "cov_peak_mem_mb"]]

    # Combine
    comparison = pd.concat([pca_mean, pca_mle, svd], ignore_index=True)

    # Save
    out_path = CFG.TABLES_DIR / "comparison_table_section1.csv"
    comparison.to_csv(out_path, index=False)

    print("Saved comparison table to:", out_path)
    print(comparison)

if __name__ == "__main__":
    main()
