import pandas as pd
import matplotlib.pyplot as plt
from utils import CFG, ensure_dirs

def main():
    ensure_dirs()

    # Load CSVs
    svd_vals = pd.read_csv(CFG.TABLES_DIR / "svd_singular_values.csv")
    svd_met  = pd.read_csv(CFG.TABLES_DIR / "svd_metrics.csv")
    pca_mean = pd.read_csv(CFG.TABLES_DIR / "pca_meanfill_metrics.csv")
    pca_mle  = pd.read_csv(CFG.TABLES_DIR / "pca_mle_metrics.csv")

    # --------------------------------------------------
    # 1. Scree plot (singular values)
    # --------------------------------------------------
    plt.figure()
    plt.plot(range(1, len(svd_vals)+1), svd_vals["singular_value"], marker="o")
    plt.xlabel("Component Index")
    plt.ylabel("Singular Value")
    plt.title("SVD Scree Plot")
    plt.grid(True)
    plt.savefig(CFG.PLOTS_DIR / "svd_scree_plot.png")
    plt.close()

    # --------------------------------------------------
    # 2. Reconstruction error vs k (Elbow plot)
    # --------------------------------------------------
    plt.figure()
    plt.plot(svd_met["k"], svd_met["reconstruction_RMSE"], marker="o")
    plt.xlabel("Number of Latent Factors (k)")
    plt.ylabel("Reconstruction RMSE")
    plt.title("SVD Reconstruction Error vs k")
    plt.grid(True)
    plt.savefig(CFG.PLOTS_DIR / "svd_elbow_plot.png")
    plt.close()

    # --------------------------------------------------
    # 3. PCA RMSE vs number of peers
    # --------------------------------------------------
    plt.figure()
    plt.plot(pca_mean["num_peers"], pca_mean["RMSE"], marker="o")
    plt.plot(pca_mle["num_peers"], pca_mle["RMSE"], marker="o")
    plt.xlabel("Number of Peers")
    plt.ylabel("RMSE")
    plt.title("PCA Prediction Error vs Number of Peers")
    plt.legend(["PCA Mean-Filling", "PCA with MLE"])
    plt.grid(True)
    plt.savefig(CFG.PLOTS_DIR / "pca_rmse_vs_peers.png")
    plt.close()

    print("All plots saved to:", CFG.PLOTS_DIR)

if __name__ == "__main__":
    main()
