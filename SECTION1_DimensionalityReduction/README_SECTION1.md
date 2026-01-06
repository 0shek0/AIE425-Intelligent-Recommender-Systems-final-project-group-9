# SECTION 1 — Dimensionality Reduction & Matrix Factorization (Parts 1, 2, 3)

## Dataset
Place MovieLens 20M files here:
SECTION1_DimensionalityReduction/data/ml-20m/ratings.csv
SECTION1_DimensionalityReduction/data/ml-20m/movies.csv

## Run (recommended)
cd SECTION1_DimensionalityReduction/code
python run_all.py

## Outputs
results/tables/
- dataset_stats.csv
- item_popularity.csv
- pca_meanfill_metrics.csv
- pca_mle_metrics.csv
- cov_mle.csv
- svd_singular_values.csv
- svd_metrics.csv
- peer lists for each target item (top-5/top-10)

## Notes
We build a manageable user-item matrix via MAX_USERS and MAX_ITEMS in utils.py.
Increase them if your machine can handle more.
