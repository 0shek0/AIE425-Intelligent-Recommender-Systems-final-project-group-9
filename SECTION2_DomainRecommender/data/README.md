# Dataset Information

## MovieLens 20M Dataset

**Source:** [GroupLens Research - MovieLens 20M](https://grouplens.org/datasets/movielens/20m/)

### Original Dataset

The MovieLens 20M dataset contains:
- **20,000,263 ratings** 
- **465,564 tag applications** 
- **27,278 movies**
- **138,493 users**
- **Rating scale:** 0.5 to 5.0 (half-star increments)
- **Time period:** Ratings from January 1995 to March 2015

### Dataset Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

---

## Processed Subset (This Project)

For this project, we created a **stratified random sample** from the MovieLens 20M dataset to meet AIE425 course requirements:

### Dataset Statistics

- **Users:** 5,347 (stratified across activity levels)
- **Movies:** 7,454
- **Interactions:** 66,000 ratings
- **Rating scale:** 1.0 to 5.0 (normalized from original 0.5-5.0 scale)
- **Sparsity:** 99.83%
- **Cold-start users:** 2,688 (50.3% with ≤5 ratings)

### Sampling Strategy

**Stratified Random Sampling** across user activity levels:
- **15% low-activity users** (≤5 ratings) → ensures cold-start presence
- **40% medium-activity users** (6-50 ratings) → CF stability
- **45% high-activity users** (>50 ratings) → robust recommendations

This approach:
- ✅ Preserves natural cold-start user distribution
- ✅ Maintains long-tail item popularity patterns
- ✅ Provides diverse user activity levels for realistic evaluation
- ✅ Meets rubric requirements (≥5,000 users, ≥500 items, ≥50,000 interactions)

### Rating Normalization

Original MovieLens ratings (0.5-5.0) were linearly transformed to (1.0-5.0):

```
rating_new = 1.0 + (rating_old - 0.5) × (4.0 / 4.5)
```

This preserves relative rating differences while meeting project requirements.

---

## Directory Structure

```
data/
├── README.md                    # This file
├── raw/                        # Original MovieLens files (symlinks)
│   ├── ratings.csv             # → ../../ml-20m/ratings.csv
│   └── movies.csv              # → ../../ml-20m/movies.csv
└── processed/                  # Processed subset for project
    ├── interactions.csv        # Canonical ratings file (66k rows)
    ├── interactions_train.csv  # Training set (80% per-user split)
    ├── interactions_test.csv   # Test set (20% per-user split)
    ├── user_map.json           # user_id → matrix index mapping
    └── item_map.json           # item_id → matrix index mapping
```

---

## Files Description

### Raw Data Files

**ratings.csv** (from MovieLens 20M)
- Columns: `userId`, `movieId`, `rating`, `timestamp`
- 20M rows, 533MB

**movies.csv** (from MovieLens 20M)
- Columns: `movieId`, `title`, `genres`
- 27,278 movies
- Genres: pipe-separated (e.g., "Action|Adventure|Sci-Fi")

### Processed Files

**interactions.csv**
- Canonical dataset for this project
- Columns: `user_id`, `item_id`, `rating`
- 66,000 rows
- Ratings normalized to 1.0-5.0 scale

**interactions_train.csv** (53,189 ratings)
- 80% of each user's ratings
- Used for model training (content-based, CF, hybrid)

**interactions_test.csv** (12,811 ratings)
- 20% of each user's ratings
- Used for evaluation (MAE, RMSE, coverage metrics)

**user_map.json** / **item_map.json**
- Mapping dictionaries for sparse matrix construction
- Format: `{original_id: matrix_index}`
- Used by collaborative filtering model

---

## Usage

### Load Dataset

```python
import pandas as pd

# Load processed interactions
interactions = pd.read_csv('data/processed/interactions.csv')

# Load train/test splits
train = pd.read_csv('data/processed/interactions_train.csv')
test = pd.read_csv('data/processed/interactions_test.csv')

# Load movies metadata
movies = pd.read_csv('data/raw/movies.csv')
```

### Dataset Statistics

See `results/stats/dataset_stats.json` for comprehensive statistics including:
- User/item/interaction counts
- Rating distribution
- Density/sparsity metrics
- Cold-start user counts
- Low-frequency item counts

---

## License & Terms of Use

The MovieLens 20M dataset is provided by GroupLens Research and is intended for **education and research purposes only**.

**Citation Required:** If you publish work using this dataset, please cite the GroupLens paper mentioned above.

**Neither the University of Minnesota nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set.**

For full license and terms of use, see: https://grouplens.org/datasets/movielens/
