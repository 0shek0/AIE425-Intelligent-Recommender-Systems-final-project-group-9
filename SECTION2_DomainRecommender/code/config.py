from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
STATS_DIR = REPORTS_DIR / "stats"

# Your uploaded zip (change if you moved it)
DATASET_ZIP_PATH = PROJECT_ROOT / "hetrec2011-lastfm-2k.zip"

# The extracted folder name we expect under data/raw
EXTRACTED_DIR_NAME = "hetrec2011-lastfm-2k"

# Files (HetRec LastFM)
USER_ARTISTS_FILE = "user_artists.dat"  # userID \t artistID \t weight(playcount)
ARTISTS_FILE = "artists.dat"

# Output files
INTERACTIONS_CSV = DATA_PROCESSED / "interactions.csv"
STATS_JSON = STATS_DIR / "part1_stats.json"
