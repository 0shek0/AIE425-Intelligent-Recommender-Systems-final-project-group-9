# Intelligent Recommender Systems — Course Project

Quick start:

1. Create and activate a Python 3.10+ virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Preprocess data (creates `data/processed/interactions.csv`, stats, and figures)

```bash
python SECTION2_DomainRecommender/code/data_preprocessing.py
```

3. Run content-based recommendations (example run)

```bash
python SECTION2_DomainRecommender/code/content_based.py
```

Notes and recommendations:
- Dependencies are pinned in `requirements.txt` for reproducibility.
- Prefer running scripts from the project root so relative paths resolve correctly.
- I recommend running `black` and `flake8` for style and adding unit tests via `pytest`.

If you want, I can next refactor `content_based.py` to avoid dense TF-IDF arrays and centralize configuration.