# Numer-Ai

My submission pipeline for the [Numerai](https://numer.ai) live equity prediction tournament. Train a LightGBM model on historical Numerai data, generate rank-normalized predictions for the current live round, and auto-upload — all in one script.

I've been running this against live rounds to get a feel for how ensemble gradient boosting handles the noisy, encrypted feature space Numerai throws at you.

---

## What it does

1. Loads feature metadata and picks the first 20 features from the `small` set (swap in `medium` or `all` for more signal)
2. Trains a LightGBM regression model on eras 1–10
3. Pulls the current live round via the Numerai API
4. Rank-normalizes predictions to `[0, 1]` — Numerai's scoring rewards this distribution
5. Saves a submission CSV and uploads it automatically

---

## Setup

```bash
pip install lightgbm numerapi pandas numpy scipy
```

Set your credentials as environment variables (don't hard-code them):

```bash
export NUMERAI_PUBLIC_ID="your_public_id"
export NUMERAI_SECRET_KEY="your_secret_key"
export NUMERAI_MODEL_ID="your_model_id"
export NUMERAI_DATA_DIR="~/Downloads"   # where your .parquet files live
```

Download your data files from the [Numerai data page](https://numer.ai/data) and drop them in `NUMERAI_DATA_DIR`.

---

## Run

```bash
python numer_ai.py
```

---

## Notes

- Feature set is intentionally small for speed — the full `all` set uses a lot of RAM
- Only eras 1–10 used for training here; add more for a more robust model
- LightGBM params are a reasonable baseline, not heavily tuned yet — that's on the roadmap
- For actual competition performance I'd add era-boosting, feature neutralization, and a proper CV split

---

## Stack

`Python` · `LightGBM` · `pandas` · `numpy` · `numerapi` · `scipy`
