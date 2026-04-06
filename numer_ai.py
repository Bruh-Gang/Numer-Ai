import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
import numerapi

# ── API setup ────────────────────────────────────────────────────────────────
# Store your credentials in environment variables or a .env file.
# Never commit your public_id/secret_key directly to version control.
napi = numerapi.NumerAPI(
    public_id=os.environ.get("NUMERAI_PUBLIC_ID", ""),
    secret_key=os.environ.get("NUMERAI_SECRET_KEY", ""),
)

# Your model ID from the Numerai dashboard
MODEL_ID = os.environ.get("NUMERAI_MODEL_ID", "")

# ── File paths ────────────────────────────────────────────────────────────────
# These default to ~/Downloads but you can override with env vars
BASE_DIR = os.environ.get("NUMERAI_DATA_DIR", os.path.expanduser("~/Downloads"))

features_path = os.path.join(BASE_DIR, "features.json")
train_path    = os.path.join(BASE_DIR, "train.parquet")

# ── 1. Load features ──────────────────────────────────────────────────────────
with open(features_path, "r") as f:
    features_metadata = json.load(f)

# Using the small feature set for a quick first run — swap "small" → "medium"
# or "all" for more signal (at the cost of memory + training time)
small_features    = features_metadata["feature_sets"]["small"]
selected_features = small_features[:20]
print(f"Using {len(selected_features)} features: {selected_features}")

# ── 2. Load training data ─────────────────────────────────────────────────────
train_data = pd.read_parquet(train_path, columns=selected_features + ["era", "target"])

# I'm only using eras 1-10 here to keep it fast for development.
# For a real submission you'd want the full history.
train_data = train_data[train_data["era"].isin([f"{i:04d}" for i in range(1, 11)])]

train_data[selected_features] = train_data[selected_features].astype(np.float32)
train_data["target"]           = train_data["target"].astype(np.float32)

print(f"Train shape: {train_data.shape}")

# ── 3. Train LightGBM ─────────────────────────────────────────────────────────
# These params are a decent starting point — tune num_leaves / learning_rate
# once you have a baseline CORR you're happy with.
params = {
    "objective":        "regression",
    "metric":           "mse",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "max_depth":        5,
}

X_train      = train_data[selected_features]
y_train      = train_data["target"]
train_ds     = lgb.Dataset(X_train, label=y_train)
model        = lgb.train(params, train_ds, num_boost_round=100)
print("Model trained.")

# ── 4. Load live round data ───────────────────────────────────────────────────
current_round = napi.get_current_round()
live_path     = os.path.join(BASE_DIR, f"live_{current_round}.parquet")
live_data     = pd.read_parquet(live_path)

# Make sure 'id' is a proper column (sometimes it lands in the index)
if "id" not in live_data.columns:
    if live_data.index.name:
        live_data = live_data.reset_index()
        print("Reset index — columns now:", list(live_data.columns))
    else:
        raise ValueError("No 'id' column found in live data — check the parquet schema.")

X_live      = live_data[selected_features].astype(np.float32)
predictions = model.predict(X_live)

# Rank-normalize so predictions are uniformly distributed in [0, 1]
# — Numerai's scoring rewards this distribution
normalized = rankdata(predictions, method="average") / len(predictions)

# ── 5. Build and save submission CSV ─────────────────────────────────────────
submission      = pd.DataFrame({"id": live_data["id"], "prediction": normalized})
submission_file = os.path.join(BASE_DIR, f"submission_round_{current_round}.csv")
submission.to_csv(submission_file, index=False)
print(f"Saved submission to {submission_file}")

# ── 6. Upload ─────────────────────────────────────────────────────────────────
napi.upload_predictions(submission_file, model_id=MODEL_ID)
print("Upload complete.")
