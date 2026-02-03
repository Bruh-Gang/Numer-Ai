# Numer-Ai

Numerai LightGBM Automated Submission Pipeline

This repository contains a complete machine learning pipeline for the Numerai tournament. The script trains a LightGBM regression model on historical Numerai data, generates predictions for the current live round, rank-normalizes those predictions, and automatically uploads the submission using the Numerai API.

Overview

The pipeline performs the following steps:

Loads Numerai feature metadata

Selects a subset of features for model training

Trains a LightGBM regression model on historical data

Loads live data for the current Numerai round

Generates predictions using the trained model

Rank-normalizes predictions to comply with Numerai requirements

Saves predictions to a submission CSV file

Uploads the submission directly to Numerai

This approach enables fully automated training and submission with minimal manual intervention.

Technologies Used

Python 3.x

pandas

numpy

LightGBM

numerapi

scipy

Required Files

The following files must be available locally before running the script:

features.json
Contains Numerai feature metadata and feature sets.

train.parquet
Historical Numerai training data.

live_<round>.parquet
Live data for the current Numerai tournament round.

Update file paths in the script as needed to match your local environment.

Pipeline Details
1. Numerai API Configuration

The script authenticates with Numerai using a public ID and secret key via numerapi. This enables programmatic access to the current tournament round and allows automated uploading of predictions.

2. Feature Selection

Feature metadata is loaded from features.json.

The "small" feature set is used.

The first 20 features from this set are selected for training.

This limited feature set is intended for simplicity and faster experimentation.

3. Training Data Preparation

Historical data is loaded from train.parquet.

Only eras 0001 through 0010 are used.

Feature columns and target values are cast to float32 for memory efficiency.

The dataset is then split into:

Features (X)

Target (y)

4. Model Training

A LightGBM regression model is trained with the following characteristics:

Gradient Boosted Decision Trees (GBDT)

Controlled tree depth to reduce overfitting

Feature and bagging fractions for regularization

The model is trained for 100 boosting rounds.

5. Live Data Loading and Prediction

The current Numerai round is retrieved via the API.

The corresponding live data file is loaded.

The script ensures that an id column exists (resetting the index if necessary).

Predictions are generated using the trained model.

6. Rank Normalization

Numerai requires submissions to be rank-normalized rather than raw predictions.
Predictions are therefore ranked and scaled to the interval [0, 1].

7. Submission File Creation

A CSV file is generated containing:

id

prediction

The file is saved as:

numerai_submission_<round>.csv

8. Automated Submission

The script uploads the submission file directly to Numerai using the provided model ID and the Numerai API.
