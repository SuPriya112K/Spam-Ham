"""
train.py

End-to-end training script for the spam classifier.

Run this file directly to:
1. Load and clean the dataset
2. Split into train/test
3. Build a Pipeline (TF-IDF + Logistic Regression)
4. Train and evaluate
5. Log the run to MLflow
6. Save the final pipeline to disk for the API to use

Usage:
    python train.py
"""

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from preprocessing import clean_text_series

# ---- Config ----
DATA_PATH = "../data/processed_spam_data.csv"
MODEL_OUTPUT_PATH = "../models/spam_classifier_pipeline.pkl"
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_clean_data(path: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(path)

    print("Cleaning text...")
    df["clean_text"] = clean_text_series(df["text"])

    # Drop rows that became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]

    return df


def main():
    os.makedirs("../models", exist_ok=True)

    df = load_and_clean_data(DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Build the combined pipeline: TF-IDF vectorizer -> Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Start an MLflow run - everything inside this block gets logged together
    with mlflow.start_run():

        # Log the hyperparameters we chose
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("ngram_range", str(NGRAM_RANGE))
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", TEST_SIZE)

        print("Training pipeline...")
        pipeline.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Spam"),
            "recall": recall_score(y_test, y_pred, pos_label="Spam"),
            "f1": f1_score(y_test, y_pred, pos_label="Spam"),
        }

        # Log metrics to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log the model itself as an MLflow artifact
        mlflow.sklearn.log_model(pipeline, "model")

        print("\n=== Results ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))

        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))

    # Save the pipeline separately too, as a plain .pkl file
    # This is what our FastAPI app will load directly (simpler than pulling from MLflow at serve time)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nPipeline saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()