"""
train.py — Fake Job Posting Detection: Training Script
=======================================================
Run this ONCE to train and save your model + vectorizer.

Usage:
    python train.py --data data/fake_job_postings.csv

Dataset: EMSCAD (Employment Scam Aegean Dataset)
Download: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
TFIDF_FEATURES  = 5000
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

# Text columns to combine into one feature string
TEXT_COLUMNS = [
    "title", "company_profile", "description",
    "requirements", "benefits"
]


# ─── Helpers ───────────────────────────────────────────────────────────────────
def combine_text(row):
    """Merge multiple text columns into a single string for TF-IDF."""
    parts = []
    for col in TEXT_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def clean_text(text: str) -> str:
    """Basic text normalization."""
    import re
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


# ─── Main ──────────────────────────────────────────────────────────────────────
def train(data_path: str):
    print(f"\n{'='*60}")
    print("  Fake Job Detection — Model Training")
    print(f"{'='*60}\n")

    # 1. Load data
    print(f"[1/6] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print(f"      Fraud distribution:\n{df['fraudulent'].value_counts()}\n")

    # 2. Build combined text feature
    print("[2/6] Combining & cleaning text columns...")
    df["combined_text"] = df.apply(combine_text, axis=1).apply(clean_text)
    df = df[df["combined_text"].str.len() > 10].reset_index(drop=True)
    print(f"      Usable rows after filter: {len(df):,}\n")

    # 3. Train / test split
    print("[3/6] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"],
        df["fraudulent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["fraudulent"],
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 4. TF-IDF vectorization
    print(f"[4/6] Fitting TF-IDF vectorizer (max_features={TFIDF_FEATURES})...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # log scaling
        min_df=2,                 # ignore very rare terms
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}\n")

    # 5. Train Logistic Regression
    print("[5/6] Training Logistic Regression (class_weight='balanced')...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_vec, y_train)

    # 6. Evaluation
    print("\n[6/6] Evaluation on held-out test set:")
    y_pred  = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    print("\n" + classification_report(y_test, y_pred,
          target_names=["Genuine", "Fraudulent"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  ROC-AUC Score:     {roc_auc_score(y_test, y_proba):.4f}\n")

    # Top suspicious words
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    top_fraud_idx  = np.argsort(coef)[-15:][::-1]
    top_legit_idx  = np.argsort(coef)[:15]
    print("  Top 15 fraud-predictive words:")
    print("  " + " | ".join(feature_names[i] for i in top_fraud_idx))
    print("\n  Top 15 genuine-predictive words:")
    print("  " + " | ".join(feature_names[i] for i in top_legit_idx))

    # Save artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n✅ Model saved to      → {MODEL_PATH}")
    print(f"✅ Vectorizer saved to → {VECTORIZER_PATH}")
    print(f"\n{'='*60}")
    print("  Training complete! Run `python app.py` to start the server.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/fake_job_postings.csv",
        help="Path to the EMSCAD CSV file",
    )
    args = parser.parse_args()
    train(args.data)
