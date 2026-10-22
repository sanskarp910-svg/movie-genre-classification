"""
Model training, evaluation metrics computation, confusion matrix generation,
and artifact serialization for Movie Genre Classification.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Ensure src directory is importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from preprocess import load_and_preprocess_data


def train_model(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
    top_n_genres: int = 10
) -> Dict[str, Any]:
    """
    Train TF-IDF + MultinomialNB classifier, compute evaluation metrics,
    plot and save confusion matrix, and serialize all artifacts.
    """
    base_dir = CURRENT_DIR.parent
    if data_dir is None:
        data_dir = str(base_dir / "data")
    if models_dir is None:
        models_dir = str(base_dir / "models")

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING MOVIE GENRE CLASSIFICATION PIPELINE")
    print("=" * 60)

    # 1. Load and preprocess data
    df = load_and_preprocess_data(data_dir=data_dir, top_n_genres=top_n_genres)

    # Use python lists or standard numpy arrays to ensure full compatibility with pandas 3.x
    X = np.array(df["cleaned_text"].tolist(), dtype=object)
    y = np.array(df["primary_genre"].tolist(), dtype=object)

    genre_counts = pd.Series(y).value_counts().to_dict()
    print(f"\n[Train] Extracted {len(y)} movies across {len(genre_counts)} genres:")
    for genre, count in genre_counts.items():
        print(f"  - {genre}: {count} movies")

    # 2. Train-test split (stratified if sample size allows)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception as e:
        print(f"[Train] Stratified split warning ({e}), falling back to standard split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    print(f"\n[Train] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. TF-IDF Vectorization (Unigrams + Bigrams, max_features=5000)
    print(f"\n[Train] Fitting TfidfVectorizer (ngram_range=(1, 2), max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[Train] TF-IDF Vocabulary size: {vocab_size} tokens")

    # 4. Multinomial Naive Bayes Classifier
    print("\n[Train] Training MultinomialNB classifier...")
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train_vec, y_train)

    # 5. Model Evaluation
    y_pred = classifier.predict(X_test_vec)
    accuracy = float(accuracy_score(y_test, y_pred))
    unique_labels = sorted(list(classifier.classes_))

    report_dict = classification_report(
        y_test, y_pred, labels=unique_labels, zero_division=0, output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, labels=unique_labels, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS (Overall Accuracy: {accuracy:.4f})")
    print("=" * 60)
    print(report_text)

    # 6. Generate and save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_path = os.path.join(models_dir, "confusion_matrix.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar=True,
        square=True
    )
    plt.title("Movie Genre Classification - Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Genre", fontsize=12)
    plt.ylabel("Actual Genre", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[Train] Confusion matrix plot saved to: {cm_path}")

    # Also save a copy in project root for convenience
    root_cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar=True,
        square=True
    )
    plt.title("Movie Genre Classification - Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Genre", fontsize=12)
    plt.ylabel("Actual Genre", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(root_cm_path, dpi=300)
    plt.close()

    # 7. Precompute TF-IDF matrix for the entire dataset for fast recommendation queries
    print("\n[Train] Computing full dataset TF-IDF vectors for recommendation engine...")
    full_tfidf_matrix = vectorizer.transform(df["cleaned_text"].tolist())

    processed_corpus = df[[
        "id_clean", "title", "primary_genre", "overview", "keywords_text", "cleaned_text"
    ]].copy()
    processed_corpus["tfidf_index"] = np.arange(len(df))

    # 8. Serialize artifacts
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    classifier_path = os.path.join(models_dir, "genre_classifier.pkl")
    corpus_path = os.path.join(models_dir, "processed_movies.pkl")
    tfidf_matrix_path = os.path.join(models_dir, "tfidf_matrix.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    print("[Train] Serializing model artifacts to disk...")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)
    joblib.dump(processed_corpus, corpus_path)
    joblib.dump(full_tfidf_matrix, tfidf_matrix_path)

    metrics_payload = {
        "accuracy": accuracy,
        "classes": unique_labels,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "total_movies": len(df),
        "vocab_size": vocab_size,
        "genre_distribution": genre_counts,
        "test_size": len(X_test),
        "train_size": len(X_train)
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"  - Vectorizer: {vectorizer_path}")
    print(f"  - Classifier: {classifier_path}")
    print(f"  - Corpus Cache: {corpus_path}")
    print(f"  - Metrics: {metrics_path}")
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return metrics_payload


if __name__ == "__main__":
    train_model()
