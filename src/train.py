"""
Model training, 5-fold cross-validation comparison, multi-label evaluation,
confusion/co-occurrence matrix generation, and artifact serialization for CineVerse AI.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from preprocess import load_dataset, TOP_12_GENRES, clean_and_lemmatize_text


def train_and_evaluate(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    max_features: int = 6000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train production multi-label classifier, run 5-fold CV model benchmark,
    compute diagnostics, and serialize both full artifacts and lightweight serverless bundle.
    """
    base_dir = CURRENT_DIR.parent
    if data_dir is None:
        data_dir = str(base_dir / "data")
    if models_dir is None:
        models_dir = str(base_dir / "models")

    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("STARTING CINEVERSE AI MULTI-LABEL TRAINING & BENCHMARK PIPELINE")
    print("=" * 65)

    # 1. Load preprocessed multi-label dataset
    df = load_dataset(data_dir=data_dir)
    print(f"\n[Train] Total training corpus: {len(df)} movies & anime")

    # 2. Multi-Label Binarization
    mlb = MultiLabelBinarizer(classes=TOP_12_GENRES)
    Y = mlb.fit_transform(df["genre_list"])

    genre_counts = {genre: int(Y[:, i].sum()) for i, genre in enumerate(TOP_12_GENRES)}
    print("\n[Train] Target Genre Frequencies:")
    for g, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {g:16s}: {count:5d} titles ({count/len(df)*100:4.1f}%)")

    # 3. TF-IDF Vectorization
    print(f"\n[Train] Fitting TfidfVectorizer (ngram_range=(1, 2), max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["cleaned_text"])
    vocab = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()
    print(f"[Train] Extracted TF-IDF Vocabulary: {len(vocab)} features")

    # 4. Train / Test Split (80 / 20)
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, np.arange(len(df)), test_size=0.2, random_state=random_state
    )
    print(f"[Train] Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # 5. Production Model Training: OneVsRest Logistic Regression
    print("\n[Train] Training Production OneVsRest Logistic Regression...")
    prod_model = OneVsRestClassifier(
        LogisticRegression(C=2.5, max_iter=600, class_weight="balanced", solver="liblinear", random_state=random_state)
    )
    t0 = time.time()
    prod_model.fit(X_train, Y_train)
    prod_fit_time = round(time.time() - t0, 3)

    # Predictions & Probabilities
    Y_pred = prod_model.predict(X_test)
    Y_probs = prod_model.predict_proba(X_test)

    # 6. Comprehensive Metrics Computation
    hamming_acc = float(1.0 - hamming_loss(Y_test, Y_pred))
    micro_f1 = float(f1_score(Y_test, Y_pred, average="micro"))
    macro_f1 = float(f1_score(Y_test, Y_pred, average="macro"))
    micro_prec = float(precision_score(Y_test, Y_pred, average="micro", zero_division=0))
    micro_rec = float(recall_score(Y_test, Y_pred, average="micro", zero_division=0))
    exact_acc = float(accuracy_score(Y_test, Y_pred))

    # Top-3 Metrics
    hits_in_top3 = 0
    total_true_in_top3 = 0
    total_true_count = 0

    for i in range(len(Y_test)):
        top3_indices = np.argsort(Y_probs[i])[::-1][:3]
        true_indices = np.where(Y_test[i] == 1)[0]
        overlap = len(set(top3_indices).intersection(set(true_indices)))
        if overlap > 0:
            hits_in_top3 += 1
        total_true_in_top3 += overlap
        total_true_count += len(true_indices)

    top3_hit_rate = float(hits_in_top3 / len(Y_test))
    top3_recall = float(total_true_in_top3 / total_true_count) if total_true_count > 0 else 0.0

    print("\n" + "=" * 65)
    print("PRODUCTION MULTI-LABEL MODEL PERFORMANCE:")
    print("=" * 65)
    print(f"  • Hamming Accuracy (Per-Label Match):  {hamming_acc * 100:.2f}%")
    print(f"  • Micro-Averaged F1 Score:             {micro_f1 * 100:.2f}%")
    print(f"  • Macro-Averaged F1 Score:             {macro_f1 * 100:.2f}%")
    print(f"  • Micro Precision / Recall:            {micro_prec * 100:.2f}% / {micro_rec * 100:.2f}%")
    print(f"  • Top-3 Any Genre Hit Rate:            {top3_hit_rate * 100:.2f}%")
    print(f"  • Top-3 Genre Recall:                  {top3_recall * 100:.2f}%")
    print(f"  • Exact Match Subset Accuracy:         {exact_acc * 100:.2f}%")

    # Per-genre classification report
    report_dict = classification_report(
        Y_test, Y_pred, target_names=TOP_12_GENRES, zero_division=0, output_dict=True
    )
    print("\nPer-Genre Classification Breakdown:")
    for g in TOP_12_GENRES:
        metrics = report_dict[g]
        print(f"  {g:16s} | Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | Support: {int(metrics['support'])}")

    # 7. 5-Fold Cross-Validation Model Benchmark
    print("\n[Benchmark] Executing 5-Fold Cross-Validation across 3 Architectures...")
    models_to_compare = {
        "Multinomial Naive Bayes": OneVsRestClassifier(MultinomialNB(alpha=0.4)),
        "Logistic Regression (Balanced)": OneVsRestClassifier(
            LogisticRegression(C=2.5, max_iter=600, class_weight="balanced", solver="liblinear", random_state=random_state)
        ),
        "LinearSVC (Balanced)": OneVsRestClassifier(
            LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=random_state)
        )
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    benchmark_results = {}

    for name, model_instance in models_to_compare.items():
        fold_hamming = []
        fold_micro_f1 = []
        fold_macro_f1 = []
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            model_instance.fit(X_tr, Y_tr)
            preds_val = model_instance.predict(X_val)

            fold_hamming.append(1.0 - hamming_loss(Y_val, preds_val))
            fold_micro_f1.append(f1_score(Y_val, preds_val, average="micro", zero_division=0))
            fold_macro_f1.append(f1_score(Y_val, preds_val, average="macro", zero_division=0))

        elapsed = round(time.time() - start_time, 2)
        benchmark_results[name] = {
            "mean_hamming_acc": round(float(np.mean(fold_hamming)), 4),
            "std_hamming_acc": round(float(np.std(fold_hamming)), 4),
            "mean_micro_f1": round(float(np.mean(fold_micro_f1)), 4),
            "std_micro_f1": round(float(np.std(fold_micro_f1)), 4),
            "mean_macro_f1": round(float(np.mean(fold_macro_f1)), 4),
            "std_macro_f1": round(float(np.std(fold_macro_f1)), 4),
            "total_cv_time_sec": elapsed
        }
        print(f"  • {name:30s} | Micro-F1: {np.mean(fold_micro_f1):.4f} ± {np.std(fold_micro_f1):.4f} | Macro-F1: {np.mean(fold_macro_f1):.4f} | Hamming: {np.mean(fold_hamming):.4f}")

    # 8. Top 10 Predictive Keywords per Genre (Feature Importance)
    print("\n[Train] Extracting Top 10 Predictive Keywords per Genre...")
    top_predictive_words = {}
    class_weights_list = []
    class_intercepts_list = []

    for i, genre in enumerate(TOP_12_GENRES):
        estimator = prod_model.estimators_[i]
        coefs = estimator.coef_[0]
        intercept = float(estimator.intercept_[0])
        class_weights_list.append([float(w) for w in coefs])
        class_intercepts_list.append(intercept)

        top_token_indices = np.argsort(coefs)[::-1][:10]
        top_words = [
            {"word": str(feature_names[idx]), "weight": round(float(coefs[idx]), 3)}
            for idx in top_token_indices
        ]
        top_predictive_words[genre] = top_words
        words_preview = ", ".join([f"{w['word']} (+{w['weight']})" for w in top_words[:5]])
        print(f"  - {genre:16s}: {words_preview}")

    # 9. Multi-Label Co-occurrence Matrix (Confusion & Correlation)
    print("\n[Train] Computing Genre Co-occurrence Matrix...")
    co_matrix = np.dot(Y.T, Y)
    genre_co_dict = {}
    for i, g1 in enumerate(TOP_12_GENRES):
        genre_co_dict[g1] = {g2: int(co_matrix[i, j]) for j, g2 in enumerate(TOP_12_GENRES)}

    # Plot Co-occurrence Matrix Heatmap
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    root_cm_path = os.path.join(base_dir, "confusion_matrix.png")

    plt.figure(figsize=(11, 9))
    norm_co = co_matrix.astype(float) / np.diag(co_matrix)[:, None]
    sns.heatmap(
        norm_co,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        xticklabels=TOP_12_GENRES,
        yticklabels=TOP_12_GENRES,
        cbar=True,
        square=True
    )
    plt.title("Multi-Label Genre Co-occurrence & Correlation Matrix", fontsize=14, pad=15)
    plt.xlabel("Associated Genre", fontsize=11)
    plt.ylabel("Base Genre", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.savefig(root_cm_path, dpi=300)
    plt.close()
    print(f"[Train] Multi-label matrix plot saved to: {cm_path}")

    # 10. Prepare Movie Database for Fast Cosine Similarity
    print("\n[Train] Precomputing Movie Database vectors for hybrid recommendations...")
    sparse_movies = []
    dense_matrix = X.toarray()

    for idx, row in df.iterrows():
        nonzeros = {int(f_idx): round(float(dense_matrix[idx, f_idx]), 4) for f_idx in np.nonzero(dense_matrix[idx])[0]}
        sparse_movies.append({
            "id": int(row["id"]),
            "title": str(row["title"]),
            "genres": row["genre_list"],
            "release_year": int(row["release_year"]),
            "vote_average": float(row["vote_average"]),
            "vote_count": int(row["vote_count"]),
            "overview": str(row["overview"]),
            "keywords": str(row["keywords_text"]),
            "tfidf_vec": nonzeros
        })

    # 11. Serialize Full Model Artifacts (.pkl)
    print("[Train] Serializing model artifacts to disk...")
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(prod_model, os.path.join(models_dir, "multi_label_model.pkl"))
    joblib.dump(df, os.path.join(models_dir, "processed_movies.pkl"))

    # 12. Serialize Metrics JSON
    metrics_payload = {
        "dataset_size": len(df),
        "vocab_size": len(vocab),
        "test_size": int(X_test.shape[0]),
        "train_size": int(X_train.shape[0]),
        "top_12_genres": TOP_12_GENRES,
        "genre_distribution": genre_counts,
        "hamming_accuracy": round(hamming_acc, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "top3_hit_rate": round(top3_hit_rate, 4),
        "top3_recall": round(top3_recall, 4),
        "exact_match_accuracy": round(exact_acc, 4),
        "classification_report": report_dict,
        "benchmark_comparison": benchmark_results,
        "top_predictive_words": top_predictive_words,
        "co_occurrence_matrix": genre_co_dict,
        "fit_time_seconds": prod_fit_time
    }

    metrics_json_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Train] Metrics saved to: {metrics_json_path}")

    # 13. Serialize Lightweight Serverless Bundle for Vercel (< 2 MB)
    serverless_bundle = {
        "vocab": {str(k): int(v) for k, v in vocab.items()},
        "idf": [round(float(x), 5) for x in vectorizer.idf_],
        "classes": TOP_12_GENRES,
        "class_weights": class_weights_list,
        "class_intercepts": class_intercepts_list,
        "top_predictive_words": top_predictive_words,
        "movies": sparse_movies,
        "metrics": metrics_payload
    }

    serverless_json_path = os.path.join(models_dir, "serverless_model.json")
    with open(serverless_json_path, "w", encoding="utf-8") as f:
        json.dump(serverless_bundle, f)

    bundle_size_mb = os.path.getsize(serverless_json_path) / (1024 * 1024)
    print(f"[Train] Serverless bundle generated: {serverless_json_path} ({bundle_size_mb:.2f} MB)")
    print("=" * 65)
    print("ALL MODEL ARTIFACTS AND BENCHMARKS READY!")
    print("=" * 65)

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
