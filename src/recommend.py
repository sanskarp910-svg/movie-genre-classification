"""
Inference & Hybrid Content Recommendation Engine for CineVerse AI.
Supports multi-label genre prediction, explainability ("Why this prediction?"),
confidence diagnostics, and hybrid recommendation scoring (plot + genres + rating).
"""

import os
import sys
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from preprocess import clean_and_lemmatize_text, TOP_12_GENRES


class MultiLabelRecommenderEngine:
    """
    Multi-label genre classification and hybrid content recommendation engine.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = CURRENT_DIR.parent
            models_dir = str(base_dir / "models")
        self.models_dir = Path(models_dir)

        self.bundle = None
        self.vectorizer = None
        self.model = None
        self.corpus_df = None

        self._load_engine()

    def _load_engine(self):
        """Load model bundle from JSON or fallback to pickle."""
        bundle_path = self.models_dir / "serverless_model.json"
        if bundle_path.exists():
            with open(bundle_path, "r", encoding="utf-8") as f:
                self.bundle = json.load(f)
            self.classes = self.bundle.get("classes", TOP_12_GENRES)
            self.vocab = self.bundle.get("vocab", {})
            self.idf = self.bundle.get("idf", [])
            self.weights = self.bundle.get("class_weights", [])
            self.intercepts = self.bundle.get("class_intercepts", [])
            self.movies = self.bundle.get("movies", [])
            self.top_predictive_words = self.bundle.get("top_predictive_words", {})
            self.metrics = self.bundle.get("metrics", {})
        else:
            # Fallback to joblib
            vec_path = self.models_dir / "tfidf_vectorizer.pkl"
            mod_path = self.models_dir / "multi_label_model.pkl"
            if not vec_path.exists() or not mod_path.exists():
                raise FileNotFoundError(f"Model artifacts not found in {self.models_dir}")
            self.vectorizer = joblib.load(vec_path)
            self.model = joblib.load(mod_path)
            self.classes = TOP_12_GENRES

    def is_ready(self) -> bool:
        return self.bundle is not None or (self.vectorizer is not None and self.model is not None)

    def vectorize_query(self, text: str) -> Tuple[dict, List[str]]:
        """Clean, lemmatize, and construct normalized TF-IDF vector."""
        cleaned = clean_and_lemmatize_text(text)
        tokens = cleaned.split()
        if not tokens:
            return {}, []

        counts = {}
        # Unigrams & bigrams
        for i in range(len(tokens)):
            t1 = tokens[i]
            if t1 in self.vocab:
                f_idx = self.vocab[t1]
                counts[f_idx] = counts.get(f_idx, 0) + 1
            if i < len(tokens) - 1:
                t2 = f"{tokens[i]} {tokens[i+1]}"
                if t2 in self.vocab:
                    f_idx = self.vocab[t2]
                    counts[f_idx] = counts.get(f_idx, 0) + 1

        vec = {}
        norm_sq = 0.0
        for f_idx, count in counts.items():
            val = (1.0 + math.log(count)) * self.idf[f_idx]
            vec[f_idx] = val
            norm_sq += val * val

        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for f_idx in vec:
                vec[f_idx] /= norm

        return vec, tokens

    def predict_genres(
        self, overview: str, keywords: str = "", top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Multi-label genre prediction returning top-k genres with probability scores,
        low-confidence alert, and feature attribution ("Why this prediction?").
        """
        combined = f"{overview} {keywords}".strip()
        vec, tokens = self.vectorize_query(combined)

        if not vec:
            return {
                "top_predictions": [],
                "primary_genre": "Unknown",
                "all_probabilities": {},
                "low_confidence_warning": False,
                "influential_words": []
            }

        # Calculate per-class probability via calibrated sigmoid
        scores = []
        for c_idx in range(len(self.classes)):
            dot = self.intercepts[c_idx] + sum(
                vec.get(f_idx, 0.0) * self.weights[c_idx][f_idx] for f_idx in vec
            )
            prob = 1.0 / (1.0 + math.exp(-max(min(dot, 15.0), -15.0)))
            scores.append((self.classes[c_idx], float(prob), c_idx))

        # Sort descending by probability
        scores.sort(key=lambda x: x[1], reverse=True)

        # Top-k genres
        top_predictions = []
        for genre, p, c_idx in scores[:top_k]:
            top_predictions.append({
                "genre": genre,
                "probability": round(p, 4),
                "percentage": f"{p * 100:.1f}%",
                "class_index": c_idx
            })

        # Low confidence detection: top 1 and top 2 are within 8% of each other
        low_confidence = False
        low_conf_msg = ""
        if len(scores) >= 2:
            prob_diff = abs(scores[0][1] - scores[1][1])
            if prob_diff < 0.08:
                low_confidence = True
                low_conf_msg = f"Close distribution: Narrative shows strong hybrid elements of both {scores[0][0]} ({scores[0][1]*100:.1f}%) and {scores[1][0]} ({scores[1][1]*100:.1f}%)."

        # "Why this prediction?" Feature Importance Attribution
        # Find which words in query had highest positive weight for the top predicted genre
        top_genre_idx = scores[0][2]
        influential_words = []
        # Invert vocab map for tokens in query
        inv_vocab = {v: k for k, v in self.vocab.items() if v in vec}

        for f_idx, tfidf_val in vec.items():
            w = self.weights[top_genre_idx][f_idx]
            if w > 0.15 and f_idx in inv_vocab:
                contribution = tfidf_val * w
                influential_words.append({
                    "word": inv_vocab[f_idx],
                    "weight": round(w, 3),
                    "impact": round(contribution, 3)
                })

        influential_words.sort(key=lambda x: x["impact"], reverse=True)
        influential_words = influential_words[:8]

        all_probs_dict = {genre: round(p, 4) for genre, p, _ in scores}

        return {
            "primary_genre": scores[0][0],
            "primary_confidence": round(scores[0][1], 4),
            "primary_percentage": f"{scores[0][1] * 100:.1f}%",
            "top_predictions": top_predictions,
            "all_probabilities": all_probs_dict,
            "low_confidence_warning": low_confidence,
            "low_confidence_message": low_conf_msg,
            "influential_words": influential_words,
            "token_count": len(tokens)
        }

    def recommend_movies(
        self,
        overview: str,
        keywords: str = "",
        top_n: int = 6,
        genre_filter: Optional[str] = None,
        min_year: int = 1900,
        max_year: int = 2030,
        min_rating: float = 0.0,
        exclude_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid content recommendation combining:
        1. TF-IDF Cosine Similarity (60% weight)
        2. Multi-label Genre Jaccard Overlap (25% weight)
        3. Rating / Quality Normalization (15% weight)
        """
        combined = f"{overview} {keywords}".strip()
        vec, _ = self.vectorize_query(combined)

        if not vec:
            return []

        # Predict query genres for genre overlap
        pred_res = self.predict_genres(overview, keywords, top_k=3)
        query_top_genres = set(p["genre"] for p in pred_res.get("top_predictions", []))

        results = []
        for m in self.movies:
            # Exclude exact match
            if exclude_title and m["title"].lower() == exclude_title.lower():
                continue

            # Filtering: year range
            year = m.get("release_year", 2010)
            if year < min_year or year > max_year:
                continue

            # Filtering: minimum rating
            rating = float(m.get("vote_average", 6.0))
            if rating < min_rating:
                continue

            # Filtering: specific genre
            movie_genres = m.get("genres", [])
            if genre_filter and genre_filter not in movie_genres and genre_filter != "All":
                continue

            # 1. Cosine similarity
            m_vec = m.get("tfidf_vec", {})
            cosine_sim = sum(vec.get(int(f_idx), 0.0) * val for f_idx, val in m_vec.items())

            # 2. Genre Jaccard Overlap
            if query_top_genres and movie_genres:
                set_m = set(movie_genres)
                jaccard = len(query_top_genres.intersection(set_m)) / len(query_top_genres.union(set_m))
            else:
                jaccard = 0.0

            # 3. Rating score (0.0 to 1.0)
            rating_score = min(max(rating / 10.0, 0.0), 1.0)

            # Hybrid Score calculation
            hybrid_score = 0.60 * cosine_sim + 0.25 * jaccard + 0.15 * rating_score

            results.append({
                "id": m["id"],
                "title": m["title"],
                "genres": movie_genres,
                "release_year": year,
                "vote_average": rating,
                "vote_count": m.get("vote_count", 0),
                "overview": m.get("overview", ""),
                "keywords": m.get("keywords", ""),
                "similarity_score": round(float(cosine_sim), 4),
                "similarity_percentage": f"{max(min(cosine_sim * 100, 100), 0):.1f}%",
                "hybrid_score": round(float(hybrid_score), 4),
                "genre_overlap": round(float(jaccard), 3)
            })

        # Rank by hybrid score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_n]


_GLOBAL_ENGINE = None


def get_engine(models_dir: Optional[str] = None) -> MultiLabelRecommenderEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = MultiLabelRecommenderEngine(models_dir=models_dir)
    return _GLOBAL_ENGINE


def predict_genres(overview: str, keywords: str = "", top_k: int = 3) -> Dict[str, Any]:
    return get_engine().predict_genres(overview, keywords, top_k=top_k)


def recommend_movies(
    overview: str,
    keywords: str = "",
    top_n: int = 6,
    genre_filter: Optional[str] = None,
    min_year: int = 1900,
    max_year: int = 2030,
    min_rating: float = 0.0
) -> List[Dict[str, Any]]:
    return get_engine().recommend_movies(
        overview=overview,
        keywords=keywords,
        top_n=top_n,
        genre_filter=genre_filter,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating
    )


if __name__ == "__main__":
    engine = get_engine()
    test_query = "A group of teenage warriors with supernatural cursed abilities battle demons in the dark city to save humanity."
    print("Test Query:", test_query)
    preds = engine.predict_genres(test_query, top_k=3)
    print("\nPredicted Top 3 Genres:")
    for p in preds["top_predictions"]:
        print(f"  - {p['genre']}: {p['percentage']}")
    print("Why this prediction:", preds["influential_words"])
    recs = engine.recommend_movies(test_query, top_n=3)
    print("\nTop 3 Hybrid Recommendations:")
    for r in recs:
        print(f"  * {r['title']} ({r['release_year']}) - Rating: {r['vote_average']} | Match: {r['similarity_percentage']} | Genres: {r['genres']}")
