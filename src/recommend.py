"""
Inference script for movie genre prediction & TF-IDF cosine similarity recommendations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ensure src directory is importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from preprocess import clean_text


class GenreRecommenderEngine:
    """
    Inference and Content Recommendation Engine using TF-IDF and Naive Bayes.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = CURRENT_DIR.parent
            models_dir = str(base_dir / "models")
        self.models_dir = Path(models_dir)

        self.vectorizer = None
        self.classifier = None
        self.processed_corpus = None
        self.tfidf_matrix = None
        self.classes = []

        self._load_artifacts()

    def _load_artifacts(self):
        """Load serialized model artifacts from disk."""
        vec_path = self.models_dir / "tfidf_vectorizer.pkl"
        clf_path = self.models_dir / "genre_classifier.pkl"
        corpus_path = self.models_dir / "processed_movies.pkl"
        matrix_path = self.models_dir / "tfidf_matrix.pkl"

        if not vec_path.exists() or not clf_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found in '{self.models_dir}'. "
                "Please run `python src/train.py` first to train the model."
            )

        self.vectorizer = joblib.load(vec_path)
        self.classifier = joblib.load(clf_path)
        self.classes = list(self.classifier.classes_)

        if corpus_path.exists():
            self.processed_corpus = joblib.load(corpus_path)
        else:
            self.processed_corpus = pd.DataFrame()

        if matrix_path.exists():
            self.tfidf_matrix = joblib.load(matrix_path)
        elif not self.processed_corpus.empty and self.vectorizer is not None:
            self.tfidf_matrix = self.vectorizer.transform(
                self.processed_corpus["cleaned_text"].tolist()
            )

    def is_ready(self) -> bool:
        """Check if all necessary artifacts are loaded."""
        return self.vectorizer is not None and self.classifier is not None

    def predict_genre(self, overview: str, keywords: str = "", top_k: int = 5) -> Dict[str, Any]:
        """
        Predict the primary movie genre with confidence probabilities.

        Parameters
        ----------
        overview : str
            Movie plot summary / synopsis.
        keywords : str
            Comma or space separated movie keywords / tags.
        top_k : int
            Number of top class probabilities to return.

        Returns
        -------
        dict
            Contains primary_genre, confidence, top_predictions list, and cleaned_query.
        """
        combined = f"{overview} {keywords}".strip()
        cleaned = clean_text(combined)

        if not cleaned:
            # Fallback for empty or all-stopword text
            cleaned = "movie film story"

        # Transform input with vectorizer
        query_vec = self.vectorizer.transform([cleaned])

        # Predict genre and class probabilities
        primary_genre = self.classifier.predict(query_vec)[0]
        probabilities = self.classifier.predict_proba(query_vec)[0]

        # Rank predictions
        classes = self.classifier.classes_
        sorted_indices = np.argsort(probabilities)[::-1]

        top_predictions = []
        for idx in sorted_indices[:top_k]:
            top_predictions.append({
                "genre": str(classes[idx]),
                "probability": float(probabilities[idx]),
                "percentage": f"{probabilities[idx] * 100:.2f}%"
            })

        confidence = float(np.max(probabilities))

        return {
            "primary_genre": str(primary_genre),
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "top_predictions": top_predictions,
            "cleaned_query": cleaned
        }

    def recommend_movies(
        self,
        overview: str,
        keywords: str = "",
        top_n: int = 5,
        exclude_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate content-based movie recommendations using TF-IDF cosine similarity.

        Parameters
        ----------
        overview : str
            Movie plot summary.
        keywords : str
            Keywords associated with the movie.
        top_n : int
            Number of similar movies to recommend (3 to 5).
        exclude_title : str, optional
            Title to exclude from recommendations (e.g. if recommending for an existing movie).

        Returns
        -------
        list of dicts
            Ranked list of similar movies with metadata and similarity scores.
        """
        if self.processed_corpus is None or self.processed_corpus.empty or self.tfidf_matrix is None:
            return []

        combined = f"{overview} {keywords}".strip()
        cleaned = clean_text(combined)
        if not cleaned:
            cleaned = "movie story"

        query_vec = self.vectorizer.transform([cleaned])

        # Compute cosine similarity between query and all movies
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort indices by descending similarity
        ranked_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            row = self.processed_corpus.iloc[idx]
            movie_title = row.get("title", f"Movie #{row.get('id_clean', idx)}")

            if exclude_title and str(movie_title).lower() == exclude_title.lower():
                continue

            sim_score = float(similarities[idx])

            recommendations.append({
                "id": int(row.get("id_clean", idx)),
                "title": str(movie_title),
                "genre": str(row.get("primary_genre", "Unknown")),
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": f"{sim_score * 100:.1f}%",
                "overview": str(row.get("overview", "")),
                "keywords": str(row.get("keywords_text", ""))
            })

            if len(recommendations) >= top_n:
                break

        return recommendations


# Convenience module-level singleton helper
_ENGINE_INSTANCE: Optional[GenreRecommenderEngine] = None


def get_engine(models_dir: Optional[str] = None) -> GenreRecommenderEngine:
    """Retrieve or initialize the global GenreRecommenderEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = GenreRecommenderEngine(models_dir=models_dir)
    return _ENGINE_INSTANCE


def predict_genre(overview: str, keywords: str = "", top_k: int = 5) -> Dict[str, Any]:
    """Helper function to predict movie genre."""
    return get_engine().predict_genre(overview, keywords, top_k=top_k)


def recommend_movies(overview: str, keywords: str = "", top_n: int = 5, exclude_title: Optional[str] = None) -> List[Dict[str, Any]]:
    """Helper function to get movie recommendations."""
    return get_engine().recommend_movies(overview, keywords, top_n=top_n, exclude_title=exclude_title)


if __name__ == "__main__":
    print("[Recommend] Testing Recommendation Engine...")
    try:
        engine = get_engine()
        test_overview = "A courageous hero enters virtual reality to battle hostile machines and liberate humanity."
        test_keywords = "cyberpunk matrix simulation artificial intelligence"

        pred = engine.predict_genre(test_overview, test_keywords)
        print(f"\nPredicted Genre: {pred['primary_genre']} (Confidence: {pred['confidence_percentage']})")
        print("Top Probabilities:")
        for p in pred["top_predictions"]:
            print(f"  - {p['genre']}: {p['percentage']}")

        recs = engine.recommend_movies(test_overview, test_keywords, top_n=4)
        print(f"\nTop {len(recs)} Recommendations:")
        for r in recs:
            print(f"  * {r['title']} [{r['genre']}] - Match: {r['similarity_percentage']}")
            print(f"    {r['overview'][:100]}...")
    except Exception as ex:
        print(f"[Recommend] Note: {ex}")
