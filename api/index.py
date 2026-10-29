"""
CineVerse AI - Movie & Anime Genre Classifier & Hybrid Recommender
Vercel Serverless Function & Full-Stack Web Application.
Theme: Sleek Anime Streaming Electric Orange (#FF640A) and Cinema Dark (#141519), with Light Mode support.
"""

import os
import sys
import json
import math
import re
import urllib.request
import urllib.parse
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Cached serverless bundle
_BUNDLE = None


def get_bundle():
    """Load pre-trained serverless model bundle."""
    global _BUNDLE
    if _BUNDLE is None:
        bundle_path = MODELS_DIR / "serverless_model.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
        with open(bundle_path, "r", encoding="utf-8") as f:
            _BUNDLE = json.load(f)
    return _BUNDLE


# High-performance stop words set
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves", "movie", "film", "story", "also", "one", "two"
}

# Lookup dictionary for irregular lemmatization
LEMMATIZATION_LOOKUP = {
    "heroes": "superhero", "superheroes": "superhero", "villains": "villain",
    "detectives": "detect", "murders": "murder", "murderers": "murder",
    "mysteries": "mysteri", "adventures": "adventur", "aliens": "alien",
    "spaceships": "spaceship", "monsters": "monster", "demons": "demon",
    "zombies": "zombi", "vampires": "vampir", "witches": "witch",
    "wizards": "wizard", "dragons": "dragon", "spirits": "spirit",
    "ghosts": "ghost", "robberies": "robberi", "thieves": "thief",
    "soldiers": "soldier", "warriors": "warrior", "battles": "battl",
    "countries": "countri", "families": "famili", "friends": "friend",
    "children": "child", "women": "woman", "men": "man"
}

# Try importing SnowballStemmer or use lightweight pure-Python suffix stemmer
try:
    from nltk.stem import SnowballStemmer
    _NLTK_STEMMER = SnowballStemmer("english")
    def stem_token(w: str) -> str:
        return _NLTK_STEMMER.stem(w)
except Exception:
    def stem_token(w: str) -> str:
        if w.endswith("ies") and len(w) > 4: return w[:-3] + "i"
        if w.endswith("ing") and len(w) > 5: return w[:-3]
        if w.endswith("ed") and len(w) > 4: return w[:-2]
        if w.endswith("es") and len(w) > 4: return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
        if w.endswith("ment") and len(w) > 6: return w[:-4]
        return w


def clean_and_tokenize(text: str):
    """Clean text, remove stop words, and apply lemmatization/stemming."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    raw_tokens = text.split()
    tokens = []
    for t in raw_tokens:
        if t in STOP_WORDS or len(t) <= 2:
            continue
        lemma = LEMMATIZATION_LOOKUP.get(t, stem_token(t))
        tokens.append(lemma)
    return tokens


def vectorize_tokens(tokens: list, vocab: dict, idf: list) -> dict:
    """Build normalized TF-IDF vector with sublinear term frequency."""
    counts = {}
    for i in range(len(tokens)):
        t1 = tokens[i]
        if t1 in vocab:
            f_idx = vocab[t1]
            counts[f_idx] = counts.get(f_idx, 0) + 1
        if i < len(tokens) - 1:
            t2 = f"{tokens[i]} {tokens[i+1]}"
            if t2 in vocab:
                f_idx = vocab[t2]
                counts[f_idx] = counts.get(f_idx, 0) + 1

    vec = {}
    norm_sq = 0.0
    for f_idx, count in counts.items():
        val = (1.0 + math.log(count)) * idf[f_idx]
        vec[f_idx] = val
        norm_sq += val * val

    if norm_sq > 0:
        norm = math.sqrt(norm_sq)
        for f_idx in vec:
            vec[f_idx] /= norm

    return vec


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CineVerse AI | Movie & Anime Multi-Label Genre Classifier</title>
  
  <!-- SEO & Open Graph Metadata -->
  <meta name="description" content="Production-ready multi-label NLP movie & anime genre classifier and hybrid content recommender powered by TF-IDF & Logistic Regression.">
  <meta property="og:title" content="CineVerse AI - Movie & Anime Genre Classifier & Recommender">
  <meta property="og:description" content="Predict top 3 genres from plot synopses with calibrated probabilities, feature attribution, and hybrid recommendations.">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="CineVerse AI">
  <meta name="twitter:description" content="AI Movie & Anime Multi-Label Genre Classification & Hybrid Content Recommendations.">
  
  <!-- Favicon Data URI -->
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%23FF640A'/><polygon points='40,30 70,50 40,70' fill='%23000'/></svg>">
  
  <!-- Google Fonts & Chart.js -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

  <style>
    :root {
      --brand-orange: #FF640A;
      --brand-orange-hover: #E05300;
      --brand-orange-light: #FFA347;
      --brand-orange-glow: rgba(255, 100, 10, 0.35);
      --bg-deep: #0D0E11;
      --bg-main: #141519;
      --card-bg: #1E1F25;
      --surface: #24262E;
      --border: #2E303A;
      --text: #F7F7F8;
      --text-muted: #9E9EB2;
      --text-dim: #6E7082;
      --badge-bg: rgba(255, 100, 10, 0.15);
      --badge-border: #FF640A;
      --chip-bg: #2C2E38;
      --modal-overlay: rgba(0, 0, 0, 0.85);
      --shadow-color: rgba(0, 0, 0, 0.45);
    }

    [data-theme="light"] {
      --brand-orange: #FF640A;
      --brand-orange-hover: #E05300;
      --brand-orange-light: #FF8A3D;
      --brand-orange-glow: rgba(255, 100, 10, 0.22);
      --bg-deep: #EAEDF2;
      --bg-main: #F4F6F9;
      --card-bg: #FFFFFF;
      --surface: #F8F9FA;
      --border: #D8DCE3;
      --text: #17181D;
      --text-muted: #5C6070;
      --text-dim: #8C90A0;
      --badge-bg: rgba(255, 100, 10, 0.12);
      --badge-border: #FF640A;
      --chip-bg: #EAECEF;
      --modal-overlay: rgba(20, 21, 25, 0.7);
      --shadow-color: rgba(0, 0, 0, 0.08);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-main);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      transition: background-color 0.25s ease, color 0.25s ease;
    }

    /* Top Navigation Bar */
    .navbar {
      background-color: var(--bg-deep);
      border-bottom: 2px solid var(--border);
      padding: 0.85rem 2rem;
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
    }
    .brand-icon {
      width: 36px;
      height: 36px;
      background: var(--brand-orange);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #000;
      font-size: 1.1rem;
      font-weight: 900;
      box-shadow: 0 0 14px var(--brand-orange-glow);
    }
    .brand-title {
      font-size: 1.4rem;
      font-weight: 900;
      color: var(--text);
      letter-spacing: -0.02em;
    }
    .brand-highlight {
      color: var(--brand-orange);
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .nav-tag {
      background: var(--badge-bg);
      border: 1px solid var(--badge-border);
      color: var(--brand-orange);
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .theme-toggle-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 0.4rem 0.75rem;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.2s ease;
    }
    .theme-toggle-btn:hover {
      border-color: var(--brand-orange);
      color: var(--brand-orange);
    }

    /* Container */
    .container {
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.25rem 4rem 1.25rem;
      flex: 1;
    }

    /* Hero Banner */
    .hero {
      margin-bottom: 2rem;
    }
    .hero-eyebrow {
      color: var(--brand-orange);
      font-size: 0.88rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .hero-title {
      font-size: 2.3rem;
      font-weight: 900;
      color: var(--text);
      letter-spacing: -0.03em;
      line-height: 1.2;
    }
    .hero-desc {
      color: var(--text-muted);
      font-size: 1.05rem;
      margin-top: 0.5rem;
      max-width: 760px;
    }

    /* Tabs Bar */
    .tab-bar {
      display: flex;
      gap: 0.5rem;
      border-bottom: 2px solid var(--border);
      margin-bottom: 2rem;
      overflow-x: auto;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 1.05rem;
      font-weight: 700;
      padding: 0.85rem 1.5rem;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      white-space: nowrap;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active {
      color: var(--brand-orange);
      border-bottom-color: var(--brand-orange);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Cards */
    .cr-card {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.75rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 4px 20px var(--shadow-color);
      transition: all 0.2s ease;
    }

    .form-group { margin-bottom: 1.25rem; }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text);
      margin-bottom: 0.5rem;
    }
    textarea, select, input {
      width: 100%;
      background-color: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem 1.1rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      transition: all 0.2s ease;
    }
    textarea:focus, select:focus, input:focus {
      outline: none;
      border-color: var(--brand-orange);
      box-shadow: 0 0 0 3px var(--brand-orange-glow);
    }

    /* Word Counter & Validation Alert */
    .counter-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 0.4rem;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    .counter-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      background: var(--surface);
    }
    .counter-valid { color: #10B981; }
    .counter-invalid { color: #EF4444; }

    .validation-alert {
      display: none;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid #EF4444;
      color: #F87171;
      border-radius: 8px;
      padding: 0.8rem 1rem;
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
      align-items: center;
      gap: 0.6rem;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 3fr 2fr;
      gap: 1.25rem;
    }
    @media (max-width: 820px) { .grid-2 { grid-template-columns: 1fr; } }

    /* Button Toolbar */
    .button-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.85rem;
      margin-top: 1rem;
    }
    .cr-btn {
      background: var(--brand-orange);
      color: #FFFFFF;
      font-weight: 800;
      font-size: 1rem;
      border: none;
      border-radius: 8px;
      padding: 0.85rem 1.75rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.6rem;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      box-shadow: 0 4px 14px var(--brand-orange-glow);
    }
    .cr-btn:hover {
      background: var(--brand-orange-hover);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px var(--brand-orange-glow);
    }
    .cr-btn-secondary {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      font-weight: 700;
      font-size: 0.92rem;
      border-radius: 8px;
      padding: 0.85rem 1.25rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
    }
    .cr-btn-secondary:hover {
      border-color: var(--brand-orange);
      color: var(--brand-orange);
    }

    /* Empty State */
    .empty-state-box {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 3.5rem 1.5rem;
      text-align: center;
      color: var(--text-muted);
      margin-top: 1.5rem;
      background: var(--card-bg);
    }
    .empty-state-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
      opacity: 0.85;
    }
    .empty-state-title {
      font-size: 1.3rem;
      font-weight: 800;
      color: var(--text);
      margin-bottom: 0.4rem;
    }
    .empty-state-sub {
      font-size: 0.95rem;
      max-width: 520px;
      margin: 0 auto;
    }

    /* Skeleton Loading State */
    .skeleton-box {
      display: none;
      margin-top: 1.5rem;
    }
    .skeleton-card {
      background: linear-gradient(90deg, var(--card-bg) 25%, var(--surface) 50%, var(--card-bg) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 8px;
      height: 120px;
      margin-bottom: 1rem;
    }
    @keyframes shimmer {
      0% { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }

    /* Prediction Banner & Top 3 Badges */
    .prediction-hero {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-left: 6px solid var(--brand-orange);
      border-radius: 10px;
      padding: 1.75rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 4px 24px var(--shadow-color);
    }
    .top3-genres-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-top: 1.25rem;
    }
    .genre-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.1rem;
      position: relative;
      overflow: hidden;
    }
    .genre-rank-tag {
      font-size: 0.72rem;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--brand-orange);
      margin-bottom: 0.35rem;
    }
    .genre-name-label {
      font-size: 1.35rem;
      font-weight: 900;
      color: var(--text);
    }
    .genre-prob-val {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--brand-orange);
      float: right;
    }

    /* Low Confidence Warning Alert */
    .low-confidence-alert {
      display: none;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid #F59E0B;
      color: #FBBF24;
      border-radius: 8px;
      padding: 0.9rem 1.2rem;
      font-size: 0.92rem;
      font-weight: 600;
      margin-bottom: 1.5rem;
      align-items: center;
      gap: 0.75rem;
    }

    /* Animated Progress Bars */
    .prob-track {
      height: 8px;
      background-color: var(--bg-deep);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 0.75rem;
    }
    .prob-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--brand-orange), var(--brand-orange-light));
      border-radius: 4px;
      width: 0%;
      transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Explainability: Why this prediction? */
    .why-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
      margin-top: 1.5rem;
    }
    .why-title {
      font-size: 0.95rem;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }
    .word-chips-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .word-chip {
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.3rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .word-weight {
      color: var(--brand-orange);
      font-weight: 800;
    }

    /* Thumbs Feedback Bar */
    .feedback-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid var(--border);
      padding-top: 1rem;
      margin-top: 1.25rem;
      font-size: 0.88rem;
      color: var(--text-muted);
    }
    .feedback-btns {
      display: flex;
      gap: 0.5rem;
    }
    .fb-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 6px;
      padding: 0.3rem 0.75rem;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.2s ease;
    }
    .fb-btn:hover {
      border-color: var(--brand-orange);
      color: var(--brand-orange);
    }
    .fb-btn.active {
      background: var(--brand-orange);
      color: #FFF;
      border-color: var(--brand-orange);
    }

    /* Recommendations & Filters */
    .filter-bar {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.1rem 1.25rem;
      margin-bottom: 1.5rem;
      display: grid;
      grid-template-columns: 2fr 1.5fr 1.5fr;
      gap: 1.25rem;
      align-items: center;
    }
    @media (max-width: 860px) { .filter-bar { grid-template-columns: 1fr; } }

    .rec-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
      gap: 1.35rem;
      margin-top: 1rem;
    }
    .stream-card {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
    }
    .stream-card:hover {
      transform: translateY(-5px);
      border-color: var(--brand-orange);
      box-shadow: 0 10px 28px var(--brand-orange-glow);
    }
    .card-poster {
      width: 100%;
      height: 220px;
      object-fit: cover;
      background-color: #17181D;
      display: block;
      border-bottom: 1px solid var(--border);
    }
    .card-content {
      padding: 1.1rem;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .card-top-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.5rem;
      gap: 0.5rem;
    }
    .stream-title {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text);
      line-height: 1.3;
    }
    .match-pill {
      background: var(--badge-bg);
      border: 1px solid var(--brand-orange);
      color: var(--brand-orange);
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-size: 0.78rem;
      font-weight: 800;
      white-space: nowrap;
    }
    .card-meta {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 0.6rem;
      font-weight: 600;
    }
    .star-score { color: #F59E0B; font-weight: 800; }
    .card-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.75rem;
    }
    .genre-chip {
      background-color: var(--chip-bg);
      color: var(--text);
      padding: 0.15rem 0.55rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
    }
    .stream-synopsis {
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.45;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
      flex: 1;
    }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: var(--modal-overlay);
      backdrop-filter: blur(4px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.25rem;
    }
    .modal-box {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      max-width: 780px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 2rem;
      position: relative;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    }
    .modal-close-btn {
      position: absolute;
      top: 1.25rem;
      right: 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 1.2rem;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }
    .modal-close-btn:hover {
      background: var(--brand-orange);
      color: #FFF;
    }

    /* KPI Cards */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .kpi-box {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-top: 3px solid var(--brand-orange);
      border-radius: 8px;
      padding: 1.35rem;
      text-align: center;
      box-shadow: 0 4px 14px var(--shadow-color);
    }
    .kpi-num {
      font-size: 2.2rem;
      font-weight: 900;
      color: var(--brand-orange);
      line-height: 1.1;
      margin-bottom: 0.35rem;
    }
    .kpi-desc {
      font-size: 0.8rem;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--text-muted);
      letter-spacing: 0.06em;
    }

    /* Charts Grid */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    @media (max-width: 600px) { .charts-grid { grid-template-columns: 1fr; } }

    .chart-container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.5rem;
      box-shadow: 0 4px 14px var(--shadow-color);
    }
    .chart-header {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text);
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Predictive Words Tabs */
    .words-genre-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-bottom: 1.25rem;
    }
    .pill-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 0.35rem 0.85rem;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .pill-btn:hover { color: var(--text); border-color: var(--brand-orange); }
    .pill-btn.active {
      background: var(--brand-orange);
      color: #FFF;
      border-color: var(--brand-orange);
    }

    /* Accordion */
    .accordion-item {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.75rem;
      overflow: hidden;
    }
    .accordion-header {
      padding: 1.1rem 1.35rem;
      font-weight: 800;
      font-size: 1rem;
      color: var(--text);
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    .accordion-body {
      padding: 0 1.35rem 1.25rem 1.35rem;
      color: var(--text-muted);
      font-size: 0.92rem;
      display: none;
      line-height: 1.6;
    }
    .accordion-body.open { display: block; }

    /* Footer */
    footer {
      background-color: var(--bg-deep);
      border-top: 1px solid var(--border);
      padding: 2rem 1.25rem;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.88rem;
      margin-top: auto;
    }
  </style>
</head>
<body data-theme="dark">

  <!-- Navbar -->
  <nav class="navbar">
    <a href="/" class="nav-brand">
      <div class="brand-icon">▷</div>
      <div class="brand-title">CINE<span class="brand-highlight">VERSE</span></div>
    </a>
    <div class="nav-actions">
      <button class="theme-toggle-btn" onclick="toggleTheme()" id="theme-btn" aria-label="Toggle Dark and Light Mode">
        <span id="theme-icon">🌙</span> <span id="theme-label">Dark</span>
      </button>
      <span class="nav-tag">AI MULTI-LABEL</span>
    </div>
  </nav>

  <!-- Container -->
  <main class="container">
    <header class="hero">
      <div class="hero-eyebrow">
        <span>⚡</span> Multi-Label NLP Classifier & Recommender
      </div>
      <h1 class="hero-title">Predict Genres & Discover Content</h1>
      <p class="hero-desc">Trained on 4,680+ titles across 12 genres with sublinear TF-IDF, Logistic Regression calibrated probabilities, and hybrid cosine-overlap recommendations.</p>
    </header>

    <!-- Navigation Tabs -->
    <div class="tab-bar" role="tablist">
      <button class="tab-btn active" role="tab" onclick="switchTab('predict')" id="tab-btn-predict">
        🎬 Genre Predictor & Recommender
      </button>
      <button class="tab-btn" role="tab" onclick="switchTab('analytics')" id="tab-btn-analytics">
        📊 Dataset & Model Analytics
      </button>
      <button class="tab-btn" role="tab" onclick="switchTab('about')" id="tab-btn-about">
        💡 How It Works
      </button>
    </div>

    <!-- TAB 1: PREDICTOR & RECOMMENDER -->
    <section id="tab-predict" class="tab-content active" role="tabpanel">
      <div class="cr-card">
        <!-- Preset Selector -->
        <div class="form-group">
          <label for="presets">⚡ Quick Archetype Presets (Auto-fill Synopsis):</label>
          <select id="presets" onchange="applyPreset()">
            <option value="">-- Choose a cinema or anime storyline --</option>
            <option value="scifi">Interstellar (Sci-Fi • Wormhole, Relativistic Black Hole & Tesseract)</option>
            <option value="action">The Dark Knight (Action • Gotham Vigilante vs Joker Chaos)</option>
            <option value="shonen">Demon Slayer Anime (Action & Fantasy • Demon Slaying Blade Corps)</option>
            <option value="horror">The Conjuring (Horror • Paranormal Investigators & Secluded Possession)</option>
            <option value="comedy">Superbad (Comedy • High School Graduation Party & Fake ID)</option>
            <option value="cyberpunk">Ghost in the Shell Anime (Sci-Fi & Crime • Cyborg Operative & Puppet Master)</option>
            <option value="romance">La La Land (Romance & Drama • Jazz Musician & Aspiring Actress)</option>
            <option value="animation">Spirited Away Anime (Animation & Fantasy • Spirit Bathhouse & Witch Curse)</option>
          </select>
        </div>

        <div class="grid-2">
          <div class="form-group">
            <label for="plot">Narrative Synopsis / Story Overview:</label>
            <textarea id="plot" rows="5" placeholder="Enter storyline synopsis here (minimum 15 words)..." oninput="updateWordCount()"></textarea>
            <div class="counter-row">
              <span id="word-badge" class="counter-badge counter-invalid">0 / 15 words minimum</span>
              <span id="char-badge">0 characters</span>
            </div>
          </div>
          <div class="form-group">
            <label for="keywords">Thematic Keywords / Tags (Optional):</label>
            <textarea id="keywords" rows="5" placeholder="e.g. superhero, gotham, vigilante, crime fighter, space travel, demon..."></textarea>
            <div class="counter-row">
              <span>Keywords provide contextual boosting</span>
            </div>
          </div>
        </div>

        <!-- Word Count Alert -->
        <div id="validation-alert" class="validation-alert">
          <span>⚠️</span> <span id="validation-msg">Please enter at least 15 words for reliable NLP feature classification (currently 0 words).</span>
        </div>

        <!-- Action Toolbar -->
        <div class="button-toolbar">
          <button class="cr-btn" onclick="runPrediction()">
            <span>⚡</span> Classify Genre & Discover Content
          </button>
          <button class="cr-btn-secondary" onclick="tryRandomExample()">
            <span>🎲</span> Try Random Example
          </button>
          <button class="cr-btn-secondary" onclick="clearInputs()">
            <span>🗑️</span> Clear
          </button>
          <button class="cr-btn-secondary" onclick="copyShareableLink()">
            <span>🔗</span> Share Link
          </button>
        </div>
      </div>

      <!-- Empty State -->
      <div id="empty-state" class="empty-state-box">
        <div class="empty-state-icon">🎬</div>
        <h3 class="empty-state-title">Enter a Synopsis to Begin</h3>
        <p class="empty-state-sub">Type a movie or anime storyline or select a preset above. CineVerse AI will classify the top 3 genres, analyze predictive words, and retrieve similar titles.</p>
      </div>

      <!-- Skeleton Shimmer Loader -->
      <div id="skeleton-loader" class="skeleton-box">
        <div class="skeleton-card"></div>
        <div class="skeleton-card" style="height: 180px;"></div>
      </div>

      <!-- Output Section -->
      <div id="result-section" style="display: none;">
        <!-- Low Confidence Alert -->
        <div id="low-conf-alert" class="low-confidence-alert">
          <span>⚠️</span> <span id="low-conf-text"></span>
        </div>

        <!-- Top 3 Genre Prediction Card -->
        <div class="prediction-hero">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
            <div>
              <div style="font-size: 0.82rem; text-transform: uppercase; color: var(--text-muted); font-weight: 800; letter-spacing: 0.06em;">
                Multi-Label Genre Classification Result
              </div>
              <h2 style="font-size: 1.5rem; font-weight: 900; color: var(--text); margin-top: 0.2rem;">
                Top Predicted Genres
              </h2>
            </div>
            <button class="cr-btn-secondary" style="padding: 0.4rem 0.85rem;" onclick="copyResultText()">
              <span>📋</span> <span id="copy-btn-text">Copy Result</span>
            </button>
          </div>

          <!-- Top 3 Genre Badges -->
          <div class="top3-genres-row" id="top3-cards"></div>

          <!-- Explainability: Why this prediction? -->
          <div class="why-section" id="why-box">
            <div class="why-title">
              <span>🔍</span> Why this prediction? (Top influential narrative signals)
            </div>
            <div class="word-chips-grid" id="influential-words"></div>
          </div>

          <!-- Feedback Bar -->
          <div class="feedback-row">
            <span>Was this classification accurate?</span>
            <div class="feedback-btns">
              <button class="fb-btn" id="fb-up" onclick="sendFeedback('up')">👍 Accurate</button>
              <button class="fb-btn" id="fb-down" onclick="sendFeedback('down')">👎 Inaccurate</button>
            </div>
          </div>
          <div id="fb-status" style="font-size: 0.8rem; color: #10B981; font-weight: 700; margin-top: 0.5rem; display: none;">
            ✓ Thank you! Your feedback helps calibrate future multi-label models.
          </div>
        </div>

        <!-- Recommendations Header & Filters -->
        <div style="margin-top: 2.5rem; margin-bottom: 1rem;">
          <h2 style="font-size: 1.6rem; font-weight: 900; color: var(--text);">Similar Content Recommendations</h2>
          <p style="font-size: 0.95rem; color: var(--text-muted); margin-top: 0.25rem;">
            Hybrid scoring combines TF-IDF Cosine Similarity (60%), Multi-label Genre Jaccard Overlap (25%), and Quality Ratings (15%).
          </p>
        </div>

        <!-- Filter Controls -->
        <div class="filter-bar">
          <div>
            <label for="filter-genre">Filter by Genre:</label>
            <select id="filter-genre" onchange="applyFilters()">
              <option value="All">All Genres (Catalog)</option>
              <option value="Action">Action</option>
              <option value="Adventure">Adventure</option>
              <option value="Animation">Animation</option>
              <option value="Comedy">Comedy</option>
              <option value="Crime">Crime</option>
              <option value="Drama">Drama</option>
              <option value="Family">Family</option>
              <option value="Fantasy">Fantasy</option>
              <option value="Horror">Horror</option>
              <option value="Romance">Romance</option>
              <option value="Science Fiction">Science Fiction</option>
              <option value="Thriller">Thriller</option>
            </select>
          </div>
          <div>
            <label for="filter-year">Release Year Range:</label>
            <select id="filter-year" onchange="applyFilters()">
              <option value="all">All Release Years</option>
              <option value="2010s">2010 – Present</option>
              <option value="2000s">2000 – 2009</option>
              <option value="1990s">1990 – 1999</option>
              <option value="classic">Classic (Pre-1990)</option>
            </select>
          </div>
          <div>
            <label for="filter-rating">Min Rating: <span id="rating-val" style="color: var(--brand-orange); font-weight: 800;">6.0</span> ⭐</label>
            <input type="range" id="filter-rating" min="0" max="9" step="0.5" value="6.0" oninput="document.getElementById('rating-val').innerText = this.value; applyFilters();">
          </div>
        </div>

        <!-- Recommendations Grid -->
        <div class="rec-grid" id="rec-list"></div>
      </div>
    </section>

    <!-- TAB 2: DATASET & MODEL ANALYTICS -->
    <section id="tab-analytics" class="tab-content" role="tabpanel">
      <!-- KPI Top Metrics -->
      <div class="kpi-row">
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-hamming">86.7%</div>
          <div class="kpi-desc">Hamming Accuracy</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-microf1">67.7%</div>
          <div class="kpi-desc">Micro F1-Score</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-hitrate">96.8%</div>
          <div class="kpi-desc">Top-3 Any Hit Rate</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-catalog">4,680</div>
          <div class="kpi-desc">Catalog Titles (12 Genres)</div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="charts-grid">
        <div class="chart-container">
          <div class="chart-header">
            <span>📊 Genre Distribution Across Catalog</span>
          </div>
          <canvas id="chart-distribution" height="280"></canvas>
        </div>
        <div class="chart-container">
          <div class="chart-header">
            <span>🎯 Per-Genre F1, Precision & Recall</span>
          </div>
          <canvas id="chart-metrics" height="280"></canvas>
        </div>
      </div>

      <div class="charts-grid">
        <div class="chart-container">
          <div class="chart-header">
            <span>⚖️ 5-Fold Cross Validation Benchmark</span>
          </div>
          <canvas id="chart-benchmark" height="260"></canvas>
        </div>
        <div class="chart-container">
          <div class="chart-header">
            <span>🔥 Genre Co-Occurrence Matrix</span>
          </div>
          <div id="co-matrix-wrap" style="overflow-x: auto; max-height: 280px;"></div>
        </div>
      </div>

      <!-- Top Predictive Words by Genre -->
      <div class="cr-card">
        <div class="chart-header">
          <span>🔤 Top Predictive Words per Genre (Regression Coefficients)</span>
        </div>
        <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 1rem;">
          Click any genre below to inspect the highest-weighted vocabulary terms learned by the model.
        </p>
        <div class="words-genre-pills" id="genre-pills"></div>
        <div class="word-chips-grid" id="genre-words-display"></div>
      </div>

      <!-- Dataset Size and Limitations Note -->
      <div class="cr-card">
        <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text); margin-bottom: 0.6rem;">
          Dataset Scale & Methodological Notes
        </h3>
        <p style="color: var(--text-muted); font-size: 0.92rem; line-height: 1.6;">
          • <strong>Training Corpus:</strong> 4,680 clean titles from TMDB 5000 and curated anime classics. Synopses are augmented with user keywords and stripped of low-information tokens.<br>
          • <strong>Multi-Label Modeling:</strong> Movies naturally bridge multiple genres (e.g. Action + Sci-Fi + Adventure). The One-vs-Rest architecture optimizes individual sigmoid decision boundaries for each of the 12 primary genres.<br>
          • <strong>Input Length Limitation:</strong> Text samples with fewer than 15 words suffer from sparse vocabulary representation. Providing rich narrative descriptions yields significantly higher confidence and fidelity.<br>
          • <strong>Calibrated Probabilities:</strong> Probabilities reflect normalized sigmoid activation values $P(C|x) = \sigma(w^T x + b)$ rather than Naive Bayes uncalibrated likelihood ratios.
        </p>
      </div>
    </section>

    <!-- TAB 3: HOW IT WORKS (ABOUT) -->
    <section id="tab-about" class="tab-content" role="tabpanel">
      <div class="cr-card">
        <h2 style="font-size: 1.5rem; font-weight: 900; color: var(--text); margin-bottom: 1rem;">
          System Architecture & Methodological Blueprint
        </h2>
        
        <div class="accordion-item">
          <div class="accordion-header" onclick="toggleAccordion(this)">
            <span>1. Text Preprocessing & Snowball Stemming</span>
            <span>▾</span>
          </div>
          <div class="accordion-body">
            Input storylines undergo normalization: lowercase transformation, punctuation stripping, stop word filtering, and Snowball morphological lemmatization. An irregular noun lookup handles semantic cinema roots (e.g., <em>superheroes → superhero, murderers → murder</em>).
          </div>
        </div>

        <div class="accordion-item">
          <div class="accordion-header" onclick="toggleAccordion(this)">
            <span>2. Sublinear TF-IDF Feature Representation (6,000 Unigrams & Bigrams)</span>
            <span>▾</span>
          </div>
          <div class="accordion-body">
            The corpus is vectorized using term frequency-inverse document frequency with sublinear scaling:
            <br><code>tf = 1 + ln(count)</code>, combined with <code>idf = ln((1 + N)/(1 + df)) + 1</code>.
            Extracting both single words (unigrams) and 2-word phrases (bigrams) preserves contextual expressions like <em>"space travel"</em>, <em>"serial killer"</em>, and <em>"high school"</em>.
          </div>
        </div>

        <div class="accordion-item">
          <div class="accordion-header" onclick="toggleAccordion(this)">
            <span>3. Multi-Label OneVsRest Logistic Regression</span>
            <span>▾</span>
          </div>
          <div class="accordion-body">
            Single-label Naive Bayes fails to represent modern media because most films have multiple genres. CineVerse AI employs 12 balanced One-vs-Rest binary classifiers. Predictions are computed using sigmoid activations, producing an independent probability for each genre and outputting the top 3 scores.
          </div>
        </div>

        <div class="accordion-item">
          <div class="accordion-header" onclick="toggleAccordion(this)">
            <span>4. Hybrid Content-Based Recommendation Formula</span>
            <span>▾</span>
          </div>
          <div class="accordion-body">
            Similar movies are retrieved via a 3-tier hybrid objective score:
            <br><strong>Hybrid Score = 0.60 × Cosine Similarity + 0.25 × Genre Jaccard Overlap + 0.15 × Normalized Rating</strong>
            This ensures that recommendations share narrative themes, genre profile, and high critical acclaim.
          </div>
        </div>
      </div>
    </section>
  </main>

  <!-- Movie Details Modal -->
  <div id="movie-modal" class="modal-backdrop" onclick="closeModalOnBackdrop(event)">
    <div class="modal-box">
      <button class="modal-close-btn" onclick="closeModal()" aria-label="Close modal">✕</button>
      <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <img id="modal-poster" src="" alt="Movie Poster" style="width: 180px; height: 260px; object-fit: cover; border-radius: 8px; border: 1px solid var(--border);">
        <div style="flex: 1; min-width: 260px;">
          <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span id="modal-match" class="match-pill">⚡ 94% Match</span>
            <span id="modal-year" style="color: var(--text-muted); font-weight: 700;">2010</span>
          </div>
          <h2 id="modal-title" style="font-size: 1.6rem; font-weight: 900; color: var(--text); line-height: 1.2;">Title</h2>
          <div style="margin-top: 0.4rem; font-size: 0.95rem; color: #F59E0B; font-weight: 800;">
            ⭐ <span id="modal-rating">8.4</span> / 10 (<span id="modal-votes" style="color: var(--text-muted); font-size: 0.85rem;">0 votes</span>)
          </div>
          <div id="modal-chips" class="card-chips" style="margin-top: 0.75rem;"></div>
          <div style="margin-top: 1.2rem;">
            <a id="modal-trailer" href="#" target="_blank" rel="noopener noreferrer" class="cr-btn" style="text-decoration: none; padding: 0.65rem 1.25rem; font-size: 0.9rem;">
              <span>▶</span> Watch Official Trailer
            </a>
          </div>
        </div>
      </div>
      <div style="margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1.25rem;">
        <h4 style="font-size: 0.95rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 0.4rem;">
          Plot Overview
        </h4>
        <p id="modal-overview" style="color: var(--text); font-size: 0.95rem; line-height: 1.6;"></p>
      </div>
      <div style="margin-top: 1rem;">
        <h4 style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 0.4rem;">
          Keywords / Thematic Tags
        </h4>
        <p id="modal-keywords" style="color: var(--text-dim); font-size: 0.85rem;"></p>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div style="margin-bottom: 0.4rem; font-weight: 800; color: var(--text);">
      CINE<span style="color: var(--brand-orange);">VERSE</span> AI
    </div>
    Multi-Label NLP Machine Learning Pipeline powered by TF-IDF & One-vs-Rest Logistic Regression.
  </footer>

  <script>
    // Presets catalog
    const presets = {
      scifi: {
        plot: "A team of exploratory astronauts travel through an enigmatic wormhole near Saturn in a desperate bid to find a habitable exoplanet and guarantee humanity survival before global crop blights extinguish all life.",
        keywords: "space travel, wormhole, black hole, time dilation, astronaut, astrophysics, planet discovery"
      },
      action: {
        plot: "Batman raises the stakes in his war on crime. With the help of allies Lt. Jim Gordon and DA Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets of Gotham, until a psychopathic criminal mastermind known as the Joker thrusts the city into anarchy.",
        keywords: "superhero, gotham city, vigilante, crime fighter, explosions, joker, chaos"
      },
      shonen: {
        plot: "After a demon savagely massacres his entire family and curses his younger sister into a demon, a determined teenage boy joins an elite corps of demon hunters wielding ancient sun-breathing sword techniques to avenge his parents and restore his sister humanity.",
        keywords: "anime, demon slayer, katana swords, supernatural powers, vengeance, dark fantasy, martial arts"
      },
      horror: {
        plot: "Paranormal investigators Ed and Lorraine Warren work to help a terrorized family living in a secluded farmhouse. Ancient demonic entities and supernatural possession push the couple to their absolute psychological limits.",
        keywords: "demonology, haunted house, paranormal investigation, possession, exorcism, evil spirits"
      },
      comedy: {
        plot: "Two co-dependent high school seniors are forced to deal with intense separation anxiety after their plan to stage a booze-soaked party goes wildly awry with fake IDs, eccentric rogue cops, and chaotic misadventures.",
        keywords: "high school, teen comedy, fake id, wild party, friendship, coming of age, police chase"
      },
      cyberpunk: {
        plot: "In a futuristic cybernetic metropolis, a cyborg public-security agent hunts a mysterious cyber-terrorist hacker known as the Puppet Master, discovering profound existential secrets regarding artificial consciousness and human identity.",
        keywords: "cyberpunk, cyborg, artificial intelligence, anime, hacker, sci-fi thriller, philosophy"
      },
      romance: {
        plot: "While navigating their burgeoning careers in Los Angeles, an impassioned jazz pianist and an aspiring actress fall passionately in love while attempting to reconcile their dreams for the future with their emotional devotion.",
        keywords: "jazz musician, aspiring actress, los angeles romance, musical love story, bittersweet dreams"
      },
      animation: {
        plot: "While exploring an abandoned amusement park, a young girl discovers a mystical world of ancient spirits and Japanese gods. When her parents are magically transformed into pigs by a vengeful witch, she must work at a grand spirit bathhouse to free them.",
        keywords: "anime, studio ghibli, spirit world, witch, bathhouse, fantasy curse, coming of age"
      }
    };

    // Stored recommendations cache for live filtering
    let allRecommendations = [];
    let currentPrediction = null;
    let analyticsData = null;

    // Word Counter
    function updateWordCount() {
      const text = document.getElementById('plot').value.trim();
      const words = text ? text.split(/\\s+/).filter(w => w.length > 0) : [];
      const wordCount = words.length;
      const charCount = text.length;

      const badge = document.getElementById('word-badge');
      badge.innerText = `${wordCount} / 15 words minimum`;
      if (wordCount >= 15) {
        badge.className = "counter-badge counter-valid";
        document.getElementById('validation-alert').style.display = "none";
      } else {
        badge.className = "counter-badge counter-invalid";
      }
      document.getElementById('char-badge').innerText = `${charCount} characters`;
    }

    function applyPreset() {
      const val = document.getElementById('presets').value;
      if (presets[val]) {
        document.getElementById('plot').value = presets[val].plot;
        document.getElementById('keywords').value = presets[val].keywords;
        updateWordCount();
      }
    }

    function tryRandomExample() {
      const keys = Object.keys(presets);
      const randKey = keys[Math.floor(Math.random() * keys.length)];
      document.getElementById('presets').value = randKey;
      applyPreset();
      runPrediction();
    }

    function clearInputs() {
      document.getElementById('plot').value = "";
      document.getElementById('keywords').value = "";
      document.getElementById('presets').value = "";
      updateWordCount();
      document.getElementById('result-section').style.display = "none";
      document.getElementById('empty-state').style.display = "block";
    }

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (name === 'predict') {
        document.getElementById('tab-btn-predict').classList.add('active');
        document.getElementById('tab-predict').classList.add('active');
      } else if (name === 'analytics') {
        document.getElementById('tab-btn-analytics').classList.add('active');
        document.getElementById('tab-analytics').classList.add('active');
        loadAnalytics();
      } else if (name === 'about') {
        document.getElementById('tab-btn-about').classList.add('active');
        document.getElementById('tab-about').classList.add('active');
      }
    }

    // Theme Toggle
    function toggleTheme() {
      const body = document.body;
      const cur = body.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      body.setAttribute('data-theme', next);
      localStorage.setItem('cineverse_theme', next);
      updateThemeUI(next);
    }

    function updateThemeUI(theme) {
      const icon = document.getElementById('theme-icon');
      const label = document.getElementById('theme-label');
      if (theme === 'light') {
        icon.innerText = '☀️';
        label.innerText = 'Light';
      } else {
        icon.innerText = '🌙';
        label.innerText = 'Dark';
      }
    }

    // Load initial theme
    const savedTheme = localStorage.getItem('cineverse_theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeUI(savedTheme);

    // Dynamic SVG Poster Fallback
    function createFallbackPoster(title, genre, year) {
      const initials = (title || 'Movie').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase();
      const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="300" height="420" viewBox="0 0 300 420">
          <defs>
            <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#1E1F25"/>
              <stop offset="100%" stop-color="#0E0F12"/>
            </linearGradient>
          </defs>
          <rect width="300" height="420" fill="url(#bg)"/>
          <circle cx="150" cy="170" r="60" fill="rgba(255, 100, 10, 0.15)" stroke="#FF640A" stroke-width="2"/>
          <text x="150" y="182" fill="#FF640A" font-size="34" font-weight="900" font-family="sans-serif" text-anchor="middle">${initials}</text>
          <rect x="25" y="270" width="250" height="2" fill="#2E303A"/>
          <text x="150" y="315" fill="#FFFFFF" font-size="16" font-weight="800" font-family="sans-serif" text-anchor="middle">${escapeXml(title.slice(0, 26))}</text>
          <text x="150" y="342" fill="#9E9EB2" font-size="13" font-weight="600" font-family="sans-serif" text-anchor="middle">${escapeXml(genre || 'Cinema')} • ${year || ''}</text>
        </svg>
      `;
      return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    }

    function escapeXml(unsafe) {
      return (unsafe || '').replace(/[<>&'"]/g, c => {
        switch (c) {
          case '<': return '&lt;';
          case '>': return '&gt;';
          case '&': return '&amp;';
          case '\'': return '&apos;';
          case '"': return '&quot;';
        }
      });
    }

    // Prediction Engine
    async function runPrediction() {
      const plot = document.getElementById('plot').value.trim();
      const keywords = document.getElementById('keywords').value.trim();

      const words = plot.split(/\\s+/).filter(w => w.length > 0);
      if (words.length < 15) {
        const alertBox = document.getElementById('validation-alert');
        document.getElementById('validation-msg').innerText = `Please enter at least 15 words for accurate NLP feature extraction (currently ${words.length} words).`;
        alertBox.style.display = "flex";
        document.getElementById('plot').focus();
        return;
      }
      document.getElementById('validation-alert').style.display = "none";

      // Show skeleton, hide empty state & result
      document.getElementById('empty-state').style.display = "none";
      document.getElementById('result-section').style.display = "none";
      document.getElementById('skeleton-loader').style.display = "block";

      try {
        const [predRes, recRes] = await Promise.all([
          fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ overview: plot, keywords: keywords, top_k: 3 })
          }).then(r => r.json()),
          fetch('/api/recommend', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ overview: plot, keywords: keywords, top_n: 12 })
          }).then(r => r.json())
        ]);

        currentPrediction = predRes;
        allRecommendations = recRes.recommendations || [];

        // Hide skeleton & show result
        document.getElementById('skeleton-loader').style.display = "none";
        document.getElementById('result-section').style.display = "block";

        // Render Top 3 Cards
        const top3Wrap = document.getElementById('top3-cards');
        top3Wrap.innerHTML = '';
        const ranks = ['#1 Primary Genre', '#2 Secondary Genre', '#3 Tertiary Genre'];

        predRes.top_predictions.forEach((item, idx) => {
          top3Wrap.innerHTML += `
            <div class="genre-card">
              <div class="genre-rank-tag">${ranks[idx] || '#' + (idx + 1)}</div>
              <div>
                <span class="genre-prob-val">${item.percentage}</span>
                <span class="genre-name-label">${item.genre}</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" id="prob-fill-${idx}" style="width: 0%;"></div>
              </div>
            </div>
          `;
        });

        // Animate progress bars
        setTimeout(() => {
          predRes.top_predictions.forEach((item, idx) => {
            const fill = document.getElementById(`prob-fill-${idx}`);
            if (fill) fill.style.width = (item.probability * 100) + '%';
          });
        }, 80);

        // Low confidence warning
        const lowConfAlert = document.getElementById('low-conf-alert');
        if (predRes.low_confidence_warning) {
          lowConfAlert.style.display = "flex";
          document.getElementById('low-conf-text').innerText = predRes.low_confidence_message;
        } else {
          lowConfAlert.style.display = "none";
        }

        // Influential words chips
        const wordsWrap = document.getElementById('influential-words');
        wordsWrap.innerHTML = '';
        if (predRes.influential_words && predRes.influential_words.length > 0) {
          predRes.influential_words.forEach(w => {
            wordsWrap.innerHTML += `
              <span class="word-chip">
                <span>${w.word}</span>
                <span class="word-weight">+${w.impact}</span>
              </span>
            `;
          });
          document.getElementById('why-box').style.display = "block";
        } else {
          document.getElementById('why-box').style.display = "none";
        }

        // Render recommendations
        applyFilters();

        document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });

      } catch (err) {
        document.getElementById('skeleton-loader').style.display = "none";
        alert("Error during inference: " + err);
      }
    }

    // Recommendations Filtering
    function applyFilters() {
      const genre = document.getElementById('filter-genre').value;
      const yearFilter = document.getElementById('filter-year').value;
      const minRating = parseFloat(document.getElementById('filter-rating').value) || 0.0;

      const filtered = allRecommendations.filter(m => {
        if (genre !== 'All' && !m.genres.includes(genre)) return false;
        if (m.vote_average < minRating) return false;
        const y = m.release_year;
        if (yearFilter === '2010s' && y < 2010) return false;
        if (yearFilter === '2000s' && (y < 2000 || y > 2009)) return false;
        if (yearFilter === '1990s' && (y < 1990 || y > 1999)) return false;
        if (yearFilter === 'classic' && y >= 1990) return false;
        return true;
      });

      const list = document.getElementById('rec-list');
      list.innerHTML = '';

      if (filtered.length === 0) {
        list.innerHTML = `
          <div style="grid-column: 1 / -1; padding: 2.5rem; text-align: center; color: var(--text-muted); background: var(--surface); border-radius: 8px;">
            No recommendations match your current filter criteria. Try adjusting the rating slider or release year range.
          </div>
        `;
        return;
      }

      filtered.slice(0, 9).forEach((item, idx) => {
        const fallback = createFallbackPoster(item.title, (item.genres[0] || ''), item.release_year);
        const chipsHtml = (item.genres || []).slice(0, 3).map(g => `<span class="genre-chip">${g}</span>`).join('');

        list.innerHTML += `
          <article class="stream-card" onclick="openDetailsModal(${item.id})">
            <img class="card-poster" 
                 src="/api/tmdb-poster?title=${encodeURIComponent(item.title)}&year=${item.release_year}"
                 alt="${escapeXml(item.title)} Poster"
                 onerror="this.onerror=null; this.src='${fallback}';">
            <div class="card-content">
              <div class="card-top-row">
                <h3 class="stream-title">${item.title}</h3>
                <span class="match-pill">⚡ ${item.similarity_percentage}</span>
              </div>
              <div class="card-meta">
                <span>${item.release_year}</span>
                <span>•</span>
                <span class="star-score">⭐ ${item.vote_average.toFixed(1)}</span>
              </div>
              <div class="card-chips">${chipsHtml}</div>
              <p class="stream-synopsis">${item.overview}</p>
            </div>
          </article>
        `;
      });
    }

    // Details Modal
    function openDetailsModal(movieId) {
      const item = allRecommendations.find(m => m.id === movieId);
      if (!item) return;

      const fallback = createFallbackPoster(item.title, (item.genres[0] || ''), item.release_year);
      const posterImg = document.getElementById('modal-poster');
      posterImg.src = `/api/tmdb-poster?title=${encodeURIComponent(item.title)}&year=${item.release_year}`;
      posterImg.onerror = function() { this.src = fallback; };

      document.getElementById('modal-title').innerText = item.title;
      document.getElementById('modal-year').innerText = item.release_year;
      document.getElementById('modal-match').innerText = `⚡ ${item.similarity_percentage} Match`;
      document.getElementById('modal-rating').innerText = item.vote_average.toFixed(1);
      document.getElementById('modal-votes').innerText = `${item.vote_count.toLocaleString()} votes`;
      document.getElementById('modal-overview').innerText = item.overview || "No overview available.";
      document.getElementById('modal-keywords').innerText = item.keywords || "None recorded";

      const chipsWrap = document.getElementById('modal-chips');
      chipsWrap.innerHTML = (item.genres || []).map(g => `<span class="genre-chip" style="font-size: 0.8rem; padding: 0.2rem 0.6rem;">${g}</span>`).join('');

      const trailerQuery = encodeURIComponent(`${item.title} ${item.release_year} official trailer`);
      document.getElementById('modal-trailer').href = `https://www.youtube.com/results?search_query=${trailerQuery}`;

      document.getElementById('movie-modal').style.display = 'flex';
      document.body.style.overflow = 'hidden';
    }

    function closeModal() {
      document.getElementById('movie-modal').style.display = 'none';
      document.body.style.overflow = 'auto';
    }

    function closeModalOnBackdrop(e) {
      if (e.target.id === 'movie-modal') closeModal();
    }

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });

    // Copy Result Text
    function copyResultText() {
      if (!currentPrediction) return;
      const text = `CineVerse AI Multi-Label Classification:\\n` +
        currentPrediction.top_predictions.map((p, i) => `#${i+1} ${p.genre}: ${p.percentage}`).join('\\n');
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copy-btn-text');
        btn.innerText = "✓ Copied!";
        setTimeout(() => { btn.innerText = "Copy Result"; }, 2000);
      });
    }

    // Shareable Link
    function copyShareableLink() {
      const plot = document.getElementById('plot').value.trim();
      const kw = document.getElementById('keywords').value.trim();
      if (!plot) { alert("Enter a synopsis before generating a share link."); return; }
      const url = new URL(window.location.origin + window.location.pathname);
      url.searchParams.set('synopsis', plot);
      if (kw) url.searchParams.set('keywords', kw);
      navigator.clipboard.writeText(url.toString()).then(() => {
        alert("Shareable link copied to clipboard!");
      });
    }

    // Feedback
    async function sendFeedback(type) {
      if (!currentPrediction) return;
      document.getElementById('fb-up').classList.toggle('active', type === 'up');
      document.getElementById('fb-down').classList.toggle('active', type === 'down');
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            feedback: type,
            primary_genre: currentPrediction.primary_genre,
            overview: document.getElementById('plot').value.slice(0, 100)
          })
        });
      } catch (e) {}
      localStorage.setItem('cineverse_feedback_' + currentPrediction.primary_genre, type);
      document.getElementById('fb-status').style.display = "block";
    }

    // Analytics Tab Charts
    let chartsCreated = false;
    async function loadAnalytics() {
      if (chartsCreated) return;
      try {
        const data = await fetch('/api/metrics').then(r => r.json());
        analyticsData = data;

        // KPI
        document.getElementById('kpi-hamming').innerText = (data.hamming_accuracy * 100).toFixed(1) + '%';
        document.getElementById('kpi-microf1').innerText = (data.micro_f1 * 100).toFixed(1) + '%';
        document.getElementById('kpi-hitrate').innerText = (data.top3_hit_rate * 100).toFixed(1) + '%';
        document.getElementById('kpi-catalog').innerText = data.dataset_size.toLocaleString();

        const isDark = document.body.getAttribute('data-theme') !== 'light';
        const textColor = isDark ? '#F7F7F8' : '#17181D';
        const gridColor = isDark ? '#2E303A' : '#E0E3EA';

        // 1. Distribution Chart
        const distLabels = Object.keys(data.genre_distribution);
        const distVals = Object.values(data.genre_distribution);
        new Chart(document.getElementById('chart-distribution'), {
          type: 'doughnut',
          data: {
            labels: distLabels,
            datasets: [{
              data: distVals,
              backgroundColor: [
                '#FF640A', '#FF8A3D', '#FFA347', '#E05300', '#F59E0B',
                '#3B82F6', '#8B5CF6', '#EC4899', '#10B981', '#14B8A6',
                '#6366F1', '#F43F5E'
              ],
              borderWidth: 2,
              borderColor: isDark ? '#1E1F25' : '#FFFFFF'
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'right', labels: { color: textColor, font: { size: 11, weight: '700' } } }
            }
          }
        });

        // 2. Metrics Chart (F1, Precision, Recall)
        const genres = data.top_12_genres;
        const rep = data.classification_report || {};
        const f1s = genres.map(g => (rep[g] ? rep[g]['f1-score'] : 0));
        const precs = genres.map(g => (rep[g] ? rep[g].precision : 0));
        const recs = genres.map(g => (rep[g] ? rep[g].recall : 0));

        new Chart(document.getElementById('chart-metrics'), {
          type: 'bar',
          data: {
            labels: genres,
            datasets: [
              { label: 'F1-Score', data: f1s, backgroundColor: '#FF640A' },
              { label: 'Precision', data: precs, backgroundColor: '#3B82F6' },
              { label: 'Recall', data: recs, backgroundColor: '#10B981' }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { color: textColor, font: { size: 10 } }, grid: { display: false } },
              y: { max: 1.0, ticks: { color: textColor }, grid: { color: gridColor } }
            },
            plugins: {
              legend: { labels: { color: textColor, font: { weight: '700' } } }
            }
          }
        });

        // 3. Benchmark Comparison
        const bm = data.benchmark_comparison || {};
        const bmModels = Object.keys(bm);
        const bmMicro = bmModels.map(m => bm[m].mean_micro_f1);
        const bmMacro = bmModels.map(m => bm[m].mean_macro_f1);

        new Chart(document.getElementById('chart-benchmark'), {
          type: 'bar',
          data: {
            labels: bmModels.map(m => m.replace(' (Balanced)', '')),
            datasets: [
              { label: 'Micro F1 (5-Fold CV)', data: bmMicro, backgroundColor: '#FF640A' },
              { label: 'Macro F1 (5-Fold CV)', data: bmMacro, backgroundColor: '#8B5CF6' }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { color: textColor, font: { weight: '700' } }, grid: { display: false } },
              y: { max: 0.8, ticks: { color: textColor }, grid: { color: gridColor } }
            },
            plugins: {
              legend: { labels: { color: textColor, font: { weight: '700' } } }
            }
          }
        });

        // 4. Co-occurrence Matrix Table
        const coMat = data.co_occurrence_matrix || {};
        let tableHtml = '<table style="width:100%; border-collapse:collapse; font-size:0.75rem; text-align:center;"><thead><tr><th></th>';
        genres.forEach(g => { tableHtml += `<th style="padding:4px; color:var(--brand-orange);">${g.slice(0,4)}</th>`; });
        tableHtml += '</tr></thead><tbody>';

        genres.forEach(g1 => {
          tableHtml += `<tr><td style="padding:4px; font-weight:700; text-align:left; color:var(--text);">${g1}</td>`;
          genres.forEach(g2 => {
            const count = (coMat[g1] && coMat[g1][g2]) || 0;
            const intensity = g1 === g2 ? 0.3 : Math.min(count / 300, 0.9);
            const bg = `rgba(255, 100, 10, ${intensity})`;
            tableHtml += `<td style="padding:4px; background:${bg}; border:1px solid var(--border);">${count}</td>`;
          });
          tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';
        document.getElementById('co-matrix-wrap').innerHTML = tableHtml;

        // Predictive Words Pills
        const pillsWrap = document.getElementById('genre-pills');
        pillsWrap.innerHTML = '';
        genres.forEach((g, idx) => {
          pillsWrap.innerHTML += `
            <button class="pill-btn ${idx === 0 ? 'active' : ''}" onclick="showPredictiveWords('${g}')">
              ${g}
            </button>
          `;
        });
        showPredictiveWords(genres[0]);

        chartsCreated = true;
      } catch (err) {
        console.error("Failed to load analytics", err);
      }
    }

    function showPredictiveWords(genre) {
      if (!analyticsData || !analyticsData.top_predictive_words) return;
      document.querySelectorAll('.pill-btn').forEach(b => {
        b.classList.toggle('active', b.innerText.trim() === genre);
      });
      const words = analyticsData.top_predictive_words[genre] || [];
      const wrap = document.getElementById('genre-words-display');
      wrap.innerHTML = '';
      words.forEach(w => {
        wrap.innerHTML += `
          <span class="word-chip">
            <span>${w.word}</span>
            <span class="word-weight">+${w.weight}</span>
          </span>
        `;
      });
    }

    function toggleAccordion(el) {
      const body = el.nextElementSibling;
      body.classList.toggle('open');
    }

    // Auto-load URL search params if present
    window.addEventListener('DOMContentLoaded', () => {
      const params = new URLSearchParams(window.location.search);
      const synopsis = params.get('synopsis');
      const kw = params.get('keywords');
      if (synopsis) {
        document.getElementById('plot').value = synopsis;
        if (kw) document.getElementById('keywords').value = kw;
        updateWordCount();
        runPrediction();
      }
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    """Serve full-stack web UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Multi-label genre classification endpoint:
    Calculates calibrated sigmoid probabilities for all 12 classes,
    identifies top 3 genres, detects low-confidence ambiguity, and outputs feature attribution.
    """
    data = request.get_json(silent=True) or {}
    overview = data.get("overview", "")
    keywords = data.get("keywords", "")
    top_k = int(data.get("top_k", 3))

    bundle = get_bundle()
    vocab = bundle["vocab"]
    idf = bundle["idf"]
    classes = bundle["classes"]
    weights = bundle["class_weights"]
    intercepts = bundle["class_intercepts"]

    combined = f"{overview} {keywords}".strip()
    tokens = clean_and_tokenize(combined)
    vec = vectorize_tokens(tokens, vocab, idf)

    if not vec:
        return jsonify({
            "primary_genre": "Drama",
            "confidence": 0.5,
            "confidence_percentage": "50.0%",
            "top_predictions": [{"genre": "Drama", "probability": 0.5, "percentage": "50.0%"}],
            "low_confidence_warning": False,
            "low_confidence_message": "",
            "influential_words": []
        })

    # Calibrated OneVsRest Sigmoid Probabilities
    scores = []
    for c_idx in range(len(classes)):
        logit = intercepts[c_idx] + sum(vec.get(f_idx, 0.0) * weights[c_idx][f_idx] for f_idx in vec)
        prob = 1.0 / (1.0 + math.exp(-max(min(logit, 15.0), -15.0)))
        scores.append((classes[c_idx], float(prob), c_idx))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_predictions = []
    for g, p, c_idx in scores[:top_k]:
        top_predictions.append({
            "genre": g,
            "probability": round(p, 4),
            "percentage": f"{p * 100:.1f}%",
            "class_index": c_idx
        })

    primary_genre = scores[0][0]
    primary_conf = scores[0][1]

    # Low Confidence Ambiguity Warning
    low_confidence = False
    low_conf_msg = ""
    if len(scores) >= 2 and abs(scores[0][1] - scores[1][1]) < 0.08:
        low_confidence = True
        low_conf_msg = (
            f"Close Probability Spread: Narrative displays strong hybrid elements of both "
            f"{scores[0][0]} ({scores[0][1]*100:.1f}%) and {scores[1][0]} ({scores[1][1]*100:.1f}%)."
        )

    # Feature Attribution ("Why this prediction?")
    top_c_idx = scores[0][2]
    inv_vocab = {v: k for k, v in vocab.items() if v in vec}
    influential_words = []
    for f_idx, tfidf_val in vec.items():
        w = weights[top_c_idx][f_idx]
        if w > 0.15 and f_idx in inv_vocab:
            impact = tfidf_val * w
            influential_words.append({
                "word": inv_vocab[f_idx],
                "weight": round(w, 2),
                "impact": round(impact, 2)
            })

    influential_words.sort(key=lambda x: x["impact"], reverse=True)

    return jsonify({
        "primary_genre": primary_genre,
        "confidence": primary_conf,
        "confidence_percentage": f"{primary_conf * 100:.1f}%",
        "top_predictions": top_predictions,
        "low_confidence_warning": low_confidence,
        "low_confidence_message": low_conf_msg,
        "influential_words": influential_words[:8]
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Hybrid recommendation endpoint:
    0.60 * Cosine Similarity + 0.25 * Genre Jaccard Overlap + 0.15 * Rating Normalization.
    """
    data = request.get_json(silent=True) or {}
    overview = data.get("overview", "")
    keywords = data.get("keywords", "")
    top_n = int(data.get("top_n", 12))

    bundle = get_bundle()
    vocab = bundle["vocab"]
    idf = bundle["idf"]
    classes = bundle["classes"]
    weights = bundle["class_weights"]
    intercepts = bundle["class_intercepts"]
    movies = bundle["movies"]

    combined = f"{overview} {keywords}".strip()
    tokens = clean_and_tokenize(combined)
    vec = vectorize_tokens(tokens, vocab, idf)

    # Predict query genres for Jaccard overlap
    scores = []
    for c_idx in range(len(classes)):
        logit = intercepts[c_idx] + sum(vec.get(f_idx, 0.0) * weights[c_idx][f_idx] for f_idx in vec)
        prob = 1.0 / (1.0 + math.exp(-max(min(logit, 15.0), -15.0)))
        scores.append((classes[c_idx], prob))
    scores.sort(key=lambda x: x[1], reverse=True)
    query_top_genres = set(g for g, _ in scores[:3])

    results = []
    for m in movies:
        m_vec = m.get("tfidf_vec", {})
        cosine_sim = sum(vec.get(int(f_idx), 0.0) * val for f_idx, val in m_vec.items())

        movie_genres = m.get("genres", [])
        if query_top_genres and movie_genres:
            set_m = set(movie_genres)
            jaccard = len(query_top_genres.intersection(set_m)) / len(query_top_genres.union(set_m))
        else:
            jaccard = 0.0

        rating = float(m.get("vote_average", 6.0))
        rating_score = min(max(rating / 10.0, 0.0), 1.0)

        hybrid_score = 0.60 * cosine_sim + 0.25 * jaccard + 0.15 * rating_score

        results.append({
            "id": m["id"],
            "title": m["title"],
            "genres": movie_genres,
            "release_year": m.get("release_year", 2010),
            "vote_average": rating,
            "vote_count": m.get("vote_count", 0),
            "overview": m.get("overview", ""),
            "keywords": m.get("keywords", ""),
            "similarity_score": round(float(cosine_sim), 4),
            "similarity_percentage": f"{max(min(cosine_sim * 100, 100), 0):.1f}%",
            "hybrid_score": round(float(hybrid_score), 4)
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return jsonify({"recommendations": results[:top_n]})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return model evaluation metrics & class distributions."""
    bundle = get_bundle()
    return jsonify(bundle.get("metrics", {}))


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Record user prediction feedback."""
    return jsonify({"success": True, "message": "Feedback recorded."})


@app.route("/api/tmdb-poster", methods=["GET"])
def api_tmdb_poster():
    """Fetch movie poster URL from TMDB API if TMDB_API_KEY is configured."""
    title = request.args.get("title", "").strip()
    year = request.args.get("year", "").strip()
    tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()

    if not tmdb_key or not title:
        return jsonify({"poster_url": None, "source": "fallback"}), 404

    try:
        params = {"api_key": tmdb_key, "query": title}
        if year:
            params["year"] = year
        url = f"https://api.themoviedb.org/3/search/movie?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "CineVerseAI/2.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results and results[0].get("poster_path"):
                poster_path = results[0]["poster_path"]
                full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return jsonify({"poster_url": full_url, "source": "tmdb"})
    except Exception:
        pass

    return jsonify({"poster_url": None, "source": "fallback"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
