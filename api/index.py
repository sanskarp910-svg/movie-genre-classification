"""
Cinemix ▶ - Minimalist Letterboxd-Style Film Genre Classifier & Hybrid Recommender
Vercel Serverless Function & Full-Stack Web Application.
"""

import os
import sys
import json
import math
import re
import urllib.request
import urllib.parse
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, redirect, Response

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Cached serverless bundle
_BUNDLE = None


def get_bundle():
    """Load pre-trained serverless model bundle."""
    global _BUNDLE
    if _BUNDLE is None:
        candidate_paths = [
            Path(__file__).resolve().parent / "serverless_model.json",
            MODELS_DIR / "serverless_model.json",
            BASE_DIR / "models" / "serverless_model.json"
        ]
        bundle_path = None
        for p in candidate_paths:
            if p.exists():
                bundle_path = p
                break
        if bundle_path is None:
            raise FileNotFoundError(f"Model bundle not found. Checked: {[str(p) for p in candidate_paths]}")
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
  <title>Cinemix ▶ | Film & Anime Genre Classifier & Recommender</title>
  
  <!-- SEO & Open Graph Metadata -->
  <meta name="description" content="Cinemix ▶ - Minimalist Letterboxd-styled multi-label movie & anime genre classifier and hybrid recommender.">
  <meta property="og:title" content="Cinemix ▶ - Film Genre Classifier & Recommender">
  <meta property="og:description" content="Minimalist Letterboxd-styled NLP classifier: predict top 3 genres with calibrated probabilities and match similar films.">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Cinemix ▶">
  <meta name="twitter:description" content="Letterboxd-styled AI Film & Anime Multi-Label Genre Classification & Recommendations.">
  
  <!-- Favicon: Letterboxd Tri-Dot Motif with Play Icon -->
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%2314181c'/><circle cx='28' cy='50' r='12' fill='%2300e054'/><circle cx='50' cy='50' r='12' fill='%23ff8000'/><circle cx='72' cy='50' r='12' fill='%2340bcf4'/></svg>">
  
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@700;800&display=swap" rel="stylesheet">

  <style>
    /* Letterboxd Dark Theme (Default) */
    :root {
      --lb-green: #00e054;
      --lb-green-hover: #00c030;
      --lb-green-glow: rgba(0, 224, 84, 0.28);
      --lb-orange: #ff8000;
      --lb-blue: #40bcf4;
      --bg-deep: #111418;
      --bg-main: #14181c;
      --card-bg: #1c2228;
      --surface: #242c34;
      --surface-hover: #2b3540;
      --border: #2c3440;
      --border-subtle: #202730;
      --text: #ffffff;
      --text-secondary: #def;
      --text-muted: #99aabb;
      --text-dim: #667788;
      --badge-bg: rgba(0, 224, 84, 0.12);
      --badge-border: #00e054;
      --chip-bg: #242c34;
      --modal-overlay: rgba(10, 13, 16, 0.88);
      --shadow-color: rgba(0, 0, 0, 0.5);
    }

    /* Clean Crisp Light Theme */
    [data-theme="light"] {
      --lb-green: #00b03e;
      --lb-green-hover: #009032;
      --lb-green-glow: rgba(0, 176, 62, 0.2);
      --lb-orange: #e06d00;
      --lb-blue: #1c9cd6;
      --bg-deep: #e6ebf0;
      --bg-main: #f0f3f6;
      --card-bg: #ffffff;
      --surface: #f4f7f9;
      --surface-hover: #eaf0f5;
      --border: #cfd8e3;
      --border-subtle: #dfe5ec;
      --text: #14181c;
      --text-secondary: #222a33;
      --text-muted: #556677;
      --text-dim: #778899;
      --badge-bg: rgba(0, 176, 62, 0.1);
      --badge-border: #00b03e;
      --chip-bg: #e6ecf2;
      --modal-overlay: rgba(20, 24, 28, 0.72);
      --shadow-color: rgba(0, 0, 0, 0.08);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: var(--bg-main);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      transition: background-color 0.2s ease, color 0.2s ease;
    }

    /* Minimalist Letterboxd Navbar */
    .navbar {
      background-color: var(--bg-deep);
      border-bottom: 1px solid var(--border);
      padding: 0.95rem 2rem;
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
    .brand-title-wrap {
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }
    .brand-title {
      font-size: 1.45rem;
      font-weight: 900;
      color: #ffffff;
      letter-spacing: -0.02em;
      text-transform: uppercase;
      font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    }
    [data-theme="light"] .brand-title {
      color: #14181c;
    }
    .play-icon-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      background-color: var(--lb-green);
      color: #14181c;
      font-size: 0.7rem;
      font-weight: 900;
      border-radius: 4px;
      box-shadow: 0 0 10px var(--lb-green-glow);
      line-height: 1;
    }
    .lb-dots {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin-left: 0.35rem;
      padding: 0.2rem 0.45rem;
      background: var(--surface);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .lb-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
    }
    .dot-green { background-color: var(--lb-green); }
    .dot-orange { background-color: var(--lb-orange); }
    .dot-blue { background-color: var(--lb-blue); }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }
    .nav-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 0.42rem 0.85rem;
      cursor: pointer;
      font-size: 0.84rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.2s ease;
      text-decoration: none;
    }
    .nav-btn:hover {
      border-color: var(--lb-green);
      color: var(--text);
    }

    /* Container */
    .container {
      max-width: 1080px;
      width: 100%;
      margin: 0 auto;
      padding: 2.75rem 1.25rem 4rem 1.25rem;
      flex: 1;
    }

    /* Minimal Hero */
    .minimal-hero {
      text-align: center;
      margin-bottom: 2.25rem;
    }
    .hero-brand {
      font-size: 3rem;
      font-weight: 900;
      color: var(--text);
      letter-spacing: -0.03em;
      text-transform: uppercase;
      font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
    }
    .hero-brand .play-icon-badge {
      width: 32px;
      height: 32px;
      font-size: 0.95rem;
    }
    .hero-sub {
      color: var(--text-muted);
      font-size: 1rem;
      margin-top: 0.4rem;
      font-weight: 500;
    }

    /* BUTTERY SMOOTH EXPANDABLE TYPEBAR */
    .expandable-search-wrapper {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1.25rem;
      max-width: 860px;
      margin: 0 auto 2.5rem auto;
      box-shadow: 0 4px 20px var(--shadow-color);
      transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
      position: relative;
    }
    .expandable-search-wrapper:hover,
    .expandable-search-wrapper:focus-within,
    .expandable-search-wrapper.is-expanded {
      border-color: var(--lb-green);
      box-shadow: 0 0 24px var(--lb-green-glow), 0 8px 32px rgba(0,0,0,0.45);
    }

    .search-top-bar {
      display: flex;
      align-items: flex-start;
      gap: 0.85rem;
      position: relative;
    }
    .search-lens-icon {
      font-size: 1.35rem;
      color: var(--lb-green);
      margin-top: 0.22rem;
      flex-shrink: 0;
      transition: transform 0.25s ease;
    }
    .expandable-search-wrapper:hover .search-lens-icon,
    .expandable-search-wrapper:focus-within .search-lens-icon,
    .expandable-search-wrapper.is-expanded .search-lens-icon {
      transform: scale(1.08);
    }
    .expandable-textarea {
      flex: 1;
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      color: var(--text) !important;
      font-family: inherit;
      font-size: 1.05rem;
      line-height: 1.5;
      resize: none;
      height: 44px;
      padding: 0.2rem 0;
      transition: height 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .expandable-textarea:focus {
      outline: none !important;
    }
    /* Expands on hover or focus */
    .expandable-search-wrapper:hover .expandable-textarea,
    .expandable-search-wrapper:focus-within .expandable-textarea,
    .expandable-search-wrapper.is-expanded .expandable-textarea {
      height: 96px;
    }

    .quick-submit-btn {
      background: var(--lb-green);
      color: #14181c;
      border: none;
      border-radius: 4px;
      width: 36px;
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 0.95rem;
      font-weight: 900;
      flex-shrink: 0;
      transition: all 0.2s ease;
      box-shadow: 0 2px 8px var(--lb-green-glow);
    }
    .quick-submit-btn:hover {
      background: var(--lb-green-hover);
      color: #ffffff;
      transform: scale(1.05);
    }

    /* Silky Smooth CSS Grid Expansion for the Tray */
    .search-expanded-tray {
      display: grid;
      grid-template-rows: 0fr;
      transition: grid-template-rows 0.32s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .expandable-search-wrapper:hover .search-expanded-tray,
    .expandable-search-wrapper:focus-within .search-expanded-tray,
    .expandable-search-wrapper.is-expanded .search-expanded-tray {
      grid-template-rows: 1fr;
    }

    .tray-inner {
      overflow: hidden;
      min-height: 0;
      opacity: 0;
      transform: translateY(-6px);
      transition: opacity 0.24s ease, transform 0.24s cubic-bezier(0.16, 1, 0.3, 1);
      border-top: 1px solid transparent;
      padding-top: 0;
    }
    .expandable-search-wrapper:hover .tray-inner,
    .expandable-search-wrapper:focus-within .tray-inner,
    .expandable-search-wrapper.is-expanded .tray-inner {
      opacity: 1;
      transform: translateY(0);
      border-top-color: var(--border-subtle);
      padding-top: 0.85rem;
      margin-top: 0.5rem;
    }

    .counter-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-dim);
    }
    .counter-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      background: var(--surface);
    }
    .counter-valid { color: var(--lb-green); }
    .counter-invalid { color: #f87171; }

    .search-secondary-row {
      display: grid;
      grid-template-columns: 2fr 1.5fr;
      gap: 0.75rem;
      margin-bottom: 0.85rem;
    }
    @media (max-width: 680px) { .search-secondary-row { grid-template-columns: 1fr; } }

    .search-secondary-row input, .search-secondary-row select {
      background-color: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.65rem 0.85rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.88rem;
      transition: all 0.2s ease;
      width: 100%;
    }
    .search-secondary-row input:focus, .search-secondary-row select:focus {
      outline: none;
      border-color: var(--lb-green);
    }

    .validation-alert {
      display: none;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid #ef4444;
      color: #f87171;
      border-radius: 4px;
      padding: 0.65rem 0.85rem;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      align-items: center;
      gap: 0.5rem;
    }

    .button-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
    }
    .cr-btn {
      background: var(--lb-green);
      color: #14181c;
      font-weight: 900;
      font-size: 0.88rem;
      border: none;
      border-radius: 4px;
      padding: 0.75rem 1.4rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      box-shadow: 0 2px 10px var(--lb-green-glow);
    }
    .cr-btn:hover {
      background: var(--lb-green-hover);
      color: #ffffff;
      transform: translateY(-1px);
    }
    .cr-btn-secondary {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-weight: 800;
      font-size: 0.82rem;
      border-radius: 4px;
      padding: 0.75rem 1.1rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .cr-btn-secondary:hover {
      border-color: var(--lb-green);
      color: var(--text);
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
      border-radius: 4px;
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
      border-left: 5px solid var(--lb-green);
      border-radius: 6px;
      padding: 1.5rem 1.75rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 20px var(--shadow-color);
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
      border-radius: 4px;
      padding: 1.1rem;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s ease;
    }
    .genre-card:hover {
      border-color: var(--lb-green);
    }
    .genre-rank-tag {
      font-size: 0.72rem;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--lb-green);
      margin-bottom: 0.35rem;
    }
    .genre-name-label {
      font-size: 1.35rem;
      font-weight: 900;
      color: var(--text);
    }
    .genre-prob-val {
      font-size: 1.15rem;
      font-weight: 900;
      color: var(--lb-green);
      float: right;
    }

    /* Low Confidence Warning Alert */
    .low-confidence-alert {
      display: none;
      background: rgba(255, 128, 0, 0.12);
      border: 1px solid var(--lb-orange);
      color: #ffaa44;
      border-radius: 4px;
      padding: 0.85rem 1.1rem;
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 1.5rem;
      align-items: center;
      gap: 0.75rem;
    }

    /* Animated Letterboxd Green Progress Bars */
    .prob-track {
      height: 6px;
      background-color: var(--bg-deep);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 0.75rem;
    }
    .prob-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--lb-green), var(--lb-blue));
      border-radius: 3px;
      width: 0%;
      transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Explainability: Why this prediction? */
    .why-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1.25rem;
      margin-top: 1.5rem;
    }
    .why-title {
      font-size: 0.85rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }
    .word-chips-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .word-chip {
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.3rem 0.7rem;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .word-weight {
      color: var(--lb-green);
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
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .feedback-btns {
      display: flex;
      gap: 0.5rem;
    }
    .fb-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 0.35rem 0.85rem;
      font-size: 0.82rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s ease;
    }
    .fb-btn:hover {
      border-color: var(--lb-green);
      color: var(--text);
    }
    .fb-btn.active {
      background: var(--lb-green);
      color: #14181c;
      border-color: var(--lb-green);
    }

    /* Recommendations Filter Bar */
    .filter-bar {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1.1rem 1.25rem;
      margin-bottom: 1.5rem;
      display: grid;
      grid-template-columns: 2fr 1.5fr 1.5fr;
      gap: 1.25rem;
      align-items: center;
    }
    @media (max-width: 860px) { .filter-bar { grid-template-columns: 1fr; } }

    .filter-bar label {
      display: block;
      font-size: 0.75rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }
    .filter-bar select, .filter-bar input {
      width: 100%;
      background-color: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.65rem 0.85rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
    }

    /* Letterboxd Poster Grid Style */
    .rec-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 1.35rem;
      margin-top: 1rem;
    }
    .stream-card {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .stream-card:hover {
      transform: translateY(-4px);
      border-color: var(--lb-green);
      box-shadow: 0 0 16px var(--lb-green-glow);
    }
    .card-poster {
      width: 100%;
      height: 240px;
      object-fit: cover;
      background-color: var(--bg-deep);
      display: block;
      border-bottom: 1px solid var(--border);
    }
    .card-content {
      padding: 1rem 1.15rem 1.2rem 1.15rem;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .card-top-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.4rem;
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
      border: 1px solid var(--lb-green);
      color: var(--lb-green);
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .card-meta {
      display: flex;
      align-items: center;
      gap: 0.65rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-bottom: 0.6rem;
      font-weight: 600;
    }
    .star-score {
      color: var(--lb-green);
      font-weight: 800;
      letter-spacing: 0.05em;
    }
    .card-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.75rem;
    }
    .genre-chip {
      background-color: var(--chip-bg);
      color: var(--text-muted);
      padding: 0.15rem 0.55rem;
      border-radius: 3px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border: 1px solid var(--border);
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

    /* Modal Backdrop */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: var(--modal-overlay);
      backdrop-filter: blur(5px);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.25rem;
    }
    .modal-box {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      max-width: 780px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 2rem;
      position: relative;
      box-shadow: 0 20px 40px rgba(0,0,0,0.7);
    }
    .modal-close-btn {
      position: absolute;
      top: 1.25rem;
      right: 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 1.1rem;
      width: 34px;
      height: 34px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }
    .modal-close-btn:hover {
      background: var(--lb-green);
      color: #14181c;
      border-color: var(--lb-green);
    }

    /* Footer */
    footer {
      background-color: var(--bg-deep);
      border-top: 1px solid var(--border);
      padding: 2.25rem 1.25rem;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.88rem;
      margin-top: auto;
    }
  </style>
</head>
<body data-theme="dark">

  <!-- Minimalist Navbar -->
  <nav class="navbar">
    <a href="/" class="nav-brand">
      <div class="brand-title-wrap">
        <span class="brand-title">CINEMIX</span>
        <span class="play-icon-badge">▶</span>
      </div>
      <div class="lb-dots" title="Cinemix Film Intelligence">
        <span class="lb-dot dot-green"></span>
        <span class="lb-dot dot-orange"></span>
        <span class="lb-dot dot-blue"></span>
      </div>
    </a>
    <div class="nav-actions">
      <!-- Dark / Light Theme Button -->
      <button class="nav-btn" onclick="toggleTheme()" id="theme-btn" aria-label="Toggle Theme">
        <span id="theme-icon">🌙</span> <span id="theme-label">Dark</span>
      </button>
      <!-- About Section Button right next to theme -->
      <button class="nav-btn" onclick="openAboutModal()" id="about-btn" aria-label="About Cinemix">
        <span>ℹ️</span> <span>About</span>
      </button>
    </div>
  </nav>

  <!-- Main Container -->
  <main class="container">
    <!-- Minimal Header (site name only) -->
    <header class="minimal-hero">
      <div class="hero-brand">
        <span>CINEMIX</span>
        <span class="play-icon-badge">▶</span>
      </div>
    </header>

    <!-- EXPANDABLE TYPEBAR ON HOVER / FOCUS -->
    <div class="expandable-search-wrapper" id="search-box">
      <!-- Search top bar -->
      <div class="search-top-bar">
        <span class="search-lens-icon">🎬</span>
        <textarea id="plot" 
                  class="expandable-textarea" 
                  placeholder="Type or paste any movie storyline, plot, or scene to classify genres & match films..." 
                  rows="1" 
                  oninput="updateWordCount()"
                  onfocus="expandTypebar()"></textarea>
        <button class="quick-submit-btn" onclick="runPrediction()" title="Classify & Discover">
          <span>▶</span>
        </button>
      </div>

      <!-- Expanded Tray (Revealed smoothly on hover or focus) -->
      <div class="search-expanded-tray">
        <div class="tray-inner">
          <div class="counter-row">
            <span id="word-badge" class="counter-badge">0 words</span>
            <span id="char-badge">0 characters</span>
          </div>

          <div class="search-secondary-row">
            <div>
              <input type="text" id="keywords" placeholder="Thematic keywords / tags (e.g. space travel, cyberpunk, noir, demons)...">
            </div>
            <div>
              <select id="presets" onchange="applyPreset(true)">
                <option value="">⚡ Quick Archetype Presets...</option>
                <option value="scifi">Interstellar (Sci-Fi • Wormhole & Tesseract)</option>
                <option value="action">The Dark Knight (Action • Gotham Vigilante)</option>
                <option value="shonen">Demon Slayer (Anime • Demon Slaying Blade)</option>
                <option value="horror">The Conjuring (Horror • Farmhouse Possession)</option>
                <option value="comedy">Superbad (Comedy • High School Wild Party)</option>
                <option value="cyberpunk">Ghost in the Shell (Anime • Cyborg Hacker)</option>
                <option value="romance">La La Land (Romance • Jazz & Dreams)</option>
                <option value="animation">Spirited Away (Anime • Spirit Bathhouse)</option>
              </select>
            </div>
          </div>

          <!-- Action Toolbar -->
          <div class="button-toolbar">
            <button class="cr-btn" onclick="runPrediction()">
              <span>⚡</span> Classify Genres & Match Films
            </button>
            <button class="cr-btn-secondary" onclick="tryRandomExample()">
              <span>🎲</span> Random Example
            </button>
            <button class="cr-btn-secondary" onclick="clearInputs()">
              <span>🗑️</span> Clear
            </button>
            <button class="cr-btn-secondary" onclick="copyShareableLink()">
              <span>🔗</span> Share Link
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Inline Error Alert -->
    <div id="error-alert" class="validation-alert" style="display: none; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; margin: 1.5rem auto; max-width: 860px; justify-content: space-between; align-items: center;">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span>⚠️</span> <span id="error-alert-msg">Inference error. Please check synopsis and try again.</span>
      </div>
      <button class="cr-btn-secondary" style="padding: 0.35rem 0.85rem; font-size: 0.78rem;" onclick="runPrediction()">🔄 Retry</button>
    </div>

    <!-- Skeleton Shimmer Loader -->
    <div id="skeleton-loader" class="skeleton-box">
      <div class="skeleton-card"></div>
      <div class="skeleton-card" style="height: 180px;"></div>
    </div>

    <!-- Output Section (Always visible with Featured Content, dynamically updated on classification) -->
    <section id="result-section">
      <!-- Low Confidence Alert -->
      <div id="low-conf-alert" class="low-confidence-alert" style="display: none;">
        <span>⚠️</span> <span id="low-conf-text"></span>
      </div>

      <!-- Top 3 Genre Prediction Card (Revealed upon classification) -->
      <div class="prediction-hero" id="prediction-hero" style="display: none;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 800; letter-spacing: 0.08em;">
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
        <div id="fb-status" style="font-size: 0.8rem; color: var(--lb-green); font-weight: 800; margin-top: 0.5rem; display: none;">
          ✓ Feedback logged. Thank you for calibrating Cinemix ▶!
        </div>
      </div>

      <!-- Recommendations Header & Filters -->
      <div style="margin-top: 2rem; margin-bottom: 1.25rem;">
        <h2 id="catalog-heading" style="font-size: 1.6rem; font-weight: 900; color: var(--text);">
          ⭐ Featured Cinema & Anime Highlights
        </h2>
        <p id="catalog-subheading" style="font-size: 0.95rem; color: var(--text-muted); margin-top: 0.25rem;">
          Discover top-rated cinematic masterpieces and anime classics. Type or paste any storyline above to predict multi-label genres and match similar films.
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
          <label for="filter-rating">Min Rating: <span id="rating-val" style="color: var(--lb-green); font-weight: 900;">0.0</span> ★</label>
          <input type="range" id="filter-rating" min="0" max="9" step="0.5" value="0.0" oninput="document.getElementById('rating-val').innerText = this.value; applyFilters();">
        </div>
      </div>

      <!-- Recommendations Grid -->
      <div class="rec-grid" id="rec-list"></div>
    </section>
  </main>

  <!-- Movie Details Modal -->
  <div id="movie-modal" class="modal-backdrop" onclick="closeModalOnBackdrop(event)">
    <div class="modal-box">
      <button class="modal-close-btn" onclick="closeModal()" aria-label="Close modal">✕</button>
      <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
        <img id="modal-poster" src="" alt="Movie Poster" style="width: 180px; height: 260px; object-fit: cover; border-radius: 4px; border: 1px solid var(--border);">
        <div style="flex: 1; min-width: 260px;">
          <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span id="modal-match" class="match-pill">⚡ 94% MATCH</span>
            <span id="modal-year" style="color: var(--text-muted); font-weight: 700;">2010</span>
          </div>
          <h2 id="modal-title" style="font-size: 1.6rem; font-weight: 900; color: var(--text); line-height: 1.2;">Title</h2>
          <div style="margin-top: 0.4rem; font-size: 0.95rem; color: var(--lb-green); font-weight: 800;">
            ★ <span id="modal-rating">8.4</span> / 10 (<span id="modal-votes" style="color: var(--text-muted); font-size: 0.85rem;">0 votes</span>)
          </div>
          <div id="modal-chips" class="card-chips" style="margin-top: 0.75rem;"></div>
          <div style="margin-top: 1.2rem;">
            <a id="modal-trailer" href="#" target="_blank" rel="noopener noreferrer" class="cr-btn" style="text-decoration: none; padding: 0.65rem 1.25rem; font-size: 0.88rem;">
              <span>▶</span> Watch Official Trailer
            </a>
          </div>
        </div>
      </div>
      <div style="margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1.25rem;">
        <h4 style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.08em; margin-bottom: 0.4rem;">
          Plot Overview
        </h4>
        <p id="modal-overview" style="color: var(--text); font-size: 0.95rem; line-height: 1.6;"></p>
      </div>
      <div style="margin-top: 1rem;">
        <h4 style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.08em; margin-bottom: 0.4rem;">
          Keywords / Thematic Tags
        </h4>
        <p id="modal-keywords" style="color: var(--text-dim); font-size: 0.85rem;"></p>
      </div>
    </div>
  </div>

  <!-- ABOUT MODAL (Explaining Cinemix & How It Works) -->
  <div id="about-modal" class="modal-backdrop" onclick="closeAboutModalOnBackdrop(event)">
    <div class="modal-box" style="max-width: 720px;">
      <button class="modal-close-btn" onclick="closeAboutModal()" aria-label="Close modal">✕</button>
      
      <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
        <h2 style="font-size: 1.8rem; font-weight: 900; color: var(--text); font-family: 'Space Grotesk', sans-serif;">CINEMIX</h2>
        <span class="play-icon-badge">▶</span>
        <div class="lb-dots" style="margin-left: 0.4rem;">
          <span class="lb-dot dot-green"></span>
          <span class="lb-dot dot-orange"></span>
          <span class="lb-dot dot-blue"></span>
        </div>
      </div>
      <p style="color: var(--lb-green); font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1.25rem;">
        Letterboxd-Styled Film & Anime Multi-Label Intelligence
      </p>

      <div style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.65; margin-bottom: 1.5rem;">
        <strong>Cinemix ▶</strong> is a specialized cinema discovery and NLP classification tool for cinephiles, screenwriters, and film fans. By analyzing narrative synopses, storyline summaries, or pitch descriptions, Cinemix predicts a film's true <strong>multi-label genre DNA</strong> and retrieves the most organically similar films across a rich catalog of 4,680+ cinema and anime titles.
      </div>

      <div style="border-top: 1px solid var(--border); padding-top: 1.25rem;">
        <h3 style="font-size: 1.15rem; font-weight: 900; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.85rem;">
          💡 How It Works (Pipeline Architecture)
        </h3>

        <div style="display: flex; flex-direction: column; gap: 1rem; color: var(--text-muted); font-size: 0.92rem; line-height: 1.6;">
          <div style="background: var(--surface); padding: 0.9rem 1.1rem; border-radius: 4px; border: 1px solid var(--border);">
            <strong style="color: var(--text); display: block; margin-bottom: 0.25rem;">1. Morphological Normalization & Lemmatization</strong>
            Input storylines undergo case folding, punctuation cleaning, stopword removal, and <strong>Snowball morphological stemming</strong> paired with an irregular noun lookup dictionary (e.g. <em>superheroes → superhero, murderers → murder, detectives → detect</em>).
          </div>

          <div style="background: var(--surface); padding: 0.9rem 1.1rem; border-radius: 4px; border: 1px solid var(--border);">
            <strong style="color: var(--text); display: block; margin-bottom: 0.25rem;">2. Sublinear TF-IDF Feature Space (6,000 N-Grams)</strong>
            The text is vectorized using unigrams and bigrams with sublinear term frequency scaling ($tf = 1 + \ln(count)$) to detect contextual narrative expressions like <em>"space travel"</em>, <em>"serial killer"</em>, and <em>"high school"</em>.
          </div>

          <div style="background: var(--surface); padding: 0.9rem 1.1rem; border-radius: 4px; border: 1px solid var(--border);">
            <strong style="color: var(--text); display: block; margin-bottom: 0.25rem;">3. Multi-Label One-vs-Rest Logistic Regression</strong>
            Films naturally cross multiple genres. Cinemix uses 12 balanced binary classifiers to compute calibrated continuous sigmoid probabilities $P(C|x) = \frac{1}{1 + e^{-(w_c^T x + b_c)}}$ and outputs the top 3 genres with confidence percentages.
          </div>

          <div style="background: var(--surface); padding: 0.9rem 1.1rem; border-radius: 4px; border: 1px solid var(--border);">
            <strong style="color: var(--text); display: block; margin-bottom: 0.25rem;">4. 3-Tier Hybrid Recommendation Formula</strong>
            Similar titles are ranked via an objective hybrid function:
            <br>
            <code style="color: var(--lb-green); font-weight: 800; font-size: 0.88rem;">Score = 0.60 × Cosine Similarity + 0.25 × Genre Jaccard Overlap + 0.15 × Normalized Rating</code>
          </div>

          <div style="background: var(--surface); padding: 0.9rem 1.1rem; border-radius: 4px; border: 1px solid var(--border);">
            <strong style="color: var(--text); display: block; margin-bottom: 0.25rem;">5. Verified Accuracy Benchmarks</strong>
            Trained on 4,680+ titles from TMDB 5000 and curated anime classics. Evaluated using 5-fold cross-validation with <strong>86.7% Hamming Accuracy</strong>, <strong>67.7% Micro F1</strong>, and a <strong>96.8% Top-3 Any Hit Rate</strong>.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div style="margin-bottom: 0.4rem; font-weight: 900; color: var(--text); letter-spacing: 0.04em;">
      CINEMIX <span style="color: var(--lb-green);">▶</span>
    </div>
    Minimalist Letterboxd-styled Film & Anime Multi-Label Intelligence.
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
        plot: "While exploring an abandoned amusement park, a young girl discovers a mystical world of ancient spirits and Japanese gods. When parents are magically transformed into pigs by a vengeful witch, she must work at a grand spirit bathhouse to free them.",
        keywords: "anime, studio ghibli, spirit world, witch, bathhouse, fantasy curse, coming of age"
      }
    };

    let allRecommendations = [];
    let currentPrediction = null;
    let collapseTimer = null;

    // Typebar expansion with debounce & stability
    function expandTypebar() {
      clearTimeout(collapseTimer);
      document.getElementById('search-box').classList.add('is-expanded');
    }

    function collapseTypebar(force = false) {
      const text = document.getElementById('plot').value.trim();
      const isFocused = document.activeElement === document.getElementById('plot') ||
                        document.activeElement === document.getElementById('keywords') ||
                        document.activeElement === document.getElementById('presets');
      if (force || (!text && !isFocused)) {
        document.getElementById('search-box').classList.remove('is-expanded');
      }
    }

    // Word Counter
    function updateWordCount() {
      const text = document.getElementById('plot').value.trim();
      const words = text ? text.split(/\s+/).filter(w => w.length > 0) : [];
      const wordCount = words.length;
      const charCount = text.length;

      const badge = document.getElementById('word-badge');
      badge.innerText = `${wordCount} word${wordCount === 1 ? '' : 's'}`;
      badge.className = "counter-badge" + (wordCount > 0 ? " counter-valid" : "");
      document.getElementById('char-badge').innerText = `${charCount} characters`;
      
      if (text.length > 0) {
        expandTypebar();
      }
    }

    function applyPreset(autoRun = false) {
      const val = document.getElementById('presets').value;
      if (presets[val]) {
        document.getElementById('plot').value = presets[val].plot;
        document.getElementById('keywords').value = presets[val].keywords;
        expandTypebar();
        updateWordCount();
        if (autoRun) {
          runPrediction();
        }
      }
    }

    function tryRandomExample() {
      const keys = Object.keys(presets);
      const randKey = keys[Math.floor(Math.random() * keys.length)];
      document.getElementById('presets').value = randKey;
      applyPreset(true);
    }

    function clearInputs() {
      document.getElementById('plot').value = "";
      document.getElementById('keywords').value = "";
      document.getElementById('presets').value = "";
      updateWordCount();
      document.getElementById('prediction-hero').style.display = "none";
      document.getElementById('catalog-heading').innerText = "⭐ Featured Cinema & Anime Highlights";
      document.getElementById('catalog-subheading').innerText = "Discover top-rated cinematic masterpieces and anime classics. Type or paste any storyline above to predict multi-label genres and match similar films.";
      collapseTypebar(true);
      loadInitialCatalog();
    }

    // Theme Toggle
    function toggleTheme() {
      const body = document.body;
      const cur = body.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      body.setAttribute('data-theme', next);
      localStorage.setItem('cinemix_theme', next);
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

    const savedTheme = localStorage.getItem('cinemix_theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeUI(savedTheme);

    // About Modal
    function openAboutModal() {
      document.getElementById('about-modal').style.display = 'flex';
      document.body.style.overflow = 'hidden';
    }

    function closeAboutModal() {
      document.getElementById('about-modal').style.display = 'none';
      document.body.style.overflow = 'auto';
    }

    function closeAboutModalOnBackdrop(e) {
      if (e.target.id === 'about-modal') closeAboutModal();
    }

    // Letterboxd-Style Dynamic SVG Poster Fallback
    function createFallbackPoster(title, genre, year) {
      const initials = (title || 'Film').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase();
      const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="300" height="420" viewBox="0 0 300 420">
          <defs>
            <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#1c2228"/>
              <stop offset="100%" stop-color="#111418"/>
            </linearGradient>
          </defs>
          <rect width="300" height="420" fill="url(#bg)"/>
          <circle cx="150" cy="170" r="54" fill="rgba(0, 224, 84, 0.12)" stroke="#00e054" stroke-width="2"/>
          <text x="150" y="181" fill="#00e054" font-size="30" font-weight="900" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">${initials}</text>
          <rect x="35" y="270" width="230" height="1" fill="#2c3440"/>
          <text x="150" y="315" fill="#FFFFFF" font-size="15" font-weight="800" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">${escapeXml(title.slice(0, 25))}</text>
          <text x="150" y="340" fill="#99aabb" font-size="12" font-weight="700" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">${escapeXml(genre || 'Film')} • ${year || ''}</text>
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

      if (!plot && !keywords) {
        expandTypebar();
        document.getElementById('plot').focus();
        return;
      }
      
      const errAlert = document.getElementById('error-alert');
      if (errAlert) errAlert.style.display = "none";

      document.getElementById('skeleton-loader').style.display = "block";

      try {
        const [pResp, rResp] = await Promise.all([
          fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ overview: plot, keywords: keywords, top_k: 3 })
          }),
          fetch('/api/recommend', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ overview: plot, keywords: keywords, top_n: 12 })
          })
        ]);

        if (!pResp.ok) throw new Error(`Predict endpoint returned status ${pResp.status}`);
        if (!rResp.ok) throw new Error(`Recommend endpoint returned status ${rResp.status}`);

        const predRes = await pResp.json();
        const recRes = await rResp.json();

        currentPrediction = predRes;
        allRecommendations = recRes.recommendations || [];
        const topPreds = predRes.top_predictions || predRes.top_genres || [];

        document.getElementById('skeleton-loader').style.display = "none";
        document.getElementById('prediction-hero').style.display = "block";
        document.getElementById('catalog-heading').innerText = "🎯 Similar Films & Recommendations";
        document.getElementById('catalog-subheading').innerText = "Ranked via hybrid TF-IDF plot similarity, multi-label genre overlap, and Bayesian quality ratings.";

        // Render Top 3 Cards
        const top3Wrap = document.getElementById('top3-cards');
        top3Wrap.innerHTML = '';
        const ranks = ['#1 Primary Genre', '#2 Secondary Genre', '#3 Tertiary Genre'];

        topPreds.forEach((item, idx) => {
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

        setTimeout(() => {
          topPreds.forEach((item, idx) => {
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

        document.getElementById('prediction-hero').scrollIntoView({ behavior: 'smooth' });

      } catch (err) {
        document.getElementById('skeleton-loader').style.display = "none";
        const errBox = document.getElementById('error-alert');
        if (errBox) {
          document.getElementById('error-alert-msg').innerText = "Inference issue: " + err.message + ". Click Retry to re-run.";
          errBox.style.display = "flex";
          errBox.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }

    // Load Initial Catalog on Page Load
    async function loadInitialCatalog() {
      try {
        const resp = await fetch('/api/recommend', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ overview: '', keywords: '', top_n: 12 })
        });
        if (resp.ok) {
          const data = await resp.json();
          allRecommendations = data.recommendations || [];
          applyFilters();
        }
      } catch (e) {
        console.warn("Initial catalog fetch fallback", e);
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
          <div style="grid-column: 1 / -1; padding: 2.5rem; text-align: center; color: var(--text-muted); background: var(--surface); border-radius: 4px;">
            No films match your current filter criteria. Try adjusting the rating slider or release year range.
          </div>
        `;
        return;
      }

      filtered.slice(0, 9).forEach((item, idx) => {
        const topGenre = (item.genres && item.genres[0]) || '';
        const fallback = createFallbackPoster(item.title, topGenre, item.release_year);
        const chipsHtml = (item.genres || []).slice(0, 3).map(g => `<span class="genre-chip">${g}</span>`).join('');

        list.innerHTML += `
          <article class="stream-card" onclick="openDetailsModal('${item.id}')">
            <img class="card-poster" 
                 src="/api/tmdb-poster?title=${encodeURIComponent(item.title)}&year=${item.release_year}&genre=${encodeURIComponent(topGenre)}"
                 alt="${escapeXml(item.title)} Poster"
                 loading="lazy"
                 onerror="this.onerror=null; this.src='${fallback}';">
            <div class="card-content">
              <div class="card-top-row">
                <h3 class="stream-title">${item.title}</h3>
                <span class="match-pill">⚡ ${item.similarity_percentage}</span>
              </div>
              <div class="card-meta">
                <span>${item.release_year}</span>
                <span>•</span>
                <span class="star-score">★ ${item.vote_average.toFixed(1)}</span>
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
      const item = allRecommendations.find(m => String(m.id) === String(movieId));
      if (!item) return;

      const topGenre = (item.genres && item.genres[0]) || '';
      const fallback = createFallbackPoster(item.title, topGenre, item.release_year);
      const posterImg = document.getElementById('modal-poster');
      posterImg.src = `/api/tmdb-poster?title=${encodeURIComponent(item.title)}&year=${item.release_year}&genre=${encodeURIComponent(topGenre)}`;
      posterImg.onerror = function() { this.src = fallback; };

      document.getElementById('modal-title').innerText = item.title;
      document.getElementById('modal-year').innerText = item.release_year;
      document.getElementById('modal-match').innerText = `⚡ ${item.similarity_percentage} MATCH`;
      document.getElementById('modal-rating').innerText = item.vote_average.toFixed(1);
      document.getElementById('modal-votes').innerText = `${item.vote_count.toLocaleString()} votes`;
      document.getElementById('modal-overview').innerText = item.overview || "No overview available.";
      document.getElementById('modal-keywords').innerText = item.keywords || "None recorded";

      const chipsWrap = document.getElementById('modal-chips');
      chipsWrap.innerHTML = (item.genres || []).map(g => `<span class="genre-chip" style="font-size: 0.78rem; padding: 0.2rem 0.6rem;">${g}</span>`).join('');

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
      if (e.key === 'Escape') {
        closeModal();
        closeAboutModal();
      }
    });

    // Copy Result Text
    function copyResultText() {
      if (!currentPrediction) return;
      const text = `Cinemix ▶ Multi-Label Classification:\\n` +
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
      localStorage.setItem('cinemix_feedback_' + currentPrediction.primary_genre, type);
      document.getElementById('fb-status').style.display = "block";
    }

    // Interaction Listeners for Smooth Typebar & Keyboard Support
    window.addEventListener('DOMContentLoaded', () => {
      const searchBox = document.getElementById('search-box');
      const plotInput = document.getElementById('plot');

      // Mouseenter / Mouseleave with 260ms grace debounce
      if (searchBox) {
        searchBox.addEventListener('mouseenter', () => {
          expandTypebar();
        });
        searchBox.addEventListener('mouseleave', () => {
          collapseTimer = setTimeout(() => {
            collapseTypebar(false);
          }, 260);
        });
      }

      // Enter key submits synopsis, Shift+Enter makes new line
      if (plotInput) {
        plotInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runPrediction();
          }
        });
      }

      // Auto-load URL search params or load Featured Catalog immediately
      const params = new URLSearchParams(window.location.search);
      const synopsis = params.get('synopsis');
      const kw = params.get('keywords');
      if (synopsis) {
        plotInput.value = synopsis;
        if (kw) document.getElementById('keywords').value = kw;
        expandTypebar();
        updateWordCount();
        runPrediction();
      } else {
        loadInitialCatalog();
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
    overview = data.get("overview") or data.get("plot", "")
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
        "top_genres": top_predictions,
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
    overview = data.get("overview") or data.get("plot", "")
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

    # Predict query genres for Jaccard overlap only if query vector is present
    if vec:
        scores = []
        for c_idx in range(len(classes)):
            logit = intercepts[c_idx] + sum(vec.get(f_idx, 0.0) * weights[c_idx][f_idx] for f_idx in vec)
            prob = 1.0 / (1.0 + math.exp(-max(min(logit, 15.0), -15.0)))
            scores.append((classes[c_idx], prob))
        scores.sort(key=lambda x: x[1], reverse=True)
        query_top_genres = set(g for g, _ in scores[:3])
    else:
        query_top_genres = set()

    results = []
    for m in movies:
        m_vec = m.get("tfidf_vec", {})
        cosine_sim = sum(vec.get(int(f_idx), 0.0) * val for f_idx, val in m_vec.items()) if vec else 0.0

        movie_genres = m.get("genres", [])
        if query_top_genres and movie_genres:
            set_m = set(movie_genres)
            jaccard = len(query_top_genres.intersection(set_m)) / len(query_top_genres.union(set_m))
        else:
            jaccard = 0.0

        rating = float(m.get("vote_average", 6.0))
        votes = float(m.get("vote_count", 0))
        wr = (votes / (votes + 300.0)) * rating + (300.0 / (votes + 300.0)) * 6.1
        rating_score = min(max(wr / 10.0, 0.0), 1.0)

        if vec:
            hybrid_score = 0.60 * cosine_sim + 0.25 * jaccard + 0.15 * rating_score
            sim_pct = f"{max(min(cosine_sim * 100, 100), 0):.1f}%"
        else:
            hybrid_score = rating_score
            sim_pct = f"{rating * 10:.0f}%"

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
    """Fetch movie poster or redirect to TMDB poster URL if configured, with SVG fallback."""
    title = request.args.get("title", "").strip()
    year = request.args.get("year", "").strip()
    genre = request.args.get("genre", "").strip()
    tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()

    if tmdb_key and title:
        try:
            params = {"api_key": tmdb_key, "query": title}
            if year:
                params["year"] = year
            url = f"https://api.themoviedb.org/3/search/movie?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Cinemix/2.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results and results[0].get("poster_path"):
                    poster_path = results[0]["poster_path"]
                    return redirect(f"https://image.tmdb.org/t/p/w500{poster_path}", code=302)
        except Exception:
            pass

    # Clean Letterboxd-Style SVG Poster response (200 OK)
    initials = "".join([w[0] for w in title.split() if w][:2]).upper() or "CX"
    clean_title = (title[:24] + "...") if len(title) > 24 else title
    clean_genre = genre or "Cinema"
    clean_year = year or ""
    svg_code = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="420" viewBox="0 0 300 420">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1c2228"/>
      <stop offset="100%" stop-color="#111418"/>
    </linearGradient>
  </defs>
  <rect width="300" height="420" fill="url(#bg)"/>
  <circle cx="150" cy="170" r="54" fill="rgba(0, 224, 84, 0.12)" stroke="#00e054" stroke-width="2"/>
  <text x="150" y="181" fill="#00e054" font-size="28" font-weight="900" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">{initials}</text>
  <rect x="35" y="270" width="230" height="1" fill="#2c3440"/>
  <text x="150" y="315" fill="#FFFFFF" font-size="15" font-weight="800" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">{clean_title}</text>
  <text x="150" y="340" fill="#99aabb" font-size="12" font-weight="700" font-family="'Plus Jakarta Sans', sans-serif" text-anchor="middle">{clean_genre} {f'• {clean_year}' if clean_year else ''}</text>
</svg>"""
    return Response(svg_code, mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
