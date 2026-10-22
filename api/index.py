"""
Vercel Serverless Entrypoint & REST API for Movie Genre Classification & Recommendation.
Designed with a sleek, vibrant anime-streaming dark & electric orange theme.
"""

import os
import sys
import json
import math
import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Preload serverless bundle
_BUNDLE = None


def get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        bundle_path = MODELS_DIR / "serverless_model.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Serverless model bundle not found at {bundle_path}")
        with open(bundle_path, "r", encoding="utf-8") as f:
            _BUNDLE = json.load(f)
    return _BUNDLE


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
    "you've", "your", "yours", "yourself", "yourselves"
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def vectorize_text(cleaned_text: str, vocab: dict, idf: list) -> dict:
    tokens = cleaned_text.split()
    unigrams = tokens
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    terms = unigrams + bigrams

    counts = {}
    for t in terms:
        if t in vocab:
            idx = vocab[t]
            counts[idx] = counts.get(idx, 0) + 1

    vec = {}
    norm_sq = 0.0
    for idx, count in counts.items():
        sub_tf = 1.0 + math.log(count)
        val = sub_tf * idf[idx]
        vec[idx] = val
        norm_sq += val * val

    if norm_sq > 0:
        norm = math.sqrt(norm_sq)
        for idx in vec:
            vec[idx] /= norm

    return vec


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CineVerse AI | Anime & Movie Genre Classifier & Recommender</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {
      --brand-orange: #FF640A;
      --brand-orange-hover: #E05300;
      --brand-orange-light: #FF8A3D;
      --brand-orange-glow: rgba(255, 100, 10, 0.28);
      --bg-deep: #000000;
      --bg-main: #141519;
      --card-bg: #1E1F25;
      --surface: #23252B;
      --border: #2D2F37;
      --text: #F7F7F8;
      --text-muted: #A0A0B2;
      --text-dim: #707080;
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
    }

    /* Top Streaming Navbar */
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
      width: 34px;
      height: 34px;
      background: var(--brand-orange);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #000000;
      font-size: 1rem;
      font-weight: 900;
      box-shadow: 0 0 14px var(--brand-orange-glow);
    }
    .brand-title {
      font-size: 1.35rem;
      font-weight: 900;
      color: #FFFFFF;
      letter-spacing: -0.02em;
    }
    .brand-highlight {
      color: var(--brand-orange);
    }
    .nav-tag {
      background: rgba(255, 100, 10, 0.12);
      border: 1px solid var(--brand-orange);
      color: var(--brand-orange);
      padding: 0.2rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 800;
      letter-spacing: 0.04em;
    }

    /* Container */
    .container {
      max-width: 1180px;
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
      font-size: 0.85rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .hero-title {
      font-size: 2.2rem;
      font-weight: 900;
      color: #FFFFFF;
      letter-spacing: -0.03em;
      line-height: 1.2;
    }
    .hero-desc {
      color: var(--text-muted);
      font-size: 1rem;
      margin-top: 0.5rem;
      max-width: 720px;
    }

    /* Stream Tabs */
    .tab-bar {
      display: flex;
      gap: 0.5rem;
      border-bottom: 2px solid var(--border);
      margin-bottom: 2rem;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 1rem;
      font-weight: 700;
      padding: 0.85rem 1.5rem;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .tab-btn:hover { color: #FFFFFF; }
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
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .form-group { margin-bottom: 1.35rem; }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #D6D6E0;
      margin-bottom: 0.5rem;
    }
    textarea, select, input {
      width: 100%;
      background-color: var(--bg-deep);
      border: 1px solid var(--border);
      border-radius: 6px;
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

    .grid-2 {
      display: grid;
      grid-template-columns: 3fr 2fr;
      gap: 1.25rem;
    }
    @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }

    /* Orange CTA Button */
    .cr-btn {
      background: var(--brand-orange);
      color: #FFFFFF;
      font-weight: 800;
      font-size: 1.05rem;
      border: none;
      border-radius: 6px;
      padding: 0.95rem 2rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.6rem;
      width: 100%;
      transition: all 0.2s ease;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      box-shadow: 0 4px 14px var(--brand-orange-glow);
    }
    .cr-btn:hover {
      background: var(--brand-orange-hover);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(255, 100, 10, 0.45);
    }
    .cr-btn:active {
      transform: translateY(0);
    }

    /* Prediction Banner */
    .prediction-hero {
      background: linear-gradient(135deg, #1E1F25 0%, #16171C 100%);
      border: 1px solid var(--border);
      border-left: 6px solid var(--brand-orange);
      border-radius: 8px;
      padding: 1.5rem 1.75rem;
      margin-bottom: 1.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1.25rem;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    }
    .genre-badge {
      background-color: var(--brand-orange);
      color: #FFFFFF;
      font-weight: 900;
      font-size: 1.4rem;
      padding: 0.45rem 1.4rem;
      border-radius: 6px;
      display: inline-block;
      letter-spacing: -0.01em;
      box-shadow: 0 0 16px var(--brand-orange-glow);
    }
    .confidence-score {
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--brand-orange);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }

    /* Orange Progress Bars */
    .bar-row {
      display: flex;
      align-items: center;
      margin-bottom: 0.75rem;
      gap: 1rem;
    }
    .bar-label {
      width: 150px;
      font-size: 0.9rem;
      font-weight: 600;
      color: #D1D1DE;
    }
    .bar-track {
      flex: 1;
      height: 12px;
      background-color: var(--bg-deep);
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #FF640A, #FFA347);
      border-radius: 6px;
    }
    .bar-pct {
      width: 65px;
      text-align: right;
      font-size: 0.85rem;
      font-weight: 800;
      color: var(--brand-orange);
    }

    /* Recommendations: Streaming Anime/Cinema Cards */
    .rec-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .stream-card {
      background-color: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: all 0.25s ease;
    }
    .stream-card:hover {
      transform: translateY(-4px);
      border-color: var(--brand-orange);
      box-shadow: 0 8px 24px var(--brand-orange-glow);
    }
    .card-top {
      background-color: #17181D;
      padding: 0.9rem 1.1rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .stream-title {
      font-size: 1.05rem;
      font-weight: 800;
      color: #FFFFFF;
      line-height: 1.3;
    }
    .match-pill {
      background: rgba(255, 100, 10, 0.15);
      border: 1px solid var(--brand-orange);
      color: var(--brand-orange);
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-size: 0.78rem;
      font-weight: 800;
      white-space: nowrap;
    }
    .card-body {
      padding: 1.1rem;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .genre-chip {
      display: inline-block;
      background-color: #2D2F37;
      color: #E2E2EC;
      padding: 0.15rem 0.65rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 700;
      margin-bottom: 0.75rem;
      align-self: flex-start;
      border: 1px solid #3A3D47;
    }
    .stream-synopsis {
      font-size: 0.88rem;
      color: var(--text-muted);
      line-height: 1.5;
      flex: 1;
    }

    /* KPI Cards */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 1.75rem;
    }
    .kpi-box {
      background-color: var(--card-bg);
      border: 1px solid var(--border);
      border-top: 3px solid var(--brand-orange);
      border-radius: 8px;
      padding: 1.5rem;
      text-align: center;
    }
    .kpi-num {
      font-size: 2.3rem;
      font-weight: 900;
      color: var(--brand-orange);
      line-height: 1.1;
      margin-bottom: 0.35rem;
    }
    .kpi-desc {
      font-size: 0.82rem;
      text-transform: uppercase;
      font-weight: 800;
      color: var(--text-muted);
      letter-spacing: 0.06em;
    }

    /* Table */
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }
    th, td {
      padding: 0.9rem 1.1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    th {
      background-color: var(--bg-deep);
      color: var(--brand-orange);
      font-weight: 800;
      text-transform: uppercase;
      font-size: 0.8rem;
      letter-spacing: 0.05em;
    }
    tr:hover td {
      background-color: rgba(255, 100, 10, 0.05);
    }

    /* Footer */
    footer {
      background-color: var(--bg-deep);
      border-top: 1px solid var(--border);
      padding: 1.75rem;
      text-align: center;
      color: var(--text-dim);
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <!-- Top Navigation Bar -->
  <nav class="navbar">
    <a href="/" class="nav-brand">
      <div class="brand-icon">▷</div>
      <div class="brand-title">CINE<span class="brand-highlight">VERSE</span></div>
    </a>
    <div style="display: flex; align-items: center; gap: 1rem;">
      <span class="nav-tag">AI INTELLIGENCE</span>
    </div>
  </nav>

  <!-- Main Container -->
  <div class="container">
    <div class="hero">
      <div class="hero-eyebrow">
        <span>⚡</span> Natural Language Processing Classifier
      </div>
      <h1 class="hero-title">Predict Genres & Discover Content</h1>
      <p class="hero-desc">Analyze narrative synopses with Multinomial Naive Bayes & discover similar movies with TF-IDF cosine recommendations.</p>
    </div>

    <!-- Tabs Header -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('predict')">
        🎬 Genre Predictor & Recommender
      </button>
      <button class="tab-btn" onclick="switchTab('analytics')">
        📊 Dataset & Model Analytics
      </button>
    </div>

    <!-- TAB 1: PREDICTOR & RECOMMENDER -->
    <div id="tab-predict" class="tab-content active">
      <div class="cr-card">
        <div class="form-group">
          <label for="presets">⚡ Quick Demonstration Presets (Auto-fill Synopsis):</label>
          <select id="presets" onchange="applyPreset()">
            <option value="">-- Select a movie / story archetype --</option>
            <option value="scifi">Interstellar (Sci-Fi • Deep Space Wormhole & Tesseract)</option>
            <option value="action">The Dark Knight (Action • Gotham Vigilante & Joker Chaos)</option>
            <option value="comedy">Superbad (Comedy • High School Party Misadventures)</option>
            <option value="horror">The Conjuring (Horror • Demonic Farmhouse Possession)</option>
            <option value="romance">La La Land (Romance • Los Angeles Jazz & Dreamers)</option>
            <option value="animation">Toy Story (Animation • Sentient Toys Coming to Life)</option>
          </select>
        </div>

        <div class="grid-2">
          <div class="form-group">
            <label for="plot">Story Overview / Plot Synopsis:</label>
            <textarea id="plot" rows="5" placeholder="Enter movie overview or narrative summary here..."></textarea>
          </div>
          <div class="form-group">
            <label for="keywords">Thematic Keywords / Tags:</label>
            <textarea id="keywords" rows="5" placeholder="e.g. superhero, gotham, vigilante, explosion, crime fighter..."></textarea>
          </div>
        </div>

        <button class="cr-btn" onclick="runPrediction()">
          <span>⚡</span> Classify Genre & Recommend Similar Titles
        </button>
      </div>

      <!-- Output Section -->
      <div id="result-section" style="display: none;">
        <div class="prediction-hero">
          <div>
            <div style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); font-weight: 800; letter-spacing: 0.05em;">
              Predicted Primary Genre
            </div>
            <div style="margin-top: 0.5rem; display: flex; align-items: center; gap: 1rem;">
              <span class="genre-badge" id="res-genre">Action</span>
            </div>
          </div>
          <div class="confidence-score">
            <span>⚡ Confidence:</span>
            <span id="res-confidence" style="color: #FFFFFF;">0%</span>
          </div>
        </div>

        <!-- Probabilities -->
        <div class="cr-card">
          <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 1.25rem;">
            Genre Probability Distribution
          </div>
          <div id="prob-bars"></div>
        </div>

        <!-- Recommendations -->
        <div style="margin-top: 2rem; margin-bottom: 1rem;">
          <h2 style="font-size: 1.4rem; font-weight: 900; color: #FFFFFF;">More Content Like This</h2>
          <p style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0.2rem;">Top content-based recommendations retrieved via TF-IDF Cosine Similarity.</p>
        </div>

        <div class="rec-grid" id="rec-list"></div>
      </div>
    </div>

    <!-- TAB 2: ANALYTICS -->
    <div id="tab-analytics" class="tab-content">
      <div class="kpi-row">
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-acc">35.0%</div>
          <div class="kpi-desc">Overall Test Accuracy</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-total">100</div>
          <div class="kpi-desc">Catalog Titles (10 Genres)</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-vocab">3,907</div>
          <div class="kpi-desc">TF-IDF Features (1 & 2-grams)</div>
        </div>
        <div class="kpi-box">
          <div class="kpi-num" id="kpi-test">20</div>
          <div class="kpi-desc">Evaluation Test Samples</div>
        </div>
      </div>

      <div class="cr-card">
        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 1rem;">
          Detailed Classification Report by Genre
        </div>
        <div style="overflow-x: auto;">
          <table id="report-table">
            <thead>
              <tr>
                <th>Genre</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div style="margin-bottom: 0.4rem; font-weight: 800; color: #FFFFFF;">
      CINE<span style="color: var(--brand-orange);">VERSE</span> AI
    </div>
    NLP Machine Learning System powered by TF-IDF & Multinomial Naive Bayes.
  </footer>

  <script>
    const presets = {
      scifi: {
        plot: "A team of exploratory astronauts travel through an enigmatic wormhole near Saturn in a desperate bid to find a habitable planet and guarantee humanity's survival before crops completely fail.",
        keywords: "space travel, wormhole, black hole, time dilation, astronaut, astrophysics"
      },
      action: {
        plot: "Batman raises the stakes in his war on crime. With the help of allies Lt. Jim Gordon and DA Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets of Gotham.",
        keywords: "superhero, gotham city, vigilante, crime fighter, explosions, joker"
      },
      comedy: {
        plot: "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry with fake IDs, eccentric cops, and wild mishaps.",
        keywords: "high school, teen comedy, fake id, wild party, friendship, coming of age"
      },
      horror: {
        plot: "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their secluded farmhouse. Demonic occurrences push the couple to their absolute limits.",
        keywords: "demonology, haunted house, paranormal investigation, possession, exorcism, evil spirits"
      },
      romance: {
        plot: "While navigating their careers in Los Angeles, a pianist and an aspiring actress fall in love while attempting to reconcile their aspirations for the future with their passionate devotion.",
        keywords: "jazz musician, aspiring actress, los angeles romance, musical love story, bittersweet dreams"
      },
      animation: {
        plot: "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom, leading to an adventure across suburban perils to reunite with their owner.",
        keywords: "toys come to life, cowboy woody, space ranger buzz lightyear, pixar animation, friendship"
      }
    };

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (name === 'predict') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('tab-predict').classList.add('active');
      } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('tab-analytics').classList.add('active');
        loadAnalytics();
      }
    }

    function applyPreset() {
      const val = document.getElementById('presets').value;
      if (presets[val]) {
        document.getElementById('plot').value = presets[val].plot;
        document.getElementById('keywords').value = presets[val].keywords;
      }
    }

    async function runPrediction() {
      const plot = document.getElementById('plot').value.trim();
      const keywords = document.getElementById('keywords').value.trim();
      if (!plot) { alert("Please enter a movie plot summary."); return; }

      const sec = document.getElementById('result-section');
      sec.style.display = 'block';

      try {
        const predRes = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ overview: plot, keywords: keywords })
        }).then(r => r.json());

        const recRes = await fetch('/api/recommend', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ overview: plot, keywords: keywords, top_n: 4 })
        }).then(r => r.json());

        document.getElementById('res-genre').innerText = predRes.primary_genre;
        document.getElementById('res-confidence').innerText = predRes.confidence_percentage;

        // Render probability bars
        const barsContainer = document.getElementById('prob-bars');
        barsContainer.innerHTML = '';
        predRes.top_predictions.forEach(item => {
          barsContainer.innerHTML += `
            <div class="bar-row">
              <div class="bar-label">${item.genre}</div>
              <div class="bar-track"><div class="bar-fill" style="width: ${item.probability * 100}%"></div></div>
              <div class="bar-pct">${item.percentage}</div>
            </div>
          `;
        });

        // Render recommendations
        const recContainer = document.getElementById('rec-list');
        recContainer.innerHTML = '';
        recRes.recommendations.forEach((item, idx) => {
          recContainer.innerHTML += `
            <div class="stream-card">
              <div class="card-top">
                <span class="stream-title">#${idx + 1}. ${item.title}</span>
                <span class="match-pill">⚡ ${item.similarity_percentage} Match</span>
              </div>
              <div class="card-body">
                <span class="genre-chip">${item.genre}</span>
                <p class="stream-synopsis">${item.overview}</p>
              </div>
            </div>
          `;
        });

        sec.scrollIntoView({ behavior: 'smooth' });

      } catch (err) {
        alert("Error during prediction: " + err);
      }
    }

    async function loadAnalytics() {
      try {
        const data = await fetch('/api/metrics').then(r => r.json());
        if (data.accuracy) {
          document.getElementById('kpi-acc').innerText = (data.accuracy * 100).toFixed(1) + '%';
          document.getElementById('kpi-total').innerText = data.total_movies;
          document.getElementById('kpi-vocab').innerText = Number(data.vocab_size).toLocaleString();
          document.getElementById('kpi-test').innerText = data.test_size;

          const tbody = document.querySelector('#report-table tbody');
          tbody.innerHTML = '';
          const rep = data.classification_report || {};
          for (const [k, v] of Object.entries(rep)) {
            if (typeof v === 'object' && !['accuracy', 'macro avg', 'weighted avg'].includes(k)) {
              tbody.innerHTML += `
                <tr>
                  <td><strong>${k}</strong></td>
                  <td>${Number(v.precision).toFixed(3)}</td>
                  <td>${Number(v.recall).toFixed(3)}</td>
                  <td>${Number(v['f1-score']).toFixed(3)}</td>
                  <td>${v.support}</td>
                </tr>
              `;
            }
          }
        }
      } catch (e) {
        console.error("Could not load metrics", e);
      }
    }
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
    """Inference endpoint: predict movie genre and class probabilities."""
    data = request.get_json(silent=True) or {}
    overview = data.get("overview", "")
    keywords = data.get("keywords", "")
    top_k = int(data.get("top_k", 5))

    bundle = get_bundle()
    vocab = bundle["vocab"]
    idf = bundle["idf"]
    classes = bundle["classes"]
    class_log_prior = bundle["class_log_prior"]
    feature_log_prob = bundle["feature_log_prob"]

    combined = f"{overview} {keywords}".strip()
    cleaned = clean_text(combined) or "movie film story"
    vec = vectorize_text(cleaned, vocab, idf)

    # Compute Naive Bayes scores
    scores = []
    for c_idx in range(len(classes)):
        score = class_log_prior[c_idx]
        for f_idx, weight in vec.items():
            score += weight * feature_log_prob[c_idx][f_idx]
        scores.append(score)

    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    probs = [s / sum_exp for s in exp_scores]

    top_indices = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)
    primary_idx = top_indices[0]

    top_predictions = []
    for idx in top_indices[:top_k]:
        top_predictions.append({
            "genre": str(classes[idx]),
            "probability": float(probs[idx]),
            "percentage": f"{probs[idx] * 100:.2f}%"
        })

    confidence = float(probs[primary_idx])

    return jsonify({
        "primary_genre": str(classes[primary_idx]),
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "top_predictions": top_predictions,
        "cleaned_query": cleaned
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Recommendation endpoint: TF-IDF cosine similarity."""
    data = request.get_json(silent=True) or {}
    overview = data.get("overview", "")
    keywords = data.get("keywords", "")
    top_n = int(data.get("top_n", 4))

    bundle = get_bundle()
    vocab = bundle["vocab"]
    idf = bundle["idf"]
    movies = bundle["movies"]

    combined = f"{overview} {keywords}".strip()
    cleaned = clean_text(combined) or "movie story"
    vec = vectorize_text(cleaned, vocab, idf)

    scored = []
    for m in movies:
        sim = sum(vec.get(int(idx), 0.0) * val for idx, val in m["tfidf_vec"].items())
        scored.append({
            "id": m["id"],
            "title": m["title"],
            "genre": m["genre"],
            "similarity_score": round(sim, 4),
            "similarity_percentage": f"{sim * 100:.1f}%",
            "overview": m["overview"],
            "keywords": m.get("keywords", "")
        })

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return jsonify({"recommendations": scored[:top_n]})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return model evaluation metrics & class distributions."""
    bundle = get_bundle()
    return jsonify(bundle.get("metrics", {}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
