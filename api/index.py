"""
Vercel Serverless Entrypoint & REST API for Movie Genre Classification & Recommendation.
Provides serverless endpoints and a self-contained full-stack interactive dashboard.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Ensure root and src are importable
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.preprocess import clean_text
from src.recommend import GenreRecommenderEngine

app = Flask(__name__)

# Global model cache for serverless execution
_ENGINE = None
_METRICS = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = GenreRecommenderEngine(models_dir=str(MODELS_DIR))
    return _ENGINE


def get_metrics():
    global _METRICS
    if _METRICS is None:
        metrics_file = MODELS_DIR / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                _METRICS = json.load(f)
        else:
            _METRICS = {}
    return _METRICS


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CineGenre AI | Movie Genre Classification & Recommender</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-border: #1e293b;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --gradient: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2rem; }
    h1 {
      font-size: 2.5rem;
      font-weight: 800;
      background: var(--gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    .subtitle { color: var(--text-muted); font-size: 1.1rem; max-width: 700px; margin: 0 auto; }
    .badge {
      display: inline-block;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.75rem;
    }
    
    /* Tabs */
    .tabs { display: flex; gap: 0.5rem; border-bottom: 1px solid var(--card-border); margin-bottom: 2rem; }
    .tab-btn {
      background: none; border: none; color: var(--text-muted); padding: 0.75rem 1.5rem;
      font-family: inherit; font-size: 1rem; font-weight: 600; cursor: pointer;
      border-bottom: 2px solid transparent; transition: all 0.2s;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: #818cf8; border-bottom-color: #818cf8; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    /* Form & Inputs */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .input-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 1.25rem; }
    @media (max-width: 768px) { .input-grid { grid-template-columns: 1fr; } }
    label { display: block; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.4rem; color: #cbd5e1; }
    textarea, select, input {
      width: 100%; background: #0b0f19; border: 1px solid #334155; border-radius: 8px;
      padding: 0.75rem; color: var(--text); font-family: inherit; font-size: 0.95rem; resize: vertical;
    }
    textarea:focus, select:focus, input:focus { outline: none; border-color: var(--accent); }
    .btn {
      display: inline-block; width: 100%; background: var(--accent); color: white;
      border: none; border-radius: 8px; padding: 0.9rem; font-size: 1rem; font-weight: 700;
      cursor: pointer; transition: background 0.2s; margin-top: 1rem;
    }
    .btn:hover { background: var(--accent-hover); }
    
    /* Results */
    .result-banner {
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
      border: 1px solid rgba(139, 92, 246, 0.4);
      border-radius: 12px;
      padding: 1.25rem;
      margin: 1.5rem 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .primary-genre {
      background: #6366f1; color: white; padding: 0.4rem 1rem; border-radius: 9999px;
      font-weight: 700; font-size: 1.2rem;
    }
    .prob-bar-container { margin-bottom: 0.75rem; }
    .prob-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .prob-bar { height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; }
    .prob-fill { height: 100%; background: var(--accent); border-radius: 4px; }
    
    /* Recommendations */
    .rec-item {
      background: #0f172a; border-left: 4px solid var(--accent);
      border-radius: 8px; padding: 1.1rem; margin-bottom: 0.9rem;
    }
    .rec-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
    .rec-title { font-weight: 700; font-size: 1.1rem; color: #f1f5f9; }
    .rec-match { background: #065f46; color: #6ee7b7; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
    
    /* Stats KPIs */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 1.25rem; text-align: center;
    }
    .kpi-val { font-size: 2rem; font-weight: 800; color: #818cf8; }
    .kpi-title { font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); }
    
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 0.9rem; }
    th { color: #cbd5e1; font-weight: 700; background: #0b0f19; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🎬 CineGenre AI</h1>
      <p class="subtitle">Movie Genre Classification & Content-Based Recommendation Engine powered by TF-IDF & Multinomial Naive Bayes</p>
      <span class="badge">🚀 Deployed & Ready on Vercel Serverless</span>
    </header>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('predict')">🎯 Genre Predictor & Recommender</button>
      <button class="tab-btn" onclick="switchTab('analytics')">📊 Dataset & Model Analytics</button>
    </div>

    <!-- TAB 1: PREDICTOR -->
    <div id="tab-predict" class="tab-content active">
      <div class="card">
        <label for="presets">⚡ Quick Demonstration Preset</label>
        <select id="presets" onchange="applyPreset()" style="margin-bottom: 1rem;">
          <option value="">-- Choose an example movie synopsis --</option>
          <option value="scifi">Sci-Fi Odyssey (Space Exploration & Wormholes)</option>
          <option value="action">Superhero Vigilante (Gotham & Urban Crime)</option>
          <option value="comedy">Teen High School Comedy (Fake ID & Chaos)</option>
          <option value="horror">Paranormal Supernatural Horror (Haunted Farmhouse)</option>
          <option value="romance">Romantic Melodrama (Los Angeles Musical Love)</option>
          <option value="animation">Sentient Animated Toys (Cowboy & Space Ranger)</option>
        </select>

        <div class="input-grid">
          <div>
            <label for="plot">Movie Plot Summary (Overview):</label>
            <textarea id="plot" rows="5" placeholder="Enter movie synopsis or story outline..."></textarea>
          </div>
          <div>
            <label for="keywords">Movie Thematic Keywords:</label>
            <textarea id="keywords" rows="5" placeholder="e.g. space, exploration, cyberpunk, superhero, romance..."></textarea>
          </div>
        </div>

        <button class="btn" onclick="runPrediction()">🚀 Classify Genre & Recommend Similar Movies</button>
      </div>

      <div id="result-section" style="display: none;">
        <div class="result-banner">
          <div>
            <div style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Predicted Primary Genre</div>
            <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.3rem;">
              <span class="primary-genre" id="res-genre">Action</span>
              <span style="font-size: 1.1rem; color: #cbd5e1;">Confidence: <strong id="res-confidence">0%</strong></span>
            </div>
          </div>
        </div>

        <div class="card">
          <h3 style="margin-bottom: 1rem;">📊 Top Genre Probabilities</h3>
          <div id="prob-bars"></div>
        </div>

        <div class="card">
          <h3 style="margin-bottom: 1rem;">🎯 Similar Content Recommendations (Cosine Similarity)</h3>
          <div id="rec-list"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: ANALYTICS -->
    <div id="tab-analytics" class="tab-content">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Model Accuracy</div>
          <div class="kpi-val" id="kpi-acc">35.0%</div>
          <div style="font-size: 0.8rem; color: var(--success);">MultinomialNB</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Corpus Size</div>
          <div class="kpi-val" id="kpi-total">100</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Top 10 Genres</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">TF-IDF Features</div>
          <div class="kpi-val" id="kpi-vocab">3,907</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">Unigram + Bigram</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Test Split Size</div>
          <div class="kpi-val" id="kpi-test">20</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);">20% Stratified Split</div>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-bottom: 0.75rem;">🎭 Genre Classification Report</h3>
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
            <div class="prob-bar-container">
              <div class="prob-label"><span>${item.genre}</span><span>${item.percentage}</span></div>
              <div class="prob-bar"><div class="prob-fill" style="width: ${item.probability * 100}%"></div></div>
            </div>
          `;
        });

        // Render recommendations
        const recContainer = document.getElementById('rec-list');
        recContainer.innerHTML = '';
        recRes.recommendations.forEach((item, idx) => {
          recContainer.innerHTML += `
            <div class="rec-item">
              <div class="rec-header">
                <span class="rec-title">#${idx + 1}. ${item.title}</span>
                <span class="rec-match">Cosine Match: ${item.similarity_percentage}</span>
              </div>
              <div style="font-size: 0.85rem; color: #a5b4fc; margin-bottom: 0.3rem;">🎭 ${item.genre}</div>
              <p style="font-size: 0.9rem; color: #cbd5e1;">${item.overview}</p>
            </div>
          `;
        });

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

    engine = get_engine()
    result = engine.predict_genre(overview, keywords, top_k=top_k)
    return jsonify(result)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Recommendation endpoint: TF-IDF cosine similarity."""
    data = request.get_json(silent=True) or {}
    overview = data.get("overview", "")
    keywords = data.get("keywords", "")
    top_n = int(data.get("top_n", 4))

    engine = get_engine()
    recs = engine.recommend_movies(overview, keywords, top_n=top_n)
    return jsonify({"recommendations": recs})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return model evaluation metrics & class distributions."""
    return jsonify(get_metrics())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
