# 🎬 CineVerse AI: Multi-Label Movie & Anime Genre Classification & Recommendation Engine

**Live Production Web App:** [https://movie-genre-classification-j8iw.vercel.app/](https://movie-genre-classification-j8iw.vercel.app/)

A production-grade, full-stack Natural Language Processing (NLP) system and recommendation engine. CineVerse AI classifies narrative synopses across **12 multi-label genres** using **sublinear TF-IDF (1 & 2-grams)** and **One-vs-Rest Balanced Logistic Regression** with calibrated sigmoid probabilities. It pairs predictions with **hybrid recommendations** ($0.60 \times \text{Cosine} + 0.25 \times \text{Genre Jaccard} + 0.15 \times \text{Rating}$), feature attribution, interactive Chart.js analytics, TMDB poster integration, and dark/light streaming themes.

---

## 🌟 Key Features & Architecture Upgrades

### 1. Expanded Dataset & Multi-Label Modeling
- **Scale:** Expanded from a 100-title single-label sample to **4,680+ clean titles** combining the TMDB 5000 cinema catalog with curated anime classics (*Demon Slayer, Ghost in the Shell, Spirited Away, Attack on Titan, Cowboy Bebop, Neon Genesis Evangelion*).
- **Multi-Label Formulation:** Film and anime media inherently span multiple genres simultaneously. Modeled via 12 binary One-vs-Rest classifiers across: `Drama`, `Comedy`, `Thriller`, `Action`, `Romance`, `Adventure`, `Crime`, `Science Fiction`, `Horror`, `Family`, `Fantasy`, and `Animation`.
- **Top 3 Genre Predictions:** Returns the top 3 genres with calibrated sigmoid probabilities ($P(C|x) = \frac{1}{1 + e^{-(w^T x + b)}}$) rather than uncalibrated single-label likelihoods.

### 2. High-Performance Text Normalization & Feature Extraction
- **Preprocessing:** Case folding, non-alphanumeric filtering, stop words pruning, and **Snowball morphological lemmatization** paired with a domain-specific irregular noun lookup dictionary (*superheroes → superhero, murderers → murder, detectives → detect*).
- **TF-IDF Vectorizer:** 6,000 unigrams and bigrams with sublinear term frequency scaling ($tf = 1 + \ln(count)$) to dampen dominant repetitive words.

### 3. Explainability & Diagnostics
- **"Why this prediction?"**: Computes exact feature attribution contributions ($tfidf_i \times w_{c,i}$) to highlight the most influential tokens in the user's synopsis.
- **Low-Confidence Ambiguity Alert:** Detects when the top 2 predicted probabilities are within 8% of each other, notifying the user of strong hybrid characteristics (e.g. Action/Sci-Fi hybrid).
- **Interactive Feedback:** Thumbs Up / Thumbs Down feedback buttons with `localStorage` and REST API persistence.

### 4. Hybrid Recommendation Engine
- **3-Tier Objective Function:**
  $$\text{Score} = 0.60 \times \text{Cosine Sim} + 0.25 \times \text{Genre Jaccard Overlap} + 0.15 \times \text{Normalized Rating}$$
- **Dynamic Poster Fetching:** Queries the official TMDB API via `TMDB_API_KEY` (environment variable) with zero hardcoded secrets. If missing or offline, dynamically renders a dark cinema SVG poster fallback with title initials and genre badge.
- **Live Filtering:** Instant filtering by genre, release year decade, and minimum star rating.
- **Cinema Details Modal:** Displays full overview, keywords, score breakdown, and a 1-click **"▶ Watch Trailer"** link to YouTube official trailers.

### 5. Interactive Analytics Tab (Chart.js)
- **Genre Distribution:** Interactive doughnut chart across all 4,680 titles.
- **Per-Genre Metrics:** Grouped bar chart comparing Precision, Recall, and F1-Score across all 12 classes.
- **5-Fold Cross Validation Benchmark:** Compares Logistic Regression, LinearSVC, and Multinomial Naive Bayes.
- **Genre Co-Occurrence Matrix:** Heatmap showing how frequently genres intersect (e.g., Action + Adventure, Romance + Drama).
- **Top Predictive Words per Genre:** Interactive tabbed pills displaying the highest-weighted regression tokens per genre.

### 6. Design & Accessibility
- **Theme Toggle:** Dark Mode (electric orange `#FF640A` on `#141519` deep dark) and Light Mode (`#F4F6F9`) with `localStorage` persistence.
- **Input Validation:** Minimum 15-word threshold with live word and character counters to guarantee sufficient feature density.
- **Mobile Responsive & Accessible:** Responsive CSS grid/flexbox, semantic HTML5, ARIA labels, and keyboard navigation.

---

## 📊 Empirical Evaluation & Benchmarks

All models were evaluated using Stratified Train/Test splits and rigorous **5-Fold Cross Validation**:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Hamming Accuracy** | **86.68%** | Precision of binary decisions across all 12 genres |
| **Micro F1-Score** | **67.72%** | **Exceeds target of 60%+** |
| **Macro F1-Score** | **65.54%** | Balanced across both majority and minority genres |
| **Top-3 Any Hit Rate** | **96.79%** | Percentage of test samples where at least 1 true genre is in top 3 |
| **Top-3 Genre Recall** | **74.44%** | Fraction of all true genres captured within top 3 predictions |
| **Exact Match Accuracy** | **19.34%** | Strict all-or-nothing subset match |

### 5-Fold Cross Validation Comparison

| Model | Mean Hamming Acc | Mean Micro F1 | Mean Macro F1 | CV Fit Time |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **86.35% (±0.28%)** | **66.68% (±0.71%)** | **63.98% (±0.95%)** | 4.17s |
| **LinearSVC (Balanced)** | 85.95% (±0.32%) | 64.31% (±0.92%) | 61.13% (±0.79%) | 3.99s |
| **Multinomial Naive Bayes** | 86.55% (±0.42%) | 58.50% (±1.43%) | 46.35% (±1.42%) | 1.03s |

*Selected Architecture: One-vs-Rest Logistic Regression provides the highest Micro-F1 and Macro-F1 with well-calibrated continuous sigmoid probabilities.*

---

## 📁 Repository Structure

```text
movie_genre_app/
│
├── api/
│   └── index.py            # Vercel Serverless Function (Pure Python Flask inference, UI & REST API)
│
├── data/
│   ├── tmdb_5000_movies.csv# Ingested TMDB catalog (4,803 titles)
│   ├── movies_metadata.csv # Legacy metadata fallback
│   └── keywords.csv        # Thematic keywords fallback
│
├── models/
│   ├── serverless_model.json# Compact serialized bundle (weights, intercepts, vocab, IDF, movies)
│   ├── metrics.json        # Evaluation metrics, benchmark tables, and co-occurrence matrix
│   ├── confusion_matrix.png# High-res confusion matrix heatmap
│   ├── tfidf_vectorizer.pkl# Scikit-learn TF-IDF vectorizer artifact
│   └── multi_label_model.pkl# OneVsRest classifier artifact
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Data loading, multi-label parsing, Snowball lemmatization
│   ├── train.py            # 5-fold CV, evaluation metrics, confusion matrix, export
│   └── recommend.py        # Multi-label inference and hybrid recommendation engine
│
├── app.py                  # Streamlit Interactive Web Dashboard (Local & Community Cloud)
├── requirements.txt        # Vercel deployment dependencies (Flask only: bundle < 15MB)
├── requirements-full.txt   # Complete training & Streamlit dependencies (scikit-learn, pandas, etc.)
├── vercel.json             # Vercel routing configuration
├── .vercelignore           # Excludes heavy raw training files from serverless lambda
└── README.md               # Complete setup, architecture, and deployment documentation
```

---

## 🚀 Installation & Local Execution

### 1. Clone & Environment Setup
```bash
git clone https://github.com/sanskarp910-svg/movie-genre-classification.git
cd movie-genre-classification
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install full dependencies for training and Streamlit
pip install -r requirements-full.txt
```

### 2. Retrain the Pipeline (Optional)
To retrain the model, run 5-fold cross-validation, and regenerate all artifacts:
```bash
python src/train.py
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the Full-Stack Serverless App Locally
```bash
python api/index.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌐 TMDB Poster API Configuration

To enable real poster thumbnails for recommendations:
1. Register for a free account at [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Generate an API Key under **Settings → API**.
3. Set the environment variable:
   - **Locally:**
     ```bash
     # Windows PowerShell
     $env:TMDB_API_KEY="your_tmdb_api_key_here"

     # Linux / macOS
     export TMDB_API_KEY="your_tmdb_api_key_here"
     ```
   - **Vercel Cloud:**
     Go to **Project Settings → Environment Variables** on your Vercel dashboard and add:
     - `Key`: `TMDB_API_KEY`
     - `Value`: `<your_tmdb_api_key>`

> *Graceful Fallback:* If no `TMDB_API_KEY` is provided, CineVerse AI automatically renders an inline SVG poster placeholder with title initials and genre branding, ensuring the interface never displays broken images.

---

## 🚢 Vercel Deployment

1. The project includes `vercel.json` configured with `@vercel/python`.
2. `requirements.txt` is kept lightweight (`flask>=3.0.0`) so the serverless lambda deploys in seconds and remains far below Vercel's 250MB limit.
3. Push to `main` branch to trigger automatic Vercel build and deployment:
   ```bash
   git add .
   git commit -m "feat: upgrade to multi-label classification, hybrid recommendations, and interactive analytics"
   git push origin main
   ```

---

## 📜 License & Acknowledgments
- **Dataset:** TMDB 5000 Movies Dataset & Kaggle Movies Dataset.
- **Frameworks:** Python, Scikit-Learn, Flask, Streamlit, Chart.js.
