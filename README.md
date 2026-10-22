# 🎬 CineGenre AI: Movie Genre Classification & Recommendation Engine

A full-stack, production-ready NLP machine learning application and interactive Streamlit web dashboard. CineGenre AI classifies movie synopses into primary genres using **TF-IDF Vectorization** and **Multinomial Naive Bayes (`MultinomialNB`)**, paired with a **Cosine Similarity Content-Based Recommendation Engine**.

---

## 🌟 Key Features

1. **Automated Dual-Mode Data Pipeline**:
   - Seamlessly ingests Kaggle's *The Movies Dataset* (`movies_metadata.csv` and `keywords.csv`).
   - Automatically initializes a high-fidelity synthetic fallback dataset across 10 top genres if CSVs are missing, ensuring instant zero-configuration execution.
2. **Robust NLP Preprocessing**:
   - Safe parsing of JSON/Python literal columns (`genres` and `keywords`) using `ast.literal_eval`.
   - Single-label primary genre extraction filtered to top 10 most frequent genres to resolve class imbalance.
   - Text cleaning (lowercasing, punctuation/numerical removal, and stopword filtering).
   - Fusion of plot synopses (`overview`) and thematic metadata (`keywords`).
3. **Machine Learning & Analytics**:
   - `TfidfVectorizer` (unigrams + bigrams, `max_features=5000`, sublinear term frequency scaling).
   - `MultinomialNB` classifier from `scikit-learn`.
   - Comprehensive model evaluation: overall Accuracy, full Classification Report (Precision, Recall, F1-Score), and high-resolution Confusion Matrix visualization (`confusion_matrix.png`).
4. **Interactive Full-Stack Web Dashboard (Streamlit)**:
   - **Tab 1: Genre Predictor & Recommender**: User-friendly input form with 1-click test presets, real-time genre prediction, probability distribution bar charts, and 3–5 content-based movie recommendations ranked by cosine similarity.
   - **Tab 2: Dataset & Model Analytics**: Metric KPI cards, interactive genre distribution charts, per-genre precision/recall/F1 metrics, confusion matrix display, and a searchable movie database explorer.

---

## 📁 Project Directory Structure

```text
movie_genre_app/
│
├── data/                   # Directory for CSV datasets (or fallback generator)
│   ├── movies_metadata.csv # Kaggle movie metadata or generated fallback
│   └── keywords.csv        # Kaggle keywords or generated fallback
│
├── models/                 # Saved model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── genre_classifier.pkl
│   ├── processed_movies.pkl
│   ├── tfidf_matrix.pkl
│   ├── metrics.json
│   └── confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Data loading, JSON parsing, text cleaning, feature combination
│   ├── train.py            # Model training, evaluation metrics, confusion matrix plot, serialization
│   └── recommend.py        # Inference script for genre prediction & TF-IDF cosine recommendations
│
├── app.py                  # Streamlit Interactive Web Dashboard (UI / Website)
├── requirements.txt        # Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, streamlit, joblib
└── README.md               # Clear step-by-step instructions to install, train, and launch the app
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### 2. Environment Setup & Dependency Installation
Navigate to the project root directory and install dependencies:

```bash
cd movie_genre_app
pip install -r requirements.txt
```

### 3. (Optional) Provide Real Kaggle Dataset
If you wish to run on Kaggle's full *Movies Metadata Dataset*:
1. Download `movies_metadata.csv` and `keywords.csv` from [Kaggle's The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
2. Place both files into the `movie_genre_app/data/` folder.
> *Note: If no CSV files are added, the system automatically creates and uses the built-in realistic fallback dataset.*

### 4. Train the ML Model & Generate Artifacts
Run the standalone training script:

```bash
python src/train.py
```

This will:
- Ingest and preprocess the dataset.
- Filter to the top 10 balanced genres.
- Extract unigram and bigram TF-IDF representations.
- Train the Naive Bayes classifier.
- Calculate and display overall accuracy and the classification report in your console.
- Generate and save `models/confusion_matrix.png` and `confusion_matrix.png`.
- Serialize the model and precomputed vectors into `models/`.

### 5. Launch the Streamlit Web Application
Start the interactive dashboard:

```bash
streamlit run app.py
```

Once launched, your web browser will automatically open to `http://localhost:8501`.

---

## 🖥️ Using the Interactive Dashboard

### Tab 1: Genre Predictor & Recommender
- **Demonstration Presets**: Select any sample from the dropdown (e.g. *Sci-Fi Odyssey*, *Superhero Vigilante*, *Teen Comedy*, *Supernatural Horror*) to auto-populate the inputs instantly.
- **Custom Input**: Type any movie plot synopsis in the **Overview** box and keywords in the **Keywords** box.
- **Predict**: Click **"Classify Genre & Recommend Similar Movies"**.
  - Review the **Predicted Primary Genre** badge and confidence percentage.
  - Explore the **Genre Probability Breakdown** chart showing confidence across all classes.
  - Browse **3–5 similar content recommendations** showing titles, genres, cosine match scores, and overviews.

### Tab 2: Dataset & Model Analytics
- View **KPI Metric Cards**: Model Accuracy, Total Movies, Feature Vocabulary Size, and Evaluation Test Split.
- Examine the **Target Genre Distribution** bar chart.
- Inspect the **Confusion Matrix** heatmap image displaying true vs. predicted classifications.
- Review the **Per-Genre Precision, Recall, and F1-score** table.
- Search the entire movie corpus using the **Corpus Database Explorer**.

---

## 🧪 Testing the Recommendation Engine in CLI
You can also run inference directly from the command line:

```bash
python src/recommend.py
```

---

## 🌐 Deployment Guide (GitHub & Vercel)

This project is dual-configured:
1. **Vercel Serverless Deployment**: Powered by `vercel.json` and `api/index.py` (Flask REST API + modern full-stack web UI with 0ms cold starts).
2. **Streamlit Deployment**: Powered by `app.py` (Streamlit UI for local execution or Streamlit Community Cloud).

---

### Step 1: Push to GitHub

1. Create a new repository on [GitHub](https://github.com/new) (e.g. `movie-genre-classification`). Leave it empty (do **not** check initialize with README/license).
2. Open PowerShell or terminal in `movie_genre_app`:
   ```bash
   cd movie_genre_app

   # Link your GitHub repository (replace with your actual GitHub username and repository name)
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git

   # Push to main branch
   git push -u origin main
   ```

---

### Step 2: Deploy to Vercel (1-Click via GitHub)

1. Go to [Vercel](https://vercel.com/) and sign in with your GitHub account.
2. Click **"Add New..."** ➔ **"Project"**.
3. Select your GitHub repository from the list and click **"Import"**.
4. In the Project Configuration:
   - **Framework Preset**: *Other* (Vercel will automatically read `vercel.json`)
   - **Root Directory**: `./` (or leave default)
5. Click **"Deploy"**.
6. Within 60 seconds, your project will be live with a production HTTPS URL (e.g. `https://movie-genre-app.vercel.app`)!

#### Deploying via Vercel CLI (Alternative):
```bash
npm install -g vercel
cd movie_genre_app
vercel
```

---

### Step 3 (Optional): Deploy Streamlit UI to Streamlit Community Cloud

If you also wish to host the native Streamlit dashboard publicly for free:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New app"**.
3. Select your repository, set **Branch** to `main`, and **Main file path** to `app.py`.
4. Click **"Deploy!"**.

---

## 🛠️ Technology Stack
- **Language**: Python 3.11
- **NLP & Feature Engineering**: TF-IDF (`scikit-learn`), Regular Expressions (`re`), Safe JSON Parsing (`ast.literal_eval`), NLTK Stopwords
- **Machine Learning**: `MultinomialNB` (`scikit-learn`)
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `streamlit`
- **Model Serialization**: `joblib`
