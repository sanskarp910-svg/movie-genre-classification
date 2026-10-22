"""
Movie Genre Classification & Recommendation Web Application
Interactive Dashboard built with Streamlit for Content Recommendation and Media Analysis.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration - Must be first Streamlit command
st.set_page_config(
    page_title="CineGenre AI | Movie Genre Classification & Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Main Theme & Typography */
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #7928CA 50%, #4338CA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }

    /* Prediction Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
    }
    .prediction-badge {
        display: inline-block;
        background-color: #6366F1;
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }
    
    /* Movie Recommendation Card */
    .rec-card {
        background: #1E293B;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    .rec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
    }
    .rec-meta {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    .genre-pill {
        background: #334155;
        color: #E2E8F0;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .sim-badge {
        background: #065F46;
        color: #6EE7B7;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .rec-overview {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.45;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_app_engine():
    """Load or initialize model artifacts and recommendation engine."""
    from recommend import GenreRecommenderEngine
    try:
        engine = GenreRecommenderEngine(models_dir=str(MODELS_DIR))
        return engine, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load evaluation metrics JSON file."""
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_pipeline_trigger():
    """Trigger the training script programmatically with UI status feedback."""
    from train import train_model
    with st.spinner("Executing Data Preprocessing, TF-IDF Vectorization, and Naive Bayes Training..."):
        metrics = train_model(data_dir=str(DATA_DIR), models_dir=str(MODELS_DIR))
    st.cache_resource.clear()
    st.cache_data.clear()
    return metrics


# Preset sample movies for one-click testing
SAMPLE_PRESETS = {
    "Select an example...": {"overview": "", "keywords": ""},
    "Sci-Fi Odyssey (Space Exploration)": {
        "overview": "A team of exploratory astronauts travel through an enigmatic wormhole near the edge of the solar system in a desperate bid to guarantee humanity's long-term survival before Earth's crops completely fail.",
        "keywords": "space travel, wormhole, black hole, time dilation, astronaut, astrophysics"
    },
    "Superhero Vigilante (Action / Urban Crime)": {
        "overview": "A masked billionaire crimefighter raises the stakes against Gotham's criminal syndicate, but faces catastrophic anarchy when a psychopathic criminal mastermind unleashes pure chaos upon the terrified citizens.",
        "keywords": "superhero, gotham city, vigilante, crime fighter, explosions, joker"
    },
    "High School Teen Comedy": {
        "overview": "Two hopelessly awkward high school seniors attempt to purchase illegal alcohol with an absurdly fake identification card for a massive graduation party, stumbling into a series of wild mishaps with eccentric local patrol officers.",
        "keywords": "high school, fake id, teen comedy, wild party, friendship, coming of age"
    },
    "Paranormal Supernatural Horror": {
        "overview": "Renowned paranormal investigators are summoned to a remote Rhode Island farmhouse where a sinister demonic spirit has possessed an innocent family, demanding an agonizing and terrifying exorcism ritual.",
        "keywords": "haunted house, demonic possession, exorcism, evil spirits, paranormal investigation"
    },
    "Romantic Melodrama (Musical & Passion)": {
        "overview": "A devoted jazz pianist and a struggling auditioning actress fall deeply in love beneath the starry skies of Los Angeles, navigating the heartbreaking conflict between their grand career ambitions and true romance.",
        "keywords": "jazz musician, aspiring actress, los angeles romance, soulmates, bittersweet dreams"
    },
    "Animated Family Adventure (Sentient Toys)": {
        "overview": "A vintage cowboy doll finds his world upended and experiences intense jealousy when his young owner receives an ultra-modern space ranger action figure, embarking on a dangerous quest across town to return safely home.",
        "keywords": "toys come to life, cowboy, space ranger, pixar animation, childhood, friendship"
    }
}


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎬 Project Architecture")
    st.markdown("""
    **Movie Genre Classification (NLP)**
    * **Algorithm:** Multinomial Naive Bayes (`MultinomialNB`)
    * **Text Vectorizer:** TF-IDF (Unigrams & Bigrams, 5000 features)
    * **Features:** Merged `overview` (plot) + `keywords`
    * **Target:** Single-label Primary Genre (Top 10 Classes)
    * **Recommendation:** Cosine Similarity Content Filtering
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Control")
    if st.button("🔄 Retrain ML Model", use_container_width=True, type="secondary"):
        try:
            metrics_res = run_pipeline_trigger()
            st.success(f"Model retrained! Accuracy: {metrics_res.get('accuracy', 0)*100:.2f}%")
            st.rerun()
        except Exception as ex:
            st.error(f"Retraining error: {ex}")

    st.markdown("---")
    metrics_data = load_metrics()
    if metrics_data:
        st.markdown("### 📊 Active Model Health")
        acc = metrics_data.get("accuracy", 0.0)
        st.metric("Test Accuracy", f"{acc * 100:.2f}%")
        st.metric("Corpus Size", f"{metrics_data.get('total_movies', 0)} movies")
        st.metric("Vocabulary", f"{metrics_data.get('vocab_size', 0)} n-grams")
    else:
        st.warning("Model artifacts not yet initialized.")


# --- MAIN HEADER ---
st.markdown('<div class="main-header">🎬 CineGenre AI: Movie Genre Classifier & Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade Natural Language Processing pipeline classifying film synopses into genres with Naive Bayes & TF-IDF similarity recommendations.</div>', unsafe_allow_html=True)

# Check model availability; offer training prompt if absent
engine, load_error = load_app_engine()
if engine is None or not engine.is_ready():
    st.warning("⚠️ **Model Artifacts Not Detected.** The Naive Bayes classifier and TF-IDF matrices have not been generated yet.")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            try:
                run_pipeline_trigger()
                st.success("Model trained successfully! Reloading...")
                st.rerun()
            except Exception as err:
                st.error(f"Training failed: {err}")
    with col_b:
        st.info("Clicking above will execute `src/train.py`, verify datasets/fallbacks, and build all artifacts automatically.")
    st.stop()


# --- TABS INTERFACE ---
tab1, tab2 = st.tabs([
    "🎯 Genre Predictor & Recommender",
    "📈 Dataset & Model Analytics"
])


# ==========================================
# TAB 1: GENRE PREDICTOR & RECOMMENDER
# ==========================================
with tab1:
    st.markdown("### 📝 Input Movie Synopsis & Keyword Themes")
    st.caption("Provide a plot overview and contextual keywords to evaluate the classifier and discover similar movies in the catalog.")

    # Preset selection for rapid demonstration
    selected_preset_key = st.selectbox(
        "⚡ Quick Demonstration Preset (select to auto-fill form):",
        options=list(SAMPLE_PRESETS.keys()),
        index=0
    )
    preset_data = SAMPLE_PRESETS[selected_preset_key]

    with st.form("prediction_form", clear_on_submit=False):
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            input_overview = st.text_area(
                "Movie Plot Summary (Overview):",
                value=preset_data["overview"],
                placeholder="e.g. In a war-torn galaxy, a farm boy discovers an ancient energy weapon and joins a rebellion against an oppressive empire...",
                height=140,
                help="The narrative summary of the film. Cleaned and processed via TF-IDF."
            )
        
        with col_input2:
            input_keywords = st.text_area(
                "Movie Keywords / Themes (comma or space separated):",
                value=preset_data["keywords"],
                placeholder="e.g. space battles, rebellion, galactic empire, chosen one, mentor",
                height=140,
                help="Key thematic tags or motifs. Extracted from the keywords corpus."
            )
            
            top_rec_count = st.slider("Number of Content Recommendations:", min_value=3, max_value=5, value=4)

        submit_btn = st.form_submit_button("🚀 Classify Genre & Recommend Similar Movies", use_container_width=True, type="primary")

    if submit_btn or (input_overview.strip() and selected_preset_key != "Select an example..."):
        if not input_overview.strip():
            st.error("Please enter a movie plot summary (overview) to proceed with classification.")
        else:
            with st.spinner("Analyzing text corpus with TF-IDF and Naive Bayes..."):
                pred_result = engine.predict_genre(input_overview, input_keywords, top_k=5)
                recs = engine.recommend_movies(input_overview, input_keywords, top_n=top_rec_count)

            # --- PREDICTION BANNER ---
            primary_genre = pred_result["primary_genre"]
            confidence = pred_result["confidence"]
            confidence_pct = pred_result["confidence_percentage"]

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600;">
                    Predicted Primary Genre
                </div>
                <div style="margin-top: 0.4rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <span class="prediction-badge">{primary_genre}</span>
                    <span style="font-size: 1.15rem; color: #E2E8F0;">
                        Confidence Score: <strong>{confidence_pct}</strong>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Prediction details layout: Probability Distribution + NLP Tokens
            col_chart, col_nlp = st.columns([3, 2])

            with col_chart:
                st.markdown("#### 📊 Genre Probability Breakdown")
                prob_df = pd.DataFrame(pred_result["top_predictions"])
                prob_df["probability_pct"] = prob_df["probability"] * 100
                st.bar_chart(
                    data=prob_df.set_index("genre")["probability_pct"],
                    color="#6366F1",
                    height=220
                )

            with col_nlp:
                st.markdown("#### 🔍 NLP Preprocessed Features")
                st.markdown(f"**Cleaned Tokens Input ({len(pred_result['cleaned_query'].split())} tokens):**")
                token_preview = " • ".join(pred_result["cleaned_query"].split()[:25])
                st.info(f"{token_preview}..." if len(pred_result["cleaned_query"].split()) > 25 else token_preview)
                st.caption("Lowercased, punctuation & numbers removed, stopwords filtered.")

            # --- RECOMMENDATION CARDS ---
            st.markdown("---")
            st.markdown(f"### 🎯 Content-Based Movie Recommendations (Top {len(recs)} Matches)")
            st.caption("Generated via TF-IDF Cosine Similarity against the processed movie database.")

            if not recs:
                st.info("No close recommendations found for this input.")
            else:
                for idx, movie in enumerate(recs, 1):
                    sim_pct = movie["similarity_percentage"]
                    sim_score = movie["similarity_score"]
                    
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;">
                            <div class="rec-title">#{idx}. {movie['title']}</div>
                            <div>
                                <span class="genre-pill">🎭 {movie['genre']}</span>
                                <span class="sim-badge">Cosine Match: {sim_pct}</span>
                            </div>
                        </div>
                        <div class="rec-overview">
                            {movie['overview']}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #94A3B8;">
                            <strong>Themes:</strong> {movie['keywords'] if movie['keywords'] else 'N/A'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ==========================================
# TAB 2: DATASET & MODEL ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📊 Dataset Statistics & Machine Learning Metrics")
    st.caption("Empirical performance evaluation, class distributions, and confusion matrix diagnostics.")

    metrics = load_metrics()

    if not metrics:
        st.info("No training metrics file located. Please click 'Retrain ML Model' on the sidebar to compute evaluation metrics.")
    else:
        # Top KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">{metrics.get('accuracy', 0.0) * 100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #34D399;">MultinomialNB</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Catalog Size</div>
                <div class="metric-value">{metrics.get('total_movies', 0)}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Filtered Top 10 Genres</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TF-IDF Features</div>
                <div class="metric-value">{metrics.get('vocab_size', 0)}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Unigrams & Bigrams</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Evaluation Test Split</div>
                <div class="metric-value">{metrics.get('test_size', 0)}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Test Samples (20%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row: Genre Distribution + Confusion Matrix
        col_dist, col_cm = st.columns([1, 1])

        with col_dist:
            st.markdown("#### 🎭 Target Genre Distribution (Top 10)")
            genre_dist = metrics.get("genre_distribution", {})
            if genre_dist:
                dist_df = pd.DataFrame(
                    list(genre_dist.items()),
                    columns=["Genre", "Movie Count"]
                ).set_index("Genre")
                st.bar_chart(dist_df, color="#8B5CF6", height=380)

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix Diagnostic")
            cm_img_path = MODELS_DIR / "confusion_matrix.png"
            if not cm_img_path.exists():
                cm_img_path = APP_DIR / "confusion_matrix.png"

            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Multinomial Naive Bayes Test Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion matrix image pending generation.")

        # Classification Report Table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Per-Genre Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            report_rows = []
            for genre_name, scores in report.items():
                if isinstance(scores, dict) and genre_name not in ["accuracy", "macro avg", "weighted avg"]:
                    report_rows.append({
                        "Genre": genre_name,
                        "Precision": f"{scores.get('precision', 0.0):.3f}",
                        "Recall": f"{scores.get('recall', 0.0):.3f}",
                        "F1-Score": f"{scores.get('f1-score', 0.0):.3f}",
                        "Support (Samples)": int(scores.get("support", 0))
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        # Interactive Dataset Explorer
        st.markdown("---")
        st.markdown("#### 🔍 Corpus Database Explorer")
        if engine and engine.processed_corpus is not None and not engine.processed_corpus.empty:
            search_query = st.text_input("Filter database by title, genre, or keyword:", placeholder="e.g. Dark Knight, Space, Comedy...")
            display_df = engine.processed_corpus[["id_clean", "title", "primary_genre", "overview", "keywords_text"]].copy()
            display_df.columns = ["ID", "Movie Title", "Primary Genre", "Overview", "Keywords"]

            if search_query.strip():
                q = search_query.strip().lower()
                display_df = display_df[
                    display_df["Movie Title"].str.lower().str.contains(q, na=False) |
                    display_df["Primary Genre"].str.lower().str.contains(q, na=False) |
                    display_df["Keywords"].str.lower().str.contains(q, na=False) |
                    display_df["Overview"].str.lower().str.contains(q, na=False)
                ]

            st.dataframe(display_df, use_container_width=True, height=280)
            st.caption(f"Showing {len(display_df)} movies in current filter.")
