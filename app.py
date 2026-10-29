"""
Movie & Anime Genre Classification & Hybrid Recommender Dashboard (Streamlit).
Multi-label NLP Classification with TF-IDF, Logistic Regression, and Cosine-Overlap Recommendations.
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
    page_title="CineVerse AI | Movie & Anime Multi-Label Genre Classifier",
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

# Custom CSS for Sleek Anime Streaming Dark & Vibrant Orange Theme
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #141519;
        color: #F7F7F8;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Brand Header */
    .brand-logo-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background-color: #FF640A;
        color: #000000 !important;
        font-weight: 900;
        font-size: 1.1rem;
        border-radius: 50%;
        margin-right: 0.6rem;
        vertical-align: middle;
        box-shadow: 0 0 16px rgba(255, 100, 10, 0.35);
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 900;
        color: #FFFFFF;
        display: inline-flex;
        align-items: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .orange-text {
        color: #FF640A;
    }
    .sub-header {
        color: #A0A0B2;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #1E1F25;
        border: 1px solid #2D2F37;
        border-top: 3px solid #FF640A;
        border-radius: 8px;
        padding: 1.25rem;
        color: white;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #FF640A;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #FF640A;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #A0A0B2;
        font-weight: 700;
    }

    /* Prediction Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, #1E1F25 0%, #16171C 100%);
        border: 1px solid #2D2F37;
        border-left: 6px solid #FF640A;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .genre-badge {
        display: inline-block;
        background-color: #FF640A;
        color: #FFFFFF;
        padding: 0.4rem 1.2rem;
        border-radius: 6px;
        font-weight: 900;
        font-size: 1.35rem;
        letter-spacing: -0.01em;
        box-shadow: 0 0 14px rgba(255, 100, 10, 0.35);
    }
    
    /* Movie Recommendation Card */
    .rec-card {
        background-color: #1E1F25;
        border: 1px solid #2D2F37;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.9rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #FF640A;
        box-shadow: 0 4px 18px rgba(255, 100, 10, 0.2);
    }
    .rec-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .genre-pill {
        background-color: #2D2F37;
        color: #E2E2EC;
        padding: 0.18rem 0.6rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.5rem;
        border: 1px solid #3A3D47;
    }
    .sim-badge {
        background-color: rgba(255, 100, 10, 0.15);
        border: 1px solid #FF640A;
        color: #FF640A;
        padding: 0.18rem 0.55rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 800;
    }
    .rec-overview {
        font-size: 0.88rem;
        color: #A0A0B2;
        line-height: 1.5;
        margin-top: 0.45rem;
    }

    /* Word Chips */
    .word-chip {
        display: inline-block;
        background: #24262E;
        border: 1px solid #2E303A;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_app_engine():
    """Load or initialize multi-label recommendation engine."""
    from recommend import MultiLabelRecommenderEngine
    try:
        engine = MultiLabelRecommenderEngine(models_dir=str(MODELS_DIR))
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
    """Trigger the multi-label training script with UI status feedback."""
    from train import train_model
    with st.spinner("Executing Data Preprocessing, TF-IDF Vectorization, and Multi-Label Training..."):
        metrics = train_model(data_dir=str(DATA_DIR), models_dir=str(MODELS_DIR))
    st.cache_resource.clear()
    st.cache_data.clear()
    return metrics


# Preset sample movies for one-click testing
SAMPLE_PRESETS = {
    "Select an archetype preset...": {"overview": "", "keywords": ""},
    "Sci-Fi Odyssey (Interstellar)": {
        "overview": "A team of exploratory astronauts travel through an enigmatic wormhole near Saturn in a desperate bid to find a habitable exoplanet and guarantee humanity survival before global crop blights extinguish all life.",
        "keywords": "space travel, wormhole, black hole, time dilation, astronaut, astrophysics, planet discovery"
    },
    "Superhero Vigilante (The Dark Knight)": {
        "overview": "Batman raises the stakes in his war on crime. With the help of allies Lt. Jim Gordon and DA Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets of Gotham, until a psychopathic criminal mastermind known as the Joker thrusts the city into anarchy.",
        "keywords": "superhero, gotham city, vigilante, crime fighter, explosions, joker, chaos"
    },
    "Demon Slayer Anime (Dark Fantasy & Action)": {
        "overview": "After a demon savagely massacres his entire family and curses his younger sister into a demon, a determined teenage boy joins an elite corps of demon hunters wielding ancient sun-breathing sword techniques to avenge his parents and restore his sister humanity.",
        "keywords": "anime, demon slayer, katana swords, supernatural powers, vengeance, dark fantasy, martial arts"
    },
    "Paranormal Horror (The Conjuring)": {
        "overview": "Paranormal investigators Ed and Lorraine Warren work to help a terrorized family living in a secluded farmhouse. Ancient demonic entities and supernatural possession push the couple to their absolute psychological limits.",
        "keywords": "demonology, haunted house, paranormal investigation, possession, exorcism, evil spirits"
    },
    "High School Teen Comedy (Superbad)": {
        "overview": "Two co-dependent high school seniors are forced to deal with intense separation anxiety after their plan to stage a booze-soaked party goes wildly awry with fake IDs, eccentric rogue cops, and chaotic misadventures.",
        "keywords": "high school, teen comedy, fake id, wild party, friendship, coming of age, police chase"
    },
    "Cyberpunk Anime (Ghost in the Shell)": {
        "overview": "In a futuristic cybernetic metropolis, a cyborg public-security agent hunts a mysterious cyber-terrorist hacker known as the Puppet Master, discovering profound existential secrets regarding artificial consciousness and human identity.",
        "keywords": "cyberpunk, cyborg, artificial intelligence, anime, hacker, sci-fi thriller, philosophy"
    },
    "Romantic Musical (La La Land)": {
        "overview": "While navigating their burgeoning careers in Los Angeles, an impassioned jazz pianist and an aspiring actress fall passionately in love while attempting to reconcile their dreams for the future with their emotional devotion.",
        "keywords": "jazz musician, aspiring actress, los angeles romance, musical love story, bittersweet dreams"
    },
    "Studio Ghibli Anime (Spirited Away)": {
        "overview": "While exploring an abandoned amusement park, a young girl discovers a mystical world of ancient spirits and Japanese gods. When her parents are magically transformed into pigs by a vengeful witch, she must work at a grand spirit bathhouse to free them.",
        "keywords": "anime, studio ghibli, spirit world, witch, bathhouse, fantasy curse, coming of age"
    }
}


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎬 Project Architecture")
    st.markdown("""
    **Multi-Label Movie & Anime Classification (NLP)**
    * **Algorithm:** One-vs-Rest Logistic Regression (Balanced)
    * **Text Vectorizer:** Sublinear TF-IDF (1 & 2-grams, 6,000 features)
    * **Stemming:** Snowball Stemmer + Lemmatization Dictionary
    * **Classes:** 12 Multi-Label Genres
    * **Probabilities:** Calibrated Sigmoids for Top 3 Genres
    * **Recommendation:** Hybrid 60% Cosine + 25% Jaccard Overlap + 15% Rating
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Control")
    if st.button("🔄 Retrain Multi-Label Model", use_container_width=True, type="secondary"):
        try:
            metrics_res = run_pipeline_trigger()
            st.success(f"Model retrained! Hamming Accuracy: {metrics_res.get('hamming_accuracy', 0)*100:.1f}%")
            st.rerun()
        except Exception as ex:
            st.error(f"Retraining error: {ex}")

    st.markdown("---")
    metrics_data = load_metrics()
    if metrics_data:
        st.markdown("### 📊 Benchmark Metrics")
        st.metric("Hamming Accuracy", f"{metrics_data.get('hamming_accuracy', 0.0) * 100:.1f}%")
        st.metric("Micro F1-Score", f"{metrics_data.get('micro_f1', 0.0) * 100:.1f}%")
        st.metric("Top-3 Hit Rate", f"{metrics_data.get('top3_hit_rate', 0.0) * 100:.1f}%")
        st.metric("Catalog Titles", f"{metrics_data.get('dataset_size', 0):,}")
    else:
        st.warning("Model artifacts not yet initialized.")


# --- MAIN HEADER ---
st.markdown('<div class="main-header"><span class="brand-logo-icon">▷</span> CINE<span class="orange-text">VERSE</span> AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Label NLP Classification & Hybrid Content Recommender across 4,680+ cinema and anime titles.</div>', unsafe_allow_html=True)

# Check model availability; offer training prompt if absent
engine, load_error = load_app_engine()
if engine is None or not engine.is_ready():
    st.warning("⚠️ **Model Artifacts Not Detected.** The multi-label classifier and matrices have not been initialized yet.")
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
        st.info("Clicking above will execute `src/train.py`, process TMDB 5000 + Anime data, and generate all artifacts.")
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
    st.markdown("### 📝 Input Movie or Anime Synopsis")
    st.caption("Provide a storyline summary (minimum 15 words) and thematic keywords to predict multi-label genres and discover similar content.")

    selected_preset_key = st.selectbox(
        "⚡ Quick Archetype Presets (Select to auto-fill form):",
        options=list(SAMPLE_PRESETS.keys()),
        index=0
    )
    preset_data = SAMPLE_PRESETS[selected_preset_key]

    with st.form("prediction_form", clear_on_submit=False):
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            input_overview = st.text_area(
                "Plot Overview / Synopsis (15 words minimum):",
                value=preset_data["overview"],
                placeholder="e.g. A young ninja with a sealed spirit beast inside him trains relentlessly to earn the respect of his village and become the ultimate leader...",
                height=150,
                help="The narrative summary of the title. Preprocessed via Snowball lemmatization and TF-IDF."
            )
        
        with col_input2:
            input_keywords = st.text_area(
                "Thematic Keywords / Tags (Optional):",
                value=preset_data["keywords"],
                placeholder="e.g. space travel, ninja, superhero, exorcism, friendship, artificial intelligence",
                height=150,
                help="Key thematic tags or motifs providing contextual weight."
            )
            
            top_rec_count = st.slider("Number of Recommendations:", min_value=3, max_value=8, value=4)

        submit_btn = st.form_submit_button("⚡ Classify Genres & Discover Similar Titles", use_container_width=True, type="primary")

    if submit_btn or (input_overview.strip() and selected_preset_key != "Select an archetype preset..."):
        words = input_overview.strip().split()
        if len(words) < 15:
            st.error(f"Please enter at least 15 words for reliable NLP feature classification (currently {len(words)} words).")
        else:
            with st.spinner("Analyzing narrative features with Multi-Label OneVsRest Pipeline..."):
                pred_result = engine.predict_genres(input_overview, input_keywords, top_k=3)
                recs = engine.recommend_movies(input_overview, input_keywords, top_n=top_rec_count)

            # --- PREDICTION BANNER ---
            primary_genre = pred_result["primary_genre"]
            primary_pct = pred_result["primary_percentage"]

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 700;">
                    Primary Genre Classification
                </div>
                <div style="margin-top: 0.4rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <span class="genre-badge">{primary_genre}</span>
                    <span style="font-size: 1.15rem; color: #E2E8F0;">
                        Calibrated Probability: <strong style="color:#FF640A;">{primary_pct}</strong>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Low Confidence Warning Alert
            if pred_result.get("low_confidence_warning"):
                st.warning(f"⚠️ {pred_result.get('low_confidence_message')}")

            # Top 3 Genres Columns
            col_t1, col_t2, col_t3 = st.columns(3)
            top_preds = pred_result.get("top_predictions", [])
            cols = [col_t1, col_t2, col_t3]
            ranks = ["#1 Primary Genre", "#2 Secondary Genre", "#3 Tertiary Genre"]

            for i, p in enumerate(top_preds[:3]):
                with cols[i]:
                    st.metric(
                        label=ranks[i],
                        value=p["genre"],
                        delta=p["percentage"]
                    )
                    st.progress(float(p["probability"]))

            # Explainability: Why this prediction?
            inf_words = pred_result.get("influential_words", [])
            if inf_words:
                st.markdown("#### 🔍 Why this prediction? (Top Influential Narrative Signals)")
                chips_html = "".join([f'<span class="word-chip">{w["word"]} <span style="color:#FF640A;">+{w["impact"]}</span></span>' for w in inf_words])
                st.markdown(chips_html, unsafe_allow_html=True)

            # --- RECOMMENDATION CARDS ---
            st.markdown("---")
            st.markdown(f"### 🎯 Content-Based Recommendations (Top {len(recs)} Matches)")
            st.caption("Hybrid Scoring = 60% TF-IDF Cosine Similarity + 25% Genre Jaccard Overlap + 15% Rating Normalization.")

            if not recs:
                st.info("No close recommendations found matching your criteria.")
            else:
                for idx, movie in enumerate(recs, 1):
                    sim_pct = movie["similarity_percentage"]
                    genres_str = " • ".join(movie.get("genres", []))
                    rating = movie.get("vote_average", 0.0)
                    year = movie.get("release_year", "N/A")
                    
                    st.markdown(f"""
                    <div class="rec-card">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;">
                            <div class="rec-title">#{idx}. {movie['title']} ({year})</div>
                            <div>
                                <span class="genre-pill">⭐ {rating:.1f}/10</span>
                                <span class="genre-pill">{genres_str}</span>
                                <span class="sim-badge">Match: {sim_pct}</span>
                            </div>
                        </div>
                        <div class="rec-overview">
                            {movie['overview']}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #94A3B8;">
                            <strong>Themes:</strong> {movie.get('keywords', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ==========================================
# TAB 2: DATASET & MODEL ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📊 Multi-Label Performance & Diagnostics")
    st.caption("Empirical metrics computed across 4,680 titles using 5-Fold Cross Validation.")

    metrics = load_metrics()

    if not metrics:
        st.info("No training metrics file located. Please click 'Retrain Multi-Label Model' on the sidebar.")
    else:
        # Top KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Hamming Accuracy</div>
                <div class="metric-value">{metrics.get('hamming_accuracy', 0.0) * 100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #34D399;">Multi-Label Precision</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Micro F1-Score</div>
                <div class="metric-value">{metrics.get('micro_f1', 0.0) * 100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #FF8A3D;">Global Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Top-3 Hit Rate</div>
                <div class="metric-value">{metrics.get('top3_hit_rate', 0.0) * 100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #34D399;">At Least 1 True Genre Hit</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Catalog Titles</div>
                <div class="metric-value">{metrics.get('dataset_size', 0):,}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">TMDB 5000 + Curated Anime</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts: Genre Distribution & 5-Fold CV Benchmark
        col_dist, col_bm = st.columns([1, 1])

        with col_dist:
            st.markdown("#### 🎭 Target Genre Distribution Across Catalog")
            genre_dist = metrics.get("genre_distribution", {})
            if genre_dist:
                dist_df = pd.DataFrame(
                    list(genre_dist.items()),
                    columns=["Genre", "Movie Count"]
                ).set_index("Genre")
                st.bar_chart(dist_df, color="#FF640A", height=320)

        with col_bm:
            st.markdown("#### ⚖️ 5-Fold Cross Validation Benchmark")
            bm = metrics.get("benchmark_comparison", {})
            if bm:
                bm_rows = []
                for mod_name, scores in bm.items():
                    bm_rows.append({
                        "Classifier": mod_name.replace(" (Balanced)", ""),
                        "Hamming Acc": f"{scores.get('mean_hamming_acc', 0)*100:.1f}%",
                        "Micro F1": f"{scores.get('mean_micro_f1', 0)*100:.1f}%",
                        "Macro F1": f"{scores.get('mean_macro_f1', 0)*100:.1f}%",
                        "CV Time": f"{scores.get('total_cv_time_sec', 0):.1f}s"
                    })
                st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)

        # Detailed Classification Report
        st.markdown("---")
        st.markdown("#### 📋 Detailed Per-Genre Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            report_rows = []
            for genre_name, scores in report.items():
                if isinstance(scores, dict) and genre_name not in ["accuracy", "macro avg", "weighted avg", "samples avg"]:
                    report_rows.append({
                        "Genre": genre_name,
                        "Precision": f"{scores.get('precision', 0.0):.3f}",
                        "Recall": f"{scores.get('recall', 0.0):.3f}",
                        "F1-Score": f"{scores.get('f1-score', 0.0):.3f}",
                        "Support (Samples)": int(scores.get("support", 0))
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        # Top Predictive Words per Genre
        pred_words = metrics.get("top_predictive_words", {})
        if pred_words:
            st.markdown("---")
            st.markdown("#### 🔤 Top Predictive Vocabulary Terms per Genre")
            selected_genre_words = st.selectbox("Inspect Top 15 Informative Words for Genre:", list(pred_words.keys()))
            words_list = pred_words.get(selected_genre_words, [])
            if words_list:
                words_df = pd.DataFrame(words_list)
                st.dataframe(words_df, use_container_width=True, hide_index=True)
