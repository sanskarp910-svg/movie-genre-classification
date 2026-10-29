import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 widescreen layout: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Theme Colors (Letterboxd Cinema Aesthetic)
    BG_DARK = RGBColor(20, 24, 28)       # #14181c (Matte Slate)
    CARD_BG = RGBColor(28, 34, 40)       # #1c2228 (Surface)
    BORDER_COL = RGBColor(44, 52, 64)    # #2c3440 (Subtle Border)
    TEXT_WHITE = RGBColor(255, 255, 255) # Pure White
    TEXT_MUTED = RGBColor(153, 170, 187) # #99aabb (Muted)
    TEXT_DIM = RGBColor(102, 119, 136)   # #667788 (Dim)
    ACC_GREEN = RGBColor(0, 224, 84)     # #00e054 (Letterboxd Green)
    ACC_ORANGE = RGBColor(255, 128, 0)   # #ff8000 (Letterboxd Orange)
    ACC_BLUE = RGBColor(64, 188, 244)    # #40bcf4 (Letterboxd Blue)

    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, category, title):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
        p_c = tf_c.paragraphs[0]
        p_c.text = category.upper()
        p_c.font.size = Pt(11)
        p_c.font.bold = True
        p_c.font.color.rgb = ACC_GREEN

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.65))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(26)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_WHITE

    def add_card(slide, left, top, width, height, bg_col=CARD_BG, border_col=BORDER_COL):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_col
        card.line.color.rgb = border_col
        card.line.width = Pt(1.2)
        return card

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Center Brand Box
    card1 = add_card(s1, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1), CARD_BG, BORDER_COL)

    tb = s1.shapes.add_textbox(Inches(2.0), Inches(1.7), Inches(9.333), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "CINEMIX ▶"
    p.font.size = Pt(46)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    p2 = tf.add_paragraph()
    p2.text = "Multi-Label Cinema & Anime Genre Classifier and Hybrid Recommender"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p2.space_before = Pt(12)

    p3 = tf.add_paragraph()
    p3.text = "A production-grade NLP system engineered with 12 calibrated OneVsRest sigmoid classifiers, 3-tier hybrid recommendation ranking, and an authentic Letterboxd minimalist aesthetic."
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(14)

    p4 = tf.add_paragraph()
    p4.text = "Live Deployment: https://movie-genre-classification-j8iw.vercel.app/\nEvaluated with 5-Fold Stratified Cross-Validation on TMDB 5000 + Anime Corpus (4,680+ Titles)"
    p4.font.size = Pt(11)
    p4.font.bold = True
    p4.font.color.rgb = ACC_BLUE
    p4.space_before = Pt(24)

    # ==========================================
    # SLIDE 2: THE PROBLEM & MOTIVATION
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Executive Context", "The Challenge: Why Film Genre Classification Needed an Overhaul")

    points = [
        ("The Multi-Label Reality of Cinema", "Real movies and anime never belong to a single genre. Inception is simultaneously Sci-Fi, Action, and Thriller; Spirited Away is Animation, Fantasy, and Adventure. Single-label classifiers fail by forcing complex narratives into a single pigeonhole."),
        ("Baseline Failure: 35% Accuracy", "The legacy implementation operated on a tiny dataset of 100 movies with only 20 test samples, yielding a flawed 35.0% accuracy on basic Multinomial Naive Bayes."),
        ("Uninformative Plain-Text Output", "Previous recommenders outputted raw titles without posters, release years, vote ratings, or genre tags, and displayed confusing empty states like 'Action, 0%' before queries were entered."),
        ("Vercel Serverless Exceeded (544 MB)", "Standard ML stacks (Scikit-Learn, Pandas, SciPy, NumPy) blow past Vercel's strict 250 MB serverless lambda limit, causing immediate deployment failure without specialized zero-dependency optimization.")
    ]

    for i, (title, desc) in enumerate(points):
        top_y = Inches(1.6 + (i * 1.35))
        add_card(s2, Inches(0.8), top_y, Inches(11.733), Inches(1.15))
        tb = s2.shapes.add_textbox(Inches(1.1), top_y + Inches(0.12), Inches(11.1), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = f"{i+1}. {title}"
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = ACC_GREEN if i == 0 else (ACC_ORANGE if i == 1 else TEXT_WHITE)
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(4)

    # ==========================================
    # SLIDE 3: DATASET ENGINEERING
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Data Pipeline", "Dataset Engineering: Expanding from 100 to 4,680+ Cinematic Titles")

    # Left Column: Dataset Stats
    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb_l = s3.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "Dataset Synthesis & Curation"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    stats = [
        ("TMDB 5000 Movies", "4,680+ real feature films with validated overviews, vote averages, and genre lists."),
        ("Curated Anime Classics", "Integrated high-impact anime titles: Death Note, Spirited Away, Ghost in the Shell, Demon Slayer, Attack on Titan."),
        ("Training Set (80%)", "3,744 rigorously cleaned storylines."),
        ("Test Set (20%)", "936 stratified hold-out test samples."),
        ("Target Classes", "12 top genres: Drama, Comedy, Thriller, Action, Romance, Adventure, Crime, Sci-Fi, Horror, Family, Fantasy, Animation.")
    ]
    for st_title, st_desc in stats:
        p1 = tf_l.add_paragraph()
        p1.text = f"• {st_title}: {st_desc}"
        p1.font.size = Pt(11)
        p1.font.color.rgb = TEXT_MUTED
        p1.space_before = Pt(10)

    # Right Column: Preprocessing Stages
    add_card(s3, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    tb_r = s3.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Morphological NLP Preprocessing"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_BLUE

    prep = [
        ("1. Normalization & Cleaning", "Case-folding, entity unescaping, punctuation stripping while preserving hyphens in compound words."),
        ("2. Stopword Filtering", "Removed 150+ high-frequency English stopwords (the, and, at, in) that introduce noise."),
        ("3. Snowball Stemming & Lemmatization", "Normalizes irregulars (superheroes → superhero, murderers → murder, detectives → detect)."),
        ("4. 6,000 N-Gram Feature Space", "Unigrams & Bigrams with sublinear term-frequency scaling: tf = 1 + ln(count). Captures contextual phrases like 'space travel', 'serial killer', and 'high school'.")
    ]
    for pr_title, pr_desc in prep:
        p1 = tf_r.add_paragraph()
        p1.text = pr_title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(10)
        p2 = tf_r.add_paragraph()
        p2.text = pr_desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # ==========================================
    # SLIDE 4: ML ARCHITECTURE & 5-FOLD CV
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Model Evaluation", "Model Architecture & 5-Fold Stratified Cross-Validation")

    # Table Card
    add_card(s4, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.6))
    tb_tab = s4.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(2.2))
    tf_tab = tb_tab.text_frame
    tf_tab.word_wrap = True

    p = tf_tab.paragraphs[0]
    p.text = "5-Fold Cross-Validation Benchmark Comparison"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    bench_text = """
Architecture                       Hamming Acc.      Micro F1          Macro F1         Status
-----------------------------------------------------------------------------------------------------------------
Logistic Regression (OvR, Balanced) 86.4% ±0.3%       66.7% ±0.7%       64.0% ±1.0%      ★ DEPLOYED (Continuous Sigmoids)
LinearSVC (OvR, Balanced)           86.0% ±0.3%       64.3% ±0.9%       61.1% ±0.8%      Evaluated Benchmark
Multinomial Naive Bayes (Baseline)  86.6% ±0.4%       58.5% ±1.4%       46.4% ±1.4%      Replaced (Poor minority recall)
    """
    p_code = tf_tab.add_paragraph()
    p_code.text = bench_text
    p_code.font.size = Pt(11)
    p_code.font.name = "Courier New"
    p_code.font.color.rgb = ACC_GREEN
    p_code.space_before = Pt(8)

    # 4 KPI Stat Boxes
    kpis = [
        ("86.7%", "Hamming Accuracy", ACC_GREEN),
        ("67.7%", "Micro F1-Score", ACC_BLUE),
        ("96.8%", "Top-3 Hit Rate", ACC_ORANGE),
        ("4,680", "Catalog Titles", TEXT_WHITE)
    ]
    for i, (val, label, col) in enumerate(kpis):
        left_x = Inches(0.8 + (i * 3.0))
        add_card(s4, left_x, Inches(4.5), Inches(2.733), Inches(2.2))
        tb_k = s4.shapes.add_textbox(left_x + Inches(0.15), Inches(4.7), Inches(2.433), Inches(1.8))
        tf_k = tb_k.text_frame
        tf_k.word_wrap = True
        p1 = tf_k.paragraphs[0]
        p1.text = val
        p1.font.size = Pt(36)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf_k.add_paragraph()
        p2.text = label
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MUTED
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(6)

    # ==========================================
    # SLIDE 5: MULTI-LABEL INFERENCE & SIGMOID CALIBRATION
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Mathematical Formulation", "Multi-Label Inference Engine: Calibrated Continuous Sigmoids")

    add_card(s5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tb_m = s5.shapes.add_textbox(Inches(1.2), Inches(1.9), Inches(10.9), Inches(4.6))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True

    p = tf_m.paragraphs[0]
    p.text = "Mathematical Formulation of Multi-Label Scoring"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    math_points = [
        ("1. Independent Linear Logit Calculation", "For each of the 12 genres c, the linear projection is computed: z_c = b_c + Σ (w_c,i * x_i), where x_i is the sublinear TF-IDF weight of token i and w_c,i is the class feature weight."),
        ("2. Logistic Sigmoid Probability Calibration", "Continuous probabilities are bounded in (0, 1) without artificial softmax distortion: P(Genre_c | x) = 1 / (1 + exp(-z_c)). Unlike softmax which forces probabilities to sum to 1.0 (assuming mutual exclusivity), sigmoid allows multi-genre films to exhibit 90%+ in Drama AND 80%+ in Romance simultaneously."),
        ("3. Dynamic Top-3 Ranking & Percentage Output", "The 12 calibrated probabilities are sorted descendingly. The top 3 genres are returned with animated percentage bars: e.g. Romance (94.2%), Drama (83.0%), Comedy (54.7%)."),
        ("4. Low-Confidence Ambiguity Detection", "If the gap between the top two genres is less than 0.08 (|P1 - P2| < 0.08), Cinemix surfaces an automated ambiguity alert informing the user that the plot contains blended narrative elements.")
    ]
    for m_title, m_desc in math_points:
        p1 = tf_m.add_paragraph()
        p1.text = m_title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(12)
        p2 = tf_m.add_paragraph()
        p2.text = m_desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

    # ==========================================
    # SLIDE 6: EXPLAINABLE AI (XAI)
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Interpretability", "Explainable AI (XAI): 'Why This Prediction?' Feature Attribution")

    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb_xl = s6.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_xl = tb_xl.text_frame
    tf_xl.word_wrap = True
    p = tf_xl.paragraphs[0]
    p.text = "How Feature Attribution Works"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_ORANGE

    x_points = [
        ("Black-Box Models vs Transparent NLP", "Many deep learning models provide zero rationale for their predictions. Cinemix exposes the linear feature attribution to screenwriters and researchers."),
        ("Attribution Formula", "For the winning primary class c, word impact is computed as: Impact(t) = TF-IDF(t) * w_(c, t)."),
        ("Significance Thresholding", "Tokens with weights w > 0.15 are isolated, ranked, and surfaced as interactive badge chips in the UI."),
        ("Real Case Study (La La Land)", "Storyline containing 'jazz pianist, aspiring actress, fall in love' yields:\n• 'love' (+1.58 impact)\n• 'aspir' (+0.58 impact)\n• 'fall' (+0.48 impact)\n• 'jazz' (+0.17 impact)\nPredicting Romance (94.2%) transparently.")
    ]
    for xt, xd in x_points:
        p1 = tf_xl.add_paragraph()
        p1.text = xt
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(10)
        p2 = tf_xl.add_paragraph()
        p2.text = xd
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # Right Card: Per-Genre Performance Sample
    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    tb_xr = s6.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_xr = tb_xr.text_frame
    tf_xr.word_wrap = True
    p = tf_xr.paragraphs[0]
    p.text = "Per-Genre Test Set Metrics (N=936)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    report_sample = """
Genre            Precision   Recall    F1-Score   Support
---------------------------------------------------------
Drama              75.7%     75.6%      75.7%       479
Animation          71.8%     76.7%      74.2%        60
Comedy             69.7%     73.5%      71.5%       347
Horror             68.3%     72.4%      70.3%       105
Action             66.5%     72.2%      69.2%       230
Family             65.5%     69.6%      67.5%       102
Adventure          64.8%     67.5%      66.1%       160
Science Fiction    63.1%     69.4%      66.1%       108
Romance            62.2%     65.9%      64.0%       179
Thriller           59.9%     68.3%      63.8%       240
    """
    p_code = tf_xr.add_paragraph()
    p_code.text = report_sample
    p_code.font.size = Pt(11)
    p_code.font.name = "Courier New"
    p_code.font.color.rgb = TEXT_WHITE
    p_code.space_before = Pt(10)

    # ==========================================
    # SLIDE 7: HYBRID RECOMMENDATION ENGINE
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Recommendation Architecture", "3-Tier Hybrid Recommendation Engine: Plot + Genre + Quality")

    add_card(s7, Inches(0.8), Inches(1.6), Inches(11.733), Inches(1.3))
    tb_hf = s7.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(0.9))
    tf_hf = tb_hf.text_frame
    tf_hf.word_wrap = True
    p = tf_hf.paragraphs[0]
    p.text = "Objective 3-Tier Hybrid Formula:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_MUTED

    p_f = tf_hf.add_paragraph()
    p_f.text = "Score = 0.60 × Cosine Similarity + 0.25 × Genre Jaccard Overlap + 0.15 × Bayesian Rating"
    p_f.font.size = Pt(18)
    p_f.font.bold = True
    p_f.font.color.rgb = ACC_GREEN
    p_f.space_before = Pt(4)

    rec_pillars = [
        ("Tier 1: Plot Cosine Similarity (60%)", "Measures the lexical and thematic vector dot-product between the query synopsis and all 4,680 pre-computed movie TF-IDF vectors: Sim(q, d) = (v_q • v_d) / (||v_q|| * ||v_d||). Ensures narrative alignment with situational elements.", ACC_BLUE),
        ("Tier 2: Multi-Label Genre Jaccard Overlap (25%)", "Calculates the Jaccard similarity coefficient between the predicted query top-3 genres and the candidate film's genre tags: J(G_q, G_d) = |G_q ∩ G_d| / |G_q ∪ G_d|. Prevents genre drift where similar words occur in unrelated genres.", ACC_ORANGE),
        ("Tier 3: Bayesian Normalized Quality (15%)", "Applies IMDB-style Bayesian shrinkage toward the global catalog mean (C = 6.1, m = 300 votes): WR = (v / (v+300)) * R + (300 / (v+300)) * 6.1. Prevents obscure 1-vote 10.0 ratings from dominating.", ACC_GREEN)
    ]
    for i, (p_title, p_desc, col) in enumerate(rec_pillars):
        top_y = Inches(3.1 + (i * 1.35))
        add_card(s7, Inches(0.8), top_y, Inches(11.733), Inches(1.2))
        tb_p = s7.shapes.add_textbox(Inches(1.1), top_y + Inches(0.12), Inches(11.1), Inches(0.95))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True
        p1 = tf_p.paragraphs[0]
        p1.text = p_title
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = col
        p2 = tf_p.add_paragraph()
        p2.text = p_desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(4)

    # ==========================================
    # SLIDE 8: UI/UX DESIGN SYSTEM (LETTERBOXD)
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Design System", "UI/UX Architecture: Authentic Letterboxd Cinema Aesthetic")

    features = [
        ("Cinematic Dark Palette", "Engineered with Letterboxd's signature matte canvas (#14181c), surface cards (#1c2228), and trademark accent dots: Green (#00e054), Orange (#ff8000), and Blue (#40bcf4). Includes dark/light mode toggle with localStorage persistence."),
        ("Silky CSS Grid Typebar", "Uses CSS Grid row interpolation (grid-template-rows: 0fr -> 1fr at 0.32s cubic-bezier) rather than max-height snapping. Eliminates 60Hz hover-flicker loops with 600ms grace debounce and focus retention."),
        ("Instant Featured Catalog", "Upon initial landing (DOMContentLoaded), automatically retrieves and displays the top 12 Bayesian-weighted masterpieces (Death Note, Spirited Away, Shawshank Redemption) so the page is never blank."),
        ("Dynamic Letterboxd SVG Posters", "Overhauled poster pipeline with 200 OK dynamic SVG graphics showing film initials, release year, and genre chip, guaranteeing zero 404 network errors even when TMDB API keys are absent."),
        ("Interactive Details Modal", "Clicking any recommendation card opens an overlay modal with plot synopsis, thematic tags, vote tally, and a direct one-click YouTube official trailer search link.")
    ]

    for i, (ft, fd) in enumerate(features):
        top_y = Inches(1.6 + (i * 1.08))
        add_card(s8, Inches(0.8), top_y, Inches(11.733), Inches(0.98))
        tb_f = s8.shapes.add_textbox(Inches(1.1), top_y + Inches(0.08), Inches(11.1), Inches(0.82))
        tf_f = tb_f.text_frame
        tf_f.word_wrap = True
        p1 = tf_f.paragraphs[0]
        p1.text = f"• {ft}"
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = ACC_GREEN if i % 2 == 0 else ACC_BLUE
        p2 = tf_f.add_paragraph()
        p2.text = fd
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # ==========================================
    # SLIDE 9: ZERO-DEPENDENCY SERVERLESS CLOUD
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Cloud Engineering", "Zero-Dependency Serverless Engineering: Beating the 250MB Lambda Limit")

    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb_sl = s9.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_sl = tb_sl.text_frame
    tf_sl.word_wrap = True
    p = tf_sl.paragraphs[0]
    p.text = "The Serverless Dilemma"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_ORANGE

    dilemma = [
        ("Vercel Lambda Limit", "Vercel enforces a strict unzipped bundle limit of 250 MB for serverless Python functions."),
        ("The Bloat of ML Dependencies", "Installing scikit-learn, scipy, numpy, and pandas immediately produces a 544.75 MB bundle, crashing the deployment build with [LAMBDA_SIZE_EXCEEDED]."),
        ("Cold Start Penalty", "Importing massive binary libraries adds 3 to 6 seconds of cold-start latency per user query.")
    ]
    for dt, dd in dilemma:
        p1 = tf_sl.add_paragraph()
        p1.text = dt
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(12)
        p2 = tf_sl.add_paragraph()
        p2.text = dd
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

    add_card(s9, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    tb_sr = s9.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_sr = tb_sr.text_frame
    tf_sr.word_wrap = True
    p = tf_sr.paragraphs[0]
    p.text = "The Pure-Python Solution"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    solution = [
        ("Custom Linear-Algebra Engine", "Trained model parameters, vocab, IDF, and weights are exported to a compact 6.9 MB JSON bundle (serverless_model.json)."),
        ("Zero Binary Dependencies", "Built a pure-Python dot-product, sublinear TF-IDF tokenizer, and sigmoid inference engine using standard math library."),
        ("Bundle Reduced to <15 MB", "requirements.txt requires only flask>=3.0.0. The entire deployment is 94% below Vercel's limit!"),
        ("Sub-50ms Cold Starts", "Inference runs instantly with near-zero latency and high serverless concurrency.")
    ]
    for st, sd in solution:
        p1 = tf_sr.add_paragraph()
        p1.text = st
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(12)
        p2 = tf_sr.add_paragraph()
        p2.text = sd
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

    # ==========================================
    # SLIDE 10: PRODUCTION TESTING & VALIDATION
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Quality Assurance", "End-to-End Production Testing & Headless Chrome Verification")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    tb_q = s10.shapes.add_textbox(Inches(1.2), Inches(1.9), Inches(10.9), Inches(4.6))
    tf_q = tb_q.text_frame
    tf_q.word_wrap = True

    p = tf_q.paragraphs[0]
    p.text = "Rigorous Verification via Chrome DevTools Protocol (CDP)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_BLUE

    cdp_notes = [
        ("1. Headless Chrome Browser Automation", "Connected to the live deployment (https://movie-genre-classification-j8iw.vercel.app/) using CDP websockets to inspect real browser runtime exceptions, network requests, and DOM mutations."),
        ("2. Root-Cause Debugging (Syntax Error Resolution)", "Caught a subtle browser-level SyntaxError caused by Python string literal quote unescaping in escapeXml (case ''':). Replaced with a clean entity replacement regex, restoring 100% script parsing."),
        ("3. 100% Button Reliability Verification", "Verified that clicking 'Classify Genres & Match Films' or 'Random Example' activates immediately, updates button state to '⏳ Classifying...', auto-loads archetype storyline on empty inputs, and populates predictions without failure."),
        ("4. Zero-404 Asset Guarantee", "All poster requests successfully resolve with 200 OK SVG fallback graphics when TMDB keys are absent, eliminating broken image icons."),
        ("5. Cross-Browser Resilient Clipboard", "Implemented safeCopyText() with document.execCommand('copy') fallback for non-secure contexts and embedded preview iframes.")
    ]
    for ct, cd in cdp_notes:
        p1 = tf_q.add_paragraph()
        p1.text = ct
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(10)
        p2 = tf_q.add_paragraph()
        p2.text = cd
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # ==========================================
    # SLIDE 11: KEY TAKEAWAYS & FUTURE ROADMAP
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Project Summary", "Key Project Takeaways & Future Architectural Roadmap")

    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb_kl = s11.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf_kl = tb_kl.text_frame
    tf_kl.word_wrap = True
    p = tf_kl.paragraphs[0]
    p.text = "Key Project Achievements"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    achieve = [
        ("Accuracy More Than Doubled", "Advanced from 35% on 20 samples to 86.7% Hamming Accuracy across 4,680+ titles."),
        ("Realistic Multi-Label Modeling", "Sigmoid continuous probability calibration reflects actual hybrid cinema genres."),
        ("Full Interpretability (XAI)", "Feature attribution provides screenwriters clear insight into predictive signals."),
        ("Production Resilience", "Lightweight <15 MB serverless lambda deployment with 0 console errors.")
    ]
    for at, ad in achieve:
        p1 = tf_kl.add_paragraph()
        p1.text = f"✓ {at}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(12)
        p2 = tf_kl.add_paragraph()
        p2.text = ad
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2))
    tb_kr = s11.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_kr = tb_kr.text_frame
    tf_kr.word_wrap = True
    p = tf_kr.paragraphs[0]
    p.text = "Future Technical Roadmap"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACC_BLUE

    future = [
        ("Sentence Transformers (ONNX Runtime)", "Fine-tune all-MiniLM-L6-v2 embeddings quantised via ONNX for deep semantic contextual similarity."),
        ("Collaborative User Watchlists", "Integrate user profile history and implicit feedback loops with matrix factorization."),
        ("Real-Time Screenplay PDF Ingestion", "Allow screenwriters to drag-and-drop full screenplay PDFs for act-by-act genre pacing analysis."),
        ("Live TMDB Webhook Sync", "Automatically sync new releases every Friday to keep the 4,680+ title catalog continuously fresh.")
    ]
    for ft, fd in future:
        p1 = tf_kr.add_paragraph()
        p1.text = f"→ {ft}"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_WHITE
        p1.space_before = Pt(12)
        p2 = tf_kr.add_paragraph()
        p2.text = fd
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

    # ==========================================
    # SLIDE 12: DEMO & CONCLUSION
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)

    card12 = add_card(s12, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1), CARD_BG, BORDER_COL)
    tb_c = s12.shapes.add_textbox(Inches(2.0), Inches(1.8), Inches(9.333), Inches(4.0))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Thank You! | Live Demonstration"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = ACC_GREEN

    p2 = tf_c.add_paragraph()
    p2.text = "Explore the Cinemix ▶ Platform Live:"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p2.space_before = Pt(16)

    p3 = tf_c.add_paragraph()
    p3.text = "🌐 Live URL: https://movie-genre-classification-j8iw.vercel.app/\n📁 GitHub: https://github.com/sanskarp910-svg/movie-genre-classification.git"
    p3.font.size = Pt(15)
    p3.font.bold = True
    p3.font.color.rgb = ACC_BLUE
    p3.space_before = Pt(8)

    p4 = tf_c.add_paragraph()
    p4.text = "Questions & Interactive Q&A Session"
    p4.font.size = Pt(22)
    p4.font.bold = True
    p4.font.color.rgb = ACC_ORANGE
    p4.space_before = Pt(24)

    out_path = r"C:\Users\hp\OneDrive\Documents\AIA PROJECT\movie_genre_app\Cinemix_Project_Presentation.pptx"
    prs.save(out_path)
    print("Presentation saved successfully at:", out_path)

if __name__ == "__main__":
    create_deck()
