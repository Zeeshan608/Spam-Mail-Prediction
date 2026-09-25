"""
Spam Mail Detector - Streamlit Frontend
Loads the trained MultinomialNB model + TF-IDF vectorizer from the
Spam_Mail_Detection.ipynb notebook and serves a clean, professional UI
for classifying email/SMS text as Spam or Ham.
"""

import pickle

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #0f1b2b 100%);
        }
        .hero {
            padding: 2.2rem 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
        }
        .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
        .hero p { margin-top: 0.4rem; font-size: 1.02rem; opacity: 0.92; }

        .card {
            background: #131c2e;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
        }
        .card h3 {
            margin-top: 0;
            color: #e5e7eb;
            font-size: 1.05rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
        }

        .result-spam {
            background: linear-gradient(135deg, #dc2626, #ef4444);
            color: white; border-radius: 16px; padding: 1.6rem 1.8rem;
            text-align: center; box-shadow: 0 10px 25px rgba(239,68,68,0.35);
        }
        .result-ham {
            background: linear-gradient(135deg, #059669, #10b981);
            color: white; border-radius: 16px; padding: 1.6rem 1.8rem;
            text-align: center; box-shadow: 0 10px 25px rgba(16,185,129,0.35);
        }
        .result-spam h2, .result-ham h2 { margin: 0; font-size: 1.6rem; }
        .result-spam p, .result-ham p { margin: 0.3rem 0 0 0; opacity: 0.95; }

        .word-chip {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            margin: 0.2rem;
            border-radius: 999px;
            font-size: 0.85rem;
            background: rgba(239,68,68,0.15);
            color: #fca5a5;
            border: 1px solid rgba(239,68,68,0.3);
        }
        .word-chip.ham {
            background: rgba(16,185,129,0.15);
            color: #6ee7b7;
            border: 1px solid rgba(16,185,129,0.3);
        }

        section[data-testid="stSidebar"] { background: #0a0f1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# LOAD MODEL + VECTORIZER
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Try joblib model first, fall back to the pickle file
    try:
        model = joblib.load("spam_ham_model.joblib")
    except FileNotFoundError:
        with open("spam_ham_model.pkl", "rb") as f:
            model = pickle.load(f)

    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_file = str(e)

CLASS_LABELS = {0: "Ham", 1: "Spam"}

EXAMPLE_HAM = (
    "Hey, are we still on for lunch tomorrow at 1pm? Let me know if that "
    "still works for you."
)
EXAMPLE_SPAM = (
    "WINNER!! You have been selected to receive a $1000 cash prize. "
    "Text WIN to 80088 now to claim before it expires!"
)

# --------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📧 Spam Mail Detector</h1>
        <p>Paste an email or SMS message below and instantly check whether it's Spam or Ham.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ok:
    st.error(
        "⚠️ Could not find **tfidf_vectorizer.pkl** (and/or the model file) in the app folder. "
        "Your notebook never saved the fitted TfidfVectorizer — add "
        "`joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')` after training, re-run that cell, "
        "then place the resulting file next to `app.py`.\n\nDetails: " + missing_file
    )
    st.stop()

# --------------------------------------------------------------------------
# SIDEBAR - ABOUT / MODEL INFO
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this tool")
    st.write(
        "This app uses a **Multinomial Naive Bayes** classifier trained on "
        "TF-IDF features to detect spam messages."
    )
    st.markdown("### 📊 Model performance")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "97.76%")
    c2.metric("Spam Precision", "100%")
    c3, c4 = st.columns(2)
    c3.metric("Spam Recall", "83%")
    c4.metric("Spam F1-Score", "91%")
    st.caption("Metrics from the model's held-out test set (1,115 messages).")
    st.divider()
    st.caption("Built with Streamlit • scikit-learn Multinomial Naive Bayes • TF-IDF")

# --------------------------------------------------------------------------
# INPUT
# --------------------------------------------------------------------------
if "message_text" not in st.session_state:
    st.session_state.message_text = ""

col_input, col_examples = st.columns([3, 1])

with col_examples:
    st.markdown('<div class="card"><h3>💡 Try an example</h3>', unsafe_allow_html=True)
    if st.button("📩 Load ham example", use_container_width=True):
        st.session_state.message_text = EXAMPLE_HAM
    if st.button("🚫 Load spam example", use_container_width=True):
        st.session_state.message_text = EXAMPLE_SPAM
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.message_text = ""
    st.markdown("</div>", unsafe_allow_html=True)

with col_input:
    st.markdown('<div class="card"><h3>✉️ Message</h3>', unsafe_allow_html=True)
    message = st.text_area(
        "Paste the email or SMS text here",
        value=st.session_state.message_text,
        height=180,
        placeholder="e.g. Congratulations! You've won a free cruise. Call now to claim your prize...",
        label_visibility="collapsed",
    )
    predict_clicked = st.button("🔍 Check Message", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PREDICTION
# --------------------------------------------------------------------------
if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message to check.")
        st.stop()

    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    ham_prob = probabilities[0] * 100
    spam_prob = probabilities[1] * 100

    st.markdown("---")
    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-spam">
                    <h2>🚫 This looks like Spam</h2>
                    <p>Spam confidence: {spam_prob:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-ham">
                    <h2>✅ This looks like Ham (Not Spam)</h2>
                    <p>Ham confidence: {ham_prob:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Spam Probability", f"{spam_prob:.1f}%")
        m2.metric("Ham Probability", f"{ham_prob:.1f}%")

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spam_prob,
            number={"suffix": "%"},
            title={"text": "Spam Likelihood"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 40], "color": "#0d3b1e"},
                    {"range": [40, 70], "color": "#4a3b0d"},
                    {"range": [70, 100], "color": "#3b0d0d"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e5e7eb"},
            height=280,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Explainability: which words in the message pushed it toward spam/ham
    # ------------------------------------------------------------------
    if hasattr(model, "feature_log_prob_"):
        feature_names = vectorizer.get_feature_names_out()
        spam_minus_ham = model.feature_log_prob_[1] - model.feature_log_prob_[0]

        nonzero_idx = vectorized.nonzero()[1]
        if len(nonzero_idx) > 0:
            scored_words = sorted(
                ((feature_names[i], spam_minus_ham[i]) for i in nonzero_idx),
                key=lambda x: x[1],
                reverse=True,
            )
            top_spam_words = [w for w, s in scored_words if s > 0][:6]
            top_ham_words = [w for w, s in reversed(scored_words) if s < 0][:6]

            with st.expander("🔎 Why this prediction? (words that influenced it)"):
                if top_spam_words:
                    st.write("**Words pushing toward Spam:**")
                    st.markdown(
                        " ".join(f'<span class="word-chip">{w}</span>' for w in top_spam_words),
                        unsafe_allow_html=True,
                    )
                if top_ham_words:
                    st.write("**Words pushing toward Ham:**")
                    st.markdown(
                        " ".join(f'<span class="word-chip ham">{w}</span>' for w in top_ham_words),
                        unsafe_allow_html=True,
                    )
                if not top_spam_words and not top_ham_words:
                    st.caption("No recognized vocabulary words found in this message.")

    st.caption(
        "⚠️ This is a demo classifier trained on a public dataset. Always use judgment "
        "and your email provider's built-in spam filtering for real-world protection."
    )