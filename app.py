import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd

st.set_page_config(
    page_title="Gene Mutation Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
    color: #e0e0ff;
}

section[data-testid="stSidebar"] { display: none; }

.hero {
    text-align: center;
    padding: 50px 20px 30px 20px;
}
.hero h1 {
    font-size: 3em;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #7c3aed, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.hero p {
    color: #9ca3af;
    font-size: 1.1em;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin: 20px 0 40px 0;
}
.stat-box {
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.2);
    border-radius: 12px;
    padding: 15px 30px;
    text-align: center;
}
.stat-box .num {
    font-size: 1.6em;
    font-weight: 700;
    color: #a855f7;
}
.stat-box .label {
    font-size: 0.75em;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(168,85,247,0.15);
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.85em;
    font-weight: 600;
    color: #a855f7;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(168,85,247,0.2);
    padding-bottom: 10px;
}

.result-pathogenic {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 14px;
    padding: 20px 25px;
    margin-top: 15px;
}
.result-pathogenic h2 { color: #f87171; margin: 0 0 5px 0; font-size: 1.6em; }
.result-pathogenic p { color: #fca5a5; margin: 0; font-size: 0.9em; }

.result-benign {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(5,150,105,0.1));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 14px;
    padding: 20px 25px;
    margin-top: 15px;
}
.result-benign h2 { color: #6ee7b7; margin: 0 0 5px 0; font-size: 1.6em; }
.result-benign p { color: #a7f3d0; margin: 0; font-size: 0.9em; }

.conf-label {
    font-size: 0.8em;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 6px 0;
}
.conf-value {
    font-size: 2em;
    font-weight: 700;
    color: #a855f7;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #9ca3af !important;
    font-size: 0.8em !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(168,85,247,0.25) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    font-size: 1em !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    margin-top: 10px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    box-shadow: 0 0 20px rgba(168,85,247,0.4) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border-radius: 10px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
}

.stDataFrame {
    background: transparent !important;
}

.about-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(168,85,247,0.1);
}
.about-icon { font-size: 1.3em; }
.about-text { color: #9ca3af; font-size: 0.9em; line-height: 1.6; }
.about-text strong { color: #e0e0ff; }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>🧬 Gene Mutation Predictor</h1>
    <p>AI-Powered Clinical Variant Classification</p>
</div>
<div class="stat-row">
    <div class="stat-box">
        <div class="num">830K+</div>
        <div class="label">Variants Trained</div>
    </div>
    <div class="stat-box">
        <div class="num">78.9%</div>
        <div class="label">AUC-ROC Score</div>
    </div>
    <div class="stat-box">
        <div class="num">73%</div>
        <div class="label">Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="num">XGBoost</div>
        <div class="label">Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('model.json')
    return model

model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">⚡ Variant Input</div>', unsafe_allow_html=True)

    variant_type = st.selectbox("Variant Type", [
        'single nucleotide variant', 'Deletion', 'Insertion',
        'Duplication', 'Indel', 'Microsatellite'
    ])
    start = st.number_input("Start Position", min_value=0, value=100000)
    stop = st.number_input("Stop Position", min_value=0, value=100001)
    predict_btn = st.button("🔍 Run Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">ℹ️ About</div>
        <div class="about-item">
            <div class="about-icon">🗄️</div>
            <div class="about-text">Trained on <strong>830,000+ real clinical variants</strong> from the ClinVar database</div>
        </div>
        <div class="about-item">
            <div class="about-icon">🤖</div>
            <div class="about-text">Uses <strong>XGBoost</strong> — a gradient boosting algorithm widely used in bioinformatics</div>
        </div>
        <div class="about-item">
            <div class="about-icon">🔴</div>
            <div class="about-text"><strong>Pathogenic</strong> — mutation is likely harmful and disease-causing</div>
        </div>
        <div class="about-item">
            <div class="about-icon">🟢</div>
            <div class="about-text"><strong>Benign</strong> — mutation is likely harmless to the organism</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    type_map = {
        'single nucleotide variant': 0,
        'Deletion': 1, 'Insertion': 2,
        'Duplication': 3, 'Indel': 4, 'Microsatellite': 5
    }

    if predict_btn:
        type_encoded = type_map.get(variant_type, 0)
        features = np.array([[type_encoded, start, stop]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        st.markdown('<div class="card"><div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            conf = probability[1]
            label = "Pathogenic"
            st.markdown(f"""
                <div class="result-pathogenic">
                    <h2>⚠️ PATHOGENIC</h2>
                    <p>This mutation is likely harmful and disease-causing</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            conf = probability[0]
            label = "Benign"
            st.markdown(f"""
                <div class="result-benign">
                    <h2>✅ BENIGN</h2>
                    <p>This mutation is likely harmless</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="conf-label">Confidence Score</div><div class="conf-value">{conf*100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(float(conf))
        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.history.append({
            "Type": variant_type,
            "Start": start,
            "Stop": stop,
            "Result": label,
            "Confidence": f"{conf*100:.1f}%"
        })

    if st.session_state.history:
        st.markdown('<div class="card"><div class="card-title">📋 Results History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)