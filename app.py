import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd

# Page config
st.set_page_config(
    page_title="Gene Mutation Predictor",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stApp { background-color: #f8f9fa; }
    
    .header-box {
        background-color: #c0392b;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .header-box h1 { color: white; font-size: 2.2em; margin: 0; }
    .header-box p { color: #f5b7b1; font-size: 1em; margin: 5px 0 0 0; }

    .result-pathogenic {
        background-color: #fdedec;
        border-left: 6px solid #c0392b;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-benign {
        background-color: #eafaf1;
        border-left: 6px solid #1e8449;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .section-title {
        color: #c0392b;
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 2px solid #f5b7b1;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-box">
        <h1>🧬 Gene Mutation Impact Predictor</h1>
        <p>AI-powered clinical variant classification system</p>
    </div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('model.json')
    return model

model = load_model()

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Variant Input</div>', unsafe_allow_html=True)

    variant_type = st.selectbox("Variant Type", [
        'single nucleotide variant', 'Deletion', 'Insertion',
        'Duplication', 'Indel', 'Microsatellite'
    ])

    start = st.number_input("Start Position", min_value=0, value=100000)
    stop = st.number_input("Stop Position", min_value=0, value=100001)

    predict_btn = st.button("🔍 Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # About section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    This tool uses an **XGBoost machine learning model** trained on **830,000+ clinical variants** 
    from the [ClinVar database](https://www.ncbi.nlm.nih.gov/clinvar/) to classify genetic mutations.

    **Labels:**
    - 🔴 **Pathogenic** — mutation is likely harmful and disease-causing
    - 🟢 **Benign** — mutation is likely harmless

    **Model Performance:**
    - AUC-ROC: 0.789
    - Accuracy: 73%
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    type_map = {
        'single nucleotide variant': 0,
        'Deletion': 1,
        'Insertion': 2,
        'Duplication': 3,
        'Indel': 4,
        'Microsatellite': 5
    }

    if predict_btn:
        type_encoded = type_map.get(variant_type, 0)
        features = np.array([[type_encoded, start, stop]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
                <div class="result-pathogenic">
                    <h3 style="color:#c0392b; margin:0">⚠️ PATHOGENIC</h3>
                    <p style="margin:5px 0 0 0">This mutation is likely <strong>harmful</strong></p>
                </div>
            """, unsafe_allow_html=True)
            conf = probability[1]
            label = "Pathogenic"
        else:
            st.markdown(f"""
                <div class="result-benign">
                    <h3 style="color:#1e8449; margin:0">✅ BENIGN</h3>
                    <p style="margin:5px 0 0 0">This mutation is likely <strong>harmless</strong></p>
                </div>
            """, unsafe_allow_html=True)
            conf = probability[0]
            label = "Benign"

        # Probability gauge
        st.markdown("<br>**Confidence Score:**", unsafe_allow_html=True)
        st.progress(float(conf))
        st.markdown(f"**{conf*100:.1f}%** confidence", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Add to history
        st.session_state.history.append({
            "Variant Type": variant_type,
            "Start": start,
            "Stop": stop,
            "Prediction": label,
            "Confidence": f"{conf*100:.1f}%"
        })

    # Results history
    if st.session_state.history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Results History</div>', unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)