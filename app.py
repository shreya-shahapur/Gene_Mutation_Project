import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('model.json')

st.title("🧬 Gene Mutation Impact Predictor")
st.write("Enter variant details below to predict whether a mutation is **Pathogenic** or **Benign**.")

# Input fields
variant_type = st.selectbox("Variant Type", [
    'single nucleotide variant', 'Deletion', 'Insertion', 
    'Duplication', 'Indel', 'Microsatellite'
])

start = st.number_input("Start Position", min_value=0, value=100000)
stop = st.number_input("Stop Position", min_value=0, value=100001)

# Encode variant type
type_map = {
    'single nucleotide variant': 0,
    'Deletion': 1,
    'Insertion': 2,
    'Duplication': 3,
    'Indel': 4,
    'Microsatellite': 5
}
type_encoded = type_map.get(variant_type, 0)

# Predict
if st.button("Predict"):
    features = np.array([[type_encoded, start, stop]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        st.error(f"⚠️ PATHOGENIC — This mutation is likely harmful")
        st.write(f"Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"✅ BENIGN — This mutation is likely harmless")
        st.write(f"Confidence: {probability[0]*100:.1f}%")