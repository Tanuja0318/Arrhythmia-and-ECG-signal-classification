import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.title("💓 AI-Based ECG Arrhythmia Detection ")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])

def extract_features(signal):
    peaks = []
    for i in range(1,len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.6:
            peaks.append(i)

    if len(peaks) < 2:
        return [0,0,0]

    rr = np.diff(peaks)
    return [np.mean(rr), np.std(rr), len(peaks)]


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "ecg" not in df.columns:
        st.error("CSV must contain 'ecg' column")
    else:
        signal = df["ecg"].values

        st.subheader("📈 ECG Signal")
        st.line_chart(signal)

        features = extract_features(signal)

        st.subheader("📊 Extracted Features")
        st.write({
            "Mean RR": features[0],
            "STD RR": features[1],
            "Peaks": features[2]
        })

        prediction = model.predict([features])

        st.subheader("🧠 Prediction")

        if prediction[0] == 0:
            st.success("Normal Rhythm ✅")
        else:
            st.error("Arrhythmia Detected ⚠️")

        # Show accuracy graph
        st.subheader("📉 Model Comparison")
        img = Image.open("models/accuracy.png")
        st.image(img)