#!/usr/bin/env python3
"""
Fake Review Detection System
Streamlit Deployment - Optimized Version
"""

import streamlit as st
import tensorflow as tf
import pickle
import json
import numpy as np
import pandas as pd
import time
from scipy.sparse import hstack

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (ENHANCED UI)
# ============================================
st.markdown("""
<style>
.main-header {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1.5rem;
}
.card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 0.75rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.result-real {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #28a745;
    font-size: 1.2rem;
    font-weight: bold;
}
.result-fake {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #dc3545;
    font-size: 1.2rem;
    font-weight: bold;
}
.small-note {
    color: #6c757d;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE (HISTORY)
# ============================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================
# OPTIMIZED LOADING WITH CACHING
# ============================================
@st.cache_resource(show_spinner=True)
def load_model_and_preprocessing():
    try:
        model = tf.keras.models.load_model("dnn_model.h5", compile=False)

        with open("preprocessing_objects.pkl", "rb") as f:
            preprocessing = pickle.load(f)

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        with open("config.json", "r") as f:
            config = json.load(f)

        return model, preprocessing, label_encoder, config

    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None, None, None, None

# ============================================
# PREPROCESSING (UNCHANGED)
# ============================================
def preprocess_input(text, rating, helpful, preprocessing, config):
    input_df = pd.DataFrame([{
        config["text_col"]: text,
        "rating": float(rating),
        "helpful": int(helpful)
    }])

    tfidf = preprocessing["tfidf_vectorizer"]
    scaler = preprocessing["scaler"]

    text_tfidf = tfidf.transform(input_df[config["text_col"]])
    numeric_scaled = scaler.transform(input_df[["rating", "helpful"]])

    combined = hstack([text_tfidf, numeric_scaled])
    return combined.toarray()

# ============================================
# PREDICTION (UNCHANGED)
# ============================================
def predict_review(text, rating, helpful, model, preprocessing, label_encoder, config):
    X = preprocess_input(text, rating, helpful, preprocessing, config)

    proba = model.predict(X, verbose=0, batch_size=1)[0][0]
    threshold = config.get("threshold", 0.5)

    prediction = 1 if proba >= threshold else 0
    label = label_encoder.inverse_transform([prediction])[0]
    confidence = float(proba) if prediction == 1 else float(1 - proba)

    return {
        "label": label,
        "probability": float(proba),
        "confidence": confidence,
    }

# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown(
        '<div class="main-header">🔍 Fake Review Detection System</div>',
        unsafe_allow_html=True
    )

    model, preprocessing, label_encoder, config = load_model_and_preprocessing()
    if model is None:
        st.stop()

    # SIDEBAR
    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.markdown(f"""
        **Model:** Deep Neural Network  
        **Features:** {config.get('n_features', 'N/A')}  
        **Classes:** {', '.join(config.get('classes', ['Real', 'Fake']))}
        """)
        st.divider()
        st.info("Model & preprocessing sudah di-cache.")

    # =======================
    # TABS
    # =======================
    tab_pred, tab_history, tab_about = st.tabs(
        ["🔍 Prediksi", "📜 Histori", "ℹ️ Tentang"]
    )

    # ===================================
    # TAB PREDIKSI
    # ===================================
    with tab_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Input Review")

        text_input = st.text_area(
            "Teks Review",
            height=150,
            placeholder="Contoh: Produk bagus, pengiriman cepat, recommended!"
        )

        col1, col2 = st.columns(2)
        with col1:
            rating_input = st.slider("Rating", 1.0, 5.0, 5.0, 0.5)
        with col2:
            helpful_input = st.number_input("Helpful Votes", 0, 1000, 0)

        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Masukkan teks review terlebih dahulu.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_review(
                        text_input,
                        rating_input,
                        helpful_input,
                        model,
                        preprocessing,
                        label_encoder,
                        config
                    )

                # SAVE HISTORY
                st.session_state.history.insert(
                    0,
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "label": result["label"],
                        "confidence": round(result["confidence"], 4),
                        "fake_probability": round(result["probability"], 4),
                        "rating": rating_input,
                        "helpful": helpful_input,
                        "text": text_input[:200] + ("..." if len(text_input) > 200 else "")
                    }
                )

                st.divider()

                if result["label"] == "Fake":
                    st.markdown(
                        f'<div class="result-fake">⚠️ FAKE REVIEW<br>'
                        f'Confidence: {result["confidence"]*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-real">✅ REAL REVIEW<br>'
                        f'Confidence: {result["confidence"]*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )

                st.progress(result["probability"])
                st.caption(
                    f"Fake Probability: {result['probability']*100:.1f}%"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ===================================
    # TAB HISTORI
    # ===================================
    with tab_history:
        st.subheader("📜 Histori Prediksi (Session Ini)")

        if len(st.session_state.history) == 0:
            st.info("Belum ada histori.")
        else:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Hapus Histori"):
                st.session_state.history = []
                st.success("Histori berhasil dihapus.")

    # ===================================
    # TAB TENTANG
    # ===================================
    with tab_about:
        st.subheader("ℹ️ Tentang Aplikasi")
        st.markdown("""
        **Fake Review Detection System**  
        Aplikasi ini mendeteksi ulasan palsu menggunakan
        **Deep Neural Network (DNN)** berbasis **TF-IDF + fitur numerik**.

        ### 🔧 Teknologi
        - TensorFlow / Keras
        - TF-IDF Vectorizer
        - Scikit-learn Scaler
        - Streamlit

        ### 🎯 Output
        - Label Real / Fake
        - Confidence Score
        - Fake Probability

        ### ⚠️ Catatan
        Aplikasi ini bersifat **research & demo**, bukan pengganti
        moderasi manusia.
        """)

if __name__ == "__main__":
    main()
