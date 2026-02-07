#!/usr/bin/env python3
"""
Fake Review Detection System
Streamlit Deployment - PKL Version
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

# Custom CSS (sama seperti sebelumnya)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-real {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .result-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL & PREPROCESSING (CACHE) - PKL VERSION
# ============================================

@st.cache_resource
def load_model_and_preprocessing():
    """
    Load model dan preprocessing objects dari PKL files
    """
    try:
        # Load DNN model
        model = tf.keras.models.load_model('dnn_model.h5')
        
        # Load preprocessing objects dari pkl
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        # Load label encoder dari pkl
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load config dari json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        return model, preprocessing, label_encoder, config
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Pastikan file berikut ada di folder yang sama:")
        st.code("""
        dnn_model.h5
        preprocessing_objects.pkl
        label_encoder.pkl
        config.json
        """)
        return None, None, None, None

# ============================================
# PREPROCESSING FUNCTION
# ============================================

def preprocess_input(text, rating, helpful, preprocessing, config):
    """
    Preprocess input untuk model prediction
    """
    # Create DataFrame
    input_df = pd.DataFrame([{
        config['text_col']: text,
        'rating': float(rating),
        'helpful': int(helpful)
    }])
    
    # Extract preprocessing objects
    tfidf = preprocessing['tfidf_vectorizer']
    scaler = preprocessing['scaler']
    
    # Transform text dengan TF-IDF
    text_tfidf = tfidf.transform(input_df[config['text_col']])
    
    # Transform numeric dengan Scaler
    numeric_scaled = scaler.transform(input_df[['rating', 'helpful']])
    
    # Combine (hstack)
    combined = hstack([text_tfidf, numeric_scaled])
    
    # Convert to dense (karena DNN butuh dense input)
    return combined.toarray()

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_review(text, rating, helpful, model, preprocessing, label_encoder, config):
    """
    Lakukan prediksi dengan timing
    """
    # Start timing
    start_preprocess = time.time()
    
    # Preprocessing
    X_processed = preprocess_input(text, rating, helpful, preprocessing, config)
    preprocess_time = (time.time() - start_preprocess) * 1000
    
    # Prediction
    start_predict = time.time()
    proba = model.predict(X_processed, verbose=0)[0][0]
    predict_time = (time.time() - start_predict) * 1000
    
    # Total time
    total_time = preprocess_time + predict_time
    
    # Determine label
    threshold = config.get('threshold', 0.5)
    prediction = 1 if proba >= threshold else 0
    label = label_encoder.inverse_transform([prediction])[0]
    
    return {
        'label': label,
        'probability': float(proba),
        'confidence': float(proba) if prediction == 1 else float(1 - proba),
        'preprocess_time_ms': preprocess_time,
        'predict_time_ms': predict_time,
        'total_time_ms': total_time
    }

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown('<div class="main-header">🔍 Fake Review Detection System</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model, preprocessing, label_encoder, config = load_model_and_preprocessing()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(f"""
        **Architecture:** Deep Neural Network (DNN)  
        **Features:** {config['n_features']} (TF-IDF + Numeric)  
        **Classes:** {', '.join(config['classes'])}  
        **Performance:** 
        - PR-AUC: 0.953
        - F1-Macro: 0.735
        """)
        
        st.divider()
        
        st.header("⚡ Speed")
        st.markdown("""
        | Method | Latency |
        |--------|---------|
        | **This System** | **~64ms** |
        | LLM API | ~185ms |
        """)
    
    # Input form
    st.subheader("Input Review")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Review Text (Bahasa Indonesia)",
            placeholder="Masukkan teks review di sini...",
            height=150
        )
    
    with col2:
        rating_input = st.slider("Rating", 1.0, 5.0, 5.0, 0.5)
        helpful_input = st.number_input("Helpful Votes", 0, 1000, 0)
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Masukkan teks review!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_review(
                    text_input, rating_input, helpful_input,
                    model, preprocessing, label_encoder, config
                )
            
            # Display result
            st.divider()
            
            if result['label'] == 'Fake':
                st.markdown(
                    f'<div class="result-fake">'
                    f'⚠️ FAKE REVIEW<br>Confidence: {result["confidence"]*100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-real">'
                    f'✅ REAL REVIEW<br>Confidence: {result["confidence"]*100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", result['label'])
            c2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            c3.metric("Time", f"{result['total_time_ms']:.1f}ms")
            
            # Progress bar
            st.progress(result['probability'], 
                       text=f"Fake Probability: {result['probability']*100:.1f}%")

if __name__ == "__main__":
    main()