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

# Custom CSS
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
# OPTIMIZED LOADING WITH BETTER CACHING
# ============================================

@st.cache_resource(show_spinner=True)
def load_model_and_preprocessing():
    """
    Load model dan preprocessing objects - CACHED for performance
    This function only runs once and caches the results
    """
    load_start = time.time()
    
    try:
        # Progress indicator
        status = st.empty()
        
        # 1. Load DNN model (usually the slowest)
        status.text("⏳ Loading model...")
        model = tf.keras.models.load_model('dnn_model.h5', compile=False)  # compile=False is faster
        model_time = time.time() - load_start
        
        # 2. Load preprocessing objects
        status.text("⏳ Loading preprocessing...")
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        prep_time = time.time() - load_start - model_time
        
        # 3. Load label encoder
        status.text("⏳ Loading label encoder...")
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        le_time = time.time() - load_start - model_time - prep_time
        
        # 4. Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        total_time = time.time() - load_start
        
        status.empty()  # Clear loading message
        
        # Log loading times (for debugging)
        print(f"✓ Model loaded in {model_time:.2f}s")
        print(f"✓ Preprocessing loaded in {prep_time:.2f}s")
        print(f"✓ Total load time: {total_time:.2f}s")
        
        return model, preprocessing, label_encoder, config
    
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Pastikan file berikut ada di folder yang sama:")
        st.code("""
        dnn_model.h5
        preprocessing_objects.pkl
        label_encoder.pkl
        config.json
        """)
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

# ============================================
# OPTIMIZED PREPROCESSING
# ============================================

def preprocess_input(text, rating, helpful, preprocessing, config):
    """
    Preprocess input untuk model prediction - OPTIMIZED
    """
    try:
        # Create DataFrame (minimal)
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
        
        # Combine using hstack (efficient for sparse matrices)
        combined = hstack([text_tfidf, numeric_scaled])
        
        # Convert to dense array for DNN
        return combined.toarray()
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# ============================================
# OPTIMIZED PREDICTION
# ============================================

def predict_review(text, rating, helpful, model, preprocessing, label_encoder, config):
    """
    Lakukan prediksi dengan timing - OPTIMIZED
    """
    try:
        # Preprocessing
        start_preprocess = time.time()
        X_processed = preprocess_input(text, rating, helpful, preprocessing, config)
        if X_processed is None:
            return None
        preprocess_time = (time.time() - start_preprocess) * 1000
        
        # Prediction (dengan batch size 1)
        start_predict = time.time()
        proba = model.predict(X_processed, verbose=0, batch_size=1)[0][0]
        predict_time = (time.time() - start_predict) * 1000
        
        # Total time
        total_time = preprocess_time + predict_time
        
        # Determine label
        threshold = config.get('threshold', 0.5)
        prediction = 1 if proba >= threshold else 0
        label = label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence
        confidence = float(proba) if prediction == 1 else float(1 - proba)
        
        return {
            'label': label,
            'probability': float(proba),  # Probability of being FAKE
            'confidence': confidence,
            'prediction': prediction,
            'preprocess_time_ms': preprocess_time,
            'predict_time_ms': predict_time,
            'total_time_ms': total_time
        }
    
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================
# MAIN APP
# ============================================

def main():
    st.markdown('<div class="main-header">🔍 Fake Review Detection System</div>', 
                unsafe_allow_html=True)
    
    # Load model (this will be cached after first run)
    with st.spinner("🚀 Loading model... (first time only)"):
        model, preprocessing, label_encoder, config = load_model_and_preprocessing()
    
    if model is None:
        st.stop()
    
    # Success message (only shows after model loads)
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(f"""
        **Architecture:** Deep Neural Network (DNN)  
        **Features:** {config.get('n_features', 'N/A')} (TF-IDF + Numeric)  
        **Classes:** {', '.join(config.get('classes', ['Real', 'Fake']))}  
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
        
        st.divider()
        
        st.header("💡 Tips")
        st.info("Model is cached - subsequent predictions will be faster!")
    
    # Input form
    st.subheader("📝 Input Review")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Review Text (Bahasa Indonesia)",
            placeholder="Contoh: Produk bagus, pengiriman cepat, recommended!",
            height=150,
            help="Masukkan teks review dalam Bahasa Indonesia"
        )
    
    with col2:
        rating_input = st.slider(
            "Rating", 
            1.0, 5.0, 5.0, 0.5,
            help="Rating produk (1-5 bintang)"
        )
        helpful_input = st.number_input(
            "Helpful Votes", 
            0, 1000, 0,
            help="Jumlah orang yang menganggap review ini membantu"
        )
    
    # Example reviews
    with st.expander("📋 Contoh Review"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Contoh Real Review"):
                st.session_state['example_text'] = "Produk sangat bagus, kualitas premium. Pengiriman cepat dan packing rapi. Seller responsif dan ramah. Highly recommended untuk yang cari kualitas terbaik!"
                st.session_state['example_rating'] = 5.0
                st.session_state['example_helpful'] = 15
                st.rerun()
        with col_b:
            if st.button("Contoh Fake Review"):
                st.session_state['example_text'] = "Bagus"
                st.session_state['example_rating'] = 5.0
                st.session_state['example_helpful'] = 0
                st.rerun()
    
    # Use example if available
    if 'example_text' in st.session_state:
        text_input = st.session_state.get('example_text', text_input)
        rating_input = st.session_state.get('example_rating', rating_input)
        helpful_input = st.session_state.get('example_helpful', helpful_input)
        # Clear example after use
        for key in ['example_text', 'example_rating', 'example_helpful']:
            if key in st.session_state:
                del st.session_state[key]
    
    # Analyze button
    if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("⚠️ Masukkan teks review terlebih dahulu!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_review(
                    text_input, rating_input, helpful_input,
                    model, preprocessing, label_encoder, config
                )
            
            if result is None:
                st.error("❌ Prediction failed. Please check the error messages above.")
                st.stop()
            
            # Display result
            st.divider()
            
            if result['label'] == 'Fake':
                st.markdown(
                    f'<div class="result-fake">'
                    f'⚠️ FAKE REVIEW DETECTED<br>Confidence: {result["confidence"]*100:.1f}%'
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
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result['label'])
            col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            col3.metric("⚡ Time", f"{result['total_time_ms']:.1f}ms")
            
            # Probability visualization
            st.write(f"**Fake Probability:** {result['probability']*100:.1f}%")
            st.progress(result['probability'])
            
            # Detailed timing (collapsible)
            with st.expander("⏱️ Performance Details"):
                st.write(f"- **Preprocessing:** {result['preprocess_time_ms']:.2f}ms")
                st.write(f"- **Prediction:** {result['predict_time_ms']:.2f}ms")
                st.write(f"- **Total:** {result['total_time_ms']:.2f}ms")
                
                # Interpretation
                st.divider()
                st.write("**Interpretation:**")
                if result['probability'] >= 0.8:
                    st.error("🚨 Very likely a fake review")
                elif result['probability'] >= 0.6:
                    st.warning("⚠️ Possibly a fake review")
                elif result['probability'] >= 0.4:
                    st.info("ℹ️ Uncertain - needs manual review")
                elif result['probability'] >= 0.2:
                    st.success("✅ Likely a genuine review")
                else:
                    st.success("✅✅ Very likely a genuine review")

if __name__ == "__main__":
    main()