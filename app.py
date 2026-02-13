#!/usr/bin/env python3
"""
Fake Review Detection System
Streamlit Deployment - With Enhanced Visualization
Model: DNN (TF-IDF + Numeric Features)
Visualization: Inspired by IndoBERT app
"""

import streamlit as st
import tensorflow as tf
import pickle
import json
import numpy as np
import pandas as pd
import time
from scipy.sparse import hstack
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fake Review Detector (DNN)",
    page_icon="🔍",
    layout="wide",
)

# ============================================
# LOAD CSS
# ============================================

# INLINE CSS - DARK THEME
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Dark App Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Section - Dark with Purple Gradient */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .badge-real {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
    }
    
    /* Dark Card Styling */
    .card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border: 1px solid #334155;
        min-height: 400px;
    }
    
    .result-row {
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        border-radius: 0.5rem;
        text-align: center;
        color: #e2e8f0;
        border: 1px solid #475569;
    }
    
    /* Dark Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    /* Dark Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        border: 1px solid #475569;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        background: #334155;
        color: #e2e8f0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        background: #475569;
        border-color: #6366f1;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    /* Dark Text Areas */
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 2px solid #475569;
        padding: 0.75rem;
        font-size: 1rem;
        background: #1e293b;
        color: #e2e8f0;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    
    .stTextArea textarea::placeholder {
        color: #64748b;
    }
    
    /* Dark Number Input */
    .stNumberInput input {
        background: #1e293b;
        color: #e2e8f0;
        border: 2px solid #475569;
        border-radius: 0.5rem;
    }
    
    .stNumberInput input:focus {
        border-color: #6366f1;
    }
    
    /* Dark Slider */
    .stSlider > div > div > div {
        background: #475569;
    }
    
    /* Dark Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #1e293b;
        padding: 0.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        color: #94a3b8;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #334155;
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    /* Dark Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Light text on dark background */
    .element-container {
        color: #e2e8f0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, span, div, label {
        color: #cbd5e1;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Dark DataFrame */
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #1e293b;
        color: #e2e8f0;
    }
    
    /* Dark Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #334155;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #1e293b;
        color: #e2e8f0;
        border: 2px solid #475569;
    }
    
    /* Caption text */
    .stCaptionContainer {
        color: #94a3b8 !important;
    }
    
    /* Code blocks */
    code {
        background: #0f172a;
        color: #f1f5f9;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #334155;
        color: #e2e8f0;
        border-radius: 0.5rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: #475569;
    }
    </style>
""", unsafe_allow_html=True)

def load_css(path: str):
    """Load external CSS file (optional enhancement)"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # CSS inline sudah cukup, tidak perlu warning
        pass

# Try to load external CSS (optional)
load_css("assets/style.css")

# ============================================
# MODEL LOADING (OPTIMIZED WITH CACHING)
# ============================================

@st.cache_resource(show_spinner=True)
def load_model_and_preprocessing():
    """
    Load model dan preprocessing objects - CACHED for performance
    """
    load_start = time.time()
    
    try:
        # 1. Load DNN model
        model = tf.keras.models.load_model('dnn_model.h5', compile=False)
        
        # 2. Load preprocessing objects
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        # 3. Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # 4. Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        total_time = time.time() - load_start
        
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
# PREPROCESSING
# ============================================

def preprocess_input(text, rating, helpful, preprocessing, config):
    """Preprocess input untuk model prediction"""
    try:
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
        
        # Combine using hstack
        combined = hstack([text_tfidf, numeric_scaled])
        
        return combined.toarray()
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# ============================================
# PREDICTION
# ============================================

def predict_review(text, rating, helpful, model, preprocessing, label_encoder, config, threshold=0.5):
    """
    Lakukan prediksi dengan timing
    Returns: dict with prediction results
    """
    try:
        # Preprocessing
        start_preprocess = time.time()
        X_processed = preprocess_input(text, rating, helpful, preprocessing, config)
        if X_processed is None:
            return None
        preprocess_time = (time.time() - start_preprocess) * 1000
        
        # Prediction
        start_predict = time.time()
        proba = model.predict(X_processed, verbose=0, batch_size=1)[0][0]
        predict_time = (time.time() - start_predict) * 1000
        
        # Total time
        total_time = preprocess_time + predict_time
        
        # Determine label with custom threshold
        prediction = 1 if proba >= threshold else 0
        label = label_encoder.inverse_transform([prediction])[0]
        
        # Calculate probabilities
        p_fake = float(proba)
        p_real = float(1 - proba)
        
        # Calculate confidence
        confidence = max(p_real, p_fake)
        
        return {
            'label': label,
            'p_real': p_real,
            'p_fake': p_fake,
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
# HELPER FUNCTIONS
# ============================================

def badge(label: str):
    """Generate HTML badge for prediction result"""
    cls = "badge-fake" if label == "Fake" else "badge-real"
    return f'<span class="badge {cls}">{label}</span>'

# ============================================
# SESSION STATE
# ============================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================
# MAIN APP
# ============================================

def main():
    # Load model (cached after first run)
    model, preprocessing, label_encoder, config = load_model_and_preprocessing()
    
    if model is None:
        st.stop()
    
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        threshold = st.slider(
            "Ambang Fake (threshold p_fake)", 
            0.30, 0.90, 0.50, 0.05,
            help="Jika p_fake ≥ threshold → Fake"
        )
        
        st.divider()
        
        st.markdown("### ℹ️ Model Info")
        st.markdown(f"""
        <div style='background: #334155; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #6366f1; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
        <strong style='color: #f1f5f9;'>Architecture:</strong> <span style='color: #cbd5e1;'>Deep Neural Network</span><br>
        <strong style='color: #f1f5f9;'>Features:</strong> <span style='color: #cbd5e1;'>{config.get('n_features', 'N/A')} (TF-IDF + Numeric)</span><br>
        <strong style='color: #f1f5f9;'>Performance:</strong><br>
        <span style='color: #cbd5e1;'>• PR-AUC: 0.953</span><br>
        <span style='color: #cbd5e1;'>• F1-Macro: 0.735</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 💡 Tips Input")
        st.markdown("""
        <div style='background: #422006; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
        <span style='color: #fde68a;'>• Tulis review minimal 5-10 kata</span><br>
        <span style='color: #fde68a;'>• Sertakan rating & helpful votes</span><br>
        <span style='color: #fde68a;'>• Gunakan Bahasa Indonesia</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🗑️ Hapus Histori", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    # ---------- HEADER ----------
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-title">🔍 Deteksi Fake Review (DNN)</div>
            <div class="hero-subtitle">
              Masukkan review produk (Bahasa Indonesia) untuk melihat prediksi Real/Fake 
              dengan Deep Neural Network berbasis TF-IDF + Numeric Features.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ---------- TABS ----------
    tab_pred, tab_history, tab_about = st.tabs(["📊 Prediksi", "📜 Histori", "ℹ️ Tentang"])
    
    # ========== TAB: PREDIKSI ==========
    with tab_pred:
        left, right = st.columns([1.2, 1])
        
        # LEFT COLUMN: INPUT
        with left:
            st.markdown("""
            <div style='background: #334155; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem; border: 1px solid #475569;'>
            <h3 style='color: #f1f5f9; margin-top: 0;'>📝 Input Review</h3>
            </div>
            """, unsafe_allow_html=True)
            
            text_input = st.text_area(
                "Masukkan review:",
                height=150,
                placeholder="Contoh: Produk bagus, pengiriman cepat, packing rapi. Seller responsif!",
                label_visibility="collapsed"
            )
            
            col_rating, col_helpful = st.columns(2)
            with col_rating:
                rating_input = st.slider(
                    "⭐ Rating", 
                    1.0, 5.0, 5.0, 0.5,
                    help="Rating produk (1-5 bintang)"
                )
            with col_helpful:
                helpful_input = st.number_input(
                    "👍 Helpful Votes", 
                    0, 1000, 0,
                    help="Jumlah yang menganggap review membantu"
                )
            
            # Example buttons
            st.markdown("""
            <div style='background: #334155; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; border: 1px solid #475569;'>
            <h5 style='color: #f1f5f9; margin-top: 0;'>📋 Contoh Review</h5>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Real Review", use_container_width=True):
                    st.session_state['example_text'] = "Produk sangat bagus, kualitas premium. Pengiriman cepat dan packing rapi. Seller responsif dan ramah. Highly recommended untuk yang cari kualitas terbaik!"
                    st.session_state['example_rating'] = 5.0
                    st.session_state['example_helpful'] = 15
                    st.rerun()
            with c2:
                if st.button("⚠️ Fake Review", use_container_width=True):
                    st.session_state['example_text'] = "Bagus"
                    st.session_state['example_rating'] = 5.0
                    st.session_state['example_helpful'] = 0
                    st.rerun()
            
            # Use example if set
            if 'example_text' in st.session_state:
                text_input = st.session_state.get('example_text', text_input)
                rating_input = st.session_state.get('example_rating', rating_input)
                helpful_input = st.session_state.get('example_helpful', helpful_input)
                # Clear example
                for key in ['example_text', 'example_rating', 'example_helpful']:
                    if key in st.session_state:
                        del st.session_state[key]
            
            # Action buttons
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                do_pred = st.button("🔍 Prediksi", use_container_width=True, type="primary")
            with btn_c2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.rerun()
        
        # RIGHT COLUMN: HASIL
        with right:
            st.markdown("""
            <div style='background: #334155; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem; border: 1px solid #475569;'>
            <h3 style='color: #f1f5f9; margin-top: 0;'>🎯 Hasil Prediksi</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            if do_pred:
                if not text_input.strip():
                    st.warning("⚠️ Teks review masih kosong!")
                else:
                    with st.spinner("Analyzing..."):
                        result = predict_review(
                            text_input, rating_input, helpful_input,
                            model, preprocessing, label_encoder, config,
                            threshold=threshold
                        )
                    
                    if result is None:
                        st.error("❌ Prediction failed")
                    else:
                        # Display prediction with badge
                        st.markdown(
                            f"<div class='result-row'>Prediksi: {badge(result['label'])}</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Prob Real", f"{result['p_real']:.4f}")
                        m2.metric("Prob Fake", f"{result['p_fake']:.4f}")
                        m3.metric("Confidence", f"{result['confidence']:.4f}")
                        
                        # Progress bar
                        st.progress(min(result['confidence'], 1.0))
                        
                        # Performance metrics
                        st.markdown("""
                        <h5 style='color: #a78bfa; margin-top: 1.5rem;'>⚡ Performance</h5>
                        """, unsafe_allow_html=True)
                        perf_c1, perf_c2, perf_c3 = st.columns(3)
                        perf_c1.metric("Preprocessing", f"{result['preprocess_time_ms']:.1f}ms")
                        perf_c2.metric("Prediction", f"{result['predict_time_ms']:.1f}ms")
                        perf_c3.metric("Total", f"{result['total_time_ms']:.1f}ms")
                        
                        # Interpretation
                        st.markdown("""
                        <h5 style='color: #a78bfa; margin-top: 1.5rem;'>📊 Interpretasi</h5>
                        """, unsafe_allow_html=True)
                        
                        if result['p_fake'] >= 0.8:
                            st.markdown("""
                            <div style='background: #7f1d1d; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ef4444; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                            <strong style='color: #fecaca;'>🚨 Sangat mungkin fake review</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        elif result['p_fake'] >= 0.6:
                            st.markdown("""
                            <div style='background: #78350f; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                            <strong style='color: #fde68a;'>⚠️ Kemungkinan fake review</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        elif result['p_fake'] >= 0.4:
                            st.markdown("""
                            <div style='background: #164e63; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #06b6d4; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                            <strong style='color: #a5f3fc;'>ℹ️ Tidak pasti - perlu review manual</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        elif result['p_fake'] >= 0.2:
                            st.markdown("""
                            <div style='background: #14532d; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                            <strong style='color: #bbf7d0;'>✅ Kemungkinan review asli</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background: #14532d; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                            <strong style='color: #bbf7d0;'>✅✅ Sangat mungkin review asli</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.history.insert(
                            0,
                            {
                                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Prediksi": result['label'],
                                "P(Real)": round(result['p_real'], 4),
                                "P(Fake)": round(result['p_fake'], 4),
                                "Confidence": round(result['confidence'], 4),
                                "Rating": rating_input,
                                "Helpful": helpful_input,
                                "Review": text_input[:100] + ("..." if len(text_input) > 100 else "")
                            }
                        )
            else:
                st.markdown("""
                <div style='background: #334155; padding: 2rem; border-radius: 0.5rem; text-align: center; border: 2px dashed #475569; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                <p style='color: #94a3b8; margin: 0; font-size: 1.1rem;'>👆 Klik tombol <strong style='color: #a78bfa;'>Prediksi</strong> untuk melihat hasil</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== TAB: HISTORI ==========
    with tab_history:
        st.markdown("""
        <div style='background: #334155; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem; border: 1px solid #475569;'>
        <h3 style='color: #f1f5f9; margin-top: 0;'>📜 Histori Prediksi (Session Ini)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.history) == 0:
            st.markdown("""
            <div style='background: #164e63; padding: 2rem; border-radius: 0.75rem; text-align: center; border-left: 4px solid #06b6d4; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
            <h4 style='color: #a5f3fc; margin: 0;'>📭 Belum ada histori prediksi</h4>
            <p style='color: #67e8f9; margin-top: 0.5rem;'>Lakukan prediksi untuk melihat histori</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                df, 
                use_container_width=True, 
                hide_index=True,
                height=400
            )
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"fake_review_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    # ========== TAB: TENTANG ==========
    with tab_about:
        st.markdown("### ℹ️ Tentang Aplikasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem; border: 1px solid #334155;'>
            <h4 style='color: #a78bfa; margin-top: 0;'>🧠 Model</h4>
            <p style='color: #cbd5e1;'>
            Aplikasi ini menggunakan <strong style='color: #f1f5f9;'>Deep Neural Network (DNN)</strong> yang 
            dilatih dengan kombinasi fitur:
            </p>
            <ul style='color: #cbd5e1;'>
            <li><strong style='color: #f1f5f9;'>TF-IDF</strong> untuk ekstraksi fitur teks</li>
            <li><strong style='color: #f1f5f9;'>Numeric features</strong> (rating, helpful votes)</li>
            </ul>
            <p style='color: #cbd5e1;'>
            Model dapat mendeteksi fake review dengan akurasi tinggi 
            berdasarkan pola linguistik dan metadata review.
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); border: 1px solid #334155;'>
            <h4 style='color: #a78bfa; margin-top: 0;'>📊 Performance</h4>
            <ul style='color: #cbd5e1;'>
            <li><strong style='color: #f1f5f9;'>PR-AUC:</strong> 0.953</li>
            <li><strong style='color: #f1f5f9;'>F1-Macro:</strong> 0.735</li>
            <li><strong style='color: #f1f5f9;'>Inference Time:</strong> ~64ms (avg)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin-bottom: 1rem; border: 1px solid #334155;'>
            <h4 style='color: #a78bfa; margin-top: 0;'>🎨 Visualisasi</h4>
            <p style='color: #cbd5e1;'>Tampilan aplikasi menggunakan:</p>
            <ul style='color: #cbd5e1;'>
            <li>Custom CSS untuk dark theme modern</li>
            <li>Tab layout untuk navigasi mudah</li>
            <li>Real-time performance metrics</li>
            <li>History tracking dengan session state</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); border: 1px solid #334155;'>
            <h4 style='color: #a78bfa; margin-top: 0;'>🔧 Tech Stack</h4>
            <ul style='color: #cbd5e1;'>
            <li><strong style='color: #f1f5f9;'>Framework:</strong> Streamlit</li>
            <li><strong style='color: #f1f5f9;'>ML:</strong> TensorFlow/Keras</li>
            <li><strong style='color: #f1f5f9;'>NLP:</strong> TF-IDF Vectorizer</li>
            <li><strong style='color: #f1f5f9;'>Deployment:</strong> CPU/GPU inference</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("""
        <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0,0,0,0.4); border: 1px solid #334155;'>
        <h4 style='color: #a78bfa; margin-top: 0;'>📖 Cara Penggunaan</h4>
        <ol style='color: #cbd5e1; line-height: 1.8;'>
        <li>Masukkan teks review di kolom input</li>
        <li>Tentukan rating (1-5 bintang) dan helpful votes</li>
        <li>Klik tombol <strong style='color: #a78bfa;'>Prediksi</strong></li>
        <li>Lihat hasil prediksi beserta confidence score</li>
        <li>Atur threshold di sidebar untuk sensitivitas deteksi</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()