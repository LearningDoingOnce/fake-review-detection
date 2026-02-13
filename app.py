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
def load_css(path: str):
    """Load external CSS file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ CSS file not found: {path}")

# Load CSS (akan dibuat di assets/style.css)
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
        **Architecture:** Deep Neural Network  
        **Features:** {config.get('n_features', 'N/A')} (TF-IDF + Numeric)  
        **Performance:**
        - PR-AUC: 0.953
        - F1-Macro: 0.735
        """)
        
        st.divider()
        
        st.markdown("### 💡 Tips Input")
        st.write("- Tulis review minimal 5-10 kata")
        st.write("- Sertakan rating & helpful votes")
        st.write("- Gunakan Bahasa Indonesia")
        
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
            st.markdown("### 📝 Input Review")
            
            text_input = st.text_area(
                "Masukkan review:",
                height=150,
                placeholder="Contoh: Produk bagus, pengiriman cepat, packing rapi. Seller responsif!"
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
            st.markdown("##### 📋 Contoh Review")
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
            st.markdown("### 🎯 Hasil Prediksi")
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
                        st.markdown("##### ⚡ Performance")
                        perf_c1, perf_c2, perf_c3 = st.columns(3)
                        perf_c1.metric("Preprocessing", f"{result['preprocess_time_ms']:.1f}ms")
                        perf_c2.metric("Prediction", f"{result['predict_time_ms']:.1f}ms")
                        perf_c3.metric("Total", f"{result['total_time_ms']:.1f}ms")
                        
                        # Interpretation
                        st.markdown("##### 📊 Interpretasi")
                        if result['p_fake'] >= 0.8:
                            st.error("🚨 Sangat mungkin fake review")
                        elif result['p_fake'] >= 0.6:
                            st.warning("⚠️ Kemungkinan fake review")
                        elif result['p_fake'] >= 0.4:
                            st.info("ℹ️ Tidak pasti - perlu review manual")
                        elif result['p_fake'] >= 0.2:
                            st.success("✅ Kemungkinan review asli")
                        else:
                            st.success("✅✅ Sangat mungkin review asli")
                        
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
                st.caption("👆 Klik tombol **Prediksi** untuk melihat hasil")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== TAB: HISTORI ==========
    with tab_history:
        st.markdown("### 📜 Histori Prediksi (Session Ini)")
        
        if len(st.session_state.history) == 0:
            st.info("📭 Belum ada histori prediksi")
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
            st.markdown("#### 🧠 Model")
            st.write("""
            Aplikasi ini menggunakan **Deep Neural Network (DNN)** yang 
            dilatih dengan kombinasi fitur:
            - **TF-IDF** untuk ekstraksi fitur teks
            - **Numeric features** (rating, helpful votes)
            
            Model dapat mendeteksi fake review dengan akurasi tinggi 
            berdasarkan pola linguistik dan metadata review.
            """)
            
            st.markdown("#### 📊 Performance")
            st.write("""
            - **PR-AUC**: 0.953
            - **F1-Macro**: 0.735
            - **Inference Time**: ~64ms (avg)
            """)
        
        with col2:
            st.markdown("#### 🎨 Visualisasi")
            st.write("""
            Tampilan aplikasi menggunakan:
            - Custom CSS untuk styling modern
            - Tab layout untuk navigasi mudah
            - Real-time performance metrics
            - History tracking dengan session state
            """)
            
            st.markdown("#### 🔧 Tech Stack")
            st.write("""
            - **Framework**: Streamlit
            - **ML**: TensorFlow/Keras
            - **NLP**: TF-IDF Vectorizer
            - **Deployment**: CPU/GPU inference
            """)
        
        st.divider()
        
        st.markdown("#### 📖 Cara Penggunaan")
        st.write("""
        1. Masukkan teks review di kolom input
        2. Tentukan rating (1-5 bintang) dan helpful votes
        3. Klik tombol **Prediksi**
        4. Lihat hasil prediksi beserta confidence score
        5. Atur threshold di sidebar untuk sensitivitas deteksi
        """)

if __name__ == "__main__":
    main()