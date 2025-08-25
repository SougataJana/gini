# ==============================================================================
# MPGEM: Molecular Prediction of Gene Expression Matrix (v17 - Static Video Final)
# A Streamlit web application for gene expression prediction using a deep
# learning model.
#
# Author: SciWhy
# Last Updated: August 26, 2025
# ==============================================================================

# --------------------
# IMPORTS
# --------------------
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import gdown
import os
import base64

# Flexible import for TensorFlow/Keras
try:
    from tensorflow.keras.utils import get_custom_objects
except ImportError:
    from keras.utils import get_custom_objects

from keras import backend as K
from keras.layers import Activation
from tensorflow.keras.models import load_model

# --------------------
# CONFIGURATION & CONSTANTS
# --------------------
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"
MPGEM_SAMPLES_FILE_ID = "1-lFwC8w_lNDLmxVsfJLQdjm9bcm5uNuO"
VIDEO_FILE_ID = "1Pzoj2inI9Y5pqltsqLQnl1QOLD-Wa6tL"
BACKGROUND_VIDEO_FILE = "dna_background.mp4" 

# --------------------
# DEEP LEARNING CUSTOM OBJECTS
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12
get_custom_objects().update({'custom_activation': Activation(custom_activation)})

# ------------------------------------------------------------------------------
# UI STYLING & APPEARANCE
# ------------------------------------------------------------------------------
def load_css_and_background():
    # This version links directly to the video file in the 'static' folder.
    # The src path is "static/filename.mp4" without a leading slash, making it a relative path.
    video_html = f"""
         <video autoplay loop muted playsinline id="background-video">
             <source src="static/{BACKGROUND_VIDEO_FILE}" type="video/mp4">
             Your browser does not support the video tag.
         </video>
    """

    st.markdown(
        f"""
        {video_html}
        <style>
        #background-video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -2;
            filter: brightness(0.4);
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            left: 0; right: 0; top: 0; bottom: 0;
            background-color: rgba(0,0,0,0.3);
            z-index: -1;
        }}
        @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes pulse {{
            0% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }}
            70% {{ transform: scale(1.05); box-shadow: 0 0 0 15px rgba(220, 53, 69, 0); }}
            100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% center; }}
            100% {{ background-position: 200% center; }}
        }}
        @keyframes draw-line {{
            from {{ width: 0; }}
            to {{ width: 300px; }}
        }}
        @keyframes gradient-shift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .main .block-container {{ background-color: transparent !important; }}
        .header-section, .stTabs, .stExpander, .st-emotion-cache-1jicfl2, [data-testid="stAlert"] {{
            background: rgba(28, 43, 56, 0.65); backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px;
            padding: 2rem; margin-bottom: 2rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            animation: fadeInUp 0.8s ease-in-out;
        }}
        [data-testid="stAlert"] {{ padding: 1rem 1.5rem !important; }}
        h1, h2, h3, p, .stMarkdown, label {{ color: #ffffff; }}
        .app-header {{ text-align: center; margin-bottom: 3rem; }}
        .main-title {{
            font-size: 4.5rem; font-weight: 800; position: relative;
            padding-bottom: 20px; margin-bottom: 1rem;
            background: linear-gradient(120deg, #f8f9fa, #ffffff, #dc3545, #f8f9fa);
            background-size: 200% auto; -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; animation: shimmer 4s linear infinite;
        }}
        .main-title::after {{
            content: ''; display: block; width: 0px; height: 3px;
            background: linear-gradient(135deg, #dc3545 0%, #f8f9fa 100%);
            position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
            border-radius: 2px; animation: draw-line 1.2s ease-out 0.5s forwards;
        }}
        .main-subtitle {{ font-size: 1.5rem; color: #a0b0c0; }}
        .stButton>button {{
            background: linear-gradient(135deg, #10b981 0%, #6ee7b7 100%);
            color: white; font-weight: bold; font-size: 16px;
            border-radius: 10px; padding: 14px 28px; border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;
        }}
        .stButton>button:hover {{ transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3); }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
        .stTabs [data-baseweb="tab"] {{ background-color: transparent; border-radius: 8px; padding: 10px 16px; color: #a0b0c0; transition: all 0.2s ease; }}
        .stTabs [aria-selected="true"] {{ background: rgba(220, 53, 69, 0.3); color: #ffffff; font-weight: bold; }}
        #fab-container {{ position: fixed; bottom: 40px; right: 40px; z-index: 1000; }}
        #fab-container .stButton>button {{
            background: linear-gradient(135deg, #dc3545 0%, #f8f9fa 100%) !important;
            color: #343a40 !important; text-shadow: 0 0 2px #ffffff;
            background-size: 300% 300%;
            animation: pulse 2.5s infinite, gradient-shift 8s ease infinite;
            width: 65px; height: 65px; border-radius: 50%; font-size: 30px; padding: 0;
        }}
        #fab-container .stButton>button:hover {{
            transform: translateY(-5px) rotate(10deg);
            box-shadow: 0px 12px 25px rgba(0, 0, 0, 0.4);
        }}
        .popup-container {{
            position: fixed; bottom: 120px; right: 40px; width: 350px; z-index: 999;
            animation: fadeInUp 0.5s ease-in-out;
        }}
        .close-button-container .stButton>button {{
            background: transparent !important; border: 1px solid rgba(255, 255, 255, 0.5) !important;
            color: rgba(255, 255, 255, 0.8) !important;
        }}
        .close-button-container .stButton>button:hover {{
            background: rgba(255, 255, 255, 0.1) !important; border-color: #ffffff !important;
            color: #ffffff !important;
        }}
        .popup-header {{ display: flex; align-items: center; justify-content: flex-start; }}
        .popup-header svg {{
            width: 24px; height: 24px; stroke: #dc3545;
            stroke-width: 2; margin-right: 10px;
        }}
        .popup-header h3 {{ margin: 0; font-size: 1.25rem; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# DATA & MODEL LOADING AND CORE LOGIC FUNCTIONS
# ------------------------------------------------------------------------------
@st.cache_data
def load_reference_genes():
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        gdown.download(f"https://drive.google.com/uc?id={REFERENCE_FILE_ID}", temp_path, quiet=True)
        reference_genes = pd.read_csv(temp_path, header=None)[1].tolist()
        return reference_genes, reference_genes[:12712]
    except Exception as e:
        st.error(f"Error downloading reference genes: {e}")
        return None, None

@st.cache_data
def load_mpgem_samples():
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        gdown.download(f"https://drive.google.com/uc?id={MPGEM_SAMPLES_FILE_ID}", temp_path, quiet=True)
        mpgem_samples = pd.read_csv(temp_path, header=None)[0].tolist()
        return mpgem_samples
    except Exception as e:
        st.error(f"Error downloading MPGEM samples list: {e}")
        return None

@st.cache_resource
def load_model_from_drive():
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", temp_path, quiet=True)
        return load_model(temp_path)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return None

@st.cache_resource
def get_video_path():
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        gdown.download(f"https://drive.google.com/uc?id={VIDEO_FILE_ID}", temp_path, quiet=True)
        return temp_path
    except Exception as e:
        st.error(f"Error downloading the tutorial video: {e}")
        return None

def create_submatrix(user_matrix, reference_genes_pred):
    user_genes = set(user_matrix.columns)
    reference_genes = set(reference_genes_pred)
    if reference_genes == user_genes:
        return user_matrix[reference_genes_pred], "equal", []
    elif reference_genes.issubset(user_genes):
        return user_matrix[reference_genes_pred], "extra", []
    else:
        missing = reference_genes - user_genes
        return None, "missing", sorted(list(missing))

def predict_and_merge(submatrix, reference_genes, model):
    input_matrix = submatrix.to_numpy()
    predicted = model.predict(input_matrix, batch_size=1, verbose=0)
    predicted_genes = reference_genes[len(submatrix.columns):]
    predicted_df = pd.DataFrame(predicted, columns=predicted_genes, index=submatrix.index)
    return pd.concat([submatrix, predicted_df], axis=1)

# ==============================================================================
# CONTACT POP-UP (with SVG Icons)
# ==============================================================================
def contact_popup():
    if "show_contact" in st.session_state and st.session_state.show_contact:
        with st.container(border=True):
            st.markdown("""
                <div class="popup-header">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
                    <h3>Contact & Support</h3>
                </div>
            """,
