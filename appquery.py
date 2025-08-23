# ==============================================================================
# MPGEM: Molecular Prediction of Gene Expression Matrix
# A Streamlit web application for gene expression prediction using a deep
# learning model.
#
# Author: SciWhy
# Last Updated: August 24, 2025
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
# --- Google Drive File IDs ---
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"
MPGEM_SAMPLES_FILE_ID = "1-lFwC8w_lNDLmxVsfJLQdjm9bcm5uNuO"
VIDEO_FILE_ID = "1Pzoj2inI9Y5pqltsqLQnl1QOLD-Wa6tL"

# --- Image file for background ---
BACKGROUND_IMAGE_FILE = "my_background.jpg"

# --------------------
# DEEP LEARNING CUSTOM OBJECTS
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# ------------------------------------------------------------------------------
# UI STYLING & APPEARANCE
# ------------------------------------------------------------------------------

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css_and_background():
    try:
        img = get_img_as_base64(BACKGROUND_IMAGE_FILE)
    except FileNotFoundError:
        st.warning(f"'{BACKGROUND_IMAGE_FILE}' not found. Using a fallback solid color.")
        img = None

    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .stApp {{
            {f'''
            background-image:
                linear-gradient(to bottom, rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpeg;base64,{img}");
            ''' if img else "background-color: #0e1117;"}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main .block-container {{
            background-color: transparent !important;
        }}
        .header-section, .stTabs, .stExpander {{
            background: rgba(28, 43, 56, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            animation: fadeInUp 0.8s ease-in-out;
        }}
        h1, h2, h3, p, .stMarkdown, label {{ color: #ffffff; }}
        .main-title {{
            font-size: 3.5rem; font-weight: 800; color: #ffffff;
            text-align: center; margin-bottom: -0.2em;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        }}
        .main-subtitle {{
            font-size: 1.5rem; color: #a0b0c0;
            text-align: center; margin-bottom: 2rem;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #1a73e8, #4285f4);
            color: white; font-weight: bold; font-size: 16px;
            border-radius: 10px; padding: 14px 28px; border: none;
            box-shadow: 0 4px 15px rgba(0, 115, 255, 0.3);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 115, 255, 0.5);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent; border-radius: 8px;
            padding: 10px 16px; color: #a0b0c0;
            transition: all 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(26, 115, 232, 0.3);
            color: #ffffff; font-weight: bold;
        }}

        /* --- NEW: Floating Action Button Styles --- */
        .floating-button {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
        }}
        .floating-button .stButton>button {{
            width: 60px;
            height: 60px;
            border-radius: 50%; /* Makes it a circle */
            font-size: 24px;
            padding: 0;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# DATA & MODEL LOADING FUNCTIONS
# ------------------------------------------------------------------------------

@st.cache_data
def load_reference_genes():
    # ... (function content unchanged)
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
    # ... (function content unchanged)
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
    # ... (function content unchanged)
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", temp_path, quiet=True)
        return load_model(temp_path)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return None

@st.cache_resource
def get_video_path():
    # ... (function content unchanged)
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        gdown.download(f"https://drive.google.com/uc?id={VIDEO_FILE_ID}", temp_path, quiet=True)
        return temp_path
    except Exception as e:
        st.error(f"Error downloading the tutorial video: {e}")
        return None

# ------------------------------------------------------------------------------
# CORE LOGIC FUNCTIONS
# ------------------------------------------------------------------------------

def create_submatrix(user_matrix, reference_genes_pred):
    # ... (function content unchanged)
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
    # ... (function content unchanged)
    input_matrix = submatrix.to_numpy()
    predicted = model.predict(input_matrix, batch_size=1, verbose=0)
    predicted_genes = reference_genes[len(submatrix.columns):]
    predicted_df = pd.DataFrame(predicted, columns=predicted_genes, index=submatrix.index)
    return pd.concat([submatrix, predicted_df], axis=1)

# ==============================================================================
# CONTACT DIALOG FUNCTION
# ==============================================================================
def contact_dialog():
    """Creates a pop-up dialog with contact and support information."""
    with st.dialog("Contact & Support"):
        st.header("📧 Contact Information")
        st.markdown(
            """
            This application is maintained by the SciWhy team. For any inquiries,
            please feel free to reach out.
            """
        )
        st.markdown('**Email:** <a href="mailto:sougataj1@gmail.com">contact@sciwhy.org</a>', unsafe_allow_html=True)
        st.markdown("**Project GitHub:** [SougataJana/gini](https://github.com/SougataJana/gini)")
        st.divider()

        st.header("🐞 Report an Issue")
        st.markdown(
            """
            Encountered a bug? Your feedback is invaluable. Please open an issue
            on our GitHub page with a detailed description.
            """
        )
        st.link_button("Submit an Issue", "https://github.com/SougataJana/gini/issues/new")

# ==============================================================================
# MAIN APPLICATION UI
# ==============================================================================

st.set_page_config(
    page_title="MPGEM Predictor",
    layout="wide",
    initial_sidebar_state="expanded" # Sidebar
