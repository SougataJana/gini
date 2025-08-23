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
## IMPORTANT: Make sure you have a 'background.jpg' file in your GitHub repository!
BACKGROUND_IMAGE_FILE = "my_background.jpg"

# --------------------
# DEEP LEARNING CUSTOM OBJECTS
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# ------------------------------------------------------------------------------
# UI STYLING & APPEARANCE
# This section injects custom CSS for a modern, visually appealing interface.
# ------------------------------------------------------------------------------

@st.cache_data
def get_img_as_base64(file):
    """Reads an image file and returns its Base64 encoded string."""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css_and_background():
    """Loads CSS styles and sets the background image."""
    try:
        img = get_img_as_base64(BACKGROUND_IMAGE_FILE)
    except FileNotFoundError:
        st.warning(f"'{BACKGROUND_IMAGE_FILE}' not found. Using a fallback solid color.")
        img = None

    st.markdown(
        f"""
        <style>
        /* --- FADE-IN ON LOAD ANIMATION --- */
        @keyframes fadeInUp {{
            0% {{
                opacity: 0;
                transform: translateY(20px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        /* --- CORE APP STYLING --- */
        .stApp {{
            {f'''
            background-image:
                linear-gradient(to bottom, rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpeg;base64,{img}");
            ''' if img else "background-color: #0e1117;"}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed; /* This creates the parallax effect */
        }}

        /* --- Ensure main content area is transparent to see the background --- */
        .main .block-container {{
            background-color: transparent !important;
        }}

        /* --- FROSTED GLASS EFFECT & FADE-IN ANIMATION FOR CONTAINERS --- */
        .header-section, .stTabs, .stExpander {{
            background: rgba(28, 43, 56, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            animation: fadeInUp 0.8s ease-out forwards; /* Apply the animation */
        }}

        /* Add a slight delay for the tabs to animate after the header */
        .stTabs {{
            animation-delay: 0.2s;
        }}

        /* --- TYPOGRAPHY --- */
        h1, h2, h3, p, .stMarkdown, label {{
            color: #ffffff;
        }}
        .main-title {{
            font-size: 3.5rem;
            font-weight: 800;
            color: #ffffff;
            text-align: center;
            margin-bottom: -0.2em;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        }}
        .main-subtitle {{
            font-size: 1.5rem;
            color: #a0b0c0;
            text-align: center;
            margin-bottom: 2rem;
        }}

        /* --- WIDGETS & INTERACTIVITY --- */
        .stButton>button {{
            background: linear-gradient(90deg, #1a73e8, #4285f4);
            color: white;
            font-weight: bold;
            font-size: 16px;
            border-radius: 10px;
            padding: 14px 28px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 115, 255, 0.3);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 115, 255, 0.5);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 8px;
            padding: 10px 16px;
            color: #a0b0c0;
            transition: all 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(26, 115, 232, 0.3);
            color: #ffffff;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# DATA & MODEL LOADING FUNCTIONS
# Cached functions to download necessary files from Google Drive.
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

# ------------------------------------------------------------------------------
# CORE LOGIC FUNCTIONS
# Functions for data processing and prediction.
# ------------------------------------------------------------------------------

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
# MAIN APPLICATION UI
# ==============================================================================

st.set_page_config(
    page_title="MPGEM Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the custom CSS and background
load_css_and_background()

# --- HEADER ---
st.markdown('<h1 class="main-title">🔬 MPGEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Molecular Prediction of Gene Expression Matrix</p>', unsafe_allow_html=True)
with st.container():
    st.markdown(
        """
        <div class="header-section">
        <p style="color: #d0d8e0; font-size: 1.1rem;">
        This application leverages a deep learning model to predict a comprehensive gene expression profile from a partial matrix.
        Upload your data to begin the analysis.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# --- TABS FOR NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "» Upload & Validate",
    "✨ Predict",
    "⤓ Download",
    "🎯 Query",
    "🗺️ Tutorial"
])


# --------------------
# TAB 1: UPLOAD & VALIDATE
# --------------------
with tab1:
    st.header("Step 1: Upload & Validate Your Matrix")
    st.markdown("### Reference Gene Information")
    st.info("Ensure gene nomenclature is correct via [HUGO Gene Nomenclature Committee](https://www.genenames.org/tools/multi-symbol-checker/)")

    ref_genes, ref_genes_pred = load_reference_genes()
    if ref_genes_pred:
        with st.expander(f"View the {len(ref_genes_pred)} required reference genes"):
            st.dataframe(pd.DataFrame(ref_genes_pred, columns=['Required Gene Names']))

    st.divider()
    st.subheader("Upload Your Data")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        user_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    with col2:
        st.markdown("##### Don't have a file?")
        st.markdown("Download a sample dataset to test the application.")
        try:
            with open("sample_csv_for_testing.csv", "rb") as f:
                st.download_button(
                    label="Download Sample CSV",
                    data=f.read(),
                    file_name="sample_csv_for_testing.csv",
                    mime="text/csv"
                )
        except FileNotFoundError:
            st.warning("Sample file not found in repository.")

    if user_file:
        try:
            with st.spinner("Processing file..."):
                user_matrix = pd.read_csv(user_file, index_col=0)
                if ref_genes is None or ref_genes_pred is None:
                    st.error("Could not load reference genes. Please try again later.")
                    st.stop()

                st.write("### Uploaded Data Preview:")
                st.dataframe(user_matrix.head())

                n_samples, n_genes = user_matrix.shape
                st.markdown(f"- **Samples:** {n_samples} | **Genes:** {n_genes}")

                # Gene compatibility check
                submatrix, status, missing_genes = create_submatrix(user_matrix, ref_genes_pred)
                st.write("### Compatibility Check:")
                if status == "equal" or status == "extra":
                    st.success("✅ Success! Your gene set is compatible. Ready for prediction.")
                    st.session_state["submatrix"] = submatrix
                    st.session_state["reference_genes"] = ref_genes
                elif status == "missing":
                    st.error(f"❌ Your matrix is missing {len(missing_genes)} required genes.")
                    st.dataframe(pd.DataFrame(missing_genes, columns=['Missing Genes']))
                    if "submatrix" in st.session_state: del st.session_state["submatrix"]
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --------------------
# TAB 2: PREDICTION
# --------------------
with tab2:
    st.header("Step 2: Run Prediction")
    if "submatrix" in st.session_state:
        st.info("Your data is ready. Click the button below to run the deep learning model.")
        if st.button("🚀 Run Model Prediction"):
            with st.spinner("Downloading model and predicting... This may take a moment."):
                model = load_model_from_drive()
                if model:
                    merged_df = predict_and_merge(st.session_state["submatrix"], st.session_state["reference_genes"], model)
                    st.session_state["merged_df"] = merged_df
                    st.success("✅ Prediction complete!")
                    st.write("### Prediction Results Preview:")
                    st.dataframe(merged_df.head())
                else:
                    st.error("Model could not be loaded. Please try again.")
    else:
        st.warning("Please upload a valid gene expression matrix in the 'Upload & Validate' tab first.")

# --------------------
# TAB 3: DOWNLOAD
# --------------------
with tab3:
    st.header("Step 3: Download Results")
    if "merged_df" in st.session_state:
        st.info("The final matrix, with original and predicted values, is ready.")
        csv = st.session_state["merged_df"].to_csv().encode('utf-8')
        st.download_button(
            label="💾 Download Full Predictions CSV",
            data=csv,
            file_name="full_gene_expression_prediction.csv",
            mime="text/csv",
        )
        st.info(f"File Size: {len(csv) / (1024*1024):.2f} MB")
    else:
        st.warning("No prediction results to download. Please run the prediction first.")

# --------------------
# TAB 4: QUERY
# --------------------
with tab4:
    st.header("Step 4: Query Results")
    if "merged_df" in st.session_state:
        df = st.session_state["merged_df"]
        st.info("Interactively filter the final gene expression matrix.")
        gene_input = st.text_input("Enter Gene Name(s) (comma-separated, e.g., 'BRCA1, TP53')")
        sample_input = st.text_input("Enter Sample ID(s) (comma-separated, e.g., 'sample_1')")

        if st.button("Run Query"):
            filtered_df = df.copy()
            if gene_input:
                gene_list = [g.strip().upper() for g in gene_input.split(",") if g.strip()]
                matching_genes = [col for col in df.columns if col.upper() in gene_list]
                if not matching_genes:
                    st.error("No matching genes found.")
                    st.stop()
                filtered_df = filtered_df[matching_genes]
            if sample_input:
                sample_list = [s.strip() for s in sample_input.split(",") if s.strip()]
                filtered_df = filtered_df[filtered_df.index.isin(sample_list)]

            if not filtered_df.empty:
                st.write("### Query Result:")
                st.dataframe(filtered_df)
                csv_query = filtered_df.to_csv().encode('utf-8')
                st.download_button("Download Query Result", csv_query, "query_result.csv", "text/csv")
            else:
                st.warning("Query returned no results.")
    else:
        st.warning("Please run a prediction to generate data for querying.")


# --------------------
# TAB 5: TUTORIAL
# --------------------
with tab5:
    st.header("How to Use the MPGEM WebApp")
    video_path = get_video_path()
    if video_path:
        st.video(video_path)

    st.subheader("Step 1: » Upload & Validate")
    st.markdown("Ensure your data is a CSV file with Sample IDs in the first column and gene names as headers. Use the **Download Sample CSV** button for a format example.")

    st.subheader("Step 2: ✨ Predict")
    st.markdown("After a successful validation, click **Run Model Prediction**. The app will download the model and generate the complete gene expression matrix.")

    st.subheader("Step 3: ⤓ Download")
    st.markdown("Once prediction is done, download the complete CSV file containing both your original and the newly predicted data.")

    st.subheader("Step 4: 🎯 Query")
    st.markdown("Interactively filter the results by gene or sample names to inspect specific data points.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by SciWhy</p>", unsafe_allow_html=True)
