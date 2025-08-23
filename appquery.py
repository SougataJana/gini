
# ==============================================================================
# MPGEM: Molecular Prediction of Gene Expression Matrix (v12 - Themed Final)
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
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"
MPGEM_SAMPLES_FILE_ID = "1-lFwC8w_lNDLmxVsfJLQdjm9bcm5uNuO"
VIDEO_FILE_ID = "1Pzoj2inI9Y5pqltsqLQnl1QOLD-Wa6tL"
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
    with open(file, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

def load_css_and_background():
    try: img = get_img_as_base64(BACKGROUND_IMAGE_FILE)
    except FileNotFoundError: st.warning(f"'{BACKGROUND_IMAGE_FILE}' not found."); img = None

    st.markdown(
        f"""
        <style>
        /* --- Animation Keyframes --- */
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

        /* --- Main App Styling --- */
        .stApp {{
            {f'''background-image: linear-gradient(to bottom, rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/jpeg;base64,{img}");''' if img else "background-color: #0e1117;"}
            background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
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
        
        /* --- Combined Header Styles --- */
        .app-header {{ text-align: center; margin-bottom: 3rem; }}
        .header-container {{
            display: flex; align-items: center; justify-content: center;
            position: relative; padding-bottom: 20px; margin-bottom: 1rem;
        }}
        .header-container::after {{
            content: ''; display: block; width: 0px; height: 3px;
            background: linear-gradient(135deg, #dc3545 0%, #f8f9fa 100%);
            position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
            border-radius: 2px; animation: draw-line 1.2s ease-out 0.5s forwards;
        }}
        .header-svg {{ stroke: #f8f9fa; }}
        .main-title {{
            font-size: 4rem; font-weight: 800;
            background: linear-gradient(120deg, #f8f9fa, #ffffff, #dc3545, #f8f9fa);
            background-size: 200% auto; -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; animation: shimmer 5s linear infinite;
        }}
        .main-subtitle {{ font-size: 1.5rem; color: #a0b0c0; }}
        
        /* --- Primary Action Button Style (Green/Teal) --- */
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
        
        /* --- ADVANCED Floating Action Button Styles (Red & White Theme) --- */
        .floating-button .stButton>button {{
            background: linear-gradient(135deg, #dc3545 0%, #f8f9fa 100%) !important;
            color: #343a40 !important;
            text-shadow: 0 0 2px #ffffff;
            background-size: 300% 300%;
            animation: pulse 2.5s infinite, gradient-shift 8s ease infinite;
            width: 65px; height: 65px; border-radius: 50%; font-size: 30px; padding: 0;
        }}
        .floating-button .stButton>button:hover {{
            transform: translateY(-5px) rotate(10deg);
            box-shadow: 0px 12px 25px rgba(0, 0, 0, 0.4);
        }}
        .fab-text {{
            visibility: hidden; width: 150px; background-color: #333; color: #fff;
            text-align: center; border-radius: 6px; padding: 5px 0;
            position: absolute; z-index: 1; bottom: 25%; right: 120%;
            opacity: 0; transition: opacity 0.3s, transform 0.3s;
            transform: translateX(10px);
        }}
        .floating-button:hover .fab-text {{ visibility: visible; opacity: 1; transform: translateX(0); }}
        #fab-st-button-container {{ position: fixed; bottom: 40px; right: 40px; z-index: 1000; }}
        
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
# CONTACT POP-UP
# ==============================================================================
def contact_popup():
    if "show_contact" in st.session_state and st.session_state.show_contact:
        with st.container(border=True):
            st.header("💬 Contact & Support")
            st.markdown(
                """
                This application is maintained by the SciWhy team. For any inquiries,
                please feel free to reach out.
                """
            )
            st.markdown('**Email:** <a href="mailto:sougataj1@gmail.com">contact@sciwhy.org</a>', unsafe_allow_html=True)
            st.markdown("**Project GitHub:** [SougataJana/gini](https://github.com/SougataJana/gini)")
            st.divider()
            st.subheader("🐞 Report an Issue")
            st.markdown("Encountered a bug? Please open an issue on our GitHub page.")
            st.link_button("Submit an Issue", "https://github.com/SougataJana/gini/issues/new")
            st.divider()
            st.subheader("🤝 About Us")
            st.markdown("Learn more about our research and other projects.")
            st.link_button("Visit Shandarlab", "http://shandarslab.org")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="close-button-container">', unsafe_allow_html=True)
            if st.button("Close", key="close_contact_popup", use_container_width=True):
                st.session_state.show_contact = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# MAIN APPLICATION UI
# ==============================================================================
st.set_page_config(page_title="MPGEM Predictor", layout="wide", initial_sidebar_state="collapsed")
load_css_and_background()

# --- COMBINED HEADER ---
st.markdown("""
    <div class="app-header">
        <div class="header-container">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="header-svg">
                <path d="M16 4.38531C15.0471 3.52877 13.8295 3 12.5 3C10.1221 3 8.12213 4.67157 7.37398 6.81816M11 21C9.02324 21 7.21884 19.8634 6.22564 18.1818M11 21L12.5 16L14 21M11 21V16.5M16.7744 18.1818C17.7812 19.8634 19.0232 21 21.5 21C22.8295 21 23.0471 20.4712 24.0000 19.6147M16.7744 18.1818C17.5225 16.0347 19.5225 14.3631 21.8999 14.3631C23.0101 14.3631 24 13.5 24 12.1818C24 10.8636 23.0101 10 21.9 10C19.6716 10 17.6716 8.32843 16.9234 6.18184M16.7744 18.1818L12.5 16M7.37398 6.81816C6.62582 8.96525 4.62582 10.6369 2.25001 10.6369C1.13989 10.6369 0.150009 11.5 0.150009 12.8182C0.150009 14.1364 1.13989 15 2.25001 15C4.47844 15 6.47843 16.6716 7.2266 18.8182M7.37398 6.81816L12.5 8.5M16.9234 6.18184L12.5 8.5M7.2266 18.8182L12.5 16M16.9234 6.18184C16.1753 8.32843 14.1753 10 11.9 10C10.7899 10 9.80001 10.8636 9.80001 12.1818C9.80001 13.5 10.7899 14.3631 11.9 14.3631C14.3768 14.3631 16.1812 15.5 17.1744 17.1818" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h1 class="main-title" style="padding-left: 15px;">MPGEM</h1>
        </div>
        <p class="main-subtitle">Molecular Prediction of Gene Expression Matrix</p>
    </div>
    """, unsafe_allow_html=True)

with st.container():
    st.markdown(
        """
        <div class="header-section">
        <p style="color: #d0d8e0; font-size: 1.1rem;">
        This application is a gene expression prediction tool powered by a deep learning model.
        It takes a partial gene expression matrix as input and predicts the expression values for a larger set of genes.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs(["» Upload & Validate", "✨ Predict", "⤓ Download", "🎯 Query", "🗺️ Tutorial"])

# --------------------
# TAB 1: UPLOAD & VALIDATE
# --------------------
with tab1:
    st.header("Step 1: Upload & Validate Your Matrix")
    st.markdown("### Reference Gene Information")
    st.info("To ensure compatibility, please verify your gene list uses the correct nomenclature at: [HUGO Gene Nomenclature Committee](https://www.genenames.org/tools/multi-symbol-checker/)")
    ref_genes, ref_genes_pred = load_reference_genes()
    if ref_genes_pred:
        with st.expander(f"View the {len(ref_genes_pred)} required reference genes"):
            st.markdown("The app will automatically re-order the genes to match the model's input format...")
            st.dataframe(pd.DataFrame(ref_genes_pred, columns=['Required Gene Names']))
    st.divider()
    st.subheader("Upload Your Data")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        user_file = st.file_uploader("Upload Your CSV File Here", type=["csv"])
    with col2:
        st.markdown("### Don't have a file? Download a sample dataset.")
        st.markdown("Use this file to understand the required input format...")
        try:
            with open("sample_csv_for_testing.csv", "rb") as f:
                sample_csv_data = f.read()
            st.download_button(label="Download Sample CSV", data=sample_csv_data, file_name="sample_csv_for_testing.csv", mime="text/csv", key="sample_download_button")
        except FileNotFoundError:
            st.warning("Sample file not found. Please ensure 'sample_csv_for_testing.csv' is in the same directory.")
    if user_file:
        with st.spinner("Validating your data... 🧬"):
            try:
                user_matrix = pd.read_csv(user_file, index_col=0)
                st.write("### Uploaded Data Preview:")
                st.dataframe(user_matrix.head())
                st.write("### Data Statistics and Sample Overlap")
                n_samples, n_genes = user_matrix.shape
                st.markdown(f"- **Number of Samples in your Matrix:** {n_samples}")
                st.markdown(f"- **Number of Genes in your Matrix:** {n_genes}")
                mpgem_samples = load_mpgem_samples()
                if mpgem_samples:
                    user_samples_set = set(user_matrix.index)
                    mpgem_samples_set = set(mpgem_samples)
                    overlapping_samples = user_samples_set.intersection(mpgem_samples_set)
                    non_overlapping_samples = user_samples_set - mpgem_samples_set
                    st.markdown(f"- **Samples already in MPGEM reference list:** {len(overlapping_samples)}")
                    st.markdown(f"- **New Samples in your Matrix:** {len(non_overlapping_samples)}")
                    if overlapping_samples:
                        with st.expander("View and Download Overlapping Sample IDs"):
                            overlapping_df = pd.DataFrame(list(overlapping_samples), columns=["Overlapping Sample IDs"])
                            st.dataframe(overlapping_df)
                            csv_overlapping = overlapping_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Overlapping Samples CSV", csv_overlapping, "overlapping_samples.csv", "text/csv", key="download_overlapping")
                    if non_overlapping_samples:
                        with st.expander("View and Download New Sample IDs"):
                            non_overlapping_df = pd.DataFrame(list(non_overlapping_samples), columns=["New Sample IDs"])
                            st.dataframe(non_overlapping_df)
                            csv_non_overlapping = non_overlapping_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download New Samples CSV", csv_non_overlapping, "new_samples.csv", "text/csv", key="download_new_samples")
                st.markdown("---")
                st.subheader("MPGEM Reference Data")
                st.info("You can download the full list of MPGEM samples for your reference.")
                mpgem_samples_df = pd.DataFrame(mpgem_samples, columns=['MPGEM Sample IDs'])
                csv_mpgem_samples = mpgem_samples_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download MPGEM Sample List", csv_mpgem_samples, "mpgem_sample_list.csv", "text/csv", key="download_mpgem_samples")
                st.divider()
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
                st.error(f"An unexpected error occurred: {e}")

# --------------------
# TAB 2: PREDICTION
# --------------------
with tab2:
    st.header("Step 2: Run Prediction")
    if "submatrix" in st.session_state:
        st.info("Your data is ready. Click the button below to run the deep learning model.")
        if st.button("🚀 Run Model Prediction"):
            with st.spinner("Running the deep learning model... 🧠"):
                model = load_model_from_drive()
                if model:
                    merged_df = predict_and_merge(st.session_state["submatrix"], st.session_state["reference_genes"], model)
                    st.session_state["merged_df"] = merged_df
                    st.success("✅ Prediction complete!")
                    st.toast('Your data has been successfully processed!', icon='🎉')
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
        st.download_button(label="💾 Download Full Predictions CSV", data=csv, file_name="full_gene_expression_prediction.csv", mime="text/csv")
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
        st.info("Interactively filter and query the final gene expression matrix.")
        query_type = st.selectbox("Select Query Type", ["Gene-based", "Sample-based", "Gene + Sample", "Threshold filter"])
        gene_input = st.text_input("Enter Gene Name(s) (comma-separated, case-insensitive, e.g., 'BRCA1, TP53')")
        sample_input = st.text_input("Enter Sample ID(s) (comma-separated, case-insensitive, e.g., 'sample_1, sample_2')")
        if st.button("Run Query"):
            filtered_df = df.copy()
            if gene_input:
                gene_list = [g.strip().upper() for g in gene_input.split(",") if g.strip()]
                matching_genes = [col for col in df.columns if col.upper() in gene_list]
                if not matching_genes:
                    st.error("No matching genes found.")
                else:
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
    st.header("🗺️ Tutorial")
    video_path = get_video_path()
    if video_path:
        st.video(video_path)
    st.markdown("Welcome to the MPGEM Gene Expression Predictor! This tutorial will guide you...")
    st.subheader("Step 1: » Upload & Validate")
    st.markdown("""1. **File Format:** ...""")
    st.subheader("Step 2: ✨ Predict")
    st.markdown("""1. After a successful validation, click ...""")
    st.subheader("Step 3: ⤓ Download")
    st.markdown("""1. Once prediction is complete, ...""")
    st.subheader("Step 4: 🎯 Query")
    st.markdown("""1. This tab allows you to filter ...""")

# --- POP-UP and FLOATING BUTTON LOGIC ---
st.markdown('<div class="popup-container">', unsafe_allow_html=True)
contact_popup()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="floating-button">
    <div class="fab-text">Contact & Support</div>
    <div id="fab-st-button-container"></div>
</div>
""", unsafe_allow_html=True)

_, fab_col = st.columns([0.92, 0.08])
with fab_col:
    if "show_contact" not in st.session_state: st.session_state.show_contact = False
    if st.button("💬", key="floating_contact"):
        st.session_state.show_contact = not st.session_state.show_contact
        st.rerun()

# --- FOOTER ---
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by SciWhy</p>", unsafe_allow_html=True)
