# ==============================================================================
# MPGEM: Molecular Prediction of Gene Expression Matrix (v18 - Syntax Fix)
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
            """, unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #a0b0c0;'>This application is maintained by the SciWhy team. For any inquiries, please feel free to reach out.</p>", unsafe_allow_html=True)
            st.markdown('**Email:** <a href="mailto:sougataj1@gmail.com">contact@sciwhy.org</a>', unsafe_allow_html=True)
            st.markdown("**Project GitHub:** [SougataJana/gini](https://github.com/SougataJana/gini)")
            st.divider()
            st.markdown("""
                <div class="popup-header">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                    <h3>Report an Issue</h3>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #a0b0c0;'>Encountered a bug? Please open an issue on our GitHub page.</p>", unsafe_allow_html=True)
            st.link_button("Submit an Issue", "https://github.com/SougataJana/gini/issues/new")
            st.divider()
            st.markdown("""
                <div class="popup-header">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    <h3>About Us</h3>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #a0b0c0;'>Learn more about our research and other projects.</p>", unsafe_allow_html=True)
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

# --- HEADER ---
st.markdown("""
    <div class="app-header">
        <h1 class="main-title">MPGEM</h1>
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

    st.markdown("Welcome to the MPGEM Gene Expression Predictor! This tutorial will guide you through each step of the application.")

    st.subheader("Step 1: » Upload & Validate")
    st.markdown(
        """
        1.  **File Format:** Ensure your gene expression data is in a CSV (`.csv`) file with the first column as Sample IDs and subsequent columns as gene names. Values should be normalized gene expression data.
        2.  **Sample Data:** If you are unsure about the format, use the **Download Sample CSV** button to get a correctly formatted file.
        3.  **Gene List Validation:** The model requires a specific set of 12,712 genes. The app will validate your data against this list.
        4.  **Upload your file:** Click **"Upload Your CSV File Here"** to upload your gene expression matrix. The app will check for compatibility and provide feedback.
        """
    )
    st.subheader("Step 2: ✨ Predict")
    st.markdown(
        """
        1.  After a successful compatibility check in Step 1, navigate to this tab.
        2.  Click the **"🚀 Run Model Prediction"** button.
        3.  The app will download the pre-trained neural network model and predict the expression values for the complete set of genes.
        4.  This process may take a few minutes. A preview of the combined matrix will be shown once the prediction is complete.
        """
    )
    st.subheader("Step 3: ⤓ Download")
    st.markdown(
        """
        1.  Once the prediction is complete, the full gene expression matrix is ready.
        2.  Click the **"💾 Download Full Predictions CSV"** button to download the complete file, including all predicted gene expression values, to your local computer.
        """
    )
    st.subheader("Step 4: 🎯 Query")
    st.markdown(
        """
        1.  This tab allows you to filter the prediction results interactively.
        2.  Select a **Query Type**, then use the text boxes to enter comma-separated gene names or sample IDs.
        3.  Click **"Run Query"** to see the filtered results, which you can also download as a new CSV file.
        """
    )

# --- POP-UP and FLOATING BUTTON LOGIC ---
st.markdown('<div class="popup-container">', unsafe_allow_html=True)
contact_popup()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div id="fab-container">', unsafe_allow_html=True)
if "show_contact" not in st.session_state:
    st.session_state.show_contact = False
if st.button("💬", key="floating_contact"):
    st.session_state.show_contact = not st.session_state.show_contact
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


# --- FOOTER ---
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by SciWhy</p>", unsafe_allow_html=True)
