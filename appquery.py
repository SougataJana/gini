
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import gdown
import os
import dask.dataframe as dd
import dask.array as da

# --------------------
# Configuration & File IDs
# --------------------
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"
MPGEM_SAMPLES_FILE_ID = "1-lFwC8w_lNDLmxVsfJLQdjm9bcm5uNuO"
VIDEO_FILE_ID = "1Pzoj2inI9Y5pqltsqLQnl1QOLD-Wa6tL"
NORMALIZATION_REFERENCE_FILE_ID = "1YyfzzbwCPZPhO6tg1Kt3-snf1iTICgOw"

# --------------------
# Flexible import for get_custom_objects
# --------------------
try:
    from tensorflow.keras.utils import get_custom_objects
except ImportError:
    from keras.utils import get_custom_objects

from keras import backend as K
from keras.layers import Activation
from tensorflow.keras.models import load_model

# --------------------
# Custom activation
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

# --------------------
# Download reference files (cached for efficiency)
# --------------------
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

@st.cache_data
def load_normalization_reference_matrix():
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        gdown.download(f"https://drive.google.com/uc?id={NORMALIZATION_REFERENCE_FILE_ID}", temp_path, quiet=True)
        return pd.read_csv(temp_path, index_col=0)
    except Exception as e:
        st.error(f"Error downloading normalization reference matrix: {e}")
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
        gdown.download(f"https://drive.google.com/uc?export=download&id={VIDEO_FILE_ID}", temp_path, quiet=True)
        return temp_path
    except Exception as e:
        st.error(f"Error downloading the tutorial video: {e}")
        return None

# --------------------
# Core Logic Functions
# --------------------
def create_submatrix(user_matrix, reference_genes_pred):
    user_genes = set(user_matrix.columns)
    reference_genes = set(reference_genes_pred)

    if reference_genes == user_genes:
        return user_matrix.copy(), "equal", []
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

# --------------------
# Quantile Normalization Pipeline
# --------------------
def find_common_genes(user_df, reference_df):
    user_genes = set(user_df.columns)
    ref_genes = set(reference_df.columns)
    common_genes = list(user_genes.intersection(ref_genes))
    if len(common_genes) == 0:
        st.error("No common genes found between user and reference matrices.")
        return None
    return sorted(common_genes)

def compute_reference_distribution(ref_df):
    try:
        sorted_values = np.sort(ref_df.values, axis=0)
        rsqd = np.mean(sorted_values, axis=1)
        return rsqd
    except Exception as e:
        st.error(f"Error computing RSQD: {e}")
        return None

def normalize_with_rsqd(user_df, rsqd):
    try:
        user_array = user_df.to_numpy()
        sorted_idx = np.argsort(user_array, axis=0)
        ranks = np.argsort(sorted_idx, axis=0)
        norm_array = np.zeros_like(user_array, dtype=float)

        for col in range(user_array.shape[1]):
            norm_array[:, col] = rsqd[ranks[:, col]]

        normalized_df = pd.DataFrame(norm_array, index=user_df.index, columns=user_df.columns)
        return normalized_df
    except Exception as e:
        st.error(f"Error during normalization: {e}")
        return None

def quantile_normalize_pipeline(user_df, reference_df):
    st.info("Finding common genes...")
    common_genes = find_common_genes(user_df, reference_df)
    if common_genes is None: return None

    st.info("Filtering matrices to common genes...")
    user_df_common = user_df[common_genes]
    reference_df_common = reference_df[common_genes]
    
    st.info("Computing Reference Subset Quantile Distribution (RSQD)...")
    rsqd = compute_reference_distribution(reference_df_common)
    if rsqd is None: return None

    st.info("Applying quantile normalization to user matrix...")
    normalized_df = normalize_with_rsqd(user_df_common, rsqd)

    return normalized_df

# --------------------
# UI Styling & Configuration
# --------------------
st.set_page_config(
    page_title="Gene Expression Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1a73e8; 
        font-weight: 700; 
        text-align: center;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #3c4043;
        border-bottom: 2px solid #e8f0fe;
        padding-bottom: 0.5em;
        margin-top: 1.5em;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    .header-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #174ea6;
        border: 2px solid #1a73e8;
    }
    .stFileUploader label {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stTextInput label {
        font-weight: bold;
    }
    .stSelectbox label {
        font-weight: bold;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1em;
        margin-bottom: 1em;
        box-shadow: none;
    }

    /* Custom title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #1a73e8;
        text-align: center;
        margin-bottom: -0.2em;
    }
    .main-subtitle {
        font-size: 1.2rem;
        color: #5f6368;
        text-align: center;
    }
    
    /* --- DARK MODE STYLES --- */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #0e1117;
        }
        .header-section {
            background-color: #1c2b38;
            box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        }
        .header-section p {
            color: #d3d3d3; /* A light gray for dark mode */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------
# Main UI Structure
# --------------------
st.markdown('<h1 class="main-title">🧬 MPGEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Molecular Prediction of Gene Expression Matrix</p>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-section">
    <p>This application is a gene expression prediction tool powered by a deep learning model.
    It takes a partial gene expression matrix as input and predicts the expression values for a larger set of genes.
    The server automatically fetches the necessary reference gene list and a pre-trained Keras model from Google Drive.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------
# Tabs for navigation - UPDATED ORDER
# --------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Normalization",  # NEW FIRST TAB
    "📁 Upload & Check", 
    "🧠 Prediction", 
    "💾 Download", 
    "🔎 Query Results", 
    "📚 Tutorial"
])

# --------------------
# Tab 1: Normalization - NEW CONTENT
# --------------------
with tab1:
    st.header("Step 1: Quantile Normalization")
    st.info("This feature will normalize your uploaded data against a large reference matrix. The normalized data is required for the prediction step.")

    st.subheader("Upload Your Matrix for Normalization")
    user_file_norm = st.file_uploader("Upload CSV File for Normalization Here", type=["csv"], key="normalization_uploader")
    
    if user_file_norm:
        file_extension = os.path.splitext(user_file_norm.name)[1]
        if file_extension.lower() != '.csv':
            st.error("Invalid file format. Please upload a CSV file with a '.csv' extension.")
            st.stop()
        
        # New chunking option for normalization upload
        st.markdown("---")
        use_chunks_norm = st.checkbox("Use memory-efficient loading for large normalization files?", value=False)
        
        with st.spinner("Processing file..."):
            try:
                if use_chunks_norm:
                    st.info("Reading file in chunks for memory-efficient processing.")
                    chunks = pd.read_csv(user_file_norm, chunksize=1000, index_col=0)
                    user_matrix_norm = pd.concat(chunks)
                else:
                    user_matrix_norm = pd.read_csv(user_file_norm, index_col=0)
                
                st.write("### Uploaded Data Preview:")
                st.dataframe(user_matrix_norm.head())
                st.session_state["user_matrix_norm"] = user_matrix_norm

                st.info("Matrix uploaded successfully. You can now run normalization.")
                
            except pd.errors.ParserError:
                st.error("Error parsing the CSV file. Please ensure it is a valid CSV with correct delimiters.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    st.markdown("---")
    
    if "user_matrix_norm" not in st.session_state:
        st.warning("Please upload a matrix in the uploader above to proceed with normalization.")
    else:
        user_matrix_norm = st.session_state["user_matrix_norm"]
        
        st.write("### Normalization Status")
        st.markdown("- **User Matrix Loaded:** ✅ Yes")
        
        if st.button("📈 Run Normalization"):
            with st.spinner("Downloading reference matrix and running normalization... This may take a few minutes."):
                reference_df = load_normalization_reference_matrix()
                if reference_df is not None:
                    normalized_df = quantile_normalize_pipeline(user_matrix_norm, reference_df)
                    
                    if normalized_df is not None:
                        st.session_state["normalized_df"] = normalized_df
                        st.success("✅ Normalization complete! The normalized data is ready.")
                        st.write("### Normalized Data Preview:")
                        st.dataframe(normalized_df.head())
                        
                        csv_normalized = normalized_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="💾 Download Normalized Matrix CSV",
                            data=csv_normalized,
                            file_name="normalized_matrix.csv",
                            mime="text/csv",
                            key="download_normalized_csv"
                        )
                else:
                    st.error("Normalization failed. Could not load reference matrix.")

# --------------------
# Tab 2: Upload & Check (Old Tab 1)
# --------------------
with tab2:
    st.header("Step 2: Upload & Check")
    st.warning("This tab is for uploading the matrix for prediction. Please ensure your matrix is already normalized.")

    st.markdown("### Reference Gene Information")
    st.info("To ensure compatibility, please verify your gene list uses the correct nomenclature at: [HUGO Gene Nomenclature Committee](https://www.genenames.org/tools/multi-symbol-checker/)")

    ref_genes, ref_genes_pred = load_reference_genes()

    if ref_genes_pred:
        with st.expander(f"View the {len(ref_genes_pred)} required reference genes"):
            st.markdown("The app will automatically re-order the genes to match the model's input format, so the original order of your genes will not be maintained in the submatrix.")
            st.dataframe(pd.DataFrame(ref_genes_pred, columns=['Required Gene Names']))
    
    st.divider()

    st.subheader("Upload Your Data")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # User file for prediction
        if "normalized_df" in st.session_state:
            st.success("Normalized data is available from Step 1. No need to upload a new file.")
            user_matrix = st.session_state["normalized_df"]
        else:
            user_file = st.file_uploader("Upload Your CSV File Here", type=["csv"], key="prediction_uploader")
            user_matrix = pd.read_csv(user_file, index_col=0) if user_file else None

        st.markdown("---")

    with col2:
        st.markdown("### Or use sample data")
        st.markdown("Download a correctly formatted file to test the app.")
        try:
            with open("sample_csv_for_testing.csv", "rb") as f:
                sample_csv_data = f.read()

            st.download_button(
                label="Download Sample CSV",
                data=sample_csv_data,
                file_name="sample_csv_for_testing.csv",
                mime="text/csv",
                key="sample_download_button"
            )
        except FileNotFoundError:
            st.warning("Sample file not found. Please ensure 'sample_csv_for_testing.csv' is in the same directory.")
    
    if user_matrix is not None:
        with st.spinner("Processing file and fetching reference genes..."):
            try:
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
                            st.download_button(
                                label="Download Overlapping Samples CSV",
                                data=csv_overlapping,
                                file_name="overlapping_samples.csv",
                                mime="text/csv",
                                key="download_overlapping"
                            )
                    
                    if non_overlapping_samples:
                        with st.expander("View and Download New Sample IDs"):
                            non_overlapping_df = pd.DataFrame(list(non_overlapping_samples), columns=["New Sample IDs"])
                            st.dataframe(non_overlapping_df)
                            csv_non_overlapping = non_overlapping_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download New Samples CSV",
                                data=csv_non_overlapping,
                                file_name="new_samples.csv",
                                mime="text/csv",
                                key="download_new_samples"
                            )

                st.markdown("---")
                st.subheader("MPGEM Reference Data")
                st.info("You can download the full list of MPGEM samples for your reference.")
                mpgem_samples_df = pd.DataFrame(mpgem_samples, columns=['MPGEM Sample IDs'])
                csv_mpgem_samples = mpgem_samples_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download MPGEM Sample List",
                    data=csv_mpgem_samples,
                    file_name="mpgem_sample_list.csv",
                    mime="text/csv",
                    key="download_mpgem_samples"
                )
                
                st.divider()
                
                submatrix, status, missing_genes = create_submatrix(user_matrix, ref_genes_pred)

                st.write("### Compatibility Check:")
                if status == "equal":
                    st.success("✅ Success! Your gene set matches the required reference genes exactly. Ready for prediction.")
                    st.session_state["submatrix"] = submatrix
                    st.session_state["reference_genes"] = ref_genes
                elif status == "extra":
                    st.info("ℹ️ Your matrix contains more genes than required. The app will automatically subset the data to match the reference gene list. Ready for prediction.")
                    st.session_state["submatrix"] = submatrix
                    st.session_state["reference_genes"] = ref_genes
                elif status == "missing":
                    st.error(f"❌ Your matrix is missing {len(missing_genes)} required genes. Prediction cannot proceed. Please upload a file with all necessary genes.")
                    st.write("**Missing Genes:**")
                    st.dataframe(pd.DataFrame(missing_genes, columns=['Gene Name']))
                    if "submatrix" in st.session_state:
                        del st.session_state["submatrix"]
            except pd.errors.ParserError:
                st.error("Error parsing the CSV file. Please ensure it is a valid CSV with correct delimiters.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --------------------
# Tab 3: Prediction (Old Tab 2)
# --------------------
with tab3:
    st.header("Step 3: Run Prediction")
    if "submatrix" in st.session_state:
        st.info("Click the button below to download the model and perform the gene expression prediction.")
        if st.button("🚀 Run Model Prediction"):
            with st.spinner("Downloading model from Google Drive and predicting... This may take a few minutes."):
                model = load_model_from_drive()
                
                if model is None:
                    st.error("Model could not be loaded. Please try again.")
                    st.stop()
                    
                merged_df = predict_and_merge(st.session_state["submatrix"], st.session_state["reference_genes"], model)
                st.session_state["merged_df"] = merged_df
                st.success("✅ Prediction complete! You can now view, download, or query the full dataset.")
                
                st.write("### Prediction Results Preview:")
                st.dataframe(merged_df.head())
    else:
        st.warning("Please upload a valid gene expression matrix in the 'Upload & Check' tab first.")

# --------------------
# Tab 4: Download (Old Tab 3)
# --------------------
with tab4:
    st.header("Step 4: Download Results")
    if "merged_df" in st.session_state:
        st.info("The final matrix, including your original data and the predicted gene expression values, is ready for download.")
        csv = st.session_state["merged_df"].to_csv().encode('utf-8')
        st.download_button(
            label="💾 Download Full Predictions CSV",
            data=csv,
            file_name="full_gene_expression_prediction.csv",
            mime="text/csv",
            key="download_full_csv_button"
        )
        st.info(f"File Size: {len(csv) / (1024*1024):.2f} MB")
    else:
        st.warning("No prediction results to download yet. Please run the prediction in the previous step.")

# --------------------
# Tab 5: Query System (Old Tab 4)
# --------------------
with tab5:
    st.header("Step 5: Query Results")
    if "merged_df" in st.session_state:
        df = st.session_state["merged_df"]
        st.info("Interactively filter and query the final gene expression matrix.")

        query_type = st.selectbox(
            "Select Query Type",
            ["Gene-based", "Sample-based", "Gene + Sample", "Threshold filter"]
        )
        
        gene_input = None
        sample_input = None
        threshold_value = None
        comparison_type = None

        if query_type in ["Gene-based", "Gene + Sample", "Threshold filter"]:
            gene_input = st.text_input("Enter Gene Name(s) (comma-separated, case-insensitive, e.g., 'BRCA1, TP53')")
        if query_type in ["Sample-based", "Gene + Sample", "Threshold filter"]:
            sample_input = st.text_input("Enter Sample ID(s) (comma-separated, case-insensitive, e.g., 'sample_1, sample_2')")
        if query_type == "Threshold filter":
            col1, col2 = st.columns(2)
            with col1:
                threshold_value = st.number_input("Enter expression value threshold", value=0.0)
            with col2:
                comparison_type = st.selectbox("Comparison Type", [">", ">=", "<", "<=", "=="])
            st.info("The threshold filter will return all samples where at least one of the specified genes meets the condition.")

        if st.button("Run Query", key="run_query_button"):
            filtered_df = df.copy()
            original_columns = df.columns

            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
                matching_genes = [col for col in df.columns if any(q in col.lower() for q in gene_list)]
                if not matching_genes:
                    st.error("No matching genes found.")
                    st.stop()
                filtered_df = filtered_df[matching_genes]

            if sample_input:
                sample_list = [s.strip().lower() for s in sample_input.split(",") if s.strip()]
                matching_samples = [idx for idx in df.index if any(q in str(idx).lower() for q in sample_list)]
                if not matching_samples:
                    st.error("No matching samples found.")
                    st.stop()
                filtered_df = filtered_df.loc[matching_samples]
            
            if query_type == "Threshold filter" and threshold_value is not None:
                if comparison_type == ">":
                    filtered_rows = (df[filtered_df.columns] > threshold_value).any(axis=1)
                elif comparison_type == ">=":
                    filtered_rows = (df[filtered_df.columns] >= threshold_value).any(axis=1)
                elif comparison_type == "<":
                    filtered_rows = (df[filtered_df.columns] < threshold_value).any(axis=1)
                elif comparison_type == "<=":
                    filtered_rows = (df[filtered_df.columns] <= threshold_value).any(axis=1)
                elif comparison_type == "==":
                    filtered_rows = (df[filtered_df.columns] == threshold_value).any(axis=1)
                
                filtered_df = df[filtered_rows]
                
                if gene_input:
                    matching_genes = [col for col in original_columns if any(q in col.lower() for q in gene_list)]
                    filtered_df = filtered_df[matching_genes]

            if not filtered_df.empty:
                st.write("### Query Result:")
                st.dataframe(filtered_df)

                csv_query = filtered_df.to_csv().encode('utf-8')
                st.download_button(
                    label="💾 Download Query Result CSV",
                    data=csv_query,
                    file_name="query_result.csv",
                    mime="text/csv",
                    key="download_query_csv_button"
                )
            else:
                st.info("The query returned no results.")
    else:
        st.warning("Please run the prediction first to generate the data for querying.")

# --------------------
# Tab 6: Tutorial (Old Tab 5)
# --------------------
with tab6:
    st.header("📚 Tutorial: How to Use the MPGEM App")
    st.markdown("Watch this video for a quick walkthrough of the app's features.")
    
    video_path = get_video_path()
    if video_path:
        st.video(video_path)

    st.markdown("Welcome to the MPGEM Gene Expression Predictor! This tutorial will guide you through each step of the application.")
    
    st.subheader("Step 1: Quantile Normalization")
    st.markdown(
        """
        1.  **Upload:** Use the uploader in the `Normalization` tab to provide your gene expression matrix.
        2.  **Run Normalization:** Click the 'Run Normalization' button. The app will download the large reference matrix, compute the RSQD, and normalize your data.
        3.  **Result:** A preview of the normalized matrix will appear, and you will have the option to download it.
        """
    )
    
    st.subheader("Step 2: Prediction")
    st.markdown(
        """
        1.  **Upload & Check:** In this tab, you can use the matrix normalized in Step 1 or upload a new one. The app will perform gene compatibility checks.
        2.  **Run Prediction:** Click the **"🚀 Run Model Prediction"** button. The app will download the deep learning model and predict expression values for the complete set of genes.
        3.  **Preview:** A preview of the combined matrix will be shown once the prediction is complete.
        """
    )

    st.subheader("Step 3: Download")
    st.markdown(
        """
        1.  When the prediction is complete, the full matrix is ready for download.
        2.  Click the **"💾 Download Full Predictions CSV"** button to save the file to your computer.
        """
    )

    st.subheader("Step 4: Query Results")
    st.markdown(
        """
        1.  This tab allows you to filter the prediction results interactively.
        2.  **Select a Query Type:** Choose from `Gene-based`, `Sample-based`, `Gene + Sample`, or `Threshold filter`.
        3.  **Enter your query:** Use the text boxes to enter comma-separated gene names or sample IDs.
        4.  **Run the query:** Click **"Run Query"**. The filtered results will be displayed in a table below, and you will have the option to download the query result.
        """
    )

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit & Keras</p>", unsafe_allow_html=True)
