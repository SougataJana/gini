import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import gdown
import os

# --------------------
# Flexible import for get_custom_objects
# --------------------
try:
    from tensorflow.keras.utils import get_custom_objects  # Preferred modern import
except ImportError:
    from keras.utils import get_custom_objects  # Fallback for older Keras

from keras import backend as K
from keras.layers import Activation
from tensorflow.keras.models import load_model

# --------------------
# Google Drive File IDs
# --------------------
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"

# --------------------
# Custom activation
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

# --------------------
# Download reference genes
# --------------------
@st.cache_data
def load_reference_genes():
    """
    Downloads and caches the reference gene list from Google Drive.
    """
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        gdown.download(f"https://drive.google.com/uc?id={REFERENCE_FILE_ID}", temp_path, quiet=True)
        reference_genes = pd.read_csv(temp_path, header=None)[1].tolist()
        # The model was trained on the first 12712 genes
        return reference_genes, reference_genes[:12712]
    except Exception as e:
        st.error(f"Error downloading reference genes: {e}")
        return None, None

# --------------------
# Download model
# --------------------
@st.cache_resource
def load_model_from_drive():
    """
    Downloads and caches the trained Keras model from Google Drive.
    """
    try:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", temp_path, quiet=True)
        return load_model(temp_path)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        return None

# --------------------
# Create submatrix
# --------------------
def create_submatrix(user_matrix, reference_genes_pred):
    """
    Checks if the user's matrix contains the required genes and creates a submatrix if needed.
    Returns the submatrix, a status string, and a list of missing genes.
    """
    user_genes = set(user_matrix.columns)
    reference_genes = set(reference_genes_pred)

    if reference_genes == user_genes:
        return user_matrix.copy(), "equal", []
    elif reference_genes.issubset(user_genes):
        return user_matrix[reference_genes_pred], "extra", []
    else:
        missing = reference_genes - user_genes
        return None, "missing", sorted(list(missing))

# --------------------
# Prediction and merge
# --------------------
def predict_and_merge(submatrix, reference_genes, model):
    """
    Runs the prediction on the submatrix and merges the results with the input.
    """
    input_matrix = submatrix.to_numpy()
    predicted = model.predict(input_matrix, batch_size=1, verbose=0)
    
    # Get the genes that were predicted (the rest of the reference genes)
    predicted_genes = reference_genes[len(submatrix.columns):]
    
    predicted_df = pd.DataFrame(predicted, columns=predicted_genes, index=submatrix.index)
    
    return pd.concat([submatrix, predicted_df], axis=1)

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
    .main {background-color: #f0f2f6;}
    h1 {color: #1a73e8; font-weight: 700; text-align: center;}
    h2 {color: #3c4043;}
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
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .css-1e6y4gb {
        padding: 2rem 1rem;
    }
    .stFileUploader label {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .header-section {
        background-color: #e8f0fe;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .header-section p {
        font-size: 1.1rem;
        color: #5f6368;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------
# Main UI Structure
# --------------------
st.markdown("<h1 style='text-align: center;'>🧬 MPGEM </h1>", unsafe_allow_html=True)
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
# Tabs for navigation
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Upload & Check", "🔮 Prediction", "📥 Download", "🔍 Query Results", "📖 Tutorial"])

# --------------------
# Tab 1: Upload & Check
# --------------------
with tab1:
    st.header("Step 1: Upload Your Matrix")
    
    # --------------------
    # Gene List Info and HUGO Link
    # --------------------
    st.markdown("### Reference Gene Information")
    st.markdown("Please verify your gene list uses the correct nomenclature at: [HUGO Gene Nomenclature Committee](https://www.genenames.org/tools/multi-symbol-checker/)")

    # Load reference genes for display
    ref_genes, ref_genes_pred = load_reference_genes()

    if ref_genes_pred:
        with st.expander(f"View the {len(ref_genes_pred)} required reference genes"):
            st.markdown(
                "Your uploaded matrix must contain these genes to proceed. "
                "The app will automatically re-order the genes to match the model's input format, so the original order of your genes will not be maintained in the submatrix."
            )
            st.dataframe(pd.DataFrame(ref_genes_pred, columns=['Required Gene Names']))
    st.markdown("---")
    # --------------------
    # End of Gene List Info
    # --------------------

    # --------------------
    # Sample Data Download Section
    # --------------------
    st.markdown("### Don't have a file? Download a sample dataset.")
    st.markdown("Use this file to understand the required input format and test the application.")

    try:
        # The file name is updated here to your specified file
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
    
    st.markdown("---")
    # --------------------
    # End of Sample Data Section
    # --------------------

    user_file = st.file_uploader("Upload Your CSV File Here", type=["csv"])
    if user_file:
        # Check if the uploaded file has a .csv extension
        file_extension = os.path.splitext(user_file.name)[1]
        if file_extension.lower() != '.csv':
            st.error("Invalid file format. Please upload a CSV file with a '.csv' extension.")
            st.stop()

        with st.spinner("Processing file and fetching reference genes..."):
            try:
                user_matrix = pd.read_csv(user_file, index_col=0)
                
                # Check if reference genes were loaded successfully
                if ref_genes is None or ref_genes_pred is None:
                    st.error("Could not load reference genes. Please try again later.")
                    st.stop()
                
                st.write("### Uploaded Data Preview:")
                st.dataframe(user_matrix.head())
                
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
# Tab 2: Prediction
# --------------------
with tab2:
    st.header("Step 2: Run Prediction")
    if "submatrix" in st.session_state:
        st.markdown("<p>Click the button below to download the model and perform the gene expression prediction.</p>", unsafe_allow_html=True)
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
# Tab 3: Download
# --------------------
with tab3:
    st.header("Step 3: Download Results")
    if "merged_df" in st.session_state:
        st.markdown("<p>The final matrix, including your original data and the predicted gene expression values, is ready for download.</p>", unsafe_allow_html=True)
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
# Tab 4: Query System
# --------------------
with tab4:
    st.header("Step 4: Query Results")
    if "merged_df" in st.session_state:
        df = st.session_state["merged_df"]
        st.markdown("<p>Interactively filter and query the final gene expression matrix.</p>", unsafe_allow_html=True)

        query_type = st.selectbox(
            "Select Query Type",
            ["Gene-based", "Sample-based", "Gene + Sample", "Threshold filter"]
        )

        gene_input = None
        sample_input = None
        threshold_value = None
        comparison_type = None

        if query_type in ["Gene-based", "Gene + Sample", "Threshold filter"]:
            gene_input = st.text_input(
                "Enter Gene Name(s) (comma-separated, case-insensitive, e.g., 'BRCA1, TP53')"
            )

        if query_type in ["Sample-based", "Gene + Sample", "Threshold filter"]:
            sample_input = st.text_input(
                "Enter Sample ID(s) (comma-separated, case-insensitive, e.g., 'sample_1, sample_2')"
            )

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

            # Handle gene filtering
            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",") if g.strip()]
                matching_genes = [col for col in df.columns if any(q in col.lower() for q in gene_list)]
                if not matching_genes:
                    st.error("No matching genes found.")
                    st.stop()
                filtered_df = filtered_df[matching_genes]

            # Handle sample filtering
            if sample_input:
                sample_list = [s.strip().lower() for s in sample_input.split(",") if s.strip()]
                matching_samples = [idx for idx in df.index if any(q in str(idx).lower() for q in sample_list)]
                if not matching_samples:
                    st.error("No matching samples found.")
                    st.stop()
                filtered_df = filtered_df.loc[matching_samples]
            
            # Handle threshold filtering
            if query_type == "Threshold filter" and threshold_value is not None:
                # Use a boolean mask to filter rows based on the condition
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
                
                # Apply the row filter to the original DataFrame
                filtered_df = df[filtered_rows]
                
                # Re-apply the column filter if gene_input was provided
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

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit & Keras</p>", unsafe_allow_html=True)

# --------------------
# Tutorial Tab - NEW CODE
# --------------------
with tab5:
    st.header("📖 Tutorial: How to Use the MPGEM App")

    st.markdown(
        """
        Welcome to the MPGEM Gene Expression Predictor! This tutorial will guide you through each step of the application.
        """
    )
    
    st.subheader("Step 1: Upload & Check")
    st.markdown(
        """
        1.  **File Format:** Ensure your gene expression data is in a CSV (`.csv`) file format. The first column should contain your sample IDs, and the subsequent columns should be your gene names. The body of the matrix should contain normalized gene expression values.
        2.  **Download Sample Data:** If you are unsure about the format, use the **Download Sample CSV** button to get a correctly formatted example file.
        3.  **Gene List Validation:** The app requires a specific set of 12,712 genes for its prediction model. You can view this list by expanding the **View the required reference genes** section.
        4.  **Upload your file:** Click **"Upload Your CSV File Here"** to upload your gene expression matrix. The app will automatically check for format and gene compatibility.
            * **Success:** If your file is compatible, a success message will appear.
            * **Missing Genes:** If genes are missing, an error will be displayed along with a list of the missing genes.
            * **Wrong Format:** An error will be shown if the file is not a valid CSV.
        """
    )

    st.subheader("Step 2: Prediction")
    st.markdown(
        """
        1.  After a successful compatibility check in Step 1, navigate to this tab.
        2.  Click the **"🚀 Run Model Prediction"** button.
        3.  The app will download the pre-trained deep learning model and use it to predict the expression values for the remaining genes in our full reference list.
        4.  This process may take a few minutes. Once complete, a preview of the combined matrix (your original data + the predicted data) will be displayed.
        """
    )

    st.subheader("Step 3: Download")
    st.markdown(
        """
        1.  Once the prediction is complete, the full gene expression matrix is ready.
        2.  Click the **"💾 Download Full Predictions CSV"** button to download the complete file, including all predicted gene expression values, to your local computer.
        """
    )

    st.subheader("Step 4: Query Results")
    st.markdown(
        """
        1.  This tab allows you to interactively filter and explore the prediction results.
        2.  **Select a Query Type:** Choose from `Gene-based`, `Sample-based`, `Gene + Sample`, or `Threshold filter`.
        3.  **Enter your query:** Use the text boxes to enter comma-separated gene names or sample IDs. The search is case-insensitive.
        4.  **Run the query:** Click **"Run Query"**. The filtered results will be displayed in a table below, and you will have the option to download the query result as a new CSV file.
        """
    )

    st.info("If you encounter any issues, please refer back to the instructions in this tutorial or check the file format of your input data.")
