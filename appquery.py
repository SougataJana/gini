import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import gdown

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
# CONFIG & THEME
# --------------------
st.set_page_config(page_title="Gene Expression Predictor", layout="wide", page_icon="🧬")

st.markdown("""
<style>
/* Background Bright */
[data-testid="stAppViewContainer"] {
    background: #ffffff;
    color: black;
}

/* Titles & Text */
.big-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: black;
    text-align: center;
    padding: 20px;
    margin-bottom: 10px;
    background: linear-gradient(270deg, #a8e6cf, #dcedc1, #ffd3b6);
    background-size: 600% 600%;
    animation: gradientAnimation 10s ease infinite;
    border-radius: 12px;
}
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Step Card */
.step-card {
    background-color: rgba(255,255,255,1);
    padding: 20px;
    margin-top: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}
.step-card:hover {
    transform: scale(1.01);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #388e3c, #66bb6a);
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 20px;
    transition: background 0.3s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #66bb6a, #388e3c);
}

/* Dataframes alternating row colors */
.dataframe tbody tr:nth-child(odd) {
    background-color: #f2f2f2;
}
.dataframe tbody tr:nth-child(even) {
    background-color: #ffffff;
}

/* Table text size */
[data-testid="stDataFrame"] div {
    font-size: 16px !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# HEADER
# --------------------
st.markdown("<div class='big-title'>🧬 Gene Expression Predictor</div>", unsafe_allow_html=True)

st.markdown("""
### ℹ️ About This App  
This tool predicts missing **gene expression values** in your dataset using a pretrained deep learning model.  
It automatically:
1. **Validates** your uploaded dataset against a reference gene list.
2. **Predicts** missing gene expression values.
3. **Merges** predicted and original values into a complete dataset.
4. **Allows** you to run queries on the results.

---

### 📂 Accepted File Formats  
- **CSV** file only (rows = sample IDs, columns = gene names).  
- First column must contain **sample IDs**.  
- Gene names must be **column headers**.  
- The matrix should be **normalized** before uploading.  

---

✅ *The server automatically downloads the correct model and reference gene list — you only need to upload your file.*
""")

# --------------------
# CONSTANTS
# --------------------
REFERENCE_FILE_ID = "1-DSpHwN4TbFvGsYEv-UboB4yrvWPKDZo"
MODEL_FILE_ID = "13N99OC_fplCKZHz2H52AFQaSeAI1Ai-v"

# --------------------
# MODEL HELPERS
# --------------------
def custom_activation(x):
    return K.sigmoid(x) * 12
get_custom_objects().update({'custom_activation': Activation(custom_activation)})

@st.cache_data
def load_reference_genes():
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    gdown.download(f"https://drive.google.com/uc?id={REFERENCE_FILE_ID}", path, quiet=False)
    reference_genes = pd.read_csv(path, header=None)[1].tolist()
    return reference_genes, reference_genes[:12712]

@st.cache_resource
def load_model_from_drive():
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", path, quiet=False)
    return load_model(path)

def create_submatrix(user_matrix, reference_genes_pred):
    user_genes = set(user_matrix.columns)
    reference_genes = set(reference_genes_pred)
    if reference_genes == user_genes:
        return user_matrix.copy(), "equal", []
    elif reference_genes.issubset(user_genes):
        return user_matrix[reference_genes_pred], "extra", []
    else:
        missing = reference_genes - user_genes
        return None, "missing", sorted(missing)

def predict_and_merge(submatrix, reference_genes, model):
    predicted = model.predict(submatrix.to_numpy(), batch_size=1, verbose=0)
    predicted_genes = reference_genes[len(submatrix.columns):]
    predicted_df = pd.DataFrame(predicted, columns=predicted_genes, index=submatrix.index)
    return pd.concat([submatrix, predicted_df], axis=1)

# --------------------
# STEP 1: UPLOAD
# --------------------
st.markdown("<div class='step-card'>", unsafe_allow_html=True)
st.subheader("📂 Step 1: Upload Your Matrix")
user_file = st.file_uploader("Upload CSV File", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

if user_file:
    reference_genes, reference_genes_pred = load_reference_genes()
    user_matrix = pd.read_csv(user_file, index_col=0)

    # 📄 Dataset Preview
    st.write("📄 **Uploaded Dataset Preview:**")
    st.dataframe(user_matrix.head())

    # STEP 2: CHECK
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.subheader("🔍 Step 2: Check Gene Compatibility")
    submatrix, status, missing_genes = create_submatrix(user_matrix, reference_genes_pred)
    if status == "equal":
        st.success("✅ Gene sets match perfectly.")
    elif status == "extra":
        st.info("ℹ️ Extra genes detected. They will be removed.")
    else:
        st.error(f"❌ Missing {len(missing_genes)} genes. Cannot proceed.")
        st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 3: PREDICT
    st.markdown("<div class='step-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Step 3: Predict Missing Genes")
    if st.button("🚀 Run Prediction"):
        with st.spinner("Running deep learning model..."):
            model = load_model_from_drive()
            merged_df = predict_and_merge(submatrix, reference_genes, model)
            st.session_state["merged_df"] = merged_df
            st.success("✅ Prediction Complete!")
            st.dataframe(merged_df.head())

            # 📥 Direct Download of Merged DF
            csv_full = merged_df.to_csv().encode('utf-8')
            st.download_button("💾 Download Full Merged Dataset", csv_full, "merged_prediction.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 4: QUERY
    if "merged_df" in st.session_state:
        st.markdown("<div class='step-card'>", unsafe_allow_html=True)
        st.subheader("🔍 Step 4: Query Results")
        gene_input = st.text_input("Search Gene Name (case-insensitive)")
        sample_input = st.text_input("Search Sample ID (case-insensitive)")
        result_df = st.session_state["merged_df"]

        if gene_input:
            result_df = result_df[[col for col in result_df.columns if gene_input.lower() in col.lower()]]
        if sample_input:
            result_df = result_df.loc[[idx for idx in result_df.index if sample_input.lower() in idx.lower()]]

        st.dataframe(result_df)
        csv = result_df.to_csv().encode('utf-8')
        st.download_button("💾 Download Query Result", csv, "query_result.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("✨ **Made for bioinformatics research** — Fast, Accurate, Reliable.")
