#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import gdown
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
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
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    gdown.download(f"https://drive.google.com/uc?id={REFERENCE_FILE_ID}", temp_path, quiet=False)
    reference_genes = pd.read_csv(temp_path, header=None)[1].tolist()
    return reference_genes, reference_genes[:12712]

# --------------------
# Download model
# --------------------
@st.cache_resource
def load_model_from_drive():
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", temp_path, quiet=False)
    return load_model(temp_path)

# --------------------
# Create submatrix
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
        return None, "missing", sorted(missing)

# --------------------
# Prediction and merge
# --------------------
def predict_and_merge(submatrix, reference_genes, model):
    input_matrix = submatrix.to_numpy()
    predicted = model.predict(input_matrix, batch_size=1, verbose=0)
    predicted_genes = reference_genes[len(submatrix.columns):]
    predicted_df = pd.DataFrame(predicted, columns=predicted_genes, index=submatrix.index)
    return pd.concat([submatrix, predicted_df], axis=1)

# --------------------
# UI Styling
# --------------------
st.set_page_config(page_title="Gene Expression Predictor", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2E8B57;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>🧬 Gene Expression Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your normalized gene expression matrix. The app will fetch the reference genes & model automatically from Google Drive.</p>", unsafe_allow_html=True)

# --------------------
# Tabs for navigation
# --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload & Check", "🔮 Prediction", "📥 Download", "🔍 Query"])

# --------------------
# Tab 1: Upload & Check
# --------------------
with tab1:
    st.subheader("📂 Step 1: Upload Your Matrix")
    user_file = st.file_uploader("Upload CSV File", type=["csv"])
    if user_file:
        with st.spinner("Fetching reference genes..."):
            reference_genes, reference_genes_pred = load_reference_genes()
        user_matrix = pd.read_csv(user_file, index_col=0)

        st.write("**Uploaded Data Preview:**")
        st.dataframe(user_matrix.head())

        with st.spinner("Checking gene compatibility..."):
            submatrix, status, missing_genes = create_submatrix(user_matrix, reference_genes_pred)

        if status == "equal":
            st.success("✅ Gene sets are equal. Ready for prediction.")
        elif status == "extra":
            st.info("ℹ️ Extra genes found. They will be removed automatically.")
        elif status == "missing":
            st.error(f"❌ Missing {len(missing_genes)} genes. Cannot proceed.")
            st.write(missing_genes)
            st.stop()

        st.session_state["submatrix"] = submatrix
        st.session_state["reference_genes"] = reference_genes

# --------------------
# Tab 2: Prediction
# --------------------
with tab2:
    st.subheader("🔮 Step 2: Run Prediction")
    if "submatrix" in st.session_state:
        if st.button("Run Model Prediction"):
            with st.spinner("Downloading model & predicting..."):
                model = load_model_from_drive()
                merged_df = predict_and_merge(st.session_state["submatrix"], st.session_state["reference_genes"], model)
                st.session_state["merged_df"] = merged_df
                st.success("✅ Prediction complete!")
                st.dataframe(merged_df.head())
    else:
        st.warning("Please upload and check your matrix in Step 1.")

# --------------------
# Tab 3: Download
# --------------------
with tab3:
    st.subheader("📥 Step 3: Download Results")
    if "merged_df" in st.session_state:
        csv = st.session_state["merged_df"].to_csv().encode('utf-8')
        st.download_button(
            label="💾 Download Predictions CSV",
            data=csv,
            file_name="merged_prediction.csv",
            mime="text/csv"
        )
    else:
        st.warning("No prediction results to download yet.")

# --------------------
# Tab 4: Query System
# --------------------
with tab4:
    st.subheader("🔍 Step 4: Query Results")
    if "merged_df" in st.session_state:
        df = st.session_state["merged_df"]

        query_type = st.radio(
            "Select Query Type",
            ["Gene-based", "Sample-based", "Gene + Sample", "Threshold filter"]
        )

        gene_input = None
        sample_input = None
        threshold_value = None

        if query_type in ["Gene-based", "Gene + Sample", "Threshold filter"]:
            gene_input = st.text_input(
                "Enter Gene Name(s) (comma-separated, case-insensitive, supports partial matches)"
            )

        if query_type in ["Sample-based", "Gene + Sample", "Threshold filter"]:
            sample_input = st.text_input(
                "Enter Sample ID(s) (comma-separated, case-insensitive, supports partial matches)"
            )

        if query_type == "Threshold filter":
            threshold_value = st.number_input("Enter expression value threshold", value=0.0)
            comparison_type = st.selectbox("Comparison Type", [">", ">=", "<", "<=", "=="])

        if st.button("Run Query"):
            filtered_df = df.copy()

            if gene_input:
                gene_list = [g.strip().lower() for g in gene_input.split(",")]
                matching_genes = [col for col in df.columns if any(q in col.lower() for q in gene_list)]
                if not matching_genes:
                    st.error("No matching genes found.")
                    st.stop()
                filtered_df = filtered_df[matching_genes]

            if sample_input:
                sample_list = [s.strip().lower() for s in sample_input.split(",")]
                matching_samples = [idx for idx in df.index if any(q in idx.lower() for q in sample_list)]
                if not matching_samples:
                    st.error("No matching samples found.")
                    st.stop()
                filtered_df = filtered_df.loc[matching_samples]

            if query_type == "Threshold filter" and threshold_value is not None:
                if comparison_type == ">":
                    filtered_df = filtered_df[(filtered_df > threshold_value).any(axis=1)]
                elif comparison_type == ">=":
                    filtered_df = filtered_df[(filtered_df >= threshold_value).any(axis=1)]
                elif comparison_type == "<":
                    filtered_df = filtered_df[(filtered_df < threshold_value).any(axis=1)]
                elif comparison_type == "<=":
                    filtered_df = filtered_df[(filtered_df <= threshold_value).any(axis=1)]
                elif comparison_type == "==":
                    filtered_df = filtered_df[(filtered_df == threshold_value).any(axis=1)]

            st.write("**Query Result:**")
            st.dataframe(filtered_df)

            csv = filtered_df.to_csv().encode('utf-8')
            st.download_button(
                label="💾 Download Query Result CSV",
                data=csv,
                file_name="query_result.csv",
                mime="text/csv"
            )
    else:
        st.warning("Please run the prediction first to generate data.")

