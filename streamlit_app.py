import streamlit as st
import pandas as pd

from src.pipelines.inference_pipeline import run_inference
from src.roi import compute_roi, decision
from src.explain import explain
from src.config import *
import matplotlib.pyplot as plt

st.set_page_config(page_title="Causal Uplift Dashboard", layout="wide")

st.title("📊 Customer Retention Uplift Dashboard")

df = run_inference(
    input_path=DATA_PATH,
    model_path=MODEL_PATH,
    feature_path=FEATURE_PATH
)

st.header("📌 Executive Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Avg Uplift", round(df["uplift_pred"].mean(), 3))
col3.metric("Avg ROI", round(df["roi"].mean(), 2))

st.header("🎯 Customer Segments")

st.bar_chart(df["segment"].value_counts())

st.header("💰 ROI Analysis")

roi_by_segment = df.groupby("segment")["roi"].mean()
st.bar_chart(roi_by_segment)

st.header("🔍 Customer Explorer")

customer_id = st.text_input("Enter Customer ID")

if st.button("Analyze Customer"):

    customer = df[df["customerID"] == customer_id]

    if customer.empty:
        st.error("Customer ID not found")

    else:
        st.write(customer)

        uplift = float(customer["uplift_pred"].iloc[0])

        st.write("### Recommendation:", decision(uplift))
        st.write("### ROI:", compute_roi(uplift))

st.header("🧠 Why This Prediction?")

if st.button("Explain Sample Customer"):

    sample = df.sample(1)

    shap_values = explain(model=None, X=sample)  # model loaded internally

    st.write("SHAP explanation generated (see plots in notebook version)")


st.header("📈 Uplift Distribution")



fig, ax = plt.subplots()
ax.hist(df["uplift_pred"], bins=30)

st.pyplot(fig)