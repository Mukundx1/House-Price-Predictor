import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# -----------------------
# Load Model and Metadata
# -----------------------

@st.cache_resource
def load_model(model_path: str):
    """Load trained LightGBM model."""
    return joblib.load(model_path)


@st.cache_data
def load_metadata(data_path: str):
    """Load cleaned dataset for metadata like unique locations."""
    df = pd.read_csv(data_path)
    metadata = {
        "locations": sorted(df["location"].dropna().unique()),
        "furnishings": sorted(df["Furnishing"].dropna().unique()),
        "ownerships": sorted(df["Ownership"].dropna().unique()),
        "facings": sorted(df["facing"].dropna().unique()),
        "transactions": sorted(df["Transaction"].dropna().unique()),
    }
    return metadata


# -----------------------
# Preprocessing
# -----------------------

def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal preprocessing to match training format."""

    # Convert categorical columns
    for col in ["location", "Furnishing", "Ownership", "facing", "Transaction"]:
        input_df[col] = input_df[col].astype("category")

    # Convert 'Floor' from string 'X out of Y' to numeric floor number
    if "Floor" in input_df.columns:
        input_df["Floor"] = input_df["Floor"].apply(lambda x: int(str(x).split(" out of ")[0]))

    return input_df


def predict_price(model, input_df: pd.DataFrame) -> float:
    """Make prediction and convert from log scale back to INR."""
    log_pred = model.predict(input_df)[0]
    price = np.expm1(log_pred)  # inverse of log1p used during training
    return price


# -----------------------
# Streamlit App
# -----------------------

def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    st.title("🏠 House Price Predictor")

    # Paths
    MODEL_PATH = "LightGBM_model.pkl"
    DATA_PATH = "CleanedData.csv"

    # Load model + metadata
    model = load_model(MODEL_PATH)
    metadata = load_metadata(DATA_PATH)

    st.sidebar.header("Input Features")

    # ---- User Inputs ----
    location = st.sidebar.selectbox("Select Location", metadata["locations"])
    area = st.sidebar.number_input("Area (sqft)", min_value=100.0, max_value=20000.0, value=1000.0, step=50.0)
    bhk = st.sidebar.slider("Bedrooms (BHK)", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.sidebar.slider("Balcony", min_value=0, max_value=5, value=1)
    floor_input = st.sidebar.text_input("Floor (e.g. '1 out of 5')", value="1 out of 5")
    furnishing = st.sidebar.selectbox("Furnishing", metadata["furnishings"])
    ownership = st.sidebar.selectbox("Ownership", metadata["ownerships"])
    facing = st.sidebar.selectbox("Facing", metadata["facings"])
    transaction = st.sidebar.selectbox("Transaction Type", metadata["transactions"])

    # Collect inputs
    input_dict = {
        "location": location,
        "Area": area,
        "BHK": bhk,
        "Bathroom": bathrooms,
        "Balcony": balcony,
        "Floor": floor_input,
        "Furnishing": furnishing,
        "Ownership": ownership,
        "facing": facing,
        "Transaction": transaction,
    }

    input_df = pd.DataFrame([input_dict])

    st.subheader("📋 User Input")
    st.write(input_df)

    # ---- Prediction ----
    if st.sidebar.button("Predict Price"):
        with st.spinner("Predicting..."):
            processed_input = preprocess_input(input_df)
            price = predict_price(model, processed_input)
            st.success(f"💰 Estimated Price: ₹ {price:,.2f}")


if __name__ == "__main__":
    main()
