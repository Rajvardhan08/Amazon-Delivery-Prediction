import streamlit as st
import pandas as pd
import joblib
from haversine import haversine, Unit

# --- 1. Load Trained Model and Columns ---
try:
    model = joblib.load('delivery_time_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please run the training script first to generate 'delivery_time_model.pkl' and 'model_columns.pkl'.")
    st.stop()

# --- 2. Streamlit Page Configuration ---
st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")
st.title("📦 Amazon Delivery Time Prediction")
st.write("Enter the details of an order to predict its delivery time using our trained machine learning model.")

# --- 3. User Input via Sidebar ---
st.sidebar.header("Delivery Details Input")

# Geospatial Inputs
st.sidebar.subheader("📍 Location Details")
store_lat = st.sidebar.number_input("Store Latitude", value=28.63, format="%.4f")
store_lon = st.sidebar.number_input("Store Longitude", value=77.22, format="%.4f")
drop_lat = st.sidebar.number_input("Drop-off Latitude", value=28.70, format="%.4f")
drop_lon = st.sidebar.number_input("Drop-off Longitude", value=77.10, format="%.4f")

# Time Inputs
st.sidebar.subheader("⏰ Order Time")
order_hour = st.sidebar.slider("Hour of Day (24h)", 0, 23, 14)
order_minute = st.sidebar.slider("Minute of Hour", 0, 59, 30)

# Agent and Order Details
st.sidebar.subheader("👤 Agent & Order Info")
agent_age = st.sidebar.slider("Agent's Age", 20, 50, 30)
agent_rating = st.sidebar.slider("Agent's Rating", 1.0, 5.0, 4.5, 0.1)
order_items = st.sidebar.slider("Number of Items", 1, 20, 5)

# Categorical Inputs
st.sidebar.subheader("📋 Categorical Features")
weather = st.sidebar.selectbox("Weather Condition", ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog'])
traffic = st.sidebar.selectbox("Traffic Condition", ['Low', 'Medium', 'Jam'])
vehicle = st.sidebar.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'van'])
area = st.sidebar.selectbox("Area Type", ['Urban', 'Semi-Urban', 'Other'])
category = st.sidebar.selectbox("Product Category", ['Grocery', 'Electronics', 'Clothing', 'Home', 'Books', 'Toys', 'Snacks', 'Pet Supplies', 'Cosmetics', 'Jewelry', 'Sports', 'Outdoors', 'Kitchen', 'Shoes', 'Skincare'])

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict Delivery Time", type="primary"):

    # Calculate distance
    distance_km = haversine((store_lat, store_lon), (drop_lat, drop_lon), unit=Unit.KILOMETERS)

    # Create a dictionary for the input data
    input_data = {
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'Order_Items': order_items,
        'Distance_km': distance_km,
        'Order_Hour': order_hour,
        'Order_Minute': order_minute,
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category
    }

    # Convert to a DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    input_df_encoded = pd.get_dummies(input_df, dtype=int)

    # Align columns with the model's training columns
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, input_df_encoded])
    final_df = final_df.fillna(0)
    final_df = final_df[model_columns] # Ensure column order is the same

    # Make prediction
    prediction = model.predict(final_df)
    predicted_time = int(prediction[0])

    # Display the prediction
    st.subheader("Prediction Result")
    st.metric(label="Predicted Delivery Time", value=f"{predicted_time} minutes")
    st.info(f"The model predicts a delivery time of approximately **{predicted_time} minutes** for the given order details.")

    with st.expander("Show Raw Input Data"):
        st.write(input_data)