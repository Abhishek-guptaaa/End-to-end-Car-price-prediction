import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Brand to models mapping
brand_to_models = {
    "Maruti": ["Swift", "Baleno", "Alto", "Dzire"],
    "Honda": ["Accord", "Civic", "Fit"],
    "Toyota": ["Camry", "Corolla", "Prius"],
    "Ford": ["Fusion", "Focus", "Mustang"],
    "BMW": ["3 Series", "5 Series", "X3"],
    "Audi": ["A3", "A4", "Q5"]
}

# Set UI
st.set_page_config(page_title="Car Price Prediction", layout="centered")



st.title("🚗 Car Price Prediction")

# Step 1: Select brand first (outside the form)
brand = st.selectbox("Select Brand", list(brand_to_models.keys()))

# Step 2: Based on brand, show relevant models
selected_model = st.selectbox("Select Model", brand_to_models.get(brand, []))

# Step 3: Now the form for other fields and submit
with st.form("prediction_form"):
    year = st.number_input("Year", min_value=1990, max_value=2030, step=1)
    mileage = st.number_input("Mileage (in km)", min_value=0)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'year': [year],
            'mileage': [mileage],
            'brand': [brand],
            'model': [selected_model],
            'fuel': [fuel]
        })

        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)

        st.success(f"💰 **Predicted Car Price:** ₹ {prediction[0]:,.2f}")
