import streamlit as st
import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.title("🏡 House Price Predictor")

# Input fields
MedInc = st.number_input("Median Income", 0.0, 20.0, 3.0)
HouseAge = st.number_input("House Age", 0.0, 100.0, 20.0)
AveRooms = st.number_input("Average Rooms", 0.0, 50.0, 5.0)
AveBedrms = st.number_input("Average Bedrooms", 0.0, 10.0, 1.0)
Population = st.number_input("Population", 0.0, 5000.0, 1000.0)
AveOccup = st.number_input("Average Occupancy", 0.0, 50.0, 3.0)
Latitude = st.number_input("Latitude", 30.0, 45.0, 35.0)
Longitude = st.number_input("Longitude", -125.0, -110.0, -120.0)

# Create a dataframe with inputs
features = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]],
                        columns=["MedInc","HouseAge","AveRooms","AveBedrms",
                                 "Population","AveOccup","Latitude","Longitude"])
import matplotlib.pyplot as plt

# Optional: show dataset chart
st.subheader("House Prices vs Median Income")
fig, ax = plt.subplots()
ax.scatter(df['MedInc'], df['Price'], alpha=0.3)
ax.set_xlabel("Median Income")
ax.set_ylabel("House Price")
st.pyplot(fig)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"Predicted House Price: ${prediction*100000:.2f}")
