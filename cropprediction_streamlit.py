import streamlit as st
import pandas as pd
import joblib

def load_data():
    """Load preprocessed data from file."""
    data = pd.read_csv(r"E:\Gopikaa\GUVI\Crop Prediction\cropdata.csv")
    required_columns = ['Area', 'Item', 'Year', 'Element', 'Value']
    data = data[required_columns]
    data = data.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value', aggfunc='sum').reset_index()
    data = data.fillna(0)
    return data

def load_model():
    """Load trained machine learning model."""
    return joblib.load("random_forest_model.pkl")

# Streamlit UI
st.title("Crop Production Prediction App")

data = load_data()
model = load_model()

# User Input
area = st.selectbox("Select Area", data['Area'].unique())
crop = st.selectbox("Select Crop", data['Item'].unique())
year = st.slider("Select Year", int(data['Year'].min()), int(data['Year'].max()), step=1)

filtered_data = data[(data['Area'] == area) & (data['Item'] == crop) & (data['Year'] == year)]

if not filtered_data.empty:
    area_harvested = filtered_data['Area harvested'].values[0]
    yield_value = filtered_data['Yield'].values[0]
    
    # Prediction
    prediction = model.predict([[area_harvested, yield_value, year]])[0]
    st.write(f"### Predicted Production: {prediction:.2f} tons")
else:
    st.write("No data available for the selected options.")
