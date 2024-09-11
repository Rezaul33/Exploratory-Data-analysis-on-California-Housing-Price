import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('model/random_forest_model.pkl')

# Define category mapping for ocean_proximity
category_mapping = {
    'NEAR BAY': 0,
    'INLAND': 1,
    '<1H OCEAN': 2,
    'NEAR OCEAN': 3,
    'ISLAND': 4
}

@st.cache
def predict_price(features):
    return model.predict(features)

# Streamlit UI
st.title('Housing Price Application')
st.header('Predict Your Local Housing Price')

# Sidebar inputs
longitude = st.sidebar.slider('Longitude', -180.0, 180.0, -122.23)
latitude = st.sidebar.slider('Latitude', -90.0, 90.0, 37.88)
housing_median_age = st.sidebar.slider('Housing Median Age', 0, 100, 41)
total_rooms = st.sidebar.slider('Total Rooms', 0, 10000, 880)
total_bedrooms = st.sidebar.slider('Total Bedrooms', 0, 5000, 129)
population = st.sidebar.slider('Population', 0, 50000, 322)
households = st.sidebar.slider('Households', 0, 5000, 126)
median_income = st.sidebar.slider('Median Income', 0.0, 15.0, 8.3252)
ocean_proximity = st.sidebar.selectbox('Ocean Proximity', ['NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'])

# Prepare the input data
data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': category_mapping[ocean_proximity]
}

# Create DataFrame
features = pd.DataFrame(data, index=[0])

# Display input features
st.dataframe(features, use_container_width=True)

# Ensure the features DataFrame has the same columns as the training data
expected_columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'median_income', 'ocean_proximity', 'feature_10', 'feature_11', 'feature_12'
]

# Add missing columns with default values if any
for column in expected_columns:
    if column not in features.columns:
        features[column] = 0  # or some other default value

# Reorder columns to match the training data
features = features[expected_columns]

# Model prediction
result = predict_price(features)
st.write(f'Predicted Price: ${result[0]:,.2f}')
