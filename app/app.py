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

# Use st.cache_data to cache the predict function
@st.cache_data
def predict_price(features):
    return model.predict(features)

# Streamlit UI
st.title('Housing Price Application')
st.header('Predict Your Local Housing Price')

# Function to sync both slider and number input
def sync_slider_and_input(label, min_value, max_value, default_value, step):
    # Use Streamlit session state to keep both inputs synced
    if f"{label}_slider" not in st.session_state:
        st.session_state[f"{label}_slider"] = default_value
    if f"{label}_number" not in st.session_state:
        st.session_state[f"{label}_number"] = default_value

    # Update slider value when number input changes
    def update_slider():
        st.session_state[f"{label}_slider"] = st.session_state[f"{label}_number"]

    # Update number input value when slider changes
    def update_number_input():
        st.session_state[f"{label}_number"] = st.session_state[f"{label}_slider"]

    # Create columns for number input and slider to be side by side
    col1, col2 = st.sidebar.columns([1, 3])

    # Display the number input in the first column
    with col1:
        st.number_input(
            label, min_value=min_value, max_value=max_value, value=st.session_state[f"{label}_number"], step=step,
            key=f"{label}_number", on_change=update_slider
        )
    
    # Display the slider in the second column
    with col2:
        st.slider(
            '', min_value, max_value, value=st.session_state[f"{label}_slider"], step=step,
            key=f"{label}_slider", on_change=update_number_input
        )

    return st.session_state[f"{label}_number"]

# Sidebar inputs with two-way sync between number input and slider
longitude = sync_slider_and_input('Longitude', -180.0, 180.0, -122.23, 0.01)
latitude = sync_slider_and_input('Latitude', -90.0, 90.0, 37.88, 0.01)
housing_median_age = sync_slider_and_input('Housing Median Age', 0, 100, 41, 1)
total_rooms = sync_slider_and_input('Total Rooms', 0, 10000, 880, 1)
total_bedrooms = sync_slider_and_input('Total Bedrooms', 0, 5000, 129, 1)
population = sync_slider_and_input('Population', 0, 50000, 322, 1)
households = sync_slider_and_input('Households', 0, 5000, 126, 1)
median_income = sync_slider_and_input('Median Income', 0.0, 15.0, 8.3252, 0.01)

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
