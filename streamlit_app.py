import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the saved model
model_path = 'diabetes_model.sav'
with open(model_path, 'rb') as file:
    diabetes_model = pickle.load(file)

# Function to predict diabetes
def predict_diabetes(data):
    prediction = diabetes_model.predict(data)
    return prediction

# Streamlit app layout
st.set_page_config(page_title="Diabetes Prediction", page_icon="🌟", layout="wide")

# Title
st.title('Diabetes Prediction Web App')

# Initialize session state for storing patient data and prediction count
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

if 'diabetes_count' not in st.session_state:
    st.session_state.diabetes_count = 0

# Sidebar Navigation
menu = st.sidebar.radio(
    "Select an option",
    ['Diabetes Prediction', 'Visualizations']
)

# Diabetes Prediction Page
if menu == 'Diabetes Prediction':
    st.subheader('Enter Patient Details')

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', [0, 1, 2], format_func=lambda x: ['Female', 'Male', 'Other'][x])
        age = st.number_input('Age', min_value=0, max_value=120, step=1)
        hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    with col2:
        heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        smoking_history = st.selectbox('Smoking History', [0, 1, 2, 3, 4, 5], format_func=lambda x: ['Current', 'Ever', 'Former', 'Never', 'No info', 'Not current'][x])
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)

    col3, col4 = st.columns(2)
    with col3:
        HbA1c_level = st.number_input('HbA1c Level (Average Blood Sugar Level)', min_value=0.0, max_value=20.0, step=0.1)
    with col4:
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=400, step=1)

    # Store input data in session state
    st.session_state.patient_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,  # Updated key
        'blood_glucose_level': blood_glucose_level
    }

    # Prediction Button
    if st.button('Predict'):
        input_data = pd.DataFrame([st.session_state.patient_data])
        prediction = predict_diabetes(input_data)
        if prediction[0] == 1:
            st.success('Diabetes prediction: Positive', icon="✅")
            st.session_state.diabetes_count += 1
        else:
            st.error('Diabetes prediction: Negative', icon="❌")

        # Real-Time Update: Display the current count of diabetes predictions
        st.session_state.diabetes_display = st.empty()
        st.session_state.diabetes_display.text(f"Total diabetes predictions so far: {st.session_state.diabetes_count}")

    # About Section
    st.write('---')
    st.subheader('About this app:')
    st.write(
        "This app predicts whether a patient has diabetes based on their medical data. "
        "Enter the details above and click 'Predict' to see the result."
    )

# Visualizations Page
if menu == 'Visualizations':
    st.subheader('Visualizations')

    # Sidebar Options for Visualizations
    viz_option = st.selectbox('Select Visualization', ['Patient Data Overview', 'Data Distribution'])

    if viz_option == 'Patient Data Overview':
        st.subheader('Patient Data Overview')
        # Extract data from session state
        patient_data = st.session_state.patient_data
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(['Age', 'BMI', 'HbA1c Level', 'Blood Glucose Level'],
               [patient_data['age'], patient_data['bmi'], patient_data['HbA1c_level'], patient_data['blood_glucose_level']],
               color=['#69b3a2', '#404080', '#ff9999', '#66b3ff'])
        ax.set_ylabel('Values')
        ax.set_title('Patient Data Overview')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif viz_option == 'Data Distribution':
        st.subheader('Data Distribution')
        patient_data = st.session_state.patient_data
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Age Distribution
        ax[0].hist([patient_data['age']], bins=10, color='#ffcc99', edgecolor='black')
        ax[0].set_title('Age Distribution')
        ax[0].set_xlabel('Age')
        ax[0].set_ylabel('Frequency')

        # Blood Glucose Level Distribution
        ax[1].hist([patient_data['blood_glucose_level']], bins=20, color='#ff6666', edgecolor='black')
        ax[1].set_title('Blood Glucose Level Distribution')
        ax[1].set_xlabel('Blood Glucose Level')
        ax[1].set_ylabel('Frequency')

        plt.tight_layout()
        st.pyplot(fig)

# CSS Styling for a better look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 1rem;
        cursor: pointer;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSelectbox > div, .stTextInput > div, .stRadio > div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stSelectbox > div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput > div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stRadio > div {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
