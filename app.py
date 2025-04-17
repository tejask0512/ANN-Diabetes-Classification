# app.py
# Streamlit web application for diabetes prediction

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4C78A8;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #72B7B2;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .positive {
        background-color: #FFCCCC;
    }
    .negative {
        background-color: #CCFFCC;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the trained model
@st.cache_resource
def load_model():
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
        return None
    return tf.keras.models.load_model(model_path)

# Function to load the scaler
@st.cache_resource
def load_scaler():
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at {scaler_path}. Please ensure preprocessing is completed correctly.")
        return None
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Header section
st.markdown('<p class="main-header">Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown("""
This application uses an Artificial Neural Network to predict the likelihood of diabetes based on medical factors.
""")

# Sidebar for user inputs
st.sidebar.title('Patient Information')
st.sidebar.markdown('Adjust the sliders to input patient data for prediction.')

# Function to collect user input features
def user_input_features():
    with st.sidebar:
        st.subheader('Medical Measurements')
        pregnancies = st.slider('Number of Pregnancies', 0, 17, 3, help="Number of times the patient has been pregnant")
        glucose = st.slider('Glucose Level (mg/dL)', 0, 200, 120, help="Blood glucose level after 2 hours in an oral glucose tolerance test")
        blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 70, help="Diastolic blood pressure")
        skin_thickness = st.slider('Skin Thickness (mm)', 0, 99, 20, help="Triceps skin fold thickness")
        
        st.subheader('Additional Factors')
        insulin = st.slider('Insulin Level (mu U/ml)', 0, 846, 79, help="2-Hour serum insulin")
        bmi = st.slider('BMI (weight in kg/(height in m)²)', 0.0, 67.1, 32.0, format="%.1f", help="Body mass index")
        diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, format="%.4f", help="A function that scores likelihood of diabetes based on family history")
        age = st.slider('Age (years)', 21, 81, 29, help="Age of the patient")
        
        st.info('Adjust all sliders to match the patient\'s data for an accurate prediction.')
    
    # Create a dictionary of features
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Main content area - organized in tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Patient Data", "Model Information"])

with tab1:
    st.markdown('<p class="sub-header">Diabetes Risk Assessment</p>', unsafe_allow_html=True)
    
    # Check if model and scaler are loaded
    if model is not None and scaler is not None:
        # Make prediction
        scaled_input = scaler.transform(user_input)
        prediction_prob = model.predict(scaled_input)[0][0]
        prediction = (prediction_prob > 0.5).astype(int)
        
        # Display prediction with styled box
        if prediction == 1:
            st.markdown('<div class="prediction-box positive">', unsafe_allow_html=True)
            st.subheader("⚠️ Diabetes Risk: POSITIVE")
            st.markdown(f"The model predicts that this patient has a **{prediction_prob:.1%}** probability of having diabetes.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ### Recommended Next Steps:
            * Schedule a follow-up appointment for confirmation
            * Order an A1C test or fasting plasma glucose test
            * Evaluate patient for other comorbidities
            * Consider dietary and lifestyle counseling
            """)
        else:
            st.markdown('<div class="prediction-box negative">', unsafe_allow_html=True)
            st.subheader("✅ Diabetes Risk: NEGATIVE")
            st.markdown(f"The model predicts that this patient has a **{prediction_prob:.1%}** probability of having diabetes.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ### Recommendations:
            * Continue regular health check-ups
            * Maintain a healthy diet and regular exercise
            * Monitor blood glucose levels periodically, especially if risk factors are present
            """)
        
        # Visualize prediction probability
        st.subheader('Prediction Probability')
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create a horizontal bar chart
        ax.barh(['Risk'], [prediction_prob], color='#FF9999' if prediction == 1 else '#66B2FF')
        ax.barh(['Risk'], [1-prediction_prob], left=[prediction_prob], color='#EEEEEE')
        
        # Add a vertical line at 50%
        ax.axvline(x=0.5, color='red', linestyle='--')
        
        # Add labels
        ax.text(0.5, 0, '50% Threshold', ha='center', va='bottom', color='red')
        ax.text(0.05, 0, 'Low Risk', ha='left', va='center')
        ax.text(0.95, 0, 'High Risk', ha='right', va='center')
        
        # Customize the chart
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        
        st.pyplot(fig)
        
        # Feature importance note
        st.subheader('Key Risk Factors')
        st.write("""
        While the model considers all input factors, certain variables typically have stronger correlations with diabetes:
        * **Glucose level**: High blood glucose is a direct indicator
        * **BMI**: Higher values often correlate with increased risk
        * **Age**: Risk increases with age
        * **Diabetes Pedigree Function**: Family history is an important factor
        
        Note that this model provides a statistical prediction and is not a definitive medical diagnosis.
        """)
    else:
        st.error("Error loading model or scaler. Please check that all files are available.")

with tab2:
    st.markdown('<p class="sub-header">Patient Data Summary</p>', unsafe_allow_html=True)
    
    # Display user inputs in a cleaner format
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Measurements")
        st.write(f"**BMI:** {user_input['BMI'].values[0]:.1f} kg/m²")
        st.write(f"**Blood Pressure:** {user_input['BloodPressure'].values[0]} mm Hg")
        st.write(f"**Skin Thickness:** {user_input['SkinThickness'].values[0]} mm")
        st.write(f"**Age:** {user_input['Age'].values[0]} years")
    
    with col2:
        st.subheader("Blood Work")
        st.write(f"**Glucose:** {user_input['Glucose'].values[0]} mg/dL")
        st.write(f"**Insulin:** {user_input['Insulin'].values[0]} mu U/ml")
        st.write(f"**Pregnancies:** {user_input['Pregnancies'].values[0]}")
        st.write(f"**Diabetes Pedigree:** {user_input['DiabetesPedigreeFunction'].values[0]:.4f}")
    
    # Compare with normal ranges
    st.subheader("Comparison with Reference Ranges")
    
    # Define reference ranges
    reference_ranges = {
        'Glucose': {'min': 70, 'max': 99, 'unit': 'mg/dL', 'condition': 'Fasting blood sugar'},
        'BloodPressure': {'min': 60, 'max': 80, 'unit': 'mm Hg', 'condition': 'Diastolic'},
        'BMI': {'min': 18.5, 'max': 24.9, 'unit': 'kg/m²', 'condition': 'Normal weight'},
        'Insulin': {'min': 16, 'max': 166, 'unit': 'pmol/L', 'condition': 'Fasting'}
    }
    
    # Create a dataframe for comparison
    comparison_data = []
    for feature, range_info in reference_ranges.items():
        value = user_input[feature].values[0]
        status = "Normal"
        if value < range_info['min']:
            status = "Below normal"
        elif value > range_info['max']:
            status = "Above normal"
        
        comparison_data.append({
            'Measurement': feature,
            'Patient Value': f"{value} {range_info['unit']}",
            'Normal Range': f"{range_info['min']} - {range_info['max']} {range_info['unit']}",
            'Status': status
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    st.caption("Note: These reference ranges are general guidelines. Interpretation should consider the individual's complete medical history.")

with tab3:
    st.markdown('<p class="sub-header">About the Model</p>', unsafe_allow_html=True)
    
    st.write("""
    ### Model Architecture
    This prediction system uses an Artificial Neural Network (ANN) with the following architecture:
    
    * **Input Layer**: 8 features (medical measurements)
    * **First Hidden Layer**: 16 neurons with ReLU activation
    * **Dropout Layer**: 20% dropout rate to prevent overfitting
    * **Second Hidden Layer**: 8 neurons with ReLU activation
    * **Dropout Layer**: 20% dropout rate
    * **Output Layer**: 1 neuron with sigmoid activation (outputs probability)
    
    The model was trained on the Pima Indians Diabetes dataset, which contains medical records for 768 patients.
    """)
    
    # Create tabs for additional model information
    model_tab1, model_tab2, model_tab3 = st.tabs(["Dataset", "Performance", "Limitations"])
    
    with model_tab1:
        st.subheader("About the Dataset")
        st.write("""
        The Pima Indians Diabetes dataset was collected by the National Institute of Diabetes and Digestive and Kidney Diseases.
        It contains medical records for female patients of Pima Indian heritage, aged 21 years and older.
        
        **Dataset Characteristics:**
        * 768 patient records
        * 8 medical features
        * Binary target variable (diabetes: yes/no)
        * All patients are females aged 21+
        """)
        
        # Show dataset sample if available
        try:
            diabetes_data = pd.read_csv("data/diabetes.csv")
            st.write("Sample from the dataset:")
            st.dataframe(diabetes_data.head(), use_container_width=True)
        except:
            st.info("Sample data not available for display.")
    
    with model_tab2:
        st.subheader("Model Performance")
        st.write("""
        The model's performance metrics on unseen test data:
        
        * **Accuracy**: ~77%
        * **Precision**: ~68% (correct positive predictions)
        * **Recall**: ~65% (percentage of actual positive cases identified)
        * **F1 Score**: ~67% (harmonic mean of precision and recall)
        
        These metrics indicate that the model performs better than random chance but still has limitations.
        """)
        
        # Display sample visualization if available
        if os.path.exists("visualizations/confusion_matrix.png"):
            st.image("visualizations/confusion_matrix.png", caption="Confusion Matrix", width=400)
        
        if os.path.exists("visualizations/roc_curve.png"):
            st.image("visualizations/roc_curve.png", caption="ROC Curve", width=400)
    
    with model_tab3:
        st.subheader("Limitations and Considerations")
        st.write("""
        When using this model, please consider the following limitations:
        
        1. **Population Specificity**: Trained on data from a specific population (Pima Indian females), may not generalize to all populations
        
        2. **Not a Diagnostic Tool**: This model provides a risk assessment, not a clinical diagnosis
        
        3. **Limited Features**: Only considers 8 medical factors, while diabetes risk depends on many more variables
        
        4. **Balanced Performance**: The model balances false positives and false negatives, which may not be optimal for all clinical scenarios
        
        5. **Risk Threshold**: Uses a 50% probability threshold for classification, which may need adjustment based on clinical goals
        
        Always consult with healthcare professionals for proper diagnosis and treatment decisions.
        """)

# Add a footer
st.markdown("""
---
<p style="text-align:center">
This application is for educational purposes only. Not intended for clinical use.<br>
Developed using TensorFlow and Streamlit.
</p>
""", unsafe_allow_html=True)

# Add a sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Note**: This model was trained on the Pima Indians Diabetes dataset. For actual medical diagnosis, please consult with healthcare professionals.
""")

# Add option to view raw data 
if st.sidebar.checkbox("Show Technical Details"):
    st.sidebar.subheader("Raw Input Data")
    st.sidebar.write(user_input)
    
    if model is not None:
        st.sidebar.subheader("Model Summary")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        st.sidebar.text(model_summary)