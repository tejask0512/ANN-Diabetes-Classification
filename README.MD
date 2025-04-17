# Diabetes Prediction System

This project implements a diabetes prediction system using an Artificial Neural Network (ANN) and provides a user-friendly web interface with Streamlit. The model analyzes medical factors to predict the likelihood of diabetes.

## Project Overview

This application uses machine learning to predict diabetes risk based on several health metrics. It features:

- An ANN model trained on the Pima Indians Diabetes dataset
- Interactive web interface built with Streamlit
- Data visualization of model performance
- TensorBoard integration for training monitoring

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system
```

2. Install required packages:
```
pip install -r requirements.txt
```

### Usage

1. Train the model (if not already trained):
```
python model_training.py
```

2. Run the Streamlit application:
```
streamlit run app.py
```

3. Open your web browser and go to `http://localhost:8501` to use the application.

## Project Structure

- `data/diabetes.csv`: The dataset used for training
- `model_training.py`: Script for preprocessing data and training the model
- `app.py`: Streamlit web application
- `models/`: Directory containing the trained model and scaler
- `visualizations/`: Directory containing generated plots and visualizations
- `logs/`: TensorBoard logs for training monitoring

## Model Architecture