import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import pickle

# Load the CatBoost model
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("CatBoost Classifier Deployment")

# Get user input
st.subheader("Enter input data:")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Create a DataFrame from user input
input_data = pd.DataFrame({'feature1': [feature1], 'feature2': [feature2], 'feature3': [feature3]})

# Make prediction
prediction = model.predict(input_data)[0]
predicted_class = model.classes_[prediction]

# Display the prediction
st.subheader("Prediction:")
st.write(f"The predicted class is: {predicted_class}")
