import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the trained model
model = tf.keras.models.load_model('modell.h5')

#load the encoders and scaler
with open('onehot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Bank Customer Churn Prediction")

# Input fields for CreditScore	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Geography
credit_score = st.number_input("Credit Score")
gender = st.selectbox("Gender",label_encoder.classes_)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 1)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")
geography = st.selectbox("Geography", one_hot_encoder.categories_[0 ])

# Prepare the input data
input_data = {
    "CreditScore": [credit_score],
    "Gender": [label_encoder.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary] 
}
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
# combine the new one-hot encoded columns with the original dataframe
input_df = pd.DataFrame(input_data)
input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
# scaling the data
input_scaled = scaler.transform(input_df)
# prediction
prediction = model.predict(input_scaled)

prediction_proba = prediction[0][0]
if prediction_proba > 0.5:
    st.write(f"The customer is likely to leave the bank with a probability of {prediction_proba:.2f}")  
else:
    st.write(f"The customer is likely to stay with the bank with a probability of {1 - prediction_proba:.2f}")  