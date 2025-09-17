#app.py
# This script serves as a Streamlit app for predicting customer churn using a pre-trained neural network
# It allows users to input customer data and receive a churn prediction.
import streamlit as st
import pandas as pd
import time
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

st.title("Churn Prediciton App")
with st.form(key='get_input_form'):
   #1
   credit_score = st.number_input("Credit Score", placeholder="Enter credit score")
   #2
   geography = st.selectbox(
      "Geography:",("France", "Spain", "Germany"))
   #3
   gender = st.selectbox(
      'Gender:',("Male","Female"))
   #4
   age = st.number_input("Age:", min_value=18, max_value=100, placeholder="Enter age")
   #5
   tenure = st.number_input("Tenure:", min_value=0, max_value=10, placeholder="Enter tenure")
   #6 
   balance = st.number_input("Balance:", min_value=0, placeholder="Enter balance")
   #7
   num_of_products = st.number_input("Number of Products:", min_value=1, max_value=4, placeholder="Enter number of products")
   #8
   has_cred_card = st.selectbox("Has credit card:", ("1","0"))
   #9
   is_active_member = st.selectbox("Is active member:", ("1","0"))
   #10
   estimated_salary = st.number_input("Estimated Salary:", min_value=0, placeholder="Enter estimated salary")

   button = st.form_submit_button(label='Submit')

   with open("geo_encoder.pkl", "rb") as f:
    geo_encoder = pickle.load(f)

   with open("gender_encoder.pkl", "rb") as f:
      gender_encoder = pickle.load(f)

   geo_encoded = geo_encoder.transform([geography])[0]
   gender_encoded = gender_encoder.transform([gender])[0]

   input_data = pd.DataFrame([{
      "credit_score": credit_score,
      "geography": geo_encoded,
      "gender": gender_encoded,
      "age": age,
      "tenure": tenure,
      "balance": balance,
      "num_of_products": num_of_products,
      "has_cred_card": has_cred_card,
      "is_active_member": is_active_member,
      "estimated_salary": estimated_salary
   }])
   model_instance = tf.keras.models.load_model("trained_ann_model.keras")
   prediction = model_instance.predict(input_data)
   # prediction = "Churn" if prediction[0][0] > 0.5
   # st.write("Prediction:", prediction)
   if button:
      'Hang in there...'
      # Add a placeholder
      latest_iteration = st.empty()
      bar = st.progress(0)
      
      for i in range(100):
      # Update the progress bar with each iteration.
         latest_iteration.text(f'Iteration {i+1}')
         bar.progress(i + 1)
         time.sleep(0.02)
      st.success("Done!")
      'Prediction: ' + ('Churn' if prediction[0][0] > 0.5 else 'No Churn') 