# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('model/model.xgb') 


# Load the pre-trained LabelEncoder
with open('label_encoder/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def preprocess_input(user_input):
    # Create an empty DataFrame with the expected column names
    columns = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 'Work_Experience', 'Spending_Score', 'Family_Size']
    # input_df = pd.DataFrame(columns=columns)

    # # Append user input to the DataFrame
    # input_df = input_df.append(user_input, ignore_index=True)

    
    input_df = pd.DataFrame(user_input, columns)
    st.dataframe(input_df)
    # input_df = pd.DataFrame([user_input])
    columns_to_encode = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']

    for column in columns_to_encode:
        input_df[column] = label_encoder.transform(input_df[column])
    return input_df

# Define the Streamlit app
def main():
    st.title('Customer Survey')

    # Define input fields for user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No', 'Prefer not to disclose'])
    age = st.number_input('Age', min_value=0, max_value=100)
    graduated = st.selectbox('Graduated', ['Yes', 'No', 'Prefer not to disclose'])
    profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'])
    work_experience = st.number_input('Work Experience', min_value=0, max_value=15, value=0)
    spending_score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])
    family_size = st.number_input('Family Size', min_value=1, max_value=9, value=1)

    # Create a dictionary from user inputs
    # user_input = {
    #     'Gender': gender,
    #     'Ever_Married': ever_married,
    #     'Age': age,
    #     'Graduated': graduated,
    #     'Profession': profession,
    #     'Work_Experience': work_experience,
    #     'Spending_Score': spending_score,
    #     'Family_Size': family_size
    # }
    input = [gender,ever_married,age,graduated,profession,work_experience,spending_score,family_size]

    # Preprocess the user input
    processed_input = preprocess_input(input)
    st.dataframe(processed_input)
    # Convert the processed input into a DataFrame
    input_df = pd.DataFrame([processed_input])

    # Display the input data
    st.subheader('User Input:')
    st.write(input_df)

    # Make predictions using the model
    prediction = model.predict(input_df)[0]

    # Display the prediction
    st.subheader('Prediction:')
    st.write(prediction)

# Run the app
if __name__ == '__main__':
    main()