import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import mail
import profiles


# Loading XGBoost model
model = xgb.XGBClassifier()
model.load_model('model/model.xgb') 


def preprocess_input(user_input):
    columns = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 'Work_Experience', 'Spending_Score', 'Family_Size']
    
    gender_map = {'Female': 0, 'Male': 1}
    ever_married_map = {'No': 0, 'Prefer not to disclose': 1, 'Yes': 2}
    graduated_map = {'No': 0, 'Prefer not to disclose': 1, 'Yes': 2}
    profession_map = {'Artist': 0, 'Doctor': 1, 'Engineer': 2, 'Entertainment': 3, 'Executive': 4, 'Healthcare': 5, 'Homemaker': 6, 'Lawyer': 7, 'Marketing': 8, 'Others': 9}
    spending_score_map = {'Average': 0, 'High': 1, 'Low': 2}
    
    # Creating DataFrame from user input
    df = pd.DataFrame([user_input], columns=columns)

    # Apply mappings
    df['Gender'] = df['Gender'].map(gender_map)
    df['Ever_Married'] = df['Ever_Married'].map(ever_married_map)
    df['Graduated'] = df['Graduated'].map(graduated_map)
    df['Profession'] = df['Profession'].map(profession_map)
    df['Spending_Score'] = df['Spending_Score'].map(spending_score_map)
    
    scaler = joblib.load('scaler/scaler.pkl')

    # Normalizing data using the loaded scaler
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df
    


# Streamlit app
def main():
    st.title('Customer Survey')

    name = st.text_input("Enter your name:")
    user_email = st.text_input("Enter your email:")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No', 'Prefer not to disclose'])
    age = st.number_input('Age', min_value=0, max_value=100)
    graduated = st.selectbox('Graduated', ['Yes', 'No', 'Prefer not to disclose'])
    profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'])
    work_experience = st.number_input('Work Experience', min_value=0, max_value=15, value=0)
    spending_score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])
    family_size = st.number_input('Family Size', min_value=1, max_value=9, value=1)


    input = [gender,ever_married,age,graduated,profession,work_experience,spending_score,family_size]
    if st.button('submit'):
        # Preprocessing user input
        processed_input = preprocess_input(input)

        # Making prediction using the model
        prediction = model.predict(processed_input)[0]

        # Display prediction
        st.subheader('Prediction:')
        if prediction==0:
            pred = 'cluster A'
            profile = profiles.cluster_A
        elif prediction==1:
            pred = 'cluster B'
            profile = profiles.cluster_B
        elif prediction==2:
            pred = 'cluster C'
            profile = profiles.cluster_C
        else:
            pred = 'cluster D'
            profile = profiles.cluster_D
        
        st.info(f'''{pred}\n {profile}''')
        if mail.send_email(user_email, name, prediction):
            st.success("Email sent successfully!")
        else:
            st.error("Failed to send email.")






# Run the app
if __name__ == '__main__':
    main()