import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("titanic_rf_model.pkl")

st.title("🚢 Titanic Survival Prediction - Random Forest")
st.write("Enter passenger details to predict survival:")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
sex_male = st.selectbox("Sex", ["Female", "Male"])
embarked_Q = st.selectbox("Embarked from Q port?", ["No", "Yes"])
embarked_S = st.selectbox("Embarked from S port?", ["No", "Yes"])

# Convert to model format
sex_male = 1 if sex_male == "Male" else 0
embarked_Q = 1 if embarked_Q == "Yes" else 0
embarked_S = 1 if embarked_S == "Yes" else 0

input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🎉 Passenger likely SURVIVED!")
    else:
        st.error("💀 Passenger likely DID NOT survive.")
