import streamlit as st
import numpy as np
import joblib 

# ----------------------------
# Page Config
st.set_page_config(page_title="Titanic Survival Predictor 🚢", page_icon="⛴", layout="centered")

# ----------------------------
# Background and custom styling
def set_custom_style(bg_url):
    st.markdown(
        f"""
        <style>
        /* Background */
        .stApp {{
            background: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Glowing title */
        @keyframes glow {{
            0% {{ text-shadow: 0 0 5px yellow, 0 0 10px gold, 0 0 20px orange, 0 0 30px red; }}
            50% {{ text-shadow: 0 0 20px yellow, 0 0 30px gold, 0 0 40px orange, 0 0 50px red; }}
            100% {{ text-shadow: 0 0 5px yellow, 0 0 10px gold, 0 0 20px orange, 0 0 30px red; }}
        }}
        .glow-text {{
            color: yellow;
            font-size: 40px;
            animation: glow 1.5s ease-in-out infinite alternate;
            text-align: center;
            font-weight: bold;
        }}

        /* Input labels */
        label, .stSlider, .stNumberInput, .stSelectbox label {{
            color: white !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_custom_style("https://t3.ftcdn.net/jpg/11/66/99/48/360_F_1166994876_rVSjhRNiNXhN6N2vztmYKXD68NnHbggn.jpg")

# ----------------------------
# Load trained model
model = joblib.load("titanic_rf_model.pkl")

# ----------------------------
# Sidebar Info
st.sidebar.markdown("## 📊 Model Information")
st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write("**Accuracy:** 81.2%")
st.sidebar.write("**Dataset:** Titanic Passenger Data (Kaggle)")

st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ About")
st.sidebar.write(
    """
    This app predicts whether a Titanic passenger would have survived
    based on personal details and travel information.
    """
)
st.sidebar.markdown("**Created by:** Sweety Pandit")

# ----------------------------
# Title
st.markdown('<p class="glow-text">🚢 Titanic Survival Prediction</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size:18px;'>Enter passenger details to see if they might have survived.</p>", unsafe_allow_html=True)

# ----------------------------
# Input form
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class 🎟", [1, 2, 3])
    age = st.slider("Age 🎂", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses 👨‍👩‍👧", 0, 10, 0)
    parch = st.number_input("Parents/Children 👨‍👩‍👦", 0, 10, 0)

with col2:
    fare = st.number_input("Fare 💵", 0.0, 600.0, 30.0)
    sex_male = st.selectbox("Sex ⚧", ["Female", "Male"])
    embarked_Q = st.selectbox("Embarked from Q port? 🛳", ["No", "Yes"])
    embarked_S = st.selectbox("Embarked from S port? 🛳", ["No", "Yes"])

# ----------------------------
# Convert to model input format
sex_male = 1 if sex_male == "Male" else 0
embarked_Q = 1 if embarked_Q == "Yes" else 0
embarked_S = 1 if embarked_S == "Yes" else 0

input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])

# ----------------------------
# Prediction button
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.markdown("<h3 style='color: limegreen;'>✅ Passenger likely SURVIVED! 🎉</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: red;'>❌ Passenger likely DID NOT survive 💀</h3>", unsafe_allow_html=True)



