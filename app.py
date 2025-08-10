# titanic_rf_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

df = load_data()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Train the tuned Random Forest model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [100],
        'criterion': ['gini'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid,
                               cv=5,
                               scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

model = train_model(X, y)

# Streamlit UI
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

# Convert inputs to model format
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
