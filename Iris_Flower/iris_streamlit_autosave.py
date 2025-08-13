import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

# File names for saved model and label encoder
MODEL_FILE = "iris_decision_tree_model.pkl"
ENCODER_FILE = "iris_label_encoder.pkl"
DATA_FILE = "IRIS.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)
X = df.drop('species', axis=1)

# Function to train and save model
def train_and_save():
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])
    y = df['species']

    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)
    return model, label_encoder

# Check if model exists
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    model = joblib.load(MODEL_FILE)
    label_encoder = joblib.load(ENCODER_FILE)
else:
    st.warning("Model not found — training and saving new model...")
    model, label_encoder = train_and_save()
    st.success("Model trained and saved successfully!")

# --- STREAMLIT UI ---
st.title("🌸 Iris Flower Classification - Decision Tree (Auto-Save)")
st.write("This app classifies Iris flowers into **Setosa**, **Versicolor**, or **Virginica** based on measurements.")

# Input for prediction
st.subheader("🔍 Predict Iris Species")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_species = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Species: **{predicted_species}**")

# Plot the decision tree
st.subheader("🌳 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    ax=ax
)
st.pyplot(fig)
