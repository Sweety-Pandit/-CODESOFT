import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ----------------------------
# Use online background image
BG_IMAGE_URL = "https://t4.ftcdn.net/jpg/14/67/13/13/360_F_1467131354_kQipCM1ORCqZHdxb1BV9mO92blFj4Tjt.jpg"  # Floral background

def set_background_url(url):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{url}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #1a1a1a !important; /* Dark text */
    }}
    /* Transparent/blur main container */
    .stApp > header, .block-container {{
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(6px);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }}
    /* Adjust heading styles */
    h1, h2, h3, h4, h5, h6, p, label {{
        color: #222 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.15);
    }}
    /* Button styling */
    .stButton button {{
        background-color: #ff69b4;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease-in-out;
    }}
    .stButton button:hover {{
        background-color: #d81b60;
        transform: scale(1.05);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------
# Page config
st.set_page_config(page_title="Iris Classifier 🌸", page_icon="🌸", layout="wide")
set_background_url(BG_IMAGE_URL)

# ----------------------------
# Load dataset
DATA_FILE = "IRIS.csv"
df = pd.read_csv(DATA_FILE)
X = df.drop('species', axis=1)

# ----------------------------
# Train or load model
MODEL_FILE = "iris_decision_tree_model.pkl"
ENCODER_FILE = "iris_label_encoder.pkl"

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

try:
    model = joblib.load(MODEL_FILE)
    label_encoder = joblib.load(ENCODER_FILE)
except:
    st.warning("⚠️ Model not found — training a new one...")
    model, label_encoder = train_and_save()
    st.success("✅ Model trained and saved successfully!")

# ----------------------------
# Title
st.markdown(
    "<h1 style='text-align: center;'>🌸 Iris Flower Classification 🌸</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Predict the species of an Iris flower based on its measurements.</p>",
    unsafe_allow_html=True
)

# ----------------------------
# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Enter Flower Measurements")
    sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("🌿 Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("🌼 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("🌼 Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

    if st.button("🌟 Predict Species", use_container_width=True):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        st.markdown(
            f"""
            <div style='background-color: rgba(255, 192, 203, 0.85); 
                        padding: 15px; border-radius: 10px; text-align: center;
                        box-shadow: 0 3px 8px rgba(0,0,0,0.2);'>
                <h3 style='color: #b30059;'>✅ Predicted Species:</h3>
                <h2 style='color: #4a148c;'><b>{predicted_species.capitalize()}</b></h2>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.subheader("🌳 Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    fig.patch.set_alpha(0)  # Transparent plot background
    ax.patch.set_alpha(0)
    st.pyplot(fig, transparent=True)
