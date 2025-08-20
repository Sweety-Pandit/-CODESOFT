import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load Saved Model
model = joblib.load("linear_regression_sales.pkl")

# Streamlit Page Config
st.set_page_config(page_title="📊 Sales Prediction App", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        .big-font {
            font-size:22px !important;
            font-weight: bold;
        }
        .medium-font {
            font-size:18px !important;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            font-size:14px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>📈 Sales Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Predict future product sales based on advertising spend using Machine Learning</p>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔧 Options")
mode = st.sidebar.radio("Choose Mode:", ["📂 Upload Dataset", "🎛 Manual Input", "ℹ️ About Project"])

# Mode 1: Upload Dataset
if mode == "📂 Upload Dataset":
    st.markdown("<p class='big-font'>📂 Upload Your Advertising Dataset</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### 👀 Preview of Uploaded Data")
        st.dataframe(data.head())

        if all(col in data.columns for col in ["TV", "Radio", "Newspaper"]):
            # Predictions
            predictions = model.predict(data[["TV", "Radio", "Newspaper"]])
            data["Predicted_Sales"] = predictions

            st.write("### 🔮 Predictions")
            st.dataframe(data)

            # Plot Actual vs Predicted (if Sales exists)
            if "Sales" in data.columns:
                st.write("### 📊 Actual vs Predicted Sales")
                plt.figure(figsize=(7,5))
                plt.scatter(data["Sales"], data["Predicted_Sales"], alpha=0.7, color="green")
                plt.xlabel("Actual Sales")
                plt.ylabel("Predicted Sales")
                plt.title("Actual vs Predicted Sales")
                st.pyplot(plt.gcf())
        else:
            st.error("❌ CSV must have columns: TV, Radio, Newspaper")

# Mode 2: Manual Input
elif mode == "🎛 Manual Input":
    st.markdown("<p class='big-font'>🎛 Enter Advertising Spend Manually</p>", unsafe_allow_html=True)

    tv = st.slider("💰 **TV Advertising Spend**", min_value=0, max_value=300, value=150, step=5)
    radio = st.slider("📻 **Radio Advertising Spend**", min_value=0, max_value=50, value=25, step=1)
    newspaper = st.slider("📰 **Newspaper Advertising Spend**", min_value=0, max_value=120, value=30, step=2)

    sample = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])

    if st.button("🔮 Predict Sales"):
        prediction = model.predict(sample)
        st.success(f"✅ Predicted Sales: **{prediction[0]:.2f} units**")

        # Show bar chart of input vs prediction
        st.write("### 📊 Advertising Spend Distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["TV", "Radio", "Newspaper"], [tv, radio, newspaper], color=["#3498DB", "#E67E22", "#2ECC71"])
        ax.set_ylabel("Amount Spent")
        ax.set_title("Advertising Spend")
        st.pyplot(fig)

# Mode 3: About Project
elif mode == "ℹ️ About Project":
    st.markdown("<p class='big-font'>ℹ️ About This Project</p>", unsafe_allow_html=True)
    st.markdown("""
    This project demonstrates how **Machine Learning** can be applied in the field of marketing and business 
    to predict **product sales** based on advertising expenditure across different channels.

    - 📺 **TV Advertising**: Budget spent on TV commercials  
    - 📻 **Radio Advertising**: Budget allocated for radio ads  
    - 📰 **Newspaper Advertising**: Budget for print media  

    Using a **Linear Regression model**, businesses can forecast how much sales they might generate by 
    adjusting their advertising strategy. This enables companies to **optimize ad spending**, **maximize revenue**, 
    and **make data-driven decisions**.

    👉 You can either **upload your dataset** or use **manual input sliders** to test predictions.
    """)

# Footer
st.markdown("---")
st.markdown("<p class='footer'>💻 Built with Streamlit | Powered by Linear Regression Model</p>", unsafe_allow_html=True)

