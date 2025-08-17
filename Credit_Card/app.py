import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =====================
# Load Model + Scaler
# =====================
model = joblib.load("fraud_model.pkl")  # or model.pkl (adjust as per your file)
scaler = joblib.load("scaler.pkl")

# =====================
# Page Config
# =====================
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# =====================
# Custom CSS for Bigger Fonts
# =====================
st.markdown(
    """
    <style>
    h1, h2, h3, h4 {
        font-size: 28px !important;
        font-weight: bold;
    }
    p, label, .stMarkdown {
        font-size: 20px !important;
    }
    .stButton button {
        font-size: 20px !important;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stDataFrame {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# Sidebar Navigation
# =====================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 About", "📂 Upload Data", "✍️ Manual Input"])

# =====================
# About Page
# =====================
if page == "🏠 About":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown(
        """
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
        <h2 style="color:#2c3e50;">ℹ️ About this Project</h2>
        <p style="color:#34495e;font-size:20px;">
        This system uses a trained <b>Machine Learning model</b> to detect fraudulent credit card transactions.  
        You can either <b>upload a CSV file</b> of transactions or <b>manually input a transaction</b> to check fraud probability.  
        The model outputs both <b>predictions</b> and <b>confidence visualizations</b> with green, yellow, and red indicators.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image("https://cdn-icons-png.flaticon.com/512/744/744502.png", width=180)

# =====================
# Upload Data Page
# =====================
elif page == "📂 Upload Data":
    st.header("📂 Upload Transactions Data")

    uploaded_file = st.file_uploader("📤 Upload your transaction CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.subheader("📑 First 5 Rows of Data")
        st.dataframe(df.head(), use_container_width=True)

        # Prepare features
        X = df.drop(columns=["Class"], errors="ignore")
        y_true = df["Class"] if "Class" in df.columns else None

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        df["Prediction"] = predictions

        # Predictions
        st.subheader("🔍 Predictions on Uploaded Data")
        st.dataframe(df.head(20), use_container_width=True)

        # Distribution Charts
        st.subheader("📊 Prediction Distribution")
        pred_counts = pd.Series(predictions).value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            pred_counts.plot(kind="bar", ax=ax, color=["green", "red"])
            ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
            ax.set_ylabel("Count")
            ax.set_title("Prediction Distribution (Bar)")
            st.pyplot(fig)
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie(pred_counts, labels=["Non-Fraud (0)", "Fraud (1)"],
                    autopct="%1.1f%%", colors=["green", "red"],
                    startangle=90, textprops={'fontsize': 14})
            ax2.set_title("Prediction Distribution (Pie)")
            st.pyplot(fig2)

        # Fraud Check
        if y_true is not None:
            st.subheader("🔍 Fraud Check (Actual Fraud Cases)")
            frauds = df[df["Class"] == 1].copy()
            if not frauds.empty:
                st.write("Showing first fraud transactions with predictions:")
                st.dataframe(frauds[["Amount", "Class", "Prediction"]].head(20))
            else:
                st.write("⚠️ No fraud cases found in uploaded dataset!")

            # Confusion Matrix
            st.subheader("📉 Confusion Matrix (Actual vs Predicted)")
            cm = confusion_matrix(y_true, predictions)
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                        xticklabels=["Non-Fraud (0)", "Fraud (1)"],
                        yticklabels=["Non-Fraud (0)", "Fraud (1)"], ax=ax3)
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")
            st.pyplot(fig3)

            # Evaluation Metrics
            st.subheader("📊 Model Evaluation Metrics")
            acc = accuracy_score(y_true, predictions)
            prec = precision_score(y_true, predictions)
            rec = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            st.write(f"✅ Accuracy: {acc:.4f}")
            st.write(f"🎯 Precision: {prec:.4f}")
            st.write(f"🔁 Recall: {rec:.4f}")
            st.write(f"📊 F1-Score: {f1:.4f}")

        # Download predictions
        st.download_button(
            label="💾 Download Predictions as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

# =====================
# Manual Input Page
# =====================
elif page == "✍️ Manual Input":
    st.header("✍️ Manual Transaction Entry")

    with st.form("manual_input_form"):
        st.markdown("Enter transaction details below:", unsafe_allow_html=True)
        amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=0.01)

        feature_inputs = []
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                val = st.number_input(f"V{i}", step=0.01)
                feature_inputs.append(val)

        submitted = st.form_submit_button("🚀 Predict Fraud")

    if submitted:
        features = np.array([feature_inputs + [amount]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.markdown("<h2 style='color:red;'>🚨 ALERT: Fraudulent Transaction Detected!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ Genuine Transaction</h2>", unsafe_allow_html=True)

        # Confidence Charts
        st.subheader("📊 Prediction Confidence")
        labels = ["Genuine (0)", "Fraud (1)"]

        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(labels, prob, color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability", fontsize=14)
        ax.set_title("Confidence Levels", fontsize=16)
        for i, v in enumerate(prob):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=12)
        st.pyplot(fig)

        # Pie Chart
        fig2, ax2 = plt.subplots()
        ax2.pie(prob, labels=labels, autopct="%1.2f%%", colors=["green", "red"],
                startangle=90, textprops={'fontsize': 14})
        ax2.set_title("Fraud vs Genuine Probability", fontsize=16)
        st.pyplot(fig2)

        # Gauge Meter
        fraud_prob = prob[1] * 100
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.axis("off")
        ax3.add_patch(plt.Circle((0, 0), 1, color="#eee", zorder=1))
        ax3.add_patch(plt.Circle((0, 0), 0.7, color="white", zorder=2))
        angle = (fraud_prob / 100) * 180
        x = np.cos(np.radians(180 - angle))
        y = np.sin(np.radians(180 - angle))
        ax3.plot([0, x], [0, y], color="red", lw=3, zorder=3)
        color = "green" if fraud_prob < 30 else "orange" if fraud_prob < 70 else "red"
        ax3.text(0, -0.2, f"Fraud Risk: {fraud_prob:.2f}%", ha="center", fontsize=16, fontweight="bold", color=color)
        st.pyplot(fig3)














