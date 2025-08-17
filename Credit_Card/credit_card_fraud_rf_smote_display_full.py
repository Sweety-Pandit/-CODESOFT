# credit_card_fraud_rf_smote_display_full.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("C:\Project 3\creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

# Drop rows with NaN in the target variable
nan_rows = y.isna()
X = X[~nan_rows]
y = y[~nan_rows]


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())

# 4. Random Forest pipeline
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", rf)
])

# 5. Train
model.fit(X_train_res, y_train_res)

# 6. Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Feature Importance
importances = model.named_steps["clf"].feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.title("Random Forest (SMOTE) - Feature Importances", fontsize=14)
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# 8. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest (SMOTE)")
plt.legend(loc="lower right")
plt.show()

# 9. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap_score:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest (SMOTE)")
plt.legend(loc="lower left")
plt.show()