import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv(".\IRIS.csv")

# Encode species
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Features & target
X = df.drop('species', axis=1)
y = df['species']

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_split=4,
    random_state=42
)
model.fit(X, y)

# Save model and label encoder
joblib.dump(model, "iris_decision_tree_model.pkl")
joblib.dump(label_encoder, "iris_label_encoder.pkl")

print("✅ Model and label encoder saved successfully!")
