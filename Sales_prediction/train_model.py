# ================================
# Train and Save Linear Regression Model
# ================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load Dataset
data = pd.read_csv(r"C:\Projrct 4\advertising.csv")
print("Dataset Preview:\n", data.head())

# Step 2: Features & Target
X = data.drop("Sales", axis=1)  
y = data["Sales"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Trained Successfully!")
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 6: Save Model
joblib.dump(model, "linear_regression_sales.pkl")
print("\nModel saved as 'linear_regression_sales.pkl'")
