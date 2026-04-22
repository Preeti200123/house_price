# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load dataset
data = pd.read_csv("housing (1).csv")

print("Dataset:\n", data.head())

# Step 3: Define features (X) and target (y)
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = data['price']

# Step 4: Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Compare actual vs predicted
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nComparison:\n", comparison)

# Step 9: Plot (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 10: Predict new house price
new_house = [[2000, 3, 2, 2, 2]]
predicted_price = model.predict(new_house)

print("\nPredicted price for new house:", predicted_price[0])
