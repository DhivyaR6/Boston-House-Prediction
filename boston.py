# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
print("Loading the dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Display dataset information
print("Dataset Information:")
print(df.info())

# Features and target
X = df.drop(columns=["MedHouseVal"])  # Features
y = df["MedHouseVal"]  # Target

# Split the dataset into training and testing sets
print("\nSplitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (optional, improves model performance)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression model
print("\nTraining Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Train a Ridge Regression model
print("\nTraining Ridge Regression...")
ridge_model = Ridge(alpha=1.0)  # You can experiment with different alpha values
ridge_model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the models
print("\nEvaluating models...")
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

print(f"Linear Regression - MSE: {linear_mse:.2f}, R²: {linear_r2:.2f}")
print(f"Ridge Regression - MSE: {ridge_mse:.2f}, R²: {ridge_r2:.2f}")

# Visualize Actual vs Predicted prices for both models
plt.figure(figsize=(12, 6))

# Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

# Ridge Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Ridge Regression: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.tight_layout()
plt.show()

# Correlation heatmap of features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
