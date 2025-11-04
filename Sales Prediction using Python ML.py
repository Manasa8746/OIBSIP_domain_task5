import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\manas\Downloads\Advertising.csv")

print("Dataset Preview:\n", data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:\n", data.isnull().sum())
print("\nStatistical Summary:\n", data.describe())

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance:")
print(f"R² Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

custom_input = pd.DataFrame({
    'TV': [150],
    'Radio': [20],
    'Newspaper': [15]
})

custom_prediction = model.predict(custom_input)
print("\nCustom Sales Prediction:")
print(f"Predicted Sales for TV=150, Radio=20, Newspaper=15 → {custom_prediction[0]:.2f} units")
