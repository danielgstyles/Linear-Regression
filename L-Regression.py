# --------------------------------------------
# Scatter Plot, Line of Best Fit & Prediction
# --------------------------------------------

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Data: x = inputs, y = outputs
x = np.array([2, 4, 6, 7, 8, 10, 12, 14, 16])
y = np.array([1, 3, 5, 8, 7, 9, 11, 13, 15])

# 2) Train a Linear Regression model
X = x.reshape(-1, 1)   # reshape into 2D for sklearn
model = LinearRegression()
model.fit(X, y)

# Get slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_
print(f"Line of best fit: y = {m:.4f}x + {b:.4f}")

# 3) Plot scatter and regression line
plt.scatter(x, y, color="blue", label="Data points")

# Create smooth x range for drawing the line
x_range = np.linspace(x.min(), x.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

plt.plot(x_range, y_range, color="red", label="Line of best fit")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot with Line of Best Fit")
plt.legend()
plt.show()   # Show chart before predicting

# 4) Predict y when x = 9
x_value = 9
y_pred = model.predict(np.array([[x_value]]))[0]
print(f"Predicted y for x={x_value} is: {y_pred:.4f}")

# --------------------------------------------
# ✨ Extension: Plot prediction point
# --------------------------------------------
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x_range, y_range, color="red", label="Line of best fit")

# Add predicted point as a green X
plt.scatter(x_value, y_pred, color="green", s=100, marker="X", label="Prediction (x=9)")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot with Prediction Point")
plt.legend()
plt.show()
