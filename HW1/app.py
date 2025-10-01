import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(layout="wide", page_title="Linear Regression Demo")
st.title("Linear Regression Visualization")
st.markdown("Use the sidebar to adjust data generation parameters (Slope, Intercept, Noise, and Data Points).")

# ------------------------------
# Sidebar - User Inputs
# ------------------------------
st.sidebar.header("Parameters")
a = st.sidebar.slider("Slope (a)", -10.0, 10.0, 2.0, 0.1)
b = st.sidebar.slider("Intercept (b)", -50.0, 50.0, 5.0, 1.0)
noise = st.sidebar.slider("Noise level (Noise)", 0.0, 20.0, 5.0, 0.5)
n_points = st.sidebar.slider("Number of data points", 10, 500, 100, 10)

# ------------------------------
# Generate Data
# ------------------------------
np.random.seed(42)
# X is 2D for scikit-learn (n_points, 1)
X = np.linspace(0, 10, n_points).reshape(-1, 1)
y_true = a * X + b
# y is also 2D (n_points, 1)
y = y_true + np.random.normal(0, noise, size=y_true.shape)

# ------------------------------
# Fit Model
# ------------------------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ------------------------------
# Residuals (for outlier detection)
# ------------------------------
residuals = np.abs(y - y_pred)
df = pd.DataFrame({
    "X": X.flatten(),
    "y": y.flatten(),
    "y_pred": y_pred.flatten(),
    "residual": residuals.flatten()
})
df_sorted = df.sort_values(by="residual", ascending=False)

# ------------------------------
# Plot Data & Regression Line
# ------------------------------
st.subheader("Data Scatter Plot and Fitted Line")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, y, label="Generated Data (y)", alpha=0.6, color='#1f77b4')
ax.plot(X, y_true, color="green", linestyle='--', label=f"True Line (y = {a}x + {b})", alpha=0.7)
ax.plot(X, y_pred, color="red", label="Fitted Regression Line", linewidth=2)

ax.set_title("Linear Regression Fit")
ax.set_xlabel("X (Feature)")
ax.set_ylabel("y (Target)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
st.pyplot(fig)

# ------------------------------
# Show Model Coefficients
# ------------------------------
st.subheader("Model Coefficients")

# 修正: 由於 y 是 2D 陣列，model.coef_ 是 [[a]] 格式，需使用 [0][0] 存取。
# 截距 model.intercept_ 是 [b] 格式，需使用 [0] 存取。
# 使用 .item() 確保它是一個標準 Python float。
estimated_slope = model.coef_[0][0].item()
estimated_intercept = model.intercept_[0].item()

st.info(f"""
- **Estimated Slope (a):** `{estimated_slope:.3f}`
- **Estimated Intercept (b):** `{estimated_intercept:.3f}`
""")
st.markdown(f"**Fitted Equation:** $y = {estimated_slope:.3f}x + {estimated_intercept:.3f}$")

# ------------------------------
# Show Top 5 Outliers
# ------------------------------
st.subheader("Top 5 Outliers (Largest Residuals)")
st.caption("The data points farthest from the fitted regression line.")
st.dataframe(df_sorted.head(5), use_container_width=True)
