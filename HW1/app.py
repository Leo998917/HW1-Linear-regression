import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Simple Linear Regression Demo")

# User input
a = st.slider("斜率 a", -10.0, 10.0, 2.0)
b = st.slider("截距 b", -5.0, 5.0, 1.0)
noise = st.slider("Noise 標準差", 0.0, 10.0, 1.0)
n_points = st.slider("資料點數", 10, 500, 100)

# Generate data
X = np.linspace(-10, 10, n_points).reshape(-1, 1)
y = a * X.flatten() + b + np.random.normal(0, noise, n_points)

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluation
r2 = model.score(X, y)

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color="red", label="Regression Line")
ax.legend()
st.pyplot(fig)

st.write(f"R² score: {r2:.3f}")
