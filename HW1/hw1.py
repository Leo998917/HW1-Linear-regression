import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("🔹 Linear Regression Demo (Streamlit)")

# 可調整參數
a = st.slider("真實斜率 a", 0.0, 5.0, 2.0)
b = st.slider("截距 b", -5.0, 5.0, 1.0)
num_points = st.slider("資料點數量", 10, 200, 50)
noise = st.slider("雜訊大小", 0.0, 5.0, 2.0)
num_outliers = st.slider("異常值數量", 0, 20, 3)

# 1. Data Preparation
X = np.linspace(0, 10, num_points).reshape(-1, 1)
y = a * X.flatten() + b + np.random.normal(0, noise, num_points)

# 加入 Outliers
outlier_indices = []
if num_outliers > 0:
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    y[outlier_indices] += np.random.normal(15, 5, num_outliers)

# 2. 建立模型並訓練
model = LinearRegression()
model.fit(X, y)

# 3. 模型參數
a_hat = model.coef_[0]
b_hat = model.intercept_

# 4. 評估
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# 顯示數值結果
st.write(f"真實參數: a={a}, b={b}")
st.write(f"模型學到的: a_hat={a_hat:.2f}, b_hat={b_hat:.2f}")
st.write(f"MSE = {mse:.2f}")

# 5. 畫圖
fig, ax = plt.subplots()

# 畫一般資料
normal_mask = np.ones(num_points, dtype=bool)
normal_mask[outlier_indices] = False
ax.scatter(X[normal_mask], y[normal_mask], color="blue", label="Data")

# 畫 Outliers（紅色圈起來）
if len(outlier_indices) > 0:
    ax.scatter(X[outlier_indices], y[outlier_indices], 
               facecolors="none", edgecolors="red", s=120, linewidths=2, label="Outliers")

# 畫迴歸線
ax.plot(X, model.predict(X), color="green", label="Fitted line")
ax.legend()

st.pyplot(fig)



