import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

st.title("📈 線性迴歸互動版 Demo (CRISP-DM)")

# Sidebar 讓使用者輸入參數
a = st.sidebar.slider("斜率 a", -5.0, 5.0, 2.0, 0.1)
b = st.sidebar.slider("截距 b", -10.0, 10.0, 1.0, 0.1)
num_points = st.sidebar.slider("資料點數量", 10, 200, 50, 10)
noise = st.sidebar.slider("雜訊大小", 0.0, 10.0, 2.0, 0.1)

# 產生資料
X = np.linspace(0, 10, num_points).reshape(-1, 1)
y = a * X.flatten() + b + np.random.normal(0, noise, num_points)

# 建立並訓練模型
model = LinearRegression()
model.fit(X, y)

# 模型參數
a_hat = model.coef_[0]
b_hat = model.intercept_
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# 顯示結果
st.write(f"✅ 真實參數: a={a}, b={b}")
st.write(f"🤖 模型學到的斜率: {a_hat:.2f}")
st.write(f"🤖 模型學到的截距: {b_hat:.2f}")
st.write(f"📊 MSE: {mse:.2f}")

# 畫圖
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data with noise")
ax.plot(X, y_pred, color="red", label="Fitted line")
ax.legend()
st.pyplot(fig)
