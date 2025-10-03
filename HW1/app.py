import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# NOTE: Removed Matplotlib Chinese Font Configuration
#       to prevent square box display issues. Plot labels are now in English.
# ------------------------------------------------

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(layout="wide", page_title="Linear Regression Demo")
st.title("線性迴歸視覺化 (含離群值分析)")
st.markdown("使用側邊欄調整資料生成參數 (斜率、截距、雜訊水準和資料點數量)。")

# ------------------------------
# Sidebar - User Inputs
# ------------------------------
st.sidebar.header("參數設定 (Parameters)")
a = st.sidebar.slider("斜率 (Slope, a)", -10.0, 10.0, 2.0, 0.1)
b = st.sidebar.slider("截距 (Intercept, b)", -50.0, 50.0, 5.0, 1.0)
noise = st.sidebar.slider("雜訊水準 (Noise level)", 0.0, 20.0, 5.0, 0.5)
n_points = st.sidebar.slider("資料點數量 (N points)", 10, 500, 100, 10)

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
# Residuals and DataFrame (for outlier detection)
# ------------------------------
residuals = np.abs(y - y_pred)
df = pd.DataFrame({
    "X": X.flatten(),
    "y": y.flatten(),
    "y_pred": y_pred.flatten(),
    "residual": residuals.flatten()
})
# 根據殘差大小排序
df_sorted = df.sort_values(by="residual", ascending=False)
top_outliers = df_sorted.head(5) # 取得前 5 個最大的離群值

# ------------------------------
# Plot Data & Regression Line (HIGHLIGHTING OUTLIERS)
# ------------------------------
st.subheader("資料點與迴歸線擬合結果")

fig, ax = plt.subplots(figsize=(10, 5))

# 1. 繪製所有資料點 (作為背景)
ax.scatter(df["X"], df["y"], label="All Data Points", alpha=0.5, color='#1f77b4')

# 2. 突出顯示殘差最大的前 5 個離群值
ax.scatter(
    top_outliers["X"], 
    top_outliers["y"], 
    color="orange", 
    s=120, # 放大標記尺寸
    marker="D", # 使用菱形標記
    edgecolors='black',
    linewidths=1.5,
    label=f"Top {len(top_outliers)} Outliers" # 標籤改為英文
)

# 3. 繪製真實線和擬合線
#ax.plot(X, y_true, color="green", linestyle='--', label=f"True Line (y = {a:.2f}x + {b:.2f})", alpha=0.7)
ax.plot(X, y_pred, color="red", label="Fitted Regression Line", linewidth=2) # 標籤改為英文

ax.set_title("Linear Regression Fit and Outliers") # 標題改為英文
ax.set_xlabel("X Value") # 軸標籤改為英文
ax.set_ylabel("Y Value") # 軸標籤改為英文
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
st.pyplot(fig)

# ------------------------------
# Show Model Coefficients (修復 TypeError)
# ------------------------------
st.subheader("模型係數結果")

# 修正: 確保從 NumPy 陣列中提取單一浮點數
estimated_slope = model.coef_[0][0].item()
estimated_intercept = model.intercept_[0].item()

st.info(f"""
- **估計斜率 (a):** `{estimated_slope:.3f}`
- **估計截距 (b):** `{estimated_intercept:.3f}`
""")
st.markdown(f"**擬合方程式:** $y = {estimated_slope:.3f}x + {estimated_intercept:.3f}$")

# ------------------------------
# Show Top 5 Outliers
# ------------------------------
st.subheader("前 5 個離群值 (最大殘差)")
st.caption("這些是距離擬合迴歸線最遠的資料點，它們在圖上以菱形橙色標記突出顯示。")
st.dataframe(top_outliers, use_container_width=True)
