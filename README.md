HW1: Simple Linear Regression with CRISP-DM
1. Business Understanding

Prompt:

我們想要建立一個簡單的線性迴歸模型，模擬方程式 $y = ax + b + \epsilon$，其中 $\epsilon$ 為隨機雜訊，並讓使用者可以調整斜率 a、截距 b、資料點數量以及 noise 大小，觀察迴歸結果。

>過程:

線性迴歸是最基礎的監督式學習方法之一，適合作為 HW1 的入門範例。

將透過一個互動式網頁 (Streamlit) 讓使用者輸入參數，實際觀察模型如何擬合不同情境的資料。

2. Data Understanding

Prompt:

產生多少筆資料點？noise 要多大才會影響模型的準確度？

>過程:

使用 numpy 產生一維輸入資料 X。

目標值 y 根據公式 $y = ax + b + \epsilon$ 生成。

noise 小 → 資料接近直線；noise 大 → 資料分散，模型準確度下降。

資料範例圖 (noise = 1)

[在這裡放上 scatter plot 截圖]

3. Data Preparation

Prompt:

如何把資料整理成 sklearn 可以使用的格式？

>過程:

X reshape 成 (n,1)，因為 sklearn 的 LinearRegression 需要二維輸入。

y 保持一維陣列。

每次根據使用者參數動態生成資料。

4. Modeling

Prompt:

使用什麼模型來擬合？

>過程:

使用 LinearRegression()。

呼叫 fit(X, y) 進行訓練。

得到模型參數 coef_ (斜率)、intercept_ (截距)。

模型視覺化 (含迴歸直線)

[在這裡放上 regression line 截圖]

5. Evaluation

Prompt:

怎麼知道模型學得好不好？

>過程:

使用決定係數 R² 作為評估指標。

觀察實際資料與回歸直線的貼合程度。

範例輸出:

R² score: 0.978

6. Deployment

Prompt:

如何讓使用者能夠互動？

過程:

使用 Streamlit 建立一個簡單的 web app。

提供 slider 讓使用者控制：

斜率 a

截距 b

noise 大小

資料點數量

動態繪製 scatter plot 與迴歸直線。

程式碼 (Streamlit)
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

Deployment

安裝需求

pip install streamlit scikit-learn matplotlib numpy


執行

streamlit run app.py


開啟瀏覽器 → http://localhost:8501
