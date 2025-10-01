作業報告：簡單線性迴歸分析 (Simple Linear Regression Analysis)
本專案旨在透過 Python (使用 Streamlit 框架) 解決一個簡單線性迴歸問題，並嚴格遵循 CRISP-DM (Cross-Industry Standard Process for Data Mining) 的六大步驟流程，達成互動式模型部署的目標。

#1. Business Understanding (商業理解)
目的 (Prompt):
目標是建立一個互動式的應用程式，讓使用者能夠模擬線性資料 Y=aX+b+ϵ 的生成過程，並訓練一個簡單線性迴歸模型來估計真實的參數 a (斜率) 和 b (截距)。最終需比較模型估計值與使用者設定的真實值之間的差異，並識別高殘差的離群值 (Outliers)。

成功標準:

模型能夠合理地擬合資料點，並能接近真實線 (True Line)。

應用程式必須允許使用者動態調整 a、雜訊水準 (Noise) 和資料點數量 (N points)。

最終應用程式使用 Streamlit 進行網頁化部署。

>2. Data Understanding (資料理解)
資料來源: 本專案的資料是合成 (Synthetic) 資料，由使用者在 Streamlit 側邊欄設定參數動態生成。

核心變數:

自變量 (Independent Variable, X): 範圍固定為 [0,10] 的等距點。

因變量 (Dependent Variable, Y): 由 Y=aX+b+ϵ 生成。

a: 使用者設定的真實斜率 (Slope)。

b: 使用者設定的真實截距 (Intercept)。

ϵ: 使用者設定的雜訊水準 (Noise)，為服從標準差為 Noise 參數的常態分佈隨機誤差。

>3. Data Preparation (資料準備)
由於資料是合成且結構化完整的，資料準備步驟相對簡單：

資料生成: 使用 numpy 的 linspace 生成 X 向量，並結合使用者定義的 a,b,noise 生成 Y 向量。

維度處理: 機器學習模型 (Scikit-learn) 要求 X 必須是二維陣列 (features matrix)，因此使用 X.reshape(-1, 1) 將 X 塑形為 (N,1)。Y 也保持 (N,1) 的形狀。

DataFrame 建立: 將 X,Y 和後續的預測值、殘差整合成 pandas DataFrame，以方便進行殘差排序和離群值分析。

>4. Modeling (模型建立)
模型選擇: 選擇 Scikit-learn 庫中的 LinearRegression 模型，這是一種基於最小平方法 (Ordinary Least Squares, OLS) 的簡單線性迴歸模型。

模型訓練過程:

初始化: 實例化模型：model = LinearRegression()

擬合: 使用準備好的 X 和 Y 資料進行訓練：model.fit(X, y)

預測: 獲得擬合線的預測值：y_pred = model.predict(X)

>5. Evaluation (模型評估)
模型評估側重於視覺化比較和參數準確性：

參數準確性 (Quantitative):

比較模型估計的斜率 (model.coef_) 和截距 (model.intercept_) 與使用者設定的真實值 (a,b)。

將結果以文字格式 (例如：Markdown st.info) 顯示給使用者。

視覺化擬合 (Qualitative):

使用 Matplotlib 繪製散點圖，圖中同時顯示 真實線 (True Line, 綠色虛線) 和 擬合迴歸線 (Fitted Regression Line, 紅色實線)，直觀判斷擬合優度。

離群值分析 (Residual Analysis):

計算殘差：∣Y−Y 
pred
​
 ∣。

將殘差最大的前 5 個資料點標記為離群值 (Outliers)，並在圖表上使用特殊的 橙色菱形 (Orange Diamond) 突出顯示，同時在下方表格中展示這些離群值的詳細資訊。

>6. Deployment (模型部署)
框架: Streamlit

本專案使用 Streamlit 框架將整個分析流程部署為一個單一的網頁應用程式：

用戶輸入介面: 側邊欄提供 a、雜訊和資料點數量的滑塊，滿足互動性要求。

即時計算與視覺化: 每當使用者調整參數時，Streamlit 會自動重新運行腳本，即時更新資料生成、模型訓練、評估結果和圖表。

結果展示: 模型係數、擬合圖表和離群值表格全部在主面板上清晰呈現。

結論:
透過 Streamlit 的部署，本簡單線性迴歸模型不僅解決了問題，還以一個高互動性的方式，滿足了作業對框架部署的要求。
