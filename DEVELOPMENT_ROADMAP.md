# V2BOT 開發路線圖

## 項目概述

建構完整的虛擬貨幣預測系統，從數據管理、特徵工程到模型訓練和回測。

---

## Phase 1: 數據基礎 ✅ (已完成)

### 已完成的任務
- [x] 下載 23 種主要加密貨幣數據
- [x] 組織成標準 OHLCV 格式 (Open, High, Low, Close, Volume)
- [x] 支援多個時間框架 (15分鐘、1小時)
- [x] 上傳到 HuggingFace Dataset
- [x] 建立 `klines/` 資料夾結構
- [x] 清理根目錄舊數據

### 文件結構
```
klines/
├── AAVE/          ← 23 種加密貨幣
├── ADA/
├── BTC/
├── ETH/
├── ...
├── XRP/
├── _combined/     ← 合併數據集
│   ├── all_symbols_15m.csv
│   └── all_symbols_1h.csv
└── README.md
```

---

## Phase 2: 數據加載與探索 🔄 (進行中)

### 任務

#### 2.1 數據加載模塊 ✅
**文件**: `ml/data_loader.py`

**功能**:
- 從 HuggingFace 下載 OHLCV 數據
- 自動數據清理 (移除重複、NaN、異常值)
- 多種正規化方法 (MinMax, ZScore)
- 時間序列窗口化
- Train/Val/Test 分割 (時間順序，無洩露)

**使用方式**:
```python
from ml.data_loader import DataLoader

loader = DataLoader(symbol='BTC', interval='15m')
df = loader.load_and_clean()
loader.summary()  # 顯示統計信息

# 創建窗口
windows = loader.create_windows(df, window_size=100)

# 分割數據
train, val, test = loader.train_val_test_split(df, train_ratio=0.7, val_ratio=0.15)
```

#### 2.2 數據探索筆記本 ✅
**文件**: `notebooks/01_data_exploration.ipynb`

**包含的步驟**:
1. 從 HF 下載 BTC 15m 數據
2. 數據清理和驗證
3. 基本統計信息
4. 視覺化 (價格趨勢、成交量、收益分佈)
5. 窗口化測試
6. Train/Val/Test 分割

**執行位置**: Google Colab

### 下一步
- [ ] 執行 `01_data_exploration.ipynb` 驗證 BTC 15m 數據
- [ ] 確認數據質量和統計特性
- [ ] 分析返回率分佈和異常值

---

## Phase 3: 特徵工程 📊 (待開始)

### 計劃的特徵

#### 3.1 基本特徵
- **價格動量**：移動平均線 (MA5, MA10, MA20)
- **波動率**：真實範圍 (ATR)、標準差
- **收益**：百分比變化、對數收益

#### 3.2 技術指標
- **RSI** (相對強度指數) - 14期
- **MACD** (指數平滑移動平均線差值)
- **布林帶** - 價格標準差帶
- **成交量指標** - OBV (能量潮指標)
- **Stochastic** - 隨機指標

#### 3.3 統計特徵
- 偏度 (Skewness)、峰度 (Kurtosis)
- 自相關係數 (ACF) 和偏自相關係數 (PACF)
- 分形維度 (Fractal Dimension)

### 預期輸出
```python
feature_df = create_features(df)
# 結果：N × (5 OHLCV + K 特徵) 的 DataFrame
```

---

## Phase 4: 模型架構 🧠 (待設計)

### 推薦方向

#### 選項 A: 價格預測 (迴歸)
**目標**: 預測下一根 K 線的收盤價

**模型候選**:
- LSTM (長短期記憶網絡)
- Transformer
- 1D-CNN
- XGBoost/LightGBM

**評估指標**:
- MAE (平均絕對誤差)
- RMSE (均方根誤差)
- MAPE (平均絕對百分比誤差)

#### 選項 B: 方向預測 (分類)
**目標**: 預測價格是上升、下降還是持平

**模型候選**:
- LSTM + Dense
- Transformer Classifier
- LightGBM

**評估指標**:
- 準確率 (Accuracy)
- F1-Score
- ROC-AUC

#### 選項 C: 交易信號生成
**目標**: 生成 買/持/賣 信號

**方法**:
- 組合技術指標
- 強化學習 (RL)

---

## Phase 5: 模型訓練 🚀 (待實現)

### 訓練框架

**預期工作流程**:
```python
# 1. 加載數據
loader = DataLoader(symbol='BTC', interval='15m')
df = loader.load_and_clean()

# 2. 創建特徵
feature_df = create_features(df)

# 3. 分割數據
train, val, test = loader.train_val_test_split(feature_df)

# 4. 正規化
train_norm = loader.normalize(train, method='minmax')
val_norm = loader.normalize(val, method='minmax')
test_norm = loader.normalize(test, method='minmax')

# 5. 創建窗口
X_train, y_train = create_windowed_dataset(train_norm, window_size=100)
X_val, y_val = create_windowed_dataset(val_norm, window_size=100)
X_test, y_test = create_windowed_dataset(test_norm, window_size=100)

# 6. 訓練模型
model = build_lstm_model(input_shape=(100, n_features))
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=50, batch_size=32)

# 7. 評估
results = model.evaluate(X_test, y_test)
print(f'Test Loss: {results[0]:.4f}')
```

### 訓練配置
- **批大小**: 32
- **Epoch**: 50-100
- **優化器**: Adam
- **學習率**: 1e-3
- **正則化**: Dropout, L2
- **早停**: 監控 validation loss

---

## Phase 6: 模型評估與最佳化 📈 (待完成)

### 評估指標
- 訓練/驗證/測試損失曲線
- 超參數調整 (Grid Search, Bayesian Optimization)
- 交叉驗證
- 特徵重要性分析

### 回測測試
- 使用歷史數據模擬交易
- 計算 Sharpe Ratio、Max Drawdown 等
- 與基準比較 (Buy & Hold)

---

## Phase 7: 生產環境部署 🌐 (未來)

### 計劃
- API 服務 (FastAPI)
- 實時預測
- 數據庫存儲
- 監控儀表板

---

## 技術棧

| 組件 | 選擇 |
|------|------|
| 環境 | Google Colab / 本地 Python |
| 數據存儲 | HuggingFace Dataset |
| 數據處理 | Pandas, NumPy |
| 可視化 | Matplotlib, Seaborn, Plotly |
| 機器學習 | TensorFlow/PyTorch, Scikit-learn, XGBoost |
| 版本控制 | Git, GitHub |

---

## 即時行動清單 (接下來的步驟)

### 短期 (本週)
- [ ] 在 Colab 執行 `01_data_exploration.ipynb`
  - 下載 BTC 15m 數據
  - 驗證數據質量
  - 分析返回率分佈
- [ ] 確認數據是否適合模型訓練

### 中期 (下週)
- [ ] 決定模型方向 (價格預測 vs 方向預測 vs 交易信號)
- [ ] 實現特徵工程模塊 (`ml/feature_engineering.py`)
- [ ] 創建特徵計算筆記本 (`notebooks/02_feature_engineering.ipynb`)

### 長期 (2-4週)
- [ ] 構建模型架構
- [ ] 訓練基準模型
- [ ] 實施超參數調整
- [ ] 評估與優化

---

## 資源連結

- **Dataset**: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- **Repository**: https://github.com/caizongxun/v2bot
- **Colab Notebook**: [01_data_exploration.ipynb]

---

## 備註

**數據周期**:
- 每根 15m K 線包含 15 分鐘的聚合交易數據
- 100 根 K 線 ≈ 1500 分鐘 ≈ 25 小時
- 一個月大約有 2880 根 15m K 線 (假設 24/7 交易)

**測試策略**:
1. 先用 BTC 15m 驗證管道
2. 逐步擴展到其他加密貨幣
3. 測試不同時間框架 (5m, 1h, 4h)
4. 評估多幣種策略

---

*最後更新: 2025-12-30*
