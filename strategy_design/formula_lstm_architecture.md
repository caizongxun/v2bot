# V2Bot 高級量化交易策略架構設計：公式逆向解析 + LSTM 融合

## 系統概述

您提出的架構是一個三層金字塔結構，具有極高的工程和理論價值：

```
層級 3 - 決策層:
    LSTM/Transformer 神經網絡
    輸入：5 個合成指標值 (由公式生成)
    輸出：交易信號 (買=1, 持=0, 賣=-1)

層級 2 - 公式層:
    符號回歸算法 (Symbolic Regression)
    輸出：5 套獨立混合公式
    每套公式將原始 K 線數據轉化為 1 個標量值
    示例: f1(x) = RSI_14 * 0.5 + MACD_diff * 0.5 + log(ATR)

層級 1 - 數據層:
    原始 OHLCV + 30+ 標準指標
    (RSI, MACD, 布林帶, ATR, OBV 等)
```

這個設計的核心創新點：

1. **特徵降維**: 30+ 維指標 → 5 維公式值 (降低過擬合風險)
2. **可解釋性**: 每個公式都是代數表達式，可完全理解決策邏輯
3. **自動發現**: 無需人工設計指標組合，符號回歸自動找最優公式
4. **時間序列建模**: LSTM 學習 5 個公式值的時間模式

---

## 完整實現路線圖

### 第一階段：符號回歸公式發現 (本地電腦，1-2 週)

**目標**: 從歷史 K 線數據中自動逆向解析出 5 套最優的混合公式

#### 1.1 環境準備

```bash
pip install pysr pandas numpy scikit-learn pandas-ta
```

#### 1.2 數據準備與指標計算

```python
import pandas as pd
from huggingface_hub import hf_hub_download
import ta

class SymbolicRegressionDataPrep:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
    
    def load_data(self, symbol='BTC', interval='1h'):
        """從 HF Dataset 加載數據"""
        file_path = hf_hub_download(
            repo_id='zongowo111/v2-crypto-ohlcv-data',
            filename=f'klines/{symbol}USDT/{symbol}_{interval}.parquet',
            repo_type='dataset',
            token=self.hf_token
        )
        return pd.read_parquet(file_path)
    
    def compute_indicators(self, df):
        """計算 30+ 基礎指標"""
        df = df.copy()
        
        # 動量指標
        df['rsi_7'] = ta.momentum.rsi(df['close'], 7)
        df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], 21)
        
        # MACD 系列
        df['macd_line'] = ta.trend.macd_line(df['close'], 12, 26)
        df['macd_signal'] = ta.trend.macd_signal_line(df['close'], 12, 26, 9)
        df['macd_diff'] = df['macd_line'] - df['macd_signal']
        
        # 移動平均
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(12).mean()
        df['ema_26'] = df['close'].ewm(26).mean()
        
        # 波動率相關
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], 20, 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # 成交量指標
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ema'] = df['obv'].ewm(14).mean()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        
        # 價格相關特徵
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def create_labels(self, df, lookahead=24, threshold=0.005):
        """
        創建訓練目標
        lookahead: 向前看 N 根 K 棒 (24h K 棒 = 1 天)
        threshold: 收益率閾值
        """
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 三分類
        df['label'] = 1  # 預設持平
        df.loc[df['future_return'] > threshold, 'label'] = 2   # 上漲
        df.loc[df['future_return'] < -threshold, 'label'] = 0  # 下跌
        
        return df
    
    def prepare(self, symbol='BTC', interval='1h'):
        """完整流程"""
        df = self.load_data(symbol, interval)
        df = self.compute_indicators(df)
        df = self.create_labels(df)
        df = df.dropna()
        return df
```

#### 1.3 符號回歸發現公式

```python
from pysr import PySRRegressor
import numpy as np

class SymbolicFormulaDiscoverer:
    def __init__(self):
        self.model = PySRRegressor(
            niterations=100,
            population_size=50,
            ncyclesperiteration=50,
            procs=4,
            binary_operators=['+', '-', '*', '/', '^'],
            unary_operators=['sin', 'cos', 'sqrt', 'exp', 'log', 'abs'],
            complexity_of_operators={
                '+': 1, '-': 1, '*': 2, '/': 3,
                'sin': 3, 'cos': 3, 'sqrt': 3, 'log': 3, '^': 3
            },
            maxsize=20,
            maxdepth=5,
            loss='mse',
            denoise=True,
        )
    
    def discover(self, X, y):
        """執行符號回歸"""
        print(f"開始符號回歸，數據形狀: {X.shape}")
        self.model.fit(X, y)
        return self.model.equations_
    
    def export_top_formulas(self, equations_df, n=5):
        """提取前 N 個最優公式"""
        top_equations = equations_df.nsmallest(n, 'loss')
        
        formulas = {}
        for i, (idx, row) in enumerate(top_equations.iterrows(), 1):
            formulas[f'formula_{i}'] = {
                'equation': row['equation'],
                'loss': float(row['loss']),
                'complexity': row['complexity']
            }
        
        return formulas
```

**本地執行腳本**：

```python
# local_symbolic_regression.py

if __name__ == '__main__':
    # 準備數據
    prep = SymbolicRegressionDataPrep()
    df = prep.prepare('BTC', '1h')
    
    # 特徵和目標
    indicator_cols = [col for col in df.columns if col not in 
                     ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X = pd.DataFrame(
        scaler.fit_transform(df[indicator_cols]),
        columns=indicator_cols
    )
    y = df['label'].values
    
    # 執行符號回歸 (耗時 30-60 分鐘)
    discoverer = SymbolicFormulaDiscoverer()
    equations = discoverer.discover(X, y)
    
    # 導出公式
    formulas = discoverer.export_top_formulas(equations, n=5)
    
    # 保存
    import json
    with open('discovered_formulas.json', 'w') as f:
        json.dump(formulas, f, indent=2)
    
    print("公式已保存到 discovered_formulas.json")
    print("\n發現的公式:")
    for name, info in formulas.items():
        print(f"{name}: {info['equation']}")
```

---

### 第二階段：LSTM 模型訓練 (Google Colab，1-2 週)

**目標**: 訓練 LSTM 模型學習 5 個公式值序列 → 交易信號的映射

#### 2.1 Colab 訓練腳本

```python
# 在 Google Colab 執行

!pip install -q tensorflow pandas numpy scikit-learn huggingface-hub pysr

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ========== 步驟 1: 加載公式 ==========
with open('discovered_formulas.json') as f:
    formulas_config = json.load(f)

print("已加載公式:")
for name, info in formulas_config.items():
    print(f"  {name}: {info['equation'][:50]}...")

# ========== 步驟 2: 加載原始數據 ==========
print("\n從 HF 加載 BTC 1h 數據...")
file_path = hf_hub_download(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    filename='klines/BTCUSDT/BTC_1h.parquet',
    repo_type='dataset'
)
df = pd.read_parquet(file_path)
print(f"加載完成: {len(df)} 行數據")

# ========== 步驟 3: 計算基礎指標 ==========
print("\n計算基礎指標...")
import ta

df['rsi_7'] = ta.momentum.rsi(df['close'], 7)
df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
df['macd_diff'] = (
    ta.trend.macd_line(df['close'], 12, 26) - 
    ta.trend.macd_signal_line(df['close'], 12, 26, 9)
)
df['sma_20'] = df['close'].rolling(20).mean()
df['bb_width'] = (
    ta.volatility.bollinger_hband(df['close'], 20, 2) -
    ta.volatility.bollinger_lband(df['close'], 20, 2)
)
df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

df = df.dropna()

# ========== 步驟 4: 應用公式生成合成指標 ==========
print("\n應用公式生成 5 個合成指標...")

# 簡化版本：直接在這裡定義公式函數
# (實際應該用 sympy 或 numexpr 動態編譯)

def formula_1(rsi_14, macd_diff, sma_20):
    return rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3

def formula_2(atr_14, volume_ratio):
    return np.log(abs(atr_14 * volume_ratio) + 1e-8)

def formula_3(bb_width, rsi_7):
    return bb_width / (rsi_7 + 1e-8)

def formula_4(macd_diff, atr_14):
    return np.where(atr_14 > 1e-8, macd_diff / atr_14, 0)

def formula_5(volume_ratio, sma_20):
    return np.tanh(volume_ratio) * sma_20

# 應用公式
formula_values = np.zeros((len(df), 5))
for i in range(len(df)):
    if i % 5000 == 0:
        print(f"  進度: {i}/{len(df)}")
    
    formula_values[i, 0] = formula_1(df.iloc[i]['rsi_14'], df.iloc[i]['macd_diff'], df.iloc[i]['sma_20'])
    formula_values[i, 1] = formula_2(df.iloc[i]['atr_14'], df.iloc[i]['volume_ratio'])
    formula_values[i, 2] = formula_3(df.iloc[i]['bb_width'], df.iloc[i]['rsi_7'])
    formula_values[i, 3] = formula_4(df.iloc[i]['macd_diff'], df.iloc[i]['atr_14'])
    formula_values[i, 4] = formula_5(df.iloc[i]['volume_ratio'], df.iloc[i]['sma_20'])

print(f"完成: {formula_values.shape}")

# ========== 步驟 5: 生成訓練目標 ==========
print("\n生成訓練目標...")
df['future_return'] = df['close'].shift(-24) / df['close'] - 1
df['label'] = 1  # 預設持平
df.loc[df['future_return'] > 0.005, 'label'] = 2   # 上漲
df.loc[df['future_return'] < -0.005, 'label'] = 0  # 下跌

y_labels = df['label'].values

# ========== 步驟 6: 數據分割 ==========
print("\n數據分割...")
train_size = int(0.7 * len(formula_values))
val_size = int(0.15 * len(formula_values))

X_train = formula_values[:train_size]
y_train = y_labels[:train_size]

X_val = formula_values[train_size:train_size+val_size]
y_val = y_labels[train_size:train_size+val_size]

X_test = formula_values[train_size+val_size:]
y_test = y_labels[train_size+val_size:]

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ========== 步驟 7: 創建時間序列 ==========
print("\n創建時間序列 (lookback=30)...")

def create_sequences(X, y, lookback=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

lookback = 30
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)

# One-hot 編碼
from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_seq, 3)
y_val_one_hot = to_categorical(y_val_seq, 3)
y_test_one_hot = to_categorical(y_test_seq, 3)

print(f"訓練集: {X_train_seq.shape}")
print(f"驗證集: {X_val_seq.shape}")
print(f"測試集: {X_test_seq.shape}")

# ========== 步驟 8: 構建 LSTM 模型 ==========
print("\n構建 LSTM 模型...")

model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True,
                     input_shape=(lookback, 5)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    
    keras.layers.LSTM(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== 步驟 9: 訓練 ==========
print("\n開始訓練...")

history = model.fit(
    X_train_seq, y_train_one_hot,
    validation_data=(X_val_seq, y_val_one_hot),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
)

# ========== 步驟 10: 評估 ==========
print("\n評估模型...")
y_pred = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test_seq, y_pred_classes)
precision = precision_score(y_test_seq, y_pred_classes, average='weighted')
recall = recall_score(y_test_seq, y_pred_classes, average='weighted')
f1 = f1_score(y_test_seq, y_pred_classes, average='weighted')

print(f"準確率: {accuracy:.4f}")
print(f"精準度: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1 分數: {f1:.4f}")

print("\n混淆矩陣:")
print(confusion_matrix(y_test_seq, y_pred_classes))

# ========== 步驟 11: 保存 ==========
model.save('formula_lstm_model.h5')
with open('scaler_config.pkl', 'wb') as f:
    import pickle
    pickle.dump(scaler, f)

print("\n模型已保存:")
print("  - formula_lstm_model.h5")
print("  - scaler_config.pkl")
```

---

### 第三階段：實時推理系統

#### 3.1 實時信號生成

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from collections import deque
import pickle

class RealTimeFormulaPredictor:
    def __init__(self, model_path='formula_lstm_model.h5', scaler_path='scaler_config.pkl'):
        """初始化預測器"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 緩衝區
        self.formula_buffer = deque(maxlen=30)
        self.last_signal = None
        self.last_confidence = None
    
    def apply_formulas(self, indicators_dict):
        """應用 5 個公式"""
        # 這裡應該應用真實的發現公式
        # 示例實現：
        
        r = indicators_dict
        
        f1 = r.get('rsi_14', 0) * 0.4 + r.get('macd_diff', 0) * 0.3 + r.get('sma_20', 0) * 0.3
        f2 = np.log(abs(r.get('atr_14', 1e-8) * r.get('volume_ratio', 1)) + 1e-8)
        f3 = r.get('bb_width', 0) / (r.get('rsi_7', 1) + 1e-8)
        f4 = r.get('macd_diff', 0) / (r.get('atr_14', 1e-8))
        f5 = np.tanh(r.get('volume_ratio', 0)) * r.get('sma_20', 0)
        
        return np.array([f1, f2, f3, f4, f5])
    
    def process_new_kline(self, kline_dict):
        """處理新 K 棒"""
        # 應用公式
        formula_values = self.apply_formulas(kline_dict)
        
        # 添加到緩衝區
        self.formula_buffer.append(formula_values)
        
        # 檢查是否有足夠數據
        if len(self.formula_buffer) < 30:
            return {
                'status': 'WARMING_UP',
                'buffer_size': len(self.formula_buffer)
            }
        
        # 準備輸入
        X = np.array(list(self.formula_buffer)).reshape(1, 30, 5)
        X_scaled = self.scaler.transform(X.reshape(-1, 5)).reshape(1, 30, 5)
        
        # 預測
        probabilities = self.model.predict(X_scaled, verbose=0)[0]
        signal_idx = np.argmax(probabilities)
        confidence = probabilities[signal_idx]
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[signal_idx]
        
        self.last_signal = signal
        self.last_confidence = float(confidence)
        
        return {
            'status': 'READY',
            'signal': signal,
            'confidence': float(confidence),
            'timestamp': kline_dict.get('timestamp'),
            'formula_values': formula_values.tolist()
        }

# 使用示例
predictor = RealTimeFormulaPredictor()

# 當新 K 棒到達時
new_kline = {
    'timestamp': '2025-12-30 20:00:00',
    'open': 42150.0,
    'high': 42250.0,
    'low': 42100.0,
    'close': 42200.0,
    'volume': 1200,
    'rsi_7': 45.2,
    'rsi_14': 48.5,
    'macd_diff': 0.0025,
    'sma_20': 42050.0,
    'bb_width': 150.0,
    'atr_14': 75.5,
    'obv': 1250000.0,
    'volume_ratio': 1.05
}

result = predictor.process_new_kline(new_kline)
print(result)
```

---

## 核心優勢總結

| 特性 | 優勢 |
| --- | --- |
| **特徵自動發現** | 無需人工設計 30+ 維指標組合，符號回歸自動找最優 5 個公式 |
| **降維效果** | 從 30+ 維降到 5 維，大幅降低過擬合風險 |
| **可解釋性** | 每個公式都是代數表達式，完全可理解 |
| **時間序列建模** | LSTM 學習公式值的時間動態 |
| **適應性** | 可定期重新執行符號回歸以發現新公式 |
| **計算效率** | 推理只需 5 個標量運算，極快 |

---

## 建議的執行順序

1. **本地準備** (1-2 週)
   - 安裝 PySR
   - 下載 BTC 歷史數據
   - 執行符號回歸發現 5 套公式

2. **Colab 訓練** (1-2 週)
   - 加載公式
   - 應用公式生成合成指標
   - 訓練 LSTM 模型

3. **實時部署** (1 週)
   - 集成 TradingView Webhook
   - 實時推理系統
   - 風控和交易執行

---

## 技術細節與最佳實踐

### 符號回歸參數優化

```python
# 更激進的搜索 (更複雜的公式)
model = PySRRegressor(
    niterations=200,
    population_size=100,
    maxsize=30,        # 允許更大的公式
    maxdepth=7,
    complexity_of_operators={'+': 1, '-': 1, '*': 2, '/': 3, '^': 4}
)

# 更保守的搜索 (簡潔的公式)
model = PySRRegressor(
    niterations=50,
    population_size=30,
    maxsize=10,        # 只允許簡潔公式
    maxdepth=3,
    complexity_of_operators={'+': 1, '-': 1, '*': 2, '/': 3, '^': 5}
)
```

### 模型超參數調優

LSTM 層數、隱層維度、Dropout 率都應根據驗證集性能調整。

---

您想先從哪個環節開始深入實現？
