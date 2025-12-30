# V2Bot 準確率優化方案

## 目標：從 38% 提升到 70%+

---

## 第一階段：診斷現狀（1-2 周）

### 1.1 分析當前模型的錯誤模式

```python
# 錯誤分析代碼
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 預測結果
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred_class)
print("\n混淆矩陣:")
print("         SELL  HOLD  BUY")
print(f"SELL    {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}")
print(f"HOLD    {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}")
print(f"BUY     {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}")

print("\n詳細分類報告:")
print(classification_report(y_test, y_pred_class, 
      target_names=['SELL', 'HOLD', 'BUY']))

# 找出預測最差的類別
for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
    class_acc = cm[i,i] / cm[i].sum()
    print(f"{label} 準確率: {class_acc:.1%}")
```

### 1.2 識別數據質量問題

```python
# 檢查標籤分佈
import matplotlib.pyplot as plt

label_counts = np.bincount(labels)
print(f"SELL: {label_counts[0]} ({100*label_counts[0]/len(labels):.1f}%)")
print(f"HOLD: {label_counts[1]} ({100*label_counts[1]/len(labels):.1f}%)")
print(f"BUY:  {label_counts[2]} ({100*label_counts[2]/len(labels):.1f}%)")

# 視覺化
plt.figure(figsize=(10, 4))
plt.bar(['SELL', 'HOLD', 'BUY'], label_counts)
plt.title('標籤分佈')
plt.ylabel('樣本數')
plt.show()

# 檢查特徵相關性
import pandas as pd
df = pd.DataFrame(formula_values, columns=['F1', 'F2', 'F3', 'F4', 'F5'])
print("\n特徵統計:")
print(df.describe())
print("\n特徵相關矩陣:")
print(df.corr())
```

### 1.3 評估當前模型的過擬合/欠擬合程度

```python
# 分析訓練歷史
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('準確率趨勢')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('損失函數趨勢')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 分析
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
gap = train_acc - val_acc

if gap < 0.02:
    print("✓ 模型狀況良好，未見明顯過擬合")
elif gap < 0.05:
    print("△ 輕微過擬合，可通過正則化改進")
else:
    print("✗ 明顯過擬合，需要重新設計架構")
```

---

## 第二階段：數據優化（2-3 周）

### 2.1 擴展數據集

**當前數據:** 54,915 根 BTC 1h K 棒 (2020 年以來)
**目標:** 150,000+ 根 K 棒

```python
# 下載更多歷史數據
import ccxt
import pandas as pd
from datetime import datetime, timedelta

exchange = ccxt.binance()
symbol = 'BTC/USDT'
interval = '1h'

all_ohlcv = []
since = exchange.parse8601('2017-01-01T00:00:00Z')  # 從 2017 開始

while since < exchange.milliseconds():
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(f"Fetched {len(all_ohlcv)} candles...")
    except:
        pass

df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"\n總共加載 {len(df)} 根 K 棒")
```

### 2.2 改進標籤生成邏輯

**當前方法：** 基於簡單的 24 小時收益率閾值
```python
future_ret = close.shift(-24) / close - 1
labels[future_ret > 0.005] = 2   # BUY
labels[future_ret < -0.005] = 0  # SELL
```

**改進方法 1：** 基於市場狀態的動態閾值

```python
def create_adaptive_labels(close, lookforward=24):
    """
    根據市場波動性創建自適應標籤
    """
    future_ret = close.shift(-lookforward) / close - 1
    atr = calculate_atr(close, 14)
    
    # 動態閾值 = ATR 的百分比
    threshold_buy = (atr / close).rolling(20).mean() * 2
    threshold_sell = -(atr / close).rolling(20).mean() * 2
    
    labels = np.ones(len(close), dtype=np.int32)
    labels[future_ret > threshold_buy] = 2    # BUY
    labels[future_ret < threshold_sell] = 0   # SELL
    
    return labels
```

**改進方法 2：** 基於風險調整收益的標籤

```python
def create_risk_adjusted_labels(close, high, low, lookforward=24):
    """
    考慮風險和回報的標籤生成
    """
    future_ret = close.shift(-lookforward) / close - 1
    max_drawdown = (low.rolling(lookforward).min() - close) / close
    
    # 計算風險調整的收益
    risk_adjusted = future_ret / (np.abs(max_drawdown) + 0.001)
    
    # 基於風險調整的分位數
    p33 = risk_adjusted.quantile(0.33)
    p67 = risk_adjusted.quantile(0.67)
    
    labels = np.ones(len(close), dtype=np.int32)
    labels[risk_adjusted > p67] = 2   # BUY
    labels[risk_adjusted < p33] = 0   # SELL
    
    return labels
```

**改進方法 3：** 基於多個時間框架的標籤

```python
def create_multi_timeframe_labels(close, lookforward=24):
    """
    使用多個時間框架的信號進行投票
    """
    labels = np.zeros(len(close), dtype=np.int32)
    
    # 短期 (12 小時)
    ret_12h = close.shift(-12) / close - 1
    signal_12h = np.sign(ret_12h)
    
    # 中期 (24 小時)
    ret_24h = close.shift(-24) / close - 1
    signal_24h = np.sign(ret_24h)
    
    # 長期 (48 小時)
    ret_48h = close.shift(-48) / close - 1
    signal_48h = np.sign(ret_48h)
    
    # 投票
    vote = signal_12h + signal_24h + signal_48h
    labels[vote > 1.5] = 2   # BUY (2 個或更多投票)
    labels[vote < -1.5] = 0  # SELL
    labels[(vote >= -1.5) & (vote <= 1.5)] = 1  # HOLD
    
    return labels
```

### 2.3 數據清理和預處理

```python
def clean_data(df):
    """
    清理和驗證數據質量
    """
    print(f"原始數據: {len(df)} 行")
    
    # 移除重複
    df = df.drop_duplicates(subset=['timestamp'])
    print(f"移除重複後: {len(df)} 行")
    
    # 移除缺失值
    df = df.dropna()
    print(f"移除缺失後: {len(df)} 行")
    
    # 移除異常值 (3-sigma)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        mean = df[col].mean()
        std = df[col].std()
        df = df[
            (df[col] >= mean - 3*std) & 
            (df[col] <= mean + 3*std)
        ]
    print(f"移除異常值後: {len(df)} 行")
    
    # 檢查數據合理性
    assert (df['high'] >= df['low']).all(), "High < Low 異常"
    assert (df['high'] >= df['close']).all(), "High < Close 異常"
    assert (df['low'] <= df['close']).all(), "Low > Close 異常"
    assert (df['volume'] >= 0).all(), "負成交量異常"
    
    return df.sort_values('timestamp').reset_index(drop=True)
```

---

## 第三階段：特徵工程（2-3 周）

### 3.1 添加更多技術指標

**當前特徵:** 5 個公式
**目標特徵:** 20+ 個多樣化指標

```python
def compute_advanced_indicators(df):
    """
    計算 20+ 高級技術指標
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 動量指標
    df['rsi_7'] = ta.momentum.rsi(close, window=7)
    df['rsi_14'] = ta.momentum.rsi(close, window=14)
    df['rsi_21'] = ta.momentum.rsi(close, window=21)
    
    # MACD 族
    df['macd'] = ta.trend.macd_diff(close)
    df['macd_signal'] = ta.trend.macd_signal(close)
    df['macd_histogram'] = ta.trend.macd_diff(close) - ta.trend.macd_signal(close)
    
    # 布林帶
    bb = ta.volatility.bollinger_bands(close, window=20, num_std=2)
    df['bb_upper'] = bb.iloc[:, 0]
    df['bb_middle'] = bb.iloc[:, 1]
    df['bb_lower'] = bb.iloc[:, 2]
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (close - df['bb_lower']) / df['bb_width']
    
    # 移動平均
    df['sma_10'] = close.rolling(10).mean()
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    df['ema_12'] = close.ewm(span=12).mean()
    df['ema_26'] = close.ewm(span=26).mean()
    
    # ATR 和波動率
    df['atr_14'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['natr'] = df['atr_14'] / close  # 標準化 ATR
    df['volatility'] = close.rolling(20).std() / close.rolling(20).mean()
    
    # 成交量指標
    df['obv'] = ta.volume.on_balance_volume(close, volume)
    df['volume_sma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma']
    df['mfi'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
    
    # 其他有用指標
    df['adx'] = ta.trend.adx(high, low, close, window=14)  # 趨勢強度
    df['cci'] = ta.momentum.cci(high, low, close, window=20)  # 商品通道指數
    df['stoch_k'] = ta.momentum.stoch(high, low, close).iloc[:, 0]  # 隨機指標
    df['williams_r'] = ta.momentum.williams_r(high, low, close, window=14)
    
    return df.fillna(method='ffill').fillna(method='bfill')
```

### 3.2 特徵選擇和重要性排名

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap

def analyze_feature_importance(X, y):
    """
    分析特徵重要性並優化特徵集
    """
    # 使用隨機森林計算特徵重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徵重要性排名:")
    print(feature_importance)
    
    # 移除低重要性特徵
    threshold = feature_importance['importance'].quantile(0.25)
    important_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()
    
    print(f"\n保留 {len(important_features)} 個特徵 (移除低於 {threshold:.4f} 的)")
    print(f"移除的特徵: {set(X.columns) - set(important_features)}")
    
    return important_features
```

### 3.3 特徵轉換和交互項

```python
def create_feature_interactions(df, features):
    """
    創建高階特徵交互項
    """
    # 比率特徵
    df['rsi_bb_ratio'] = df['rsi_14'] / (df['bb_width'] + 1e-8)
    df['price_bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['atr_volatility_ratio'] = df['atr_14'] / df['volatility']
    
    # 交叉特徵
    df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
    df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['price_above_sma'] = (df['close'] > df['sma_20']).astype(int)
    
    # 動量交互
    df['rsi_macd_agreement'] = np.sign(df['rsi_14'] - 50) == np.sign(df['macd'])
    df['rsi_trend_agreement'] = np.sign(df['rsi_14'] - 50) == np.sign(df['adx'])
    
    # 成交量確認
    df['volume_rsi_confirmation'] = (df['volume_ratio'] > 1.2) & (df['rsi_14'] > 60) | \
                                    (df['volume_ratio'] > 1.2) & (df['rsi_14'] < 40)
    
    return df
```

---

## 第四階段：模型架構優化（2-3 周）

### 4.1 增強模型設計

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_advanced_lstm_model(lookback=30, n_features=20):
    """
    高級 LSTM 架構 - 為了提升準確率
    """
    model = keras.Sequential([
        # 第一層: 雙向 LSTM
        layers.Bidirectional(
            layers.LSTM(128, activation='relu', return_sequences=True),
            input_shape=(lookback, n_features)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第二層: 雙向 LSTM
        layers.Bidirectional(
            layers.LSTM(64, activation='relu', return_sequences=True)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第三層: 標準 LSTM
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 密集層
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # 輸出層
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 4.2 Transformer 架構（更先進）

```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

def create_transformer_model(lookback=30, n_features=20):
    """
    Transformer 架構 - 當前 SOTA
    """
    inputs = keras.Input(shape=(lookback, n_features))
    
    # 多頭注意力層
    x = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.2
    )(inputs, inputs)
    x = LayerNormalization()(x)
    
    # LSTM 層
    x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)
    
    # 密集層
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 4.3 集成學習（Ensemble）

```python
def create_ensemble_model(lookback=30, n_features=20):
    """
    多模型集成 - 最高準確率
    """
    # 創建多個模型
    model1 = create_advanced_lstm_model(lookback, n_features)  # LSTM
    model2 = create_transformer_model(lookback, n_features)    # Transformer
    
    # 另一個 LSTM 變體
    model3 = keras.Sequential([
        layers.LSTM(96, activation='relu', return_sequences=True, input_shape=(lookback, n_features)),
        layers.Dropout(0.4),
        layers.LSTM(48, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 創建集成層
    inputs = keras.Input(shape=(lookback, n_features))
    
    out1 = model1(inputs)
    out2 = model2(inputs)
    out3 = model3(inputs)
    
    # 平均集成
    ensemble_output = layers.Average()([out1, out2, out3])
    
    ensemble_model = keras.Model(inputs, ensemble_output)
    ensemble_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble_model
```

---

## 第五階段：訓練策略優化（2 周）

### 5.1 進階訓練配置

```python
def train_with_advanced_strategy(model, X_train, y_train, X_val, y_val):
    """
    使用進階訓練策略
    """
    
    # 類別權重平衡 (處理不平衡數據)
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\n類別權重: {class_weight_dict}")
    
    # 回調函數
    callbacks = [
        # 早停機制
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # 最小改進
        ),
        
        # 學習率調整
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # 模型檢查點
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 訓練
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

### 5.2 超參數優化

```python
from keras_tuner import RandomSearch, Hyperband

def create_tunable_model(hp):
    """
    創建可調整超參數的模型
    """
    model = keras.Sequential([
        layers.LSTM(
            units=hp.Int('lstm_1_units', min_value=64, max_value=256, step=32),
            activation='relu',
            return_sequences=True,
            input_shape=(30, 20)
        ),
        layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
        
        layers.LSTM(
            units=hp.Int('lstm_2_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
        
        layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 執行超參數搜索
tuner = Hyperband(
    create_tunable_model,
    objective='val_accuracy',
    max_epochs=30,
    directory='hyperparameter_tuning',
    project_name='v2bot'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)

# 獲得最佳模型
best_model = tuner.get_best_models(num_models=1)[0]
```

---

## 第六階段：後期微調（1-2 周）

### 6.1 交叉驗證

```python
from sklearn.model_selection import KFold

def k_fold_validation(X, y, n_splits=5):
    """
    K 折交叉驗證
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n折 {fold+1}/{n_splits}")
        
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_vl = X[val_idx]
        y_vl = y[val_idx]
        
        model = create_advanced_lstm_model()
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        val_acc = history.history['val_accuracy'][-1]
        fold_scores.append(val_acc)
        print(f"折驗證準確率: {val_acc:.4f}")
    
    print(f"\n平均準確率: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    return fold_scores
```

### 6.2 TTA (Test Time Augmentation)

```python
def test_time_augmentation(model, X_test, n_augmentations=5):
    """
    測試時間數據增強 - 提升準確率 1-3%
    """
    predictions = []
    
    for i in range(n_augmentations):
        # 添加輕微噪聲
        noise = np.random.normal(0, 0.01, X_test.shape)
        X_augmented = X_test + noise
        X_augmented = np.clip(X_augmented, -5, 5)  # 限制範圍
        
        pred = model.predict(X_augmented, verbose=0)
        predictions.append(pred)
    
    # 平均預測
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction
```

---

## 完整實現路線圖

### 第 1 個月：數據和特徵優化
- [ ] 擴展數據集到 150,000+ K 棒
- [ ] 實現 3 種改進的標籤生成方法
- [ ] 添加 20+ 技術指標
- [ ] 進行特徵選擇和重要性分析
- [ ] 預期準確率提升：38% → 48-52%

### 第 2 個月：模型優化
- [ ] 測試進階 LSTM 架構
- [ ] 實現 Transformer 模型
- [ ] 創建集成學習系統
- [ ] 進行超參數優化
- [ ] 預期準確率提升：48-52% → 58-65%

### 第 3 個月：微調和驗證
- [ ] K 折交叉驗證
- [ ] 測試時間增強
- [ ] 長期回測驗證
- [ ] 實盤微調
- [ ] 預期準確率提升：58-65% → 70%+

---

## 重要提示

⚠️ **過擬合風險**
- 監控訓練/驗證差距
- 使用早停機制
- 添加充足的 Dropout 和 L2 正則化
- 定期在新數據上驗證

⚠️ **準確率提升的現實預期**
- 從 38% → 70% 需要 2-3 個月努力
- 不是所有改進都能帶來相同效果
- 實際交易性能可能低於准確率指標
- 專注於 Sharpe 比率和最大回撤，而不僅僅是準確率

---

最後更新：2025-12-30
