#!/usr/bin/env python3
"""
V2Bot Advanced Training - 提升到 70%+ 準確率

Usage in Colab:
    SYMBOL = 'BTC'
    INTERVAL = '1h'
    EPOCHS = 50
    LOOKBACK = 40
    
    import requests
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_advanced_training.py'
    script = requests.get(url).text
    exec(script, {'SYMBOL': SYMBOL, 'INTERVAL': INTERVAL, 'EPOCHS': EPOCHS, 'LOOKBACK': LOOKBACK})
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT ADVANCED TRAINING - 70%+ ACCURACY")
print("="*80)

SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
EPOCHS = globals().get('EPOCHS', 50)
LOOKBACK = globals().get('LOOKBACK', 40)

print(f"\nParameters: {SYMBOL} {INTERVAL} | Epochs: {EPOCHS} | Lookback: {LOOKBACK}")

# Import
print("\nImporting libraries...")
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print(f"  NumPy {np.__version__} | Pandas {pd.__version__} | TF {tf.__version__}")

# Scaler
class DictScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_
    
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_
    
    def to_dict(self):
        return {'mean': self.mean_.tolist(), 'scale': self.scale_.tolist()}

# Load data
print(f"\nLoading {SYMBOL} {INTERVAL} data...")
try:
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'klines/{SYMBOL}USDT/{SYMBOL}_{INTERVAL}.parquet',
        repo_type='dataset',
        cache_dir='/tmp/hf_cache'
    )
    df = pd.read_parquet(file_path)
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")
except Exception as e:
    print(f"  Fallback: Creating synthetic data")
    prices = 40000 + np.cumsum(np.random.randn(5000) * 100)
    df = pd.DataFrame({
        'open': prices + np.random.randn(5000) * 50,
        'high': prices + np.abs(np.random.randn(5000) * 100),
        'low': prices - np.abs(np.random.randn(5000) * 100),
        'close': prices,
        'volume': np.random.uniform(100, 5000, 5000)
    })

close = pd.to_numeric(df['close'], errors='coerce')
high = pd.to_numeric(df['high'], errors='coerce')
low = pd.to_numeric(df['low'], errors='coerce')
volume = pd.to_numeric(df['volume'], errors='coerce')

print(f"\nComputing advanced indicators...")

# 延伸技術指標
def rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# 計算基本技術指標
df['rsi_7'] = rsi(close, 7)
df['rsi_14'] = rsi(close, 14)
df['rsi_21'] = rsi(close, 21)
macd_line, signal_line, macd_diff = macd(close)
df['macd_line'] = macd_line
df['macd_signal'] = signal_line
df['macd_diff'] = macd_diff
df['sma_10'] = close.rolling(10).mean()
df['sma_20'] = close.rolling(20).mean()
df['sma_50'] = close.rolling(50).mean()
df['ema_12'] = close.ewm(12).mean()
df['ema_26'] = close.ewm(26).mean()
df['atr_14'] = atr(high, low, close, 14)
df['bb_upper'] = df['sma_20'] + (close.rolling(20).std() * 2)
df['bb_lower'] = df['sma_20'] - (close.rolling(20).std() * 2)
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_pct'] = (close - df['bb_lower']) / (df['bb_width'] + 1e-8)
df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
df['volume_sma'] = volume.rolling(20).mean()
df['volume_ratio'] = volume / (df['volume_sma'].replace(0, 1))
df['volatility'] = close.rolling(20).std() / close.rolling(20).mean()
df['natr'] = df['atr_14'] / close

# 增強特徵
df['rsi_bb_ratio'] = df['rsi_14'] / (df['bb_width'] + 1e-8)
df['price_bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(float)
df['price_above_sma'] = (close > df['sma_20']).astype(float)

df = df.fillna(method='ffill').fillna(method='bfill')
print(f"  Advanced indicators computed")

# 决定批次 - 綠色旗信號
print(f"\nApplying advanced labeling...")

def create_smart_labels(close, high, low, lookforward=24):
    """
    覺技者指標標籤 - 使用市場狀態自適應
    """
    future_ret = close.shift(-lookforward) / close - 1
    max_drawdown = (low.rolling(lookforward).min() - close) / close
    
    # 風險調整的收益
    risk_adjusted = future_ret / (np.abs(max_drawdown) + 0.001)
    
    # 根據風險調整的收益分位
    p33 = risk_adjusted.quantile(0.33)
    p67 = risk_adjusted.quantile(0.67)
    
    labels = np.ones(len(close), dtype=np.int32)
    labels[risk_adjusted > p67] = 2   # BUY
    labels[risk_adjusted < p33] = 0   # SELL
    
    return labels

labels = create_smart_labels(close, high, low, 24)
print(f"  SELL: {(labels == 0).sum()} | HOLD: {(labels == 1).sum()} | BUY: {(labels == 2).sum()}")

# 選擇特徵
feature_cols = [
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd_line', 'macd_signal', 'macd_diff',
    'sma_20', 'sma_50',
    'atr_14', 'bb_width', 'bb_pct', 'volatility',
    'volume_ratio', 'ema_cross', 'price_above_sma',
    'rsi_bb_ratio', 'price_bb_position'
]

df_features = df[feature_cols].dropna()
print(f"\n選擇 {len(feature_cols)} 個特徵")
print(f"  {', '.join(feature_cols[:5])}...")

# 提取特徵
formula_values = df_features.values.astype(np.float32)
formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
labels = labels[:len(formula_values)]

print(f"\nDataset shape: {formula_values.shape}")

# 數據分割
print(f"\nSplitting data (70% / 15% / 15%)...")
train_sz = int(0.7 * len(formula_values))
val_sz = int(0.15 * len(formula_values))

X_tr = formula_values[:train_sz]
y_tr = labels[:train_sz]
X_vl = formula_values[train_sz:train_sz+val_sz]
y_vl = labels[train_sz:train_sz+val_sz]
X_ts = formula_values[train_sz+val_sz:]
y_ts = labels[train_sz+val_sz:]

scaler = DictScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_vl_sc = scaler.transform(X_vl)
X_ts_sc = scaler.transform(X_ts)

print(f"  Train: {X_tr_sc.shape} | Val: {X_vl_sc.shape} | Test: {X_ts_sc.shape}")

# 序列化
print(f"\nCreating sequences (lookback={LOOKBACK})...")

def create_seq(X, y, lb=30):
    Xs, ys = [], []
    for i in range(len(X) - lb):
        Xs.append(X[i:i+lb])
        ys.append(y[i+lb])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

X_tr_sq, y_tr_sq = create_seq(X_tr_sc, y_tr, LOOKBACK)
X_vl_sq, y_vl_sq = create_seq(X_vl_sc, y_vl, LOOKBACK)
X_ts_sq, y_ts_sq = create_seq(X_ts_sc, y_ts, LOOKBACK)

y_tr_oh = to_categorical(y_tr_sq, 3)
y_vl_oh = to_categorical(y_vl_sq, 3)

print(f"  Sequences created successfully")

# 類別權重
print(f"\nComputing class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_tr_sq),
    y=y_tr_sq
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"  Weights: SELL={class_weight_dict[0]:.2f}, HOLD={class_weight_dict[1]:.2f}, BUY={class_weight_dict[2]:.2f}")

# 模型
print(f"\nBuilding advanced LSTM model...")

model = keras.Sequential([
    keras.layers.Bidirectional(
        keras.layers.LSTM(128, activation='relu', return_sequences=True),
        input_shape=(LOOKBACK, len(feature_cols))
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.LSTM(64, activation='relu', return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.LSTM(32, activation='relu', return_sequences=False),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Model: {model.count_params():,} parameters")

# 回調函數
print(f"\nSetting up callbacks...")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# 訓練
print(f"\nTraining ({EPOCHS} epochs)...")

history = model.fit(
    X_tr_sq, y_tr_oh,
    validation_data=(X_vl_sq, y_vl_oh),
    epochs=EPOCHS,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("  Training completed")

# 評估
print(f"\nEvaluating...")
y_pd = model.predict(X_ts_sq, verbose=0)
y_pc = np.argmax(y_pd, axis=1)
test_ac = np.mean(y_pc == y_ts_sq)

print(f"  Test Accuracy: {test_ac:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(
    y_ts_sq, y_pc,
    target_names=['SELL', 'HOLD', 'BUY']
))

# 业业准確率
cm = confusion_matrix(y_ts_sq, y_pc)
print(f"\n混淆矩陣:")
print(f"        SELL  HOLD   BUY")
for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
    print(f"{label:5} {cm[i,0]:4d} {cm[i,1]:4d} {cm[i,2]:4d}")

# 保存
print(f"\nSaving artifacts...")

model.save(f'formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras')

scaler_dict = scaler.to_dict()
with open(f'scaler_advanced_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(scaler_dict, f)

metadata = {
    'lookback': LOOKBACK,
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'n_params': int(model.count_params()),
    'train_accuracy': float(history.history['accuracy'][-1]),
    'val_accuracy': float(history.history['val_accuracy'][-1]),
    'test_accuracy': float(test_ac)
}

with open(f'metadata_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved: formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras")
print(f"  Scaler saved: scaler_advanced_{SYMBOL}_{INTERVAL}.json")
print(f"  Metadata saved: metadata_{SYMBOL}_{INTERVAL}.json")

print("\n" + "="*80)
print(f"SUCCESS! ADVANCED MODEL TRAINED")
print("="*80)

print(f"\nPerformance Summary:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:  {test_ac:.4f}")

if test_ac >= 0.70:
    print(f"\n✅ 目標達成! 準確率 >= 70%")
elif test_ac >= 0.65:
    print(f"\n✔ 優選結果! 準確率 >= 65%")
elif test_ac >= 0.60:
    print(f"\n△ 良好結果! 準確率 >= 60%")
else:
    print(f"\n→ 需要進一步優化...")

print(f"\nFiles Ready for Download:")
print(f"  1. formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras")
print(f"  2. scaler_advanced_{SYMBOL}_{INTERVAL}.json")
print(f"  3. metadata_{SYMBOL}_{INTERVAL}.json")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
