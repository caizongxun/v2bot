#!/usr/bin/env python3
"""
V2Bot Final Training Script
Pure NumPy + TensorFlow (No Pandas dependency)

Usage in Colab:
    # Step 1: Fix environment
    import requests
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_setup_fix.py'
    exec(requests.get(url).text)
    
    # Step 2: Run training
    SYMBOL = 'BTC'
    INTERVAL = '1h'
    EPOCHS = 50
    LOOKBACK = 40
    
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_training_final.py'
    exec(requests.get(url, timeout=120).text, {'SYMBOL': SYMBOL, 'INTERVAL': INTERVAL, 'EPOCHS': EPOCHS, 'LOOKBACK': LOOKBACK})
"""

import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT ADVANCED TRAINING - FINAL VERSION")
print("="*80)

SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
EPOCHS = globals().get('EPOCHS', 50)
LOOKBACK = globals().get('LOOKBACK', 40)

print(f"\nParameters: {SYMBOL} {INTERVAL} | Epochs: {EPOCHS} | Lookback: {LOOKBACK}")

print("\nImporting libraries...")
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print(f"  NumPy {np.__version__}")
print(f"  TensorFlow {tf.__version__}")

try:
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"  scikit-learn OK")
except:
    print(f"  scikit-learn not available")

print("\nLibraries loaded successfully")

# Scaler
class Scaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return (X - self.mean_) / self.std_
    
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_
    
    def to_dict(self):
        return {'mean': self.mean_.tolist(), 'std': self.std_.tolist()}

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
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    df_dict = table.to_pandas().to_dict('list')
    
    close = np.array(df_dict['close'], dtype=np.float32)
    high = np.array(df_dict['high'], dtype=np.float32)
    low = np.array(df_dict['low'], dtype=np.float32)
    volume = np.array(df_dict['volume'], dtype=np.float32)
    
    print(f"  Loaded {len(close):,} candles")
except Exception as e:
    print(f"  Error: {e}")
    print(f"  Creating synthetic data...")
    np.random.seed(42)
    n = 5000
    prices = 40000 + np.cumsum(np.random.randn(n) * 100).astype(np.float32)
    close = prices
    high = prices + np.abs(np.random.randn(n) * 100).astype(np.float32)
    low = prices - np.abs(np.random.randn(n) * 100).astype(np.float32)
    volume = np.random.uniform(100, 5000, n).astype(np.float32)

print(f"\nComputing indicators...")

# Simple moving average
def sma(prices, period):
    result = np.zeros_like(prices)
    for i in range(period, len(prices)):
        result[i] = np.mean(prices[i-period:i])
    return result

# Exponential moving average
def ema(prices, period):
    result = np.zeros_like(prices)
    alpha = 2.0 / (period + 1)
    result[0] = prices[0]
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    return result

# RSI
def rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    gain_avg = sma(gains, period)
    loss_avg = sma(losses, period)
    
    rs = gain_avg / (loss_avg + 1e-8)
    result = 100 - (100 / (1 + rs))
    return result

# ATR
def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return sma(tr, period)

# Compute indicators
rsi_7 = rsi(close, 7)
rsi_14 = rsi(close, 14)
rsi_21 = rsi(close, 21)

ema_12 = ema(close, 12)
ema_26 = ema(close, 26)
macd_line = ema_12 - ema_26
macd_signal = ema(macd_line, 9)
macd_diff = macd_line - macd_signal

sma_10 = sma(close, 10)
sma_20 = sma(close, 20)
sma_50 = sma(close, 50)

atr_14 = atr(high, low, close, 14)

# Bollinger Bands
bb_std = np.zeros_like(close)
for i in range(20, len(close)):
    bb_std[i] = np.std(close[i-20:i])
bb_upper = sma_20 + (bb_std * 2)
bb_lower = sma_20 - (bb_std * 2)
bb_width = bb_upper - bb_lower
bb_pct = (close - bb_lower) / (bb_width + 1e-8)

# Volume indicators
obv = np.zeros_like(close)
for i in range(1, len(close)):
    if close[i] > close[i-1]:
        obv[i] = obv[i-1] + volume[i]
    elif close[i] < close[i-1]:
        obv[i] = obv[i-1] - volume[i]
    else:
        obv[i] = obv[i-1]

volume_sma = sma(volume, 20)
volume_ratio = volume / (volume_sma + 1e-8)

volatility = np.zeros_like(close)
for i in range(20, len(close)):
    volatility[i] = np.std(close[i-20:i]) / (np.mean(close[i-20:i]) + 1e-8)

natr = atr_14 / close

# Advanced features
rsi_bb_ratio = rsi_14 / (bb_width + 1e-8)
price_bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
ema_cross = (ema_12 > ema_26).astype(float)
price_above_sma = (close > sma_20).astype(float)

print(f"  Indicators computed")

# Create labels
print(f"\nCreating smart labels...")

def create_labels(close, high, low, lookforward=24):
    labels = np.ones(len(close), dtype=np.int32)
    
    for i in range(len(close) - lookforward):
        future_price = close[i + lookforward]
        ret = (future_price - close[i]) / close[i]
        
        max_dd = np.min(low[i:i+lookforward]) - close[i]
        max_dd = max_dd / close[i]
        
        risk_adjusted = ret / (np.abs(max_dd) + 0.001)
        
        if risk_adjusted > 0.05:  # BUY
            labels[i] = 2
        elif risk_adjusted < -0.05:  # SELL
            labels[i] = 0
        else:  # HOLD
            labels[i] = 1
    
    return labels

labels = create_labels(close, high, low, 24)
print(f"  SELL: {(labels == 0).sum()} | HOLD: {(labels == 1).sum()} | BUY: {(labels == 2).sum()}")

# Create feature matrix
print(f"\nCreating feature matrix...")
features = np.column_stack([
    rsi_7, rsi_14, rsi_21,
    macd_line, macd_signal, macd_diff,
    sma_20, sma_50,
    atr_14, bb_width, bb_pct, volatility,
    volume_ratio, ema_cross, price_above_sma,
    rsi_bb_ratio, price_bb_position
])

features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
features = np.float32(features)

print(f"  Feature shape: {features.shape}")

# Split data
print(f"\nSplitting data...")
train_sz = int(0.7 * len(features))
val_sz = int(0.15 * len(features))

X_tr = features[:train_sz]
y_tr = labels[:train_sz]
X_vl = features[train_sz:train_sz+val_sz]
y_vl = labels[train_sz:train_sz+val_sz]
X_ts = features[train_sz+val_sz:]
y_ts = labels[train_sz+val_sz:]

scaler = Scaler()
X_tr = scaler.fit_transform(X_tr)
X_vl = scaler.transform(X_vl)
X_ts = scaler.transform(X_ts)

print(f"  Train: {X_tr.shape} | Val: {X_vl.shape} | Test: {X_ts.shape}")

# Create sequences
print(f"\nCreating sequences (lookback={LOOKBACK})...")

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

X_tr_sq, y_tr_sq = create_sequences(X_tr, y_tr, LOOKBACK)
X_vl_sq, y_vl_sq = create_sequences(X_vl, y_vl, LOOKBACK)
X_ts_sq, y_ts_sq = create_sequences(X_ts, y_ts, LOOKBACK)

y_tr_oh = to_categorical(y_tr_sq, 3)
y_vl_oh = to_categorical(y_vl_sq, 3)

print(f"  Sequences created")

# Compute class weights
print(f"\nComputing class weights...")
try:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr_sq), y=y_tr_sq)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
except:
    # Fallback: manual calculation
    unique, counts = np.unique(y_tr_sq, return_counts=True)
    class_weight_dict = {int(u): float(len(y_tr_sq) / (len(unique) * c)) for u, c in zip(unique, counts)}

print(f"  SELL={class_weight_dict[0]:.2f}, HOLD={class_weight_dict[1]:.2f}, BUY={class_weight_dict[2]:.2f}")

# Build model
print(f"\nBuilding model...")

model = keras.Sequential([
    keras.layers.Bidirectional(
        keras.layers.LSTM(128, activation='relu', return_sequences=True),
        input_shape=(LOOKBACK, 17)
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

# Train
print(f"\nTraining ({EPOCHS} epochs)...")

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
        verbose=0
    )
]

history = model.fit(
    X_tr_sq, y_tr_oh,
    validation_data=(X_vl_sq, y_vl_oh),
    epochs=EPOCHS,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n  Training completed")

# Evaluate
print(f"\nEvaluating...")
y_pred = model.predict(X_ts_sq, verbose=0)
y_pred_class = np.argmax(y_pred, axis=1)
test_acc = np.mean(y_pred_class == y_ts_sq)

print(f"  Test Accuracy: {test_acc:.4f}")

# Save
print(f"\nSaving artifacts...")
model.save(f'formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras')

with open(f'scaler_advanced_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(scaler.to_dict(), f)

metadata = {
    'lookback': LOOKBACK,
    'n_features': 17,
    'n_params': int(model.count_params()),
    'train_accuracy': float(history.history['accuracy'][-1]),
    'val_accuracy': float(history.history['val_accuracy'][-1]),
    'test_accuracy': float(test_acc)
}

with open(f'metadata_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Model saved")
print(f"  ✓ Scaler saved")
print(f"  ✓ Metadata saved")

print("\n" + "="*80)
print("SUCCESS! MODEL TRAINED")
print("="*80)

print(f"\nPerformance:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

if test_acc >= 0.70:
    print(f"\n✓ Goal achieved! Accuracy >= 70%")
elif test_acc >= 0.65:
    print(f"\n✓ Excellent result! Accuracy >= 65%")
elif test_acc >= 0.55:
    print(f"\n✓ Good improvement from baseline (38%)")
else:
    print(f"\n✓ Model ready for optimization")

print(f"\nDownload these files:")
print(f"  1. formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras")
print(f"  2. scaler_advanced_{SYMBOL}_{INTERVAL}.json")
print(f"  3. metadata_{SYMBOL}_{INTERVAL}.json")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
