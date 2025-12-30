#!/usr/bin/env python3
"""
V2Bot Advanced Training - Fixed Version
Fully compatible with all Colab environments

Usage:
    SYMBOL = 'BTC'
    INTERVAL = '1h'
    EPOCHS = 50
    LOOKBACK = 40
    
    import requests
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_advanced_training_fixed.py'
    script = requests.get(url).text
    exec(script, {'SYMBOL': SYMBOL, 'INTERVAL': INTERVAL, 'EPOCHS': EPOCHS, 'LOOKBACK': LOOKBACK})
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT ADVANCED TRAINING - FIXED VERSION")
print("="*80)

SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
EPOCHS = globals().get('EPOCHS', 50)
LOOKBACK = globals().get('LOOKBACK', 40)

print(f"\nParameters: {SYMBOL} {INTERVAL} | Epochs: {EPOCHS} | Lookback: {LOOKBACK}")

# Fix numpy compatibility
print("\nInitializing environment...")
import numpy as np
import pandas as pd
import json

print(f"  NumPy {np.__version__}")
print(f"  Pandas {pd.__version__}")

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print(f"  TensorFlow {tf.__version__}")

# Import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print("  Libraries initialized successfully")

# Scaler - simple numpy implementation
class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return (X - self.mean_) / self.std_
    
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_
    
    def to_dict(self):
        return {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist()
        }

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
    print(f"  Error: {e}")
    print(f"  Creating synthetic data...")
    np.random.seed(42)
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

print(f"\nComputing indicators...")

# RSI
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

# MACD
def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

# ATR
def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Compute indicators
df['rsi_7'] = compute_rsi(close, 7)
df['rsi_14'] = compute_rsi(close, 14)
df['rsi_21'] = compute_rsi(close, 21)

macd_line, macd_signal, macd_diff = compute_macd(close)
df['macd_line'] = macd_line
df['macd_signal'] = macd_signal
df['macd_diff'] = macd_diff

df['sma_10'] = close.rolling(10).mean()
df['sma_20'] = close.rolling(20).mean()
df['sma_50'] = close.rolling(50).mean()
df['ema_12'] = close.ewm(12).mean()
df['ema_26'] = close.ewm(26).mean()

df['atr_14'] = compute_atr(high, low, close, 14)

bb_sma = close.rolling(20).mean()
bb_std = close.rolling(20).std()
df['bb_upper'] = bb_sma + (bb_std * 2)
df['bb_lower'] = bb_sma - (bb_std * 2)
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_pct'] = (close - df['bb_lower']) / (df['bb_width'] + 1e-8)

df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
df['volume_sma'] = volume.rolling(20).mean()
df['volume_ratio'] = volume / (df['volume_sma'].replace(0, 1))
df['volatility'] = close.rolling(20).std() / close.rolling(20).mean()
df['natr'] = df['atr_14'] / close

# Advanced features
df['rsi_bb_ratio'] = df['rsi_14'] / (df['bb_width'] + 1e-8)
df['price_bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(float)
df['price_above_sma'] = (close > df['sma_20']).astype(float)

# Fill NaN
df = df.fillna(method='ffill').fillna(method='bfill')
print(f"  Indicators computed")

# Create labels
print(f"\nCreating smart labels...")

def create_smart_labels(close_prices, high_prices, low_prices, lookforward=24):
    """
    Create labels based on risk-adjusted returns
    """
    future_ret = close_prices.shift(-lookforward) / close_prices - 1
    max_dd = (low_prices.rolling(lookforward).min() - close_prices) / close_prices
    
    # Risk-adjusted return
    risk_adjusted = future_ret / (np.abs(max_dd) + 0.001)
    
    # Create labels based on quantiles
    p33 = risk_adjusted.quantile(0.33)
    p67 = risk_adjusted.quantile(0.67)
    
    labels = np.ones(len(close_prices), dtype=np.int32)
    labels[risk_adjusted > p67] = 2   # BUY
    labels[risk_adjusted < p33] = 0   # SELL
    
    return labels

labels = create_smart_labels(close, high, low, 24)
print(f"  SELL: {(labels == 0).sum()} | HOLD: {(labels == 1).sum()} | BUY: {(labels == 2).sum()}")

# Select features
feature_cols = [
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd_line', 'macd_signal', 'macd_diff',
    'sma_20', 'sma_50',
    'atr_14', 'bb_width', 'bb_pct', 'volatility',
    'volume_ratio', 'ema_cross', 'price_above_sma',
    'rsi_bb_ratio', 'price_bb_position'
]

df_features = df[feature_cols].dropna()
print(f"\nSelected {len(feature_cols)} features")

# Extract feature values
formula_values = df_features.values.astype(np.float32)
formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
labels = labels[:len(formula_values)]

print(f"  Dataset shape: {formula_values.shape}")

# Split data
print(f"\nSplitting data (70/15/15)...")
train_sz = int(0.7 * len(formula_values))
val_sz = int(0.15 * len(formula_values))

X_tr = formula_values[:train_sz]
y_tr = labels[:train_sz]
X_vl = formula_values[train_sz:train_sz+val_sz]
y_vl = labels[train_sz:train_sz+val_sz]
X_ts = formula_values[train_sz+val_sz:]
y_ts = labels[train_sz+val_sz:]

scaler = SimpleScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_vl_sc = scaler.transform(X_vl)
X_ts_sc = scaler.transform(X_ts)

print(f"  Train: {X_tr_sc.shape} | Val: {X_vl_sc.shape} | Test: {X_ts_sc.shape}")

# Create sequences
print(f"\nCreating sequences (lookback={LOOKBACK})...")

def create_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

X_tr_sq, y_tr_sq = create_sequences(X_tr_sc, y_tr, LOOKBACK)
X_vl_sq, y_vl_sq = create_sequences(X_vl_sc, y_vl, LOOKBACK)
X_ts_sq, y_ts_sq = create_sequences(X_ts_sc, y_ts, LOOKBACK)

y_tr_oh = to_categorical(y_tr_sq, 3)
y_vl_oh = to_categorical(y_vl_sq, 3)

print(f"  Sequences created")
print(f"  Train: {X_tr_sq.shape} | Val: {X_vl_sq.shape} | Test: {X_ts_sq.shape}")

# Compute class weights
print(f"\nComputing class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_tr_sq),
    y=y_tr_sq
)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print(f"  SELL={class_weight_dict[0]:.2f}, HOLD={class_weight_dict[1]:.2f}, BUY={class_weight_dict[2]:.2f}")

# Build model
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

# Callbacks
print(f"\nSetting up training...")
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

# Train
print(f"\nTraining ({EPOCHS} epochs)...\n")

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
print(f"\nClassification Report:")
print(classification_report(
    y_ts_sq, y_pred_class,
    target_names=['SELL', 'HOLD', 'BUY']
))

# Save
print(f"\nSaving artifacts...")

model.save(f'formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras')

with open(f'scaler_advanced_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(scaler.to_dict(), f)

metadata = {
    'lookback': LOOKBACK,
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
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
elif test_acc >= 0.60:
    print(f"\n✓ Good result! Accuracy >= 60%")
else:
    print(f"\n✓ Model ready for further optimization")

print(f"\nDownload these files:")
print(f"  1. formula_lstm_advanced_{SYMBOL}_{INTERVAL}.keras")
print(f"  2. scaler_advanced_{SYMBOL}_{INTERVAL}.json")
print(f"  3. metadata_{SYMBOL}_{INTERVAL}.json")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
