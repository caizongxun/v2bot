#!/usr/bin/env python3
"""
Final Working Colab Solution - Compatible ta-lib API

Usage:
    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_final_working.py').text,
         {'SYMBOL': 'BTC', 'INTERVAL': '1h', 'EPOCHS': 20})
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT FORMULA-LSTM - FINAL WORKING SOLUTION")
print("="*80)

SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
EPOCHS = globals().get('EPOCHS', 20)

print(f"\nParameters: {SYMBOL} {INTERVAL} | Epochs: {EPOCHS}")

# Step 1: Import core
print("\nStep 1: Importing core libraries...")

import numpy as np
import pandas as pd
import json
import pickle

print(f"  ✓ NumPy {np.__version__}")
print(f"  ✓ Pandas {pd.__version__}")

# Step 2: TensorFlow
print("\nStep 2: Loading TensorFlow/Keras...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
print(f"  ✓ TensorFlow {tf.__version__}")

# Step 3: Custom scaler
print("\nStep 3: Initializing custom StandardScaler...")

class SimpleStandardScaler:
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

print("  ✓ Custom StandardScaler ready")

# Step 4: Load data
print("\nStep 4: Loading data from Hugging Face...")

try:
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'klines/{SYMBOL}USDT/{SYMBOL}_{INTERVAL}.parquet',
        repo_type='dataset',
        cache_dir='/tmp/hf_cache'
    )
    df = pd.read_parquet(file_path)
    print(f"  ✓ Loaded {len(df):,} rows")
except Exception as e:
    print(f"  Fallback: Creating synthetic data")
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    prices = 40000 + np.cumsum(np.random.randn(2000) * 100)
    df = pd.DataFrame({
        'open': prices + np.random.randn(2000) * 50,
        'high': prices + np.abs(np.random.randn(2000) * 100),
        'low': prices - np.abs(np.random.randn(2000) * 100),
        'close': prices,
        'volume': np.random.uniform(100, 5000, 2000)
    }, index=dates)

# Step 5: Custom indicators (avoid ta-lib compatibility issues)
print("\nStep 5: Computing technical indicators...")

close = pd.to_numeric(df['close'], errors='coerce')
high = pd.to_numeric(df['high'], errors='coerce')
low = pd.to_numeric(df['low'], errors='coerce')
volume = pd.to_numeric(df['volume'], errors='coerce')

df_ind = df.copy().astype(float)

# Manual indicator implementations
def rsi(prices, period=14):
    """Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(prices, fast=12, slow=26, signal=9):
    """MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger_bands(prices, period=20, num_std=2):
    """Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

# Calculate indicators
df_ind['rsi_7'] = rsi(close, 7)
df_ind['rsi_14'] = rsi(close, 14)
macd_line, signal_line, macd_diff = macd(close)
df_ind['macd_line'] = macd_line
df_ind['macd_signal'] = signal_line
df_ind['macd_diff'] = macd_diff
df_ind['sma_20'] = close.rolling(20).mean()
df_ind['ema_12'] = close.ewm(12).mean()
df_ind['atr_14'] = atr(high, low, close, 14)
df_ind['bb_upper'], df_ind['bb_lower'] = bollinger_bands(close, 20, 2)
df_ind['bb_width'] = df_ind['bb_upper'] - df_ind['bb_lower']
df_ind['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
df_ind['volume_sma'] = volume.rolling(20).mean()
df_ind['volume_ratio'] = volume / (df_ind['volume_sma'].replace(0, 1))

df_ind = df_ind.fillna(method='ffill').fillna(method='bfill')
print(f"  ✓ Computed for {len(df_ind):,} bars")

# Step 6: Apply formulas
print("\nStep 6: Applying 5 trading formulas...")

n = len(df_ind)
formula_values = np.zeros((n, 5), dtype=np.float32)

for i in range(n):
    if i % max(1, n//10) == 0 and i > 0:
        print(f"    {100*i/n:.0f}%", end=' ')
    
    row = df_ind.iloc[i]
    
    def safe_get(key, default=0):
        try:
            val = float(row[key])
            return val if val == val and np.isfinite(val) else default
        except:
            return default
    
    rsi_14 = safe_get('rsi_14', 50)
    macd_d = safe_get('macd_diff', 0)
    sma_20 = safe_get('sma_20', float(close.iloc[i]) if i < len(close) else 40000)
    atr_14 = safe_get('atr_14', 1)
    vol_ratio = safe_get('volume_ratio', 1)
    rsi_7 = safe_get('rsi_7', 50)
    bb_w = safe_get('bb_width', 1)
    
    # 5 formulas
    f1 = rsi_14 * 0.4 + macd_d * 0.3 + sma_20 * 0.3
    f2 = np.log(abs(atr_14 * vol_ratio) + 1e-8)
    f3 = bb_w / (rsi_7 + 1e-8)
    f4 = macd_d / (atr_14 + 1e-8)
    f5 = np.tanh(vol_ratio) * sma_20
    
    formula_values[i] = [f1, f2, f3, f4, f5]

formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
print("\n  ✓ Formulas applied")

# Step 7: Labels
print("\nStep 7: Creating labels...")

close_num = pd.to_numeric(df_ind['close'], errors='coerce')
future_ret = close_num.shift(-24) / close_num - 1
labels = np.ones(len(df_ind), dtype=np.int32)
labels[future_ret > 0.005] = 2
labels[future_ret < -0.005] = 0

print(f"  SELL: {(labels == 0).sum()} | HOLD: {(labels == 1).sum()} | BUY: {(labels == 2).sum()}")

# Step 8: Split
print("\nStep 8: Splitting and scaling...")

train_sz = int(0.7 * len(formula_values))
val_sz = int(0.15 * len(formula_values))

X_tr = formula_values[:train_sz]
y_tr = labels[:train_sz]
X_vl = formula_values[train_sz:train_sz+val_sz]
y_vl = labels[train_sz:train_sz+val_sz]
X_ts = formula_values[train_sz+val_sz:]
y_ts = labels[train_sz+val_sz:]

scaler = SimpleStandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_vl_sc = scaler.transform(X_vl)
X_ts_sc = scaler.transform(X_ts)

print(f"  Train: {X_tr_sc.shape} | Val: {X_vl_sc.shape} | Test: {X_ts_sc.shape}")

# Step 9: Sequences
print("\nStep 9: Creating LSTM sequences...")

lookback = 30

def create_seq(X, y, lb=30):
    Xs, ys = [], []
    for i in range(len(X) - lb):
        Xs.append(X[i:i+lb])
        ys.append(y[i+lb])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

X_tr_sq, y_tr_sq = create_seq(X_tr_sc, y_tr, lookback)
X_vl_sq, y_vl_sq = create_seq(X_vl_sc, y_vl, lookback)
X_ts_sq, y_ts_sq = create_seq(X_ts_sc, y_ts, lookback)

y_tr_oh = to_categorical(y_tr_sq, 3)
y_vl_oh = to_categorical(y_vl_sq, 3)

print(f"  ✓ Sequences: {X_tr_sq.shape}")

# Step 10: Build
print("\nStep 10: Building LSTM...")

model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 5)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(32, activation='relu', return_sequences=False),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  ✓ Model: {model.count_params():,} params")

# Step 11: Train
print(f"\nStep 11: Training ({EPOCHS} epochs)...")

history = model.fit(
    X_tr_sq, y_tr_oh,
    validation_data=(X_vl_sq, y_vl_oh),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
    ],
    verbose=1
)

print("  ✓ Done")

# Step 12: Eval
print("\nStep 12: Evaluating...")

y_pd = model.predict(X_ts_sq, verbose=0)
y_pc = np.argmax(y_pd, axis=1)
test_ac = np.mean(y_pc == y_ts_sq)
print(f"  Test Accuracy: {test_ac:.4f}")

# Step 13: Save
print("\nStep 13: Saving artifacts...")

model.save(f'formula_lstm_model_{SYMBOL}_{INTERVAL}.h5')
with open(f'scaler_config_{SYMBOL}_{INTERVAL}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

formulas = {
    'formula_1': {'equation': 'rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3'},
    'formula_2': {'equation': 'log(abs(atr_14 * volume_ratio) + 1e-8)'},
    'formula_3': {'equation': 'bb_width / (rsi_7 + 1e-8)'},
    'formula_4': {'equation': 'macd_diff / (atr_14 + 1e-8)'},
    'formula_5': {'equation': 'tanh(volume_ratio) * sma_20'}
}
with open(f'discovered_formulas_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(formulas, f, indent=2)

print(f"  ✓ formula_lstm_model_{SYMBOL}_{INTERVAL}.h5")
print(f"  ✓ scaler_config_{SYMBOL}_{INTERVAL}.pkl")
print(f"  ✓ discovered_formulas_{SYMBOL}_{INTERVAL}.json")

print("\n" + "="*80)
print(f"SUCCESS! {SYMBOL} {INTERVAL} MODEL TRAINED")
print("="*80)

print(f"\nPerformance:")
print(f"  Train: {history.history['accuracy'][-1]:.4f}")
print(f"  Val:   {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test:  {test_ac:.4f}")

print(f"\nFiles Ready for Download:")
print(f"  1. formula_lstm_model_{SYMBOL}_{INTERVAL}.h5")
print(f"  2. scaler_config_{SYMBOL}_{INTERVAL}.pkl")
print(f"  3. discovered_formulas_{SYMBOL}_{INTERVAL}.json")

print(f"\nNext: Download & use real_time_predictor.py for live trading")
print("="*80 + "\n")
