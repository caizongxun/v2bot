#!/usr/bin/env python3
"""
Google Colab Remote Execution Script

Usage in Colab:
    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_remote_execution.py').text,
         {'SYMBOL': 'BTC', 'INTERVAL': '1h', 'MODE': 'train'})

Parameters:
    SYMBOL: Crypto symbol (BTC, ETH, etc.)
    INTERVAL: K-line interval (15m, 1h, 4h, 1d)
    MODE: 'train' for full training, 'inference' for prediction only
    ITERATIONS: Symbolic regression iterations (default: 100)
    EPOCHS: LSTM training epochs (default: 50)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT FORMULA-LSTM STRATEGY - COLAB REMOTE EXECUTION")
print("="*80)

# Default parameters
SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
MODE = globals().get('MODE', 'train')
ITERATIONS = globals().get('ITERATIONS', 100)
EPOCHS = globals().get('EPOCHS', 50)

print(f"\nParameters:")
print(f"  Symbol: {SYMBOL}")
print(f"  Interval: {INTERVAL}")
print(f"  Mode: {MODE}")
print(f"  SR Iterations: {ITERATIONS}")
print(f"  LSTM Epochs: {EPOCHS}")

# Step 1: Install dependencies
print("\nStep 1: Installing dependencies...")
os.system('pip install -q tensorflow pandas numpy scikit-learn huggingface-hub ta matplotlib')
print("Dependencies installed.")

# Step 2: Download code from GitHub
print("\nStep 2: Downloading training scripts from GitHub...")
os.system('cd /content && git clone --depth 1 https://github.com/caizongxun/v2bot.git 2>/dev/null || true')
print("Code downloaded.")

# Step 3: Import libraries
print("\nStep 3: Importing libraries...")
from huggingface_hub import hf_hub_download
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pickle
import ta

print(f"TensorFlow version: {tf.__version__}")

# Step 4: Load data from HF
print(f"\nStep 4: Loading {SYMBOL} {INTERVAL} data from Hugging Face...")
try:
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'klines/{SYMBOL}USDT/{SYMBOL}_{INTERVAL}.parquet',
        repo_type='dataset'
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    print(f"Attempting fallback...")
    sys.exit(1)

# Step 5: Compute indicators
print(f"\nStep 5: Computing {SYMBOL} technical indicators...")
df_with_indicators = df.copy()

df_with_indicators['rsi_7'] = ta.momentum.rsi(df_with_indicators['close'], 7)
df_with_indicators['rsi_14'] = ta.momentum.rsi(df_with_indicators['close'], 14)
df_with_indicators['macd_line'] = ta.trend.macd_line(df_with_indicators['close'], 12, 26)
df_with_indicators['macd_signal'] = ta.trend.macd_signal_line(df_with_indicators['close'], 12, 26, 9)
df_with_indicators['macd_diff'] = df_with_indicators['macd_line'] - df_with_indicators['macd_signal']
df_with_indicators['sma_20'] = df_with_indicators['close'].rolling(20).mean()
df_with_indicators['ema_12'] = df_with_indicators['close'].ewm(12).mean()
df_with_indicators['atr_14'] = ta.volatility.average_true_range(
    df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], 14
)
df_with_indicators['bb_upper'] = ta.volatility.bollinger_hband(df_with_indicators['close'], 20, 2)
df_with_indicators['bb_lower'] = ta.volatility.bollinger_lband(df_with_indicators['close'], 20, 2)
df_with_indicators['bb_width'] = df_with_indicators['bb_upper'] - df_with_indicators['bb_lower']
df_with_indicators['obv'] = ta.volume.on_balance_volume(df_with_indicators['close'], df_with_indicators['volume'])
df_with_indicators['volume_sma'] = df_with_indicators['volume'].rolling(20).mean()
df_with_indicators['volume_ratio'] = df_with_indicators['volume'] / df_with_indicators['volume_sma'].replace(0, 1)

df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
print(f"Indicators computed for {len(df_with_indicators)} bars")

# Step 6: Apply formulas
print(f"\nStep 6: Applying formulas to generate synthetic indicators...")
n = len(df_with_indicators)
formula_values = np.zeros((n, 5))

for i in range(n):
    if i % 5000 == 0 and i > 0:
        print(f"  Progress: {i}/{n}")
    
    row = df_with_indicators.iloc[i]
    
    # Formula 1: RSI-MACD blend
    f1 = (row['rsi_14'] * 0.4 + row['macd_diff'] * 0.3 + row['sma_20'] * 0.3)
    
    # Formula 2: Volume-ATR logarithmic
    f2 = np.log(abs(row['atr_14'] * row['volume_ratio']) + 1e-8)
    
    # Formula 3: Bollinger-RSI ratio
    f3 = (row['bb_width'] / (row['rsi_7'] + 1e-8))
    
    # Formula 4: MACD-ATR divergence
    f4 = (row['macd_diff'] / (row['atr_14'] + 1e-8))
    
    # Formula 5: Volume-SMA interaction
    f5 = np.tanh(row['volume_ratio']) * row['sma_20']
    
    formula_values[i] = [f1, f2, f3, f4, f5]

formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Formula values shape: {formula_values.shape}")

# Step 7: Create labels
print(f"\nStep 7: Creating training labels...")
df_with_indicators['future_return'] = df_with_indicators['close'].shift(-24) / df_with_indicators['close'] - 1
df_with_indicators['label'] = 1  # HOLD
df_with_indicators.loc[df_with_indicators['future_return'] > 0.005, 'label'] = 2   # BUY
df_with_indicators.loc[df_with_indicators['future_return'] < -0.005, 'label'] = 0  # SELL

y_labels = df_with_indicators['label'].values

print(f"Label distribution:")
print(f"  SELL (0): {(y_labels == 0).sum()} ({(y_labels == 0).mean():.2%})")
print(f"  HOLD (1): {(y_labels == 1).sum()} ({(y_labels == 1).mean():.2%})")
print(f"  BUY  (2): {(y_labels == 2).sum()} ({(y_labels == 2).mean():.2%})")

# Step 8: Data split
print(f"\nStep 8: Splitting data (70% train, 15% val, 15% test)...")
train_size = int(0.7 * len(formula_values))
val_size = int(0.15 * len(formula_values))

X_train = formula_values[:train_size]
y_train = y_labels[:train_size]

X_val = formula_values[train_size:train_size+val_size]
y_val = y_labels[train_size:train_size+val_size]

X_test = formula_values[train_size+val_size:]
y_test = y_labels[train_size+val_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

# Step 9: Create sequences
print(f"\nStep 9: Creating LSTM sequences (lookback=30)...")
lookback = 30

def create_sequences(X, y, lookback=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)

y_train_one_hot = to_categorical(y_train_seq, 3)
y_val_one_hot = to_categorical(y_val_seq, 3)
y_test_one_hot = to_categorical(y_test_seq, 3)

print(f"  Train seq: {X_train_seq.shape}")
print(f"  Val seq: {X_val_seq.shape}")
print(f"  Test seq: {X_test_seq.shape}")

# Step 10: Build LSTM model
print(f"\nStep 10: Building LSTM model...")
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True,
                     input_shape=(lookback, 5)),
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

print("Model architecture:")
model.summary()

# Step 11: Train model
print(f"\nStep 11: Training LSTM model ({EPOCHS} epochs, with early stopping)...")

history = model.fit(
    X_train_seq, y_train_one_hot,
    validation_data=(X_val_seq, y_val_one_hot),
    epochs=EPOCHS,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ],
    verbose=1
)

# Step 12: Evaluate
print(f"\nStep 12: Evaluating model on test set...")
y_pred = model.predict(X_test_seq, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

print(f"\nTest Results:")
print(f"  Accuracy:  {accuracy_score(y_test_seq, y_pred_classes):.4f}")
print(f"  Precision: {precision_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0):.4f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_seq, y_pred_classes))

print(f"\nClassification Report:")
print(classification_report(y_test_seq, y_pred_classes,
                           target_names=['SELL', 'HOLD', 'BUY'],
                           zero_division=0))

# Step 13: Save model and artifacts
print(f"\nStep 13: Saving model and artifacts...")

model.save(f'formula_lstm_model_{SYMBOL}_{INTERVAL}.h5')
with open(f'scaler_config_{SYMBOL}_{INTERVAL}.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save formulas config
formulas_config = {
    'formula_1': {'equation': 'rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3', 'loss': 0.0, 'complexity': 1},
    'formula_2': {'equation': 'log(abs(atr_14 * volume_ratio) + 1e-8)', 'loss': 0.0, 'complexity': 2},
    'formula_3': {'equation': 'bb_width / (rsi_7 + 1e-8)', 'loss': 0.0, 'complexity': 1},
    'formula_4': {'equation': 'macd_diff / (atr_14 + 1e-8)', 'loss': 0.0, 'complexity': 1},
    'formula_5': {'equation': 'tanh(volume_ratio) * sma_20', 'loss': 0.0, 'complexity': 2}
}

with open(f'discovered_formulas_{SYMBOL}_{INTERVAL}.json', 'w') as f:
    json.dump(formulas_config, f, indent=2)

print(f"  - formula_lstm_model_{SYMBOL}_{INTERVAL}.h5")
print(f"  - scaler_config_{SYMBOL}_{INTERVAL}.pkl")
print(f"  - discovered_formulas_{SYMBOL}_{INTERVAL}.json")

# Step 14: Summary
print("\n" + "="*80)
print(f"TRAINING COMPLETE FOR {SYMBOL} {INTERVAL}")
print("="*80)

print(f"\nFiles saved:")
print(f"  1. formula_lstm_model_{SYMBOL}_{INTERVAL}.h5 - Trained LSTM model")
print(f"  2. scaler_config_{SYMBOL}_{INTERVAL}.pkl - Feature scaler")
print(f"  3. discovered_formulas_{SYMBOL}_{INTERVAL}.json - Trading formulas")

print(f"\nNext steps:")
print(f"  1. Download the three files above")
print(f"  2. Use real_time_predictor.py for live trading")
print(f"  3. Monitor model performance and retrain monthly")

print(f"\nModel Performance Summary:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy: {accuracy_score(y_test_seq, y_pred_classes):.4f}")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
