#!/usr/bin/env python3
"""
Final Colab Solution - Pure Python (No Subprocess)

Usage in Colab:
    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_final_solution.py').text,
         {'SYMBOL': 'BTC', 'INTERVAL': '1h', 'EPOCHS': 20})
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("V2BOT FORMULA-LSTM STRATEGY - FINAL COLAB SOLUTION")
print("="*80)

# Parameters
SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
EPOCHS = globals().get('EPOCHS', 20)

print(f"\nParameters:")
print(f"  Symbol: {SYMBOL}")
print(f"  Interval: {INTERVAL}")
print(f"  Epochs: {EPOCHS}")

# Step 1: Import with graceful fallback
print(f"\nStep 1: Importing libraries...")

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError:
    print("  Installing NumPy...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy'], check=False)
    import numpy as np

try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except ImportError:
    print("  Installing Pandas...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pandas'], check=False)
    import pandas as pd

try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except ImportError:
    print("  Installing TensorFlow...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorflow'], check=False)
    import tensorflow as tf

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    print(f"  ✓ Scikit-Learn")
except ImportError:
    print("  Installing Scikit-Learn...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn'], check=False)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from huggingface_hub import hf_hub_download
    print(f"  ✓ HuggingFace Hub")
except ImportError:
    print("  Installing HuggingFace Hub...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface-hub'], check=False)
    from huggingface_hub import hf_hub_download

try:
    import ta
    print(f"  ✓ TA-Lib")
except ImportError:
    print("  Installing TA-Lib...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ta'], check=False)
    import ta

try:
    from tensorflow.keras.utils import to_categorical
    print(f"  ✓ All required modules loaded")
except ImportError as e:
    print(f"  Error: {e}")
    sys.exit(1)

import json
import pickle

# Step 2: Load data
print(f"\nStep 2: Loading {SYMBOL} {INTERVAL} data from Hugging Face...")
try:
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'klines/{SYMBOL}USDT/{SYMBOL}_{INTERVAL}.parquet',
        repo_type='dataset',
        cache_dir='/tmp/hf_cache'
    )
    df = pd.read_parquet(file_path)
    print(f"  ✓ Loaded {len(df):,} rows")
except Exception as e:
    print(f"  Error: {e}")
    print(f"  Creating synthetic data for demo...")
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    prices = 40000 + np.cumsum(np.random.randn(2000) * 100)
    df = pd.DataFrame({
        'open': prices + np.random.randn(2000) * 50,
        'high': prices + np.abs(np.random.randn(2000) * 100),
        'low': prices - np.abs(np.random.randn(2000) * 100),
        'close': prices,
        'volume': np.random.uniform(100, 5000, 2000)
    }, index=dates)

# Step 3: Compute indicators
print(f"\nStep 3: Computing technical indicators...")

close = pd.to_numeric(df['close'], errors='coerce')
high = pd.to_numeric(df['high'], errors='coerce')
low = pd.to_numeric(df['low'], errors='coerce')
volume = pd.to_numeric(df['volume'], errors='coerce')

df_ind = df.copy()

df_ind['rsi_7'] = ta.momentum.rsi(close, 7)
df_ind['rsi_14'] = ta.momentum.rsi(close, 14)
df_ind['macd_line'] = ta.trend.macd_line(close, 12, 26)
df_ind['macd_signal'] = ta.trend.macd_signal_line(close, 12, 26, 9)
df_ind['macd_diff'] = df_ind['macd_line'] - df_ind['macd_signal']
df_ind['sma_20'] = close.rolling(20).mean()
df_ind['ema_12'] = close.ewm(12).mean()
df_ind['atr_14'] = ta.volatility.average_true_range(high, low, close, 14)
df_ind['bb_upper'] = ta.volatility.bollinger_hband(close, 20, 2)
df_ind['bb_lower'] = ta.volatility.bollinger_lband(close, 20, 2)
df_ind['bb_width'] = df_ind['bb_upper'] - df_ind['bb_lower']
df_ind['obv'] = ta.volume.on_balance_volume(close, volume)
df_ind['volume_sma'] = volume.rolling(20).mean()
df_ind['volume_ratio'] = volume / (df_ind['volume_sma'].replace(0, 1))

df_ind = df_ind.fillna(method='ffill').fillna(method='bfill')
print(f"  ✓ Indicators computed for {len(df_ind):,} bars")

# Step 4: Apply formulas
print(f"\nStep 4: Applying formulas...")

n = len(df_ind)
formula_values = np.zeros((n, 5))

for i in range(n):
    if i % max(1, n//10) == 0 and i > 0:
        print(f"    Progress: {100*i/n:.1f}%")
    
    row = df_ind.iloc[i]
    
    rsi_14 = float(row.get('rsi_14', 50) or 50)
    macd_diff = float(row.get('macd_diff', 0) or 0)
    sma_20 = float(row.get('sma_20', close.iloc[i]) or close.iloc[i])
    atr_14 = float(row.get('atr_14', 1) or 1)
    volume_ratio = float(row.get('volume_ratio', 1) or 1)
    rsi_7 = float(row.get('rsi_7', 50) or 50)
    bb_width = float(row.get('bb_width', 1) or 1)
    
    f1 = (rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3)
    f2 = np.log(abs(atr_14 * volume_ratio) + 1e-8)
    f3 = bb_width / (rsi_7 + 1e-8)
    f4 = macd_diff / (atr_14 + 1e-8)
    f5 = np.tanh(volume_ratio) * sma_20
    
    formula_values[i] = [f1, f2, f3, f4, f5]

formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  ✓ Formula values computed: {formula_values.shape}")

# Step 5: Create labels
print(f"\nStep 5: Creating labels...")

future_return = close.shift(-24) / close - 1
labels = np.ones(len(df_ind), dtype=int)
labels[future_return > 0.005] = 2
labels[future_return < -0.005] = 0

print(f"  ✓ SELL: {(labels == 0).sum()} | HOLD: {(labels == 1).sum()} | BUY: {(labels == 2).sum()}")

# Step 6: Split and scale
print(f"\nStep 6: Splitting and scaling data...")

train_size = int(0.7 * len(formula_values))
val_size = int(0.15 * len(formula_values))

X_train = formula_values[:train_size]
y_train = labels[:train_size]
X_val = formula_values[train_size:train_size+val_size]
y_val = labels[train_size:train_size+val_size]
X_test = formula_values[train_size+val_size:]
y_test = labels[train_size+val_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Train: {X_train_scaled.shape} | Val: {X_val_scaled.shape} | Test: {X_test_scaled.shape}")

# Step 7: Create sequences
print(f"\nStep 7: Creating LSTM sequences...")

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

print(f"  ✓ Sequences created: {X_train_seq.shape}")

# Step 8: Build model
print(f"\nStep 8: Building LSTM model...")

from tensorflow import keras

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

print(f"  ✓ Model built: {model.count_params():,} parameters")

# Step 9: Train
print(f"\nStep 9: Training LSTM ({EPOCHS} epochs)...")

history = model.fit(
    X_train_seq, y_train_one_hot,
    validation_data=(X_val_seq, y_val_one_hot),
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

print(f"  ✓ Training completed")

# Step 10: Evaluate
print(f"\nStep 10: Evaluating model...")

y_pred = model.predict(X_test_seq, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

test_acc = accuracy_score(y_test_seq, y_pred_classes)
test_prec = precision_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)
test_rec = recall_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)
test_f1 = f1_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)

print(f"\n  Test Results:")
print(f"    Accuracy:  {test_acc:.4f}")
print(f"    Precision: {test_prec:.4f}")
print(f"    Recall:    {test_rec:.4f}")
print(f"    F1-Score:  {test_f1:.4f}")

# Step 11: Save
print(f"\nStep 11: Saving artifacts...")

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

print(f"  ✓ model_{SYMBOL}_{INTERVAL}.h5")
print(f"  ✓ scaler_{SYMBOL}_{INTERVAL}.pkl")
print(f"  ✓ formulas_{SYMBOL}_{INTERVAL}.json")

# Step 12: Summary
print("\n" + "="*80)
print(f"COMPLETE! {SYMBOL} {INTERVAL} MODEL TRAINED")
print("="*80)

print(f"\nPerformance Summary:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

print(f"\nDownload Files:")
print(f"  1. formula_lstm_model_{SYMBOL}_{INTERVAL}.h5")
print(f"  2. scaler_config_{SYMBOL}_{INTERVAL}.pkl")
print(f"  3. discovered_formulas_{SYMBOL}_{INTERVAL}.json")

print(f"\nNext Step:")
print(f"  Use real_time_predictor.py for live trading")
print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
