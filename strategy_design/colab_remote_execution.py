#!/usr/bin/env python3
"""
Google Colab Remote Execution Script (Fixed)

Usage in Colab:
    import requests
    exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_remote_execution_fixed.py').text,
         {'SYMBOL': 'BTC', 'INTERVAL': '1h', 'MODE': 'train'})

Parameters:
    SYMBOL: Crypto symbol (BTC, ETH, etc.)
    INTERVAL: K-line interval (15m, 1h, 4h, 1d)
    MODE: 'train' for full training, 'inference' for prediction only
    EPOCHS: LSTM training epochs (default: 50)
"""

import os
import sys
import subprocess

print("\n" + "="*80)
print("V2BOT FORMULA-LSTM STRATEGY - COLAB REMOTE EXECUTION (FIXED)")
print("="*80)

# Get parameters
SYMBOL = globals().get('SYMBOL', 'BTC')
INTERVAL = globals().get('INTERVAL', '1h')
MODE = globals().get('MODE', 'train')
EPOCHS = globals().get('EPOCHS', 20)

print(f"\nParameters:")
print(f"  Symbol: {SYMBOL}")
print(f"  Interval: {INTERVAL}")
print(f"  Mode: {MODE}")
print(f"  LSTM Epochs: {EPOCHS}")

# Step 1: Fix NumPy and install clean dependencies
print("\nStep 1: Fixing dependencies (NumPy compatibility)...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', '-q', 'numpy==1.24.3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tensorflow==2.13.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pandas==1.5.3'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn==1.3.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface-hub==0.16.4'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ta==0.10.2'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib==3.7.1'])
print("Dependencies fixed and installed.")

# Step 2: Now import all libraries
print("\nStep 2: Importing libraries...")
try:
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    
    from huggingface_hub import hf_hub_download
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    import pickle
    import json
    import ta
    import matplotlib.pyplot as plt
    
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ Pandas version: {pd.__version__}")
    print(f"✓ TensorFlow version: {tf.__version__}")
    print("✓ All libraries imported successfully")
    
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Attempting to fix...")
    sys.exit(1)

# Step 3: Load data
print(f"\nStep 3: Loading {SYMBOL} {INTERVAL} data from Hugging Face...")
try:
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'klines/{SYMBOL}USDT/{SYMBOL}_{INTERVAL}.parquet',
        repo_type='dataset',
        cache_dir='/tmp/hf_cache'
    )
    df = pd.read_parquet(file_path)
    print(f"✓ Loaded {len(df):,} rows of {SYMBOL} {INTERVAL} data")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
except Exception as e:
    print(f"Error loading data: {e}")
    print(f"Using synthetic data for demonstration...")
    # Create synthetic data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, 1000),
        'high': np.random.uniform(40000, 45000, 1000),
        'low': np.random.uniform(40000, 45000, 1000),
        'close': np.random.uniform(40000, 45000, 1000),
        'volume': np.random.uniform(100, 5000, 1000)
    }, index=dates)

# Step 4: Ensure data quality
print(f"\nStep 4: Validating and cleaning data...")
df = df.fillna(method='ffill').fillna(method='bfill')
df = df[df['close'] > 0]  # Remove invalid prices
if len(df) < 100:
    print(f"Error: Insufficient data ({len(df)} rows, need at least 100)")
    sys.exit(1)
print(f"✓ Data validated: {len(df):,} valid rows")

# Step 5: Compute indicators
print(f"\nStep 5: Computing technical indicators...")
try:
    # Ensure close price is numeric
    close = pd.to_numeric(df['close'], errors='coerce')
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    volume = pd.to_numeric(df['volume'], errors='coerce')
    
    df_with_indicators = df.copy()
    
    # Momentum indicators
    df_with_indicators['rsi_7'] = ta.momentum.rsi(close, 7)
    df_with_indicators['rsi_14'] = ta.momentum.rsi(close, 14)
    
    # Trend indicators
    df_with_indicators['macd_line'] = ta.trend.macd_line(close, 12, 26)
    df_with_indicators['macd_signal'] = ta.trend.macd_signal_line(close, 12, 26, 9)
    df_with_indicators['macd_diff'] = df_with_indicators['macd_line'] - df_with_indicators['macd_signal']
    df_with_indicators['sma_20'] = close.rolling(20).mean()
    df_with_indicators['ema_12'] = close.ewm(12).mean()
    
    # Volatility indicators
    df_with_indicators['atr_14'] = ta.volatility.average_true_range(high, low, close, 14)
    df_with_indicators['bb_upper'] = ta.volatility.bollinger_hband(close, 20, 2)
    df_with_indicators['bb_lower'] = ta.volatility.bollinger_lband(close, 20, 2)
    df_with_indicators['bb_width'] = df_with_indicators['bb_upper'] - df_with_indicators['bb_lower']
    
    # Volume indicators
    df_with_indicators['obv'] = ta.volume.on_balance_volume(close, volume)
    df_with_indicators['volume_sma'] = volume.rolling(20).mean()
    df_with_indicators['volume_ratio'] = volume / (df_with_indicators['volume_sma'].replace(0, 1))
    
    # Fill NaN values
    df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✓ Indicators computed for {len(df_with_indicators):,} bars")
    
except Exception as e:
    print(f"Error computing indicators: {e}")
    sys.exit(1)

# Step 6: Apply formulas
print(f"\nStep 6: Applying formulas to generate synthetic indicators...")
n = len(df_with_indicators)
formula_values = np.zeros((n, 5))

try:
    for i in range(n):
        if i % max(1, n//10) == 0 and i > 0:
            print(f"  Progress: {i:,}/{n:,} ({100*i/n:.1f}%)")
        
        row = df_with_indicators.iloc[i]
        
        # Safe access with defaults
        rsi_14 = float(row.get('rsi_14', 50) or 50)
        macd_diff = float(row.get('macd_diff', 0) or 0)
        sma_20 = float(row.get('sma_20', close.iloc[i]) or close.iloc[i])
        atr_14 = float(row.get('atr_14', 1) or 1)
        volume_ratio = float(row.get('volume_ratio', 1) or 1)
        rsi_7 = float(row.get('rsi_7', 50) or 50)
        bb_width = float(row.get('bb_width', 1) or 1)
        
        # Formula 1: RSI-MACD blend
        f1 = (rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3)
        
        # Formula 2: Volume-ATR logarithmic
        f2 = np.log(abs(atr_14 * volume_ratio) + 1e-8)
        
        # Formula 3: Bollinger-RSI ratio
        f3 = bb_width / (rsi_7 + 1e-8)
        
        # Formula 4: MACD-ATR divergence
        f4 = macd_diff / (atr_14 + 1e-8)
        
        # Formula 5: Volume-SMA interaction
        f5 = np.tanh(volume_ratio) * sma_20
        
        formula_values[i] = [f1, f2, f3, f4, f5]
    
    formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"✓ Formula values computed: shape {formula_values.shape}")
    
except Exception as e:
    print(f"Error applying formulas: {e}")
    sys.exit(1)

# Step 7: Create labels
print(f"\nStep 7: Creating training labels...")
try:
    close_numeric = pd.to_numeric(df_with_indicators['close'], errors='coerce')
    future_return = close_numeric.shift(-24) / close_numeric - 1
    
    labels = np.ones(len(df_with_indicators), dtype=int)  # Default HOLD
    labels[future_return > 0.005] = 2   # BUY
    labels[future_return < -0.005] = 0  # SELL
    
    print(f"✓ Label distribution:")
    print(f"  SELL (0): {(labels == 0).sum():,} ({(labels == 0).mean():.1%})")
    print(f"  HOLD (1): {(labels == 1).sum():,} ({(labels == 1).mean():.1%})")
    print(f"  BUY  (2): {(labels == 2).sum():,} ({(labels == 2).mean():.1%})")
    
except Exception as e:
    print(f"Error creating labels: {e}")
    sys.exit(1)

# Step 8: Split and scale data
print(f"\nStep 8: Splitting and scaling data...")
try:
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
    
    print(f"✓ Data split:")
    print(f"  Train: {X_train_scaled.shape}")
    print(f"  Val:   {X_val_scaled.shape}")
    print(f"  Test:  {X_test_scaled.shape}")
    
except Exception as e:
    print(f"Error splitting data: {e}")
    sys.exit(1)

# Step 9: Create LSTM sequences
print(f"\nStep 9: Creating LSTM sequences (lookback=30)...")
try:
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
    
    print(f"✓ Sequences created:")
    print(f"  Train: {X_train_seq.shape}")
    print(f"  Val:   {X_val_seq.shape}")
    print(f"  Test:  {X_test_seq.shape}")
    
except Exception as e:
    print(f"Error creating sequences: {e}")
    sys.exit(1)

# Step 10: Build LSTM model
print(f"\nStep 10: Building LSTM model...")
try:
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
    
    print(f"✓ Model built successfully")
    print(f"  Total parameters: {model.count_params():,}")
    
except Exception as e:
    print(f"Error building model: {e}")
    sys.exit(1)

# Step 11: Train model
print(f"\nStep 11: Training LSTM model ({EPOCHS} epochs)...")
print(f"  (This may take 10-60 minutes depending on data size)")
try:
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
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            )
        ],
        verbose=1
    )
    print(f"✓ Training completed")
    
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# Step 12: Evaluate
print(f"\nStep 12: Evaluating model on test set...")
try:
    y_pred = model.predict(X_test_seq, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    test_acc = accuracy_score(y_test_seq, y_pred_classes)
    test_prec = precision_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)
    test_rec = recall_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_seq, y_pred_classes, average='weighted', zero_division=0)
    
    print(f"✓ Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    print(f"\n✓ Confusion Matrix:")
    cm = confusion_matrix(y_test_seq, y_pred_classes)
    print(cm)
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    sys.exit(1)

# Step 13: Save artifacts
print(f"\nStep 13: Saving model and artifacts...")
try:
    model.save(f'formula_lstm_model_{SYMBOL}_{INTERVAL}.h5')
    with open(f'scaler_config_{SYMBOL}_{INTERVAL}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    formulas_config = {
        'formula_1': {'equation': 'rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3'},
        'formula_2': {'equation': 'log(abs(atr_14 * volume_ratio) + 1e-8)'},
        'formula_3': {'equation': 'bb_width / (rsi_7 + 1e-8)'},
        'formula_4': {'equation': 'macd_diff / (atr_14 + 1e-8)'},
        'formula_5': {'equation': 'tanh(volume_ratio) * sma_20'}
    }
    
    with open(f'discovered_formulas_{SYMBOL}_{INTERVAL}.json', 'w') as f:
        json.dump(formulas_config, f, indent=2)
    
    print(f"✓ Files saved:")
    print(f"  - formula_lstm_model_{SYMBOL}_{INTERVAL}.h5")
    print(f"  - scaler_config_{SYMBOL}_{INTERVAL}.pkl")
    print(f"  - discovered_formulas_{SYMBOL}_{INTERVAL}.json")
    
except Exception as e:
    print(f"Error saving files: {e}")
    sys.exit(1)

# Step 14: Summary
print("\n" + "="*80)
print(f"TRAINING COMPLETE FOR {SYMBOL} {INTERVAL}")
print("="*80)

print(f"\nModel Performance:")
print(f"  Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")

print(f"\nNext Steps:")
print(f"  1. Download the three .h5, .pkl, .json files above")
print(f"  2. Use real_time_predictor.py for live trading")
print(f"  3. Monitor performance and retrain monthly")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
