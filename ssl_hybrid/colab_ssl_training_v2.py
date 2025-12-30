#!/usr/bin/env python3
"""
SSL Hybrid Signal Filter Training v2
Fixed SSL Hybrid + Proper signal extraction
"""

print("\n" + "="*80)
print("SSL HYBRID SIGNAL FILTER v2")
print("="*80)

print("\nImporting libraries...")
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("  ✓ Done")

# Load data
print(f"\nLoading BTC 1h data...")
try:
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename='klines/BTCUSDT/BTC_1h.parquet',
        repo_type='dataset',
        cache_dir='/tmp/hf_cache'
    )
    import pyarrow.parquet as pq
    table = pq.read_table(file_path)
    df_dict = table.to_pandas().to_dict('list')
    
    close = np.array(df_dict['close'], dtype=np.float64)
    high = np.array(df_dict['high'], dtype=np.float64)
    low = np.array(df_dict['low'], dtype=np.float64)
    volume = np.array(df_dict['volume'], dtype=np.float64)
    
    print(f"  Loaded {len(close):,} candles")
except Exception as e:
    print(f"  Error loading data: {e}")
    print(f"  Creating synthetic data...")
    np.random.seed(42)
    n = 5000
    prices = 40000 + np.cumsum(np.random.randn(n) * 100).astype(np.float64)
    close = prices.copy()
    high = prices + np.abs(np.random.randn(n) * 50).astype(np.float64)
    low = np.maximum(prices - np.abs(np.random.randn(n) * 50).astype(np.float64), 100)
    volume = np.random.uniform(1000, 100000, n).astype(np.float64)
    print(f"  Created {n} synthetic candles")

print(f"\nData validation:")
print(f"  close: {len(close)}")
print(f"  high: {len(high)}")
print(f"  low: {len(low)}")
print(f"  volume: {len(volume)}")

# Download and load fixed SSL Hybrid
print(f"\nLoading SSL Hybrid implementation...")
import requests

try:
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_hybrid_fixed.py'
    response = requests.get(url, timeout=30)
    exec(response.text, globals())
    print("  ✓ Downloaded and loaded")
except Exception as e:
    print(f"  Error: {e}")
    print("  Please check github connection")
    raise

# Calculate SSL Hybrid
print(f"\nCalculating SSL Hybrid...")
try:
    indicator = SSLHybrid(
        close, high, low, volume,
        baseline_type="HMA",
        baseline_len=60,
        ssl2_type="JMA",
        ssl2_len=5,
        exit_type="HMA",
        exit_len=15,
        atr_len=14,
        atr_mult=1.0,
        channel_mult=0.2,
        atr_crit=0.9
    )
    print("  ✓ Calculated successfully")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    raise

# Extract signals
print(f"\nExtracting signals...")
try:
    signals = extract_signals(close, high, low, volume, indicator, lookforward=5)
    print(f"  Total signals: {len(signals)}")
    
    if len(signals) > 0:
        true_signals = [s for s in signals if s['is_true']]
        false_signals = [s for s in signals if not s['is_true']]
        
        print(f"  True signals:  {len(true_signals)} ({100*len(true_signals)/len(signals):.1f}%)")
        print(f"  False signals: {len(false_signals)} ({100*len(false_signals)/len(signals):.1f}%)")
        
        # Analyze false signals
        if len(false_signals) > 0:
            false_returns = [s['return'] for s in false_signals]
            print(f"\n  False signal analysis:")
            print(f"    Avg return: {np.mean(false_returns)*100:.2f}%")
            print(f"    Std dev: {np.std(false_returns)*100:.2f}%")
        
        if len(true_signals) > 0:
            true_returns = [s['return'] for s in true_signals]
            print(f"\n  True signal analysis:")
            print(f"    Avg return: {np.mean(true_returns)*100:.2f}%")
            print(f"    Std dev: {np.std(true_returns)*100:.2f}%")
    else:
        print("  WARNING: No signals found!")
        
except Exception as e:
    print(f"  Error extracting signals: {e}")
    import traceback
    traceback.print_exc()
    raise

# Extract features
print(f"\nExtracting features for each signal...")
try:
    features_list = []
    labels = []
    
    for signal in signals:
        try:
            feat = extract_features(signal, close, high, low, volume, indicator, lookback=40)
            features_list.append(feat)
            labels.append(1 if signal['is_true'] else 0)
        except Exception as e:
            print(f"  Error on signal {signal['index']}: {e}")
            continue
    
    if len(features_list) == 0:
        raise ValueError("No features extracted!")
    
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(labels)
    
    print(f"  Features extracted: {X.shape}")
    print(f"  Feature names: {feature_names}")
    print(f"  Labels: {len(y)} ({(y == 1).sum()} true, {(y == 0).sum()} false)")
    
    # Ensure minimum data
    if len(X) < 50:
        print(f"  WARNING: Only {len(X)} signals, need at least 50 for proper training")
        print(f"  Consider using longer historical data")
        
except Exception as e:
    print(f"  Error extracting features: {e}")
    import traceback
    traceback.print_exc()
    raise

# Split data
print(f"\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"  Train: {X_train.shape} ({(y_train==1).sum()} true)")
print(f"  Val:   {X_val.shape} ({(y_val==1).sum()} true)")
print(f"  Test:  {X_test.shape} ({(y_test==1).sum()} true)")

# Normalize
print(f"\nNormalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Done")

# Build model
print(f"\nBuilding model...")
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(),
        keras.metrics.Precision(),
        keras.metrics.Recall()
    ]
)

print(f"  Parameters: {model.count_params():,}")

# Class weights
from sklearn.utils.class_weight import compute_class_weight
try:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"  Class weights: {class_weight_dict}")
except:
    class_weight_dict = None
    print("  No class weighting")

# Train
print(f"\nTraining...\n")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=0
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n  Training completed")

# Evaluate
print(f"\nEvaluating...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

acc = np.mean(y_pred == y_test)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"  Accuracy: {acc:.4f}")
print(f"  AUC: {auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['False', 'True']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]} FP={cm[0,1]}")
print(f"  FN={cm[1,0]} TP={cm[1,1]}")

tnr = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
tpr = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0

print(f"\nPerformance:")
print(f"  False signal filtration: {tnr*100:.1f}%")
print(f"  True signal preservation: {tpr*100:.1f}%")

# Save
print(f"\nSaving...")
model.save('ssl_filter_v2.keras')

with open('ssl_scaler_v2.json', 'w') as f:
    json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)

with open('ssl_metadata_v2.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'n_features': len(feature_names),
        'n_params': int(model.count_params()),
        'accuracy': float(acc),
        'auc': float(auc),
        'false_filter_rate': float(tnr),
        'true_keep_rate': float(tpr),
        'n_signals': len(signals),
        'n_true': len(true_signals) if len(signals) > 0 else 0,
        'n_false': len(false_signals) if len(signals) > 0 else 0,
    }, f, indent=2)

print("  ✓ Saved")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nDownload:")
print(f"  1. ssl_filter_v2.keras")
print(f"  2. ssl_scaler_v2.json")
print(f"  3. ssl_metadata_v2.json")
print()
