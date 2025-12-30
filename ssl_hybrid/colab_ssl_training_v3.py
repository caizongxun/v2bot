#!/usr/bin/env python3
"""
SSL Hybrid Signal Filter v3 Training
Improved signal definition + stronger features
"""

print("\n" + "="*80)
print("SSL HYBRID SIGNAL FILTER v3 - IMPROVED")
print("="*80)

print("\nImporting libraries...")
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
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
    print(f"  Error: {e}")
    print(f"  Using fallback data...")
    np.random.seed(42)
    n = 10000
    prices = 40000 + np.cumsum(np.random.randn(n) * 100).astype(np.float64)
    close = prices.copy()
    high = prices + np.abs(np.random.randn(n) * 50).astype(np.float64)
    low = np.maximum(prices - np.abs(np.random.randn(n) * 50).astype(np.float64), 100)
    volume = np.random.uniform(1000, 100000, n).astype(np.float64)

print(f"\nData shape: {len(close)} candles")

# Download and load v3 implementation
print(f"\nLoading SSL Hybrid v3...")
import requests

try:
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_hybrid_v3_improved.py'
    response = requests.get(url, timeout=30)
    exec(response.text, globals())
    print("  ✓ Loaded SSL Hybrid V3")
except Exception as e:
    print(f"  Error: {e}")
    raise

# Calculate SSL Hybrid v3
print(f"\nCalculating SSL Hybrid v3 indicators...")
try:
    indicator = SSLHybridV3(close, high, low, volume)
    print("  ✓ Calculated")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    raise

# Extract signals
print(f"\nExtracting multi-timeframe signals...")
try:
    signals = extract_signals_v3(close, high, low, volume, indicator)
    print(f"  Total signals: {len(signals)}")
    
    if len(signals) > 0:
        true_signals = [s for s in signals if s['is_true']]
        false_signals = [s for s in signals if not s['is_true']]
        
        print(f"  True signals:  {len(true_signals)} ({100*len(true_signals)/len(signals):.1f}%)")
        print(f"  False signals: {len(false_signals)} ({100*len(false_signals)/len(signals):.1f}%)")
        
        # Analysis
        if len(true_signals) > 0:
            true_returns = [s['avg_return'] for s in true_signals]
            print(f"\n  TRUE signals:")
            print(f"    Avg return: {np.mean(true_returns)*100:.3f}%")
            print(f"    Std dev: {np.std(true_returns)*100:.3f}%")
            print(f"    Min/Max: {np.min(true_returns)*100:.3f}% / {np.max(true_returns)*100:.3f}%")
        
        if len(false_signals) > 0:
            false_returns = [s['avg_return'] for s in false_signals]
            print(f"\n  FALSE signals:")
            print(f"    Avg return: {np.mean(false_returns)*100:.3f}%")
            print(f"    Std dev: {np.std(false_returns)*100:.3f}%")
            print(f"    Min/Max: {np.min(false_returns)*100:.3f}% / {np.max(false_returns)*100:.3f}%")
    else:
        print("  WARNING: No signals found!")
        
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    raise

# Extract features
print(f"\nExtracting improved features...")
try:
    features_list = []
    labels = []
    
    for signal in signals:
        try:
            feat = extract_features_v3(signal, close, high, low, volume, indicator)
            features_list.append(feat)
            labels.append(1 if signal['is_true'] else 0)
        except Exception as e:
            continue
    
    if len(features_list) == 0:
        raise ValueError("No features extracted!")
    
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(labels)
    
    print(f"  Features: {X.shape}")
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Labels: {len(y)} ({(y == 1).sum()} true, {(y == 0).sum()} false)")
    
    # Feature summary
    print(f"\n  Feature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"    {i:2d}. {name}")
        
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    raise

# Split data
print(f"\nSplitting data (64% train, 16% val, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"  Train: {X_train.shape[0]:4d} samples ({(y_train==1).sum():4d} true)")
print(f"  Val:   {X_val.shape[0]:4d} samples ({(y_val==1).sum():4d} true)")
print(f"  Test:  {X_test.shape[0]:4d} samples ({(y_test==1).sum():4d} true)")

# Normalize
print(f"\nNormalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Done")

# Build improved model
print(f"\nBuilding neural network...")
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    
    # First block
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    # Second block
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    # Third block
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    # Fourth block
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    
    # Output
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

print(f"  Parameters: {model.count_params():,}")

# Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print(f"  Class weights: {class_weight_dict}")

# Train
print(f"\nTraining (250 epochs with early stopping)...\n")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=30,
        restore_best_weights=True,
        mode='max',
        verbose=0
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-8,
        verbose=0
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=250,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=0
)

print("  ✓ Training completed")

# Evaluate
print(f"\nEvaluating on test set...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

acc = np.mean(y_pred == y_test)
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy: {acc:.4f}")
print(f"  AUC:      {auc:.4f}")
print(f"  F1-Score: {f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['False', 'True'], digits=3))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:4d} | FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d} | TP={cm[1,1]:4d}")

tnr = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
tpr = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
ppv = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
npv = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0

print(f"\nDiagnostic Performance:")
print(f"  False signal filtration rate (TNR): {tnr*100:.1f}%")
print(f"  True signal preservation (TPR):    {tpr*100:.1f}%")
print(f"  Positive Predictive Value (PPV):   {ppv*100:.1f}%")
print(f"  Negative Predictive Value (NPV):   {npv*100:.1f}%")

# Save
print(f"\nSaving model artifacts...")
model.save('ssl_filter_v3.keras')

with open('ssl_scaler_v3.json', 'w') as f:
    json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)

with open('ssl_metadata_v3.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'n_features': len(feature_names),
        'n_params': int(model.count_params()),
        'accuracy': float(acc),
        'auc': float(auc),
        'f1_score': float(f1),
        'sensitivity': float(tpr),
        'specificity': float(tnr),
        'ppv': float(ppv),
        'npv': float(npv),
        'n_signals': len(signals),
        'n_true': len(true_signals),
        'n_false': len(false_signals),
        'true_avg_return': float(np.mean([s['avg_return'] for s in true_signals])) if true_signals else 0,
        'false_avg_return': float(np.mean([s['avg_return'] for s in false_signals])) if false_signals else 0,
    }, f, indent=2)

print("  ✓ Saved")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nArtifacts:")
print(f"  1. ssl_filter_v3.keras")
print(f"  2. ssl_scaler_v3.json")
print(f"  3. ssl_metadata_v3.json")
print()
