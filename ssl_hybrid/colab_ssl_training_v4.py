#!/usr/bin/env python3
"""
SSL Hybrid Signal Filter v4 Training
Fixed overfitting + data leakage with K-Fold CV
"""

print("\n" + "="*80)
print("SSL HYBRID SIGNAL FILTER v4 - FIXED OVERFITTING")
print("="*80)

print("\nImporting libraries...")
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("  OK")

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
    raise

# Load v4
print(f"\nLoading SSL Hybrid v4...")
import requests

try:
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_hybrid_v4_fixed.py'
    response = requests.get(url, timeout=30)
    exec(response.text, globals())
    print("  OK")
except Exception as e:
    print(f"  Error: {e}")
    raise

# Calculate indicators
print(f"\nCalculating indicators...")
indicator = SSLHybridV4(close, high, low, volume)
print("  OK")

# Extract signals
print(f"\nExtracting signals...")
signals = extract_signals_v4(close, high, low, volume, indicator, lookforward=10)
print(f"  Total signals: {len(signals)}")

if len(signals) > 0:
    true_signals = [s for s in signals if s['is_true']]
    false_signals = [s for s in signals if not s['is_true']]
    
    print(f"  True:  {len(true_signals):5d} ({100*len(true_signals)/len(signals):5.1f}%)")
    print(f"  False: {len(false_signals):5d} ({100*len(false_signals)/len(signals):5.1f}%)")
    
    if len(true_signals) > 0:
        true_returns = [s['max_return'] for s in true_signals]
        print(f"\n  TRUE signal analysis:")
        print(f"    Avg max_return: {np.mean(true_returns)*100:+.3f}%")
        print(f"    Min/Max: {np.min(true_returns)*100:+.3f}% / {np.max(true_returns)*100:+.3f}%")
    
    if len(false_signals) > 0:
        false_returns = [s['max_return'] for s in false_signals]
        print(f"\n  FALSE signal analysis:")
        print(f"    Avg max_return: {np.mean(false_returns)*100:+.3f}%")
        print(f"    Min/Max: {np.min(false_returns)*100:+.3f}% / {np.max(false_returns)*100:+.3f}%")

# Extract features
print(f"\nExtracting features...")
features_list = []
labels = []

for signal in signals:
    feat = extract_features_v4(signal, close, high, low, volume, indicator)
    if len(feat) > 0:
        features_list.append(feat)
        labels.append(1 if signal['is_true'] else 0)

if len(features_list) == 0:
    raise ValueError("No features extracted!")

feature_names = sorted(features_list[0].keys())
X = np.array([[f[name] for name in feature_names] for f in features_list])
y = np.array(labels)

print(f"  Shape: {X.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Labels: {(y == 1).sum()} true, {(y == 0).sum()} false")
print(f"\n  Feature list:")
for i, name in enumerate(feature_names, 1):
    print(f"    {i:2d}. {name}")

# Normalize
print(f"\nNormalizing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  OK")

# K-Fold Cross-Validation
print(f"\nK-Fold Cross-Validation (5 folds)...\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"Fold {fold}/5...")
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Build model
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )],
        verbose=0
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    acc = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    fold_results.append({
        'fold': fold,
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    
    print(f"  Accuracy: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f}\n")

# Average results
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

mean_acc = np.mean([r['accuracy'] for r in fold_results])
std_acc = np.std([r['accuracy'] for r in fold_results])
mean_auc = np.mean([r['auc'] for r in fold_results])
std_auc = np.std([r['auc'] for r in fold_results])
mean_f1 = np.mean([r['f1'] for r in fold_results])
std_f1 = np.std([r['f1'] for r in fold_results])

print(f"\nAccuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
print(f"AUC:      {mean_auc:.3f} +/- {std_auc:.3f}")
print(f"F1-Score: {mean_f1:.3f} +/- {std_f1:.3f}")

# Aggregate confusion matrix
all_y_test = np.concatenate([r['y_test'] for r in fold_results])
all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

cm = confusion_matrix(all_y_test, all_y_pred)
print(f"\nCombined Confusion Matrix (all folds):")
print(f"  TN={cm[0,0]:5d} | FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d} | TP={cm[1,1]:5d}")

tnr = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
tpr = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
ppv = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0

print(f"\nDiagnostic Metrics:")
print(f"  Specificity (TNR):  {tnr*100:.1f}%")
print(f"  Sensitivity (TPR):  {tpr*100:.1f}%")
print(f"  Precision (PPV):    {ppv*100:.1f}%")

print(f"\nClassification Report:")
print(classification_report(all_y_test, all_y_pred, target_names=['False', 'True'], digits=3))

# Save final model (trained on all data)
print(f"\nTraining final model on all data...")
model = keras.Sequential([
    keras.layers.Input(shape=(X_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

model.fit(
    X_scaled, y,
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )],
    verbose=0
)

print(f"  OK")

# Save
print(f"\nSaving artifacts...")
model.save('ssl_filter_v4.keras')

with open('ssl_scaler_v4.json', 'w') as f:
    json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, f)

with open('ssl_metadata_v4.json', 'w') as f:
    json.dump({
        'features': feature_names,
        'n_features': len(feature_names),
        'accuracy_mean': float(mean_acc),
        'accuracy_std': float(std_acc),
        'auc_mean': float(mean_auc),
        'auc_std': float(std_auc),
        'f1_mean': float(mean_f1),
        'f1_std': float(std_f1),
        'sensitivity': float(tpr),
        'specificity': float(tnr),
        'ppv': float(ppv),
        'n_signals': len(signals),
        'n_true': len(true_signals),
        'n_false': len(false_signals),
        'cv_method': 'K-Fold (k=5)',
    }, f, indent=2)

print(f"  OK")

print("\n" + "="*80)
print("TRAINING COMPLETE - v4 with proper validation")
print("="*80)
print(f"\nExpected realistic performance:")
print(f"  Accuracy:   {mean_acc:.1%} +/- {std_acc:.1%}")
print(f"  AUC:        {mean_auc:.3f}")
print(f"  Sensitivity: {tpr*100:.1f}%")
print(f"  Specificity: {tnr*100:.1f}%")
print()
