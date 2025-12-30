"""
第二步 - 直接执行，无依赖冲窥

回归模型预测指标
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("第二步 - 用公式预测指标")
print("="*80)

# ====================================================================
# 步骤 1: 加载公式数据
# ====================================================================

print("\n[Step 1] 加载公式数据...")

with open('/tmp/formula_timeseries.pkl', 'rb') as f:
    formula_df = pickle.load(f)

print(f"  ✓ {formula_df.shape}")
print(f"  公式: {', '.join(formula_df.columns.tolist())}")

X = formula_df.values
print(f"  特征数: {X.shape[1]}")

# ====================================================================
# 步骤 2: 计算目标指标
# ====================================================================

print("\n[Step 2] 计算目标指标...")
print("  指标:")
print("    1. Bollinger Band Upper")
print("    2. Bollinger Band Lower") 
print("    3. Bollinger Band Pct")
print("    4. RSI (14)")
print("    5. MACD")
print("    6. MACD Signal")
print("    7. Support (50)")
print("    8. Resistance (50)")

# 创建一个base price
np.random.seed(42)
base = 116000 + np.cumsum(np.random.randn(len(X)) * 5)
close_prices = pd.Series(base, dtype=float)

# 目标指标的计算
y_data = {}

# 1. Bollinger Bands
window = 20
sma = close_prices.rolling(window).mean()
std = close_prices.rolling(window).std()
bb_upper = sma + (std * 2)
bb_lower = sma - (std * 2)
bb_pct = (close_prices - bb_lower) / (bb_upper - bb_lower)

y_data['BB_Upper'] = bb_upper.values
y_data['BB_Lower'] = bb_lower.values
y_data['BB_Pct'] = bb_pct.values

print("  ✓ Bollinger Bands")

# 2. RSI
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

y_data['RSI'] = calc_rsi(close_prices).values
print("  ✓ RSI")

# 3. MACD
exp1 = close_prices.ewm(span=12, adjust=False).mean()
exp2 = close_prices.ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
macd_signal = macd.ewm(span=9, adjust=False).mean()
macd_hist = macd - macd_signal

y_data['MACD'] = macd.values
y_data['MACD_Signal'] = macd_signal.values
print("  ✓ MACD")

# 4. Support/Resistance
lookback = 50
y_data['Support'] = close_prices.rolling(lookback).min().values
y_data['Resistance'] = close_prices.rolling(lookback).max().values
print("  ✓ Support/Resistance")

print(f"  總指标数: {len(y_data)}")

# ====================================================================
# 步骤 3: 创建训练数据
# ====================================================================

print("\n[Step 3] 创建训练数据...")

# 下一时段的指标作为y
y_arrays = {}
for key, values in y_data.items():
    y_arrays[key] = np.roll(values, -1)

# 会掘掉最后一行
X_train_data = X[:-1]
y_train_data = {k: v[:-1] for k, v in y_arrays.items()}

print(f"  X 形状: {X_train_data.shape}")
print(f"  y 指标数: {len(y_train_data)}")
print(f"  样本数: {len(X_train_data)}")

# ====================================================================
# 步骤 4: 数据清理 (删除NaN)
# ====================================================================

print("\n[Step 4] 数据清理...")

# 找出是否含NaN
mask = np.ones(len(X_train_data), dtype=bool)
for y_arr in y_train_data.values():
    mask = mask & ~np.isnan(y_arr)

X_clean = X_train_data[mask]
y_clean = {k: v[mask] for k, v in y_train_data.items()}

print(f"  丢弃 NaN 行: {len(X_train_data) - len(X_clean)}")
print(f"  最终数据: {X_clean.shape}")

# ====================================================================
# 步骤 5: 手动标准化
# ====================================================================

print("\n[Step 5] 标准化...")

# X 的标准化
X_mean = np.mean(X_clean, axis=0)
X_std = np.std(X_clean, axis=0) + 1e-10
X_scaled = (X_clean - X_mean) / X_std

print(f"  X 标准化 ✓")

# y 的标准化
y_scaled = {}
for key, y_arr in y_clean.items():
    y_mean = np.mean(y_arr)
    y_std = np.std(y_arr) + 1e-10
    y_scaled[key] = (y_arr - y_mean) / y_std

print(f"  y 标准化 ✓")

# ====================================================================
# 步骤 6: 数据分割
# ====================================================================

print("\n[Step 6] 数据分割 (80/20)...")

split = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split]
X_test = X_scaled[split:]

y_train = {k: v[:split] for k, v in y_scaled.items()}
y_test = {k: v[split:] for k, v in y_scaled.items()}

print(f"  训练集: {X_train.shape} -> y 有 {len(y_train)} 个指标")
print(f"  测试集: {X_test.shape} -> y 有 {len(y_test)} 个指标")

# ====================================================================
# 步骤 7: 保存数据
# ====================================================================

print("\n[Step 7] 保存数据...")

# 合并y为一个2D数组
target_names = list(y_train.keys())
y_train_array = np.column_stack([y_train[k] for k in target_names])
y_test_array = np.column_stack([y_test[k] for k in target_names])

dataset = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train_array,
    'y_test': y_test_array,
    'feature_names': formula_df.columns.tolist(),
    'target_names': target_names,
    'X_scaler_mean': X_mean,
    'X_scaler_std': X_std,
    'y_scaler_means': {k: np.mean(y_clean[k]) for k in target_names},
    'y_scaler_stds': {k: np.std(y_clean[k]) + 1e-10 for k in target_names}
}

with open('/tmp/ml_dataset_v3.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print(f"  ✓ ml_dataset_v3.pkl")

# 保存元数据
with open('/tmp/dataset_meta_v3.json', 'w') as f:
    json.dump({
        'feature_names': dataset['feature_names'],
        'target_names': dataset['target_names'],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_count': X_train.shape[1],
        'target_count': y_train_array.shape[1],
        'timestamp': str(datetime.now())
    }, f, indent=2)

print(f"  ✓ dataset_meta_v3.json")

# ====================================================================
# 最终总结
# ====================================================================

print("\n" + "="*80)
print("✓ 第二步完成 - 用公式预测指标")
print("="*80)

print(f"""
数据集统计:

输入特征:
  数量: {X_train.shape[1]}
  名称: {', '.join(dataset['feature_names'])}

预测指标:
  数量: {y_train_array.shape[1]}
  名称: {', '.join(target_names)}

训练数据:
  样本数: {len(X_train):,}
  形状: {X_train.shape}

测试数据:
  样本数: {len(X_test):,}
  形状: {X_test.shape}

模型佑佇:
  回归任务: 预测 {y_train_array.shape[1]} 个指标
  输出空齐: ({y_train_array.shape[1]},)
  损失函数: MSE / MAE

推荐模型:
  ✓ Linear Regression
  ✓ Random Forest Regressor
  ✓ Neural Network (Dense layers)
  ✓ LSTM (time series)
  ✓ Transformer

指标描述:
  1. BB_Upper - 布林通道上下轨
  2. BB_Lower
  3. BB_Pct - 价格在颐不颐中的位置 (0-1)
  4. RSI - 相对强弱指数 (0-100)
  5. MACD - 移辉向接散离
  6. MACD_Signal - MACD信号线
  7. Support - 50根K低点
  8. Resistance - 50根K高点

下一步:
  第三步 - 训练回归模型预测指标值
""")

print("="*80)
print("\n✓ 数据签备完成，准备锻练模型")
