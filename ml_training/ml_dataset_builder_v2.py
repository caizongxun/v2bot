"""
ML模型训练数据集构建器 - 第二步

功能:
1. 加载9个公式的时间序列
2. 加载标签化的K线数据
3. 对齐时间戳和数据
4. 处理缺失值和异常值
5. 创建训练/测试集
6. 生成特征工程后的数据集
7. 进行特征标准化和归一化
8. 输出用于ML模型的最终数据集
"""

import os
import sys

print("[Setup] Installing compatible dependencies...")
os.system('pip install --upgrade -q numpy pandas scikit-learn')

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ML DATASET BUILDER - FORMULA VALUES + PRICE LABELS")
print("="*80)

# ====================================================================
# STEP 1: 加载公式时间序列
# ====================================================================

print("\n[STEP 1] Load formula timeseries...")

try:
    with open('/tmp/formula_timeseries.pkl', 'rb') as f:
        formula_df = pickle.load(f)
    
    print(f"[Loader] Formula data shape: {formula_df.shape}")
    print(f"[Loader] Formulas: {', '.join(formula_df.columns.tolist())}")
    
except Exception as e:
    print(f"[ERROR] Loading pickle failed: {e}")
    print("[INFO] Trying to load from CSV...")
    try:
        formula_df = pd.read_csv('/tmp/formula_timeseries.csv')
        print(f"[Loader] Loaded from CSV: {formula_df.shape}")
    except Exception as e2:
        print(f"[ERROR] Both methods failed: {e2}")
        raise

# ====================================================================
# STEP 2: 加载标签化的K线数据
# ====================================================================

print("\n[STEP 2] Load labeled klines data...")

try:
    labeled_df = pd.read_csv('/tmp/labeled_klines_phase1.csv')
    print(f"[Loader] Labeled data shape: {labeled_df.shape}")
    print(f"[Loader] Columns: {labeled_df.columns.tolist()}")
    
    # 转换时间戳
    labeled_df['timestamp'] = pd.to_datetime(labeled_df['timestamp'])
    labeled_df = labeled_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"[Loader] Time range: {labeled_df['timestamp'].min()} to {labeled_df['timestamp'].max()}")
    print(f"[Loader] Unique labels: {labeled_df['Reversal_Label'].unique()}")
    
except Exception as e:
    print(f"[ERROR] Loading labeled data: {e}")
    raise

# ====================================================================
# STEP 3: 准备数据对齐
# ====================================================================

print("\n[STEP 3] Data alignment preparation...")

# 公式数据需要索引，假设是连续的
print(f"[Info] Formula data rows: {len(formula_df)}")
print(f"[Info] Labeled data rows: {len(labeled_df)}")

# 计算对齐策略
data_len = min(len(formula_df), len(labeled_df))
print(f"[Info] Using minimum length: {data_len} rows")

# 截断到相同长度
formula_df_aligned = formula_df.iloc[:data_len].copy()
labeled_df_aligned = labeled_df.iloc[:data_len].copy()

print(f"[Aligned] Formula: {formula_df_aligned.shape}")
print(f"[Aligned] Labels: {labeled_df_aligned.shape}")

# ====================================================================
# STEP 4: 合并数据集
# ====================================================================

print("\n[STEP 4] Merge datasets...")

# 重置索引
formula_df_aligned.reset_index(drop=True, inplace=True)
labeled_df_aligned.reset_index(drop=True, inplace=True)

# 合并
merged_df = pd.concat([
    labeled_df_aligned[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'Regime', 'Reversal_Label', 'Reversal_Type', 'Reversal_Strength']],
    formula_df_aligned
], axis=1)

print(f"[Merged] Shape: {merged_df.shape}")
print(f"[Merged] Columns: {len(merged_df.columns)}")

# ====================================================================
# STEP 5: 处理缺失值和异常值
# ====================================================================

print("\n[STEP 5] Handle missing values and outliers...")

missing_counts = merged_df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"[Missing values] Total: {missing_counts.sum()}")
    # 填充缺失值（向前填充）
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.fillna(method='bfill')
    print(f"[After filling] Missing values: {merged_df.isnull().sum().sum()}")
else:
    print("[Missing values] None found")

# ====================================================================
# STEP 6: 创建特征工程
# ====================================================================

print("\n[STEP 6] Feature engineering...")

# 创建目标变量（1=上涨, 0=平盘, -1=下跌）
merged_df['price_change'] = merged_df['close'].diff()
merged_df['future_return'] = merged_df['close'].pct_change().shift(-1) * 100

# 三分类标签（基于未来收益）
def create_label(future_return):
    if pd.isna(future_return):
        return np.nan
    elif future_return > 0.05:  # 上涨阈值
        return 1
    elif future_return < -0.05:  # 下跌阈值
        return -1
    else:
        return 0

merged_df['target_label'] = merged_df['future_return'].apply(create_label)

print(f"[Features] Created target label")

# 提取原始公式列
formula_cols = formula_df_aligned.columns.tolist()
print(f"[Features] Formula columns: {len(formula_cols)}")

# 创建滞后特征（前N个周期的公式值）
print(f"[Features] Creating lag features...")
for col in formula_cols:
    for lag in [1, 2, 3, 5]:
        merged_df[f'{col}_lag{lag}'] = merged_df[col].shift(lag)

# 创建滚动统计特征
print(f"[Features] Creating rolling statistics...")
for col in formula_cols:
    merged_df[f'{col}_ma5'] = merged_df[col].rolling(5).mean()
    merged_df[f'{col}_std5'] = merged_df[col].rolling(5).std()
    merged_df[f'{col}_ma20'] = merged_df[col].rolling(20).mean()

print(f"[Features] Total columns after engineering: {merged_df.shape[1]}")

# ====================================================================
# STEP 7: 删除包含NaN的行
# ====================================================================

print("\n[STEP 7] Remove rows with NaN...")

initial_rows = len(merged_df)
merged_df = merged_df.dropna()
final_rows = len(merged_df)

print(f"[Cleaned] Rows removed: {initial_rows - final_rows}")
print(f"[Cleaned] Final rows: {final_rows}")

# ====================================================================
# STEP 8: 分离特征和目标
# ====================================================================

print("\n[STEP 8] Separate features and target...")

# 特征列（排除ID和目标列）
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'Regime', 'Reversal_Label', 'Reversal_Type', 'Reversal_Strength',
                'price_change', 'future_return', 'target_label']

feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
print(f"[Features] Total feature columns: {len(feature_cols)}")

X = merged_df[feature_cols].values
y = merged_df['target_label'].values
timestamps = merged_df['timestamp'].values

print(f"[Data] X shape: {X.shape}")
print(f"[Data] y shape: {y.shape}")
print(f"[Target] Distribution:")
target_dist = pd.Series(y).value_counts().sort_index()
for label, count in target_dist.items():
    pct = count / len(y) * 100
    label_name = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}.get(label, str(label))
    print(f"  {label_name:5s}: {count:6d} ({pct:5.2f}%)")

# ====================================================================
# STEP 9: 特征标准化
# ====================================================================

print("\n[STEP 9] Feature scaling...")

# 使用StandardScaler进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"[Scaling] Mean of scaled features: {X_scaled.mean(axis=0)[:5].round(4)}")
print(f"[Scaling] Std of scaled features:  {X_scaled.std(axis=0)[:5].round(4)}")

# ====================================================================
# STEP 10: 划分训练集和测试集
# ====================================================================

print("\n[STEP 10] Train-test split...")

# 使用时间序列分割（80% 训练，20% 测试）
split_idx = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]
ts_train = timestamps[:split_idx]
ts_test = timestamps[split_idx:]

print(f"[Split] Train: {X_train.shape}")
print(f"[Split] Test:  {X_test.shape}")
print(f"[Split] Train labels distribution:")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
for label, pct in train_dist.items():
    label_name = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}.get(label, str(label))
    print(f"  {label_name:5s}: {pct*100:5.2f}%")

print(f"[Split] Test labels distribution:")
test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
for label, pct in test_dist.items():
    label_name = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}.get(label, str(label))
    print(f"  {label_name:5s}: {pct*100:5.2f}%")

# ====================================================================
# STEP 11: 保存数据集
# ====================================================================

print("\n[STEP 11] Save datasets...")

# 保存为pickle格式（用于快速加载）
dataset = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'ts_train': ts_train,
    'ts_test': ts_test,
    'feature_names': feature_cols,
    'scaler': scaler,
    'formula_pool': formula_cols,
}

with open('/tmp/ml_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print("[Save] ML dataset -> /tmp/ml_dataset.pkl")

# 保存特征名称
with open('/tmp/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("[Save] Feature names -> /tmp/feature_names.json")

# 保存训练集信息
train_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'feature_count': len(feature_cols),
    'formula_count': len(formula_cols),
    'target_distribution': {
        'train': pd.Series(y_train).value_counts().to_dict(),
        'test': pd.Series(y_test).value_counts().to_dict()
    },
    'feature_scaling': {
        'method': 'StandardScaler',
        'mean': X_scaled.mean(axis=0).tolist()[:10],
        'std': X_scaled.std(axis=0).tolist()[:10]
    }
}

with open('/tmp/dataset_info.json', 'w') as f:
    json.dump(train_info, f, indent=2, default=str)

print("[Save] Dataset info -> /tmp/dataset_info.json")

# 保存为CSV用于其他工具
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df['timestamp'] = ts_train
train_df['target'] = y_train
train_df.to_csv('/tmp/train_dataset.csv', index=False)

test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df['timestamp'] = ts_test
test_df['target'] = y_test
test_df.to_csv('/tmp/test_dataset.csv', index=False)

print("[Save] Train dataset CSV -> /tmp/train_dataset.csv")
print("[Save] Test dataset CSV  -> /tmp/test_dataset.csv")

# ====================================================================
# STEP 12: 生成数据分析报告
# ====================================================================

print("\n[STEP 12] Generate analysis report...")

report = f"""
ML DATASET BUILDER REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY
{'-'*80}

1. Input Data:
   - Formula time series: {len(formula_df)} rows x {len(formula_cols)} formulas
   - Labeled klines: {len(labeled_df)} rows
   - Aligned data: {data_len} rows

2. Feature Engineering:
   - Base formulas: {len(formula_cols)}
   - Lag features (1,2,3,5): {len(formula_cols) * 4}
   - Rolling statistics (MA5, Std5, MA20): {len(formula_cols) * 3}
   - Total features: {len(feature_cols)}

3. Data Cleaning:
   - Rows with missing values removed: {initial_rows - final_rows}
   - Final dataset size: {final_rows}
   - Missing value handling: Forward-fill + backward-fill

DATASET SPLIT
{'-'*80}

Training Set: {len(X_train)} rows ({len(X_train)/len(X_scaled)*100:.1f}%)
Test Set:     {len(X_test)} rows ({len(X_test)/len(X_scaled)*100:.1f}%)

CLASS DISTRIBUTION
{'-'*80}

Training Set:
  -1 (DOWN):  {(y_train == -1).sum():6d} ({(y_train == -1).sum()/len(y_train)*100:5.2f}%)
   0 (FLAT):  {(y_train == 0).sum():6d} ({(y_train == 0).sum()/len(y_train)*100:5.2f}%)
  +1 (UP):    {(y_train == 1).sum():6d} ({(y_train == 1).sum()/len(y_train)*100:5.2f}%)

Test Set:
  -1 (DOWN):  {(y_test == -1).sum():6d} ({(y_test == -1).sum()/len(y_test)*100:5.2f}%)
   0 (FLAT):  {(y_test == 0).sum():6d} ({(y_test == 0).sum()/len(y_test)*100:5.2f}%)
  +1 (UP):    {(y_test == 1).sum():6d} ({(y_test == 1).sum()/len(y_test)*100:5.2f}%)

FEATURE STATISTICS
{'-'*80}

Scaled Feature Ranges (StandardScaler):
  Mean: {X_scaled.mean(axis=0).mean():.6f}
  Std:  {X_scaled.std(axis=0).mean():.6f}
  Min:  {X_scaled.min():.6f}
  Max:  {X_scaled.max():.6f}

OUTPUT FILES
{'-'*80}

1. ml_dataset.pkl
   - Complete dataset with train/test split
   - Includes scaler and metadata
   - Format: Python pickle

2. train_dataset.csv
   - Training features with target labels
   - Rows: {len(X_train)}
   - Columns: {len(feature_cols) + 2}

3. test_dataset.csv
   - Test features with target labels
   - Rows: {len(X_test)}
   - Columns: {len(feature_cols) + 2}

4. feature_names.json
   - List of all feature column names
   - For reference and model interpretation

5. dataset_info.json
   - Metadata and statistics
   - Train/test sizes and distributions

NEXT STEPS
{'-'*80}

1. Train machine learning models:
   - LightGBM for classification
   - XGBoost for classification
   - Neural Network (LSTM/GRU) for time series
   - Random Forest for interpretability

2. Model evaluation:
   - Cross-validation
   - Confusion matrix
   - ROC-AUC curves
   - Feature importance analysis

3. Hyperparameter tuning:
   - Grid search or Bayesian optimization
   - Early stopping on validation set
   - Monitor for overfitting

4. Model deployment:
   - Save best model
   - Create prediction pipeline
   - Generate real-time trading signals
"""

with open('/tmp/ML_DATASET_REPORT.txt', 'w') as f:
    f.write(report)

print("[Save] Report -> /tmp/ML_DATASET_REPORT.txt")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("ML DATASET BUILDER COMPLETE")
print("="*80)

print(f"""

成功创建ML训练数据集

输出文件:
  1. ml_dataset.pkl          - 完整数据集 (Python pickle)
  2. train_dataset.csv       - 训练集 ({len(X_train)} rows x {len(feature_cols)+2} cols)
  3. test_dataset.csv        - 测试集 ({len(X_test)} rows x {len(feature_cols)+2} cols)
  4. feature_names.json      - 特征名称列表
  5. dataset_info.json       - 数据集信息和统计
  6. ML_DATASET_REPORT.txt   - 完整报告

数据集信息:
  - 总特征数: {len(feature_cols)}
  - 训练样本: {len(X_train)}
  - 测试样本: {len(X_test)}
  - 目标类别: 3 (DOWN/-1, FLAT/0, UP/+1)

特征构成:
  - 原始公式: {len(formula_cols)}
  - 滞后特征 (lag 1-5): {len(formula_cols) * 4}
  - 滚动统计: {len(formula_cols) * 3}
  - 其他特征: {len(feature_cols) - len(formula_cols) * (1 + 4 + 3)}

下一步:
  准备进行第三步 - 机器学习模型训练
  选择合适的模型框架:
    - LightGBM (推荐,速度快)
    - XGBoost (高精度)
    - Neural Networks (深度学习)
    - Ensemble (组合模型)

""")

print("="*80)
