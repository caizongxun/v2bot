"""
最终版本 - 自动检测文件位置
支持 /tmp, /content, 当前目录等多个位置
"""

import subprocess
import sys
import os
import glob

print("[Setup] 使用当前环境，最小化依赖")

try:
    import pickle
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    print("[Info] 所有模块已可用")
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    raise

try:
    from sklearn.preprocessing import StandardScaler
    USE_SKLEARN = True
except:
    print("[Warn] StandardScaler 不可用，使用手动标准化")
    USE_SKLEARN = False
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit_transform(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0) + 1e-10
            return (X - self.mean_) / self.scale_
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_

print("="*80)
print("ML DATASET BUILDER - 第二步")
print("="*80)

# ====================================================================
# 文件位置检测
# ====================================================================

print("\n[Setup] 检测文件位置...")

def find_file(filename, search_dirs=None):
    """在多个目录中搜索文件"""
    if search_dirs is None:
        search_dirs = ['/tmp', '/content', '.', os.path.expanduser('~')]
    
    for directory in search_dirs:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            print(f"  找到: {filepath}")
            return filepath
        
        # 也尝试模糊搜索
        pattern = os.path.join(directory, f'*{filename}')
        matches = glob.glob(pattern)
        if matches:
            print(f"  找到: {matches[0]}")
            return matches[0]
    
    return None

# 查找公式文件
formula_pkl_path = find_file('formula_timeseries.pkl')
if not formula_pkl_path:
    raise FileNotFoundError("找不到 formula_timeseries.pkl")

# 查找标签文件
labeled_csv_path = find_file('labeled_klines_phase1.csv')
if not labeled_csv_path:
    # 尝试其他可能的名字
    labeled_csv_path = find_file('labeled_klines.csv')
if not labeled_csv_path:
    raise FileNotFoundError("找不到 labeled_klines_phase1.csv")

# ====================================================================
# STEP 1: 加载公式时间序列
# ====================================================================

print("\n[Step 1] 加载公式时间序列...")

try:
    with open(formula_pkl_path, 'rb') as f:
        formula_df = pickle.load(f)
    print(f"  ✓ 公式数据: {formula_df.shape}")
    print(f"    列: {', '.join(formula_df.columns.tolist()[:3])}...")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    raise

# ====================================================================
# STEP 2: 加载标签数据
# ====================================================================

print("\n[Step 2] 加载标签数据...")

try:
    labeled_df = pd.read_csv(labeled_csv_path)
    labeled_df['timestamp'] = pd.to_datetime(labeled_df['timestamp'])
    print(f"  ✓ 标签数据: {labeled_df.shape}")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    raise

# ====================================================================
# STEP 3: 数据对齐
# ====================================================================

print("\n[Step 3] 数据对齐...")

data_len = min(len(formula_df), len(labeled_df))
print(f"  对齐长度: {data_len}")

formula_df = formula_df.iloc[:data_len].reset_index(drop=True)
labeled_df = labeled_df.iloc[:data_len].reset_index(drop=True)

print(f"  ✓ 对齐完成")

# ====================================================================
# STEP 4: 合并数据集
# ====================================================================

print("\n[Step 4] 合并数据集...")

merged_df = pd.concat([
    labeled_df[['timestamp', 'close']],
    formula_df
], axis=1)

print(f"  ✓ 合并后: {merged_df.shape}")

# ====================================================================
# STEP 5: 创建会标签
# ====================================================================

print("\n[Step 5] 创建会标签...")

merged_df['future_return'] = merged_df['close'].pct_change().shift(-1) * 100

def create_label(ret):
    if pd.isna(ret):
        return np.nan
    elif ret > 0.05:  # 上涨
        return 1
    elif ret < -0.05:  # 下跌
        return -1
    else:  # 平盘
        return 0

merged_df['target'] = merged_df['future_return'].apply(create_label)

target_dist = merged_df['target'].value_counts().to_dict()
print(f"  ✓ 目标分布: {target_dist}")

# ====================================================================
# STEP 6: 特征工程
# ====================================================================

print("\n[Step 6] 特征工程...")

formula_cols = formula_df.columns.tolist()
print(f"  原始公式: {len(formula_cols)}")

# 滞后特征
print(f"  添加滞后特征 (lag 1,2,3,5)...", end='')
for col in formula_cols:
    for lag in [1, 2, 3, 5]:
        merged_df[f'{col}_lag{lag}'] = merged_df[col].shift(lag)
print(f" ✓")

# 滚动统计
print(f"  添加滚动统计 (MA5, Std5, MA20)...", end='')
for col in formula_cols:
    merged_df[f'{col}_ma5'] = merged_df[col].rolling(5).mean()
    merged_df[f'{col}_std5'] = merged_df[col].rolling(5).std()
    merged_df[f'{col}_ma20'] = merged_df[col].rolling(20).mean()
print(f" ✓")

total_features = merged_df.shape[1] - 2 - 1  # 排除 timestamp, close, target
print(f"  总特征数: {total_features}")

# ====================================================================
# STEP 7: 数据清理
# ====================================================================

print("\n[Step 7] 数据清理...")

initial_len = len(merged_df)
merged_df = merged_df.dropna()
final_len = len(merged_df)

print(f"  移除行数: {initial_len - final_len}")
print(f"  最终行数: {final_len}")

# ====================================================================
# STEP 8: 提取特征
# ====================================================================

print("\n[Step 8] 提取特征...")

exclude_cols = ['timestamp', 'close', 'future_return', 'target']
feature_cols = [c for c in merged_df.columns if c not in exclude_cols]

X = merged_df[feature_cols].values
y = merged_df['target'].values

print(f"  特征数: {len(feature_cols)}")
print(f"  样本数: {len(X)}")
print(f"  X 形状: {X.shape}")
print(f"  y 形状: {y.shape}")

# ====================================================================
# STEP 9: 特征标准化
# ====================================================================

print("\n[Step 9] 特征标准化...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ 标准化完成")
if USE_SKLEARN:
    print(f"    使用: sklearn StandardScaler")
else:
    print(f"    使用: 手动标准化")

# ====================================================================
# STEP 10: 数据分割
# ====================================================================

print("\n[Step 10] 数据分割...")

split_idx = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")

# 分布统计
train_dist = {}
test_dist = {}
for label in [-1, 0, 1]:
    train_dist[label] = int((y_train == label).sum())
    test_dist[label] = int((y_test == label).sum())

print(f"  训练集分布: {train_dist}")
print(f"  测试集分布: {test_dist}")

# ====================================================================
# STEP 11: 保存数据集
# ====================================================================

print("\n[Step 11] 保存数据集...")

# Pickle
dataset = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': feature_cols,
    'scaler': scaler,
    'formula_cols': formula_cols
}

with open('/tmp/ml_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("  ✓ ml_dataset.pkl")

# JSON - Feature Names
with open('/tmp/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("  ✓ feature_names.json")

# CSV - Train
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df['target'] = y_train
train_df.to_csv('/tmp/train_dataset.csv', index=False)
print("  ✓ train_dataset.csv")

# CSV - Test
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df['target'] = y_test
test_df.to_csv('/tmp/test_dataset.csv', index=False)
print("  ✓ test_dataset.csv")

# ====================================================================
# STEP 12: 保存数据信息
# ====================================================================

print("\n[Step 12] 保存数据信息...")

dataset_info = {
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'feature_count': len(feature_cols),
    'formula_count': len(formula_cols),
    'total_samples': int(len(X_train) + len(X_test)),
    'train_split_pct': 0.8,
    'test_split_pct': 0.2,
    'target_classes': 3,
    'class_labels': {'1': 'UP', '0': 'FLAT', '-1': 'DOWN'},
    'class_distribution_train': train_dist,
    'class_distribution_test': test_dist,
    'feature_scaling': 'StandardScaler (sklearn)' if USE_SKLEARN else 'StandardScaler (manual)',
    'timestamp': str(datetime.now())
}

with open('/tmp/dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)
print("  ✓ dataset_info.json")

# ====================================================================
# 最终总结
# ====================================================================

print("\n" + "="*80)
print("✓ ML数据集构建完成")
print("="*80)

print(f"""
输出文件:
  1. ml_dataset.pkl       - 完整数据集 (pickle)
  2. train_dataset.csv    - 训练集 ({len(X_train)} rows x {len(feature_cols)+1} cols)
  3. test_dataset.csv     - 测试集 ({len(X_test)} rows x {len(feature_cols)+1} cols)
  4. feature_names.json   - 特征名称 ({len(feature_cols)} features)
  5. dataset_info.json    - 数据统计信息

数据统计:
  总特征: {len(feature_cols)}
  - 原始公式: {len(formula_cols)}
  - 滞后特征 (1,2,3,5): {len(formula_cols) * 4}
  - 滚动统计 (MA5, Std5, MA20): {len(formula_cols) * 3}
  
  训练集: {len(X_train)} 样本 ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)
  测试集: {len(X_test)} 样本 ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)
  
  训练集目标分布:
    UP   (1):   {train_dist.get(1, 0):6d} ({train_dist.get(1, 0)/len(y_train)*100:5.2f}%)
    FLAT (0):   {train_dist.get(0, 0):6d} ({train_dist.get(0, 0)/len(y_train)*100:5.2f}%)
    DOWN (-1):  {train_dist.get(-1, 0):6d} ({train_dist.get(-1, 0)/len(y_train)*100:5.2f}%)
  
  测试集目标分布:
    UP   (1):   {test_dist.get(1, 0):6d} ({test_dist.get(1, 0)/len(y_test)*100:5.2f}%)
    FLAT (0):   {test_dist.get(0, 0):6d} ({test_dist.get(0, 0)/len(y_test)*100:5.2f}%)
    DOWN (-1):  {test_dist.get(-1, 0):6d} ({test_dist.get(-1, 0)/len(y_test)*100:5.2f}%)

下一步:
  第三步 - 机器学习模型训练
  选择模型算法:
    ✓ LightGBM (推荐 - 速度快)
    ✓ XGBoost (高精度)
    ✓ Random Forest (可解释性强)
    ✓ Neural Network (深度学习)
""")

print("="*80)
print("\n✓ ML数据集准备完成，可以进行第三步")
