"""
Colab特效扵步 - 内嵌的ML数据集构建

不需要任何外部脚本，直接在下一个不同的Colab单元中运行
"""

# ===========================
# 第一个单元：备优化环境
# ===========================

# 重新启动 Colab 并重新安装一切
!pip uninstall -y numpy pandas -q
!pip install numpy==1.26.4 pandas==2.1.4 scikit-learn==1.3.2 -q

print("[✓] 环境优化完成")

# ===========================
# 第二个单元：数据加载和合并
# ===========================

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("\n[Step 1] 加载公式时间序列...")

try:
    with open('/tmp/formula_timeseries.pkl', 'rb') as f:
        formula_df = pickle.load(f)
    print(f"  公式数据: {formula_df.shape}")
except:
    print("  [✗] 加载失败，请确保上传了 formula_timeseries.pkl")
    raise

print("\n[Step 2] 加载标签数据...")

try:
    labeled_df = pd.read_csv('/tmp/labeled_klines_phase1.csv')
    labeled_df['timestamp'] = pd.to_datetime(labeled_df['timestamp'])
    print(f"  标签数据: {labeled_df.shape}")
except:
    print("  [✗] 加载失败，请确保上传了 labeled_klines_phase1.csv")
    raise

print("\n[Step 3] 数据对齐...")

# 对齐了鬼数据是手工上传的，仅上传有数据中的一个子集
data_len = min(len(formula_df), len(labeled_df))
print(f"  对齐长度: {data_len}")

formula_df = formula_df.iloc[:data_len].reset_index(drop=True)
labeled_df = labeled_df.iloc[:data_len].reset_index(drop=True)

print("\n[Step 4] 合并数据集...")

merged_df = pd.concat([
    labeled_df[['timestamp', 'close']],
    formula_df
], axis=1)

print(f"  合并后的数据: {merged_df.shape}")

# ===========================
# 第三个单元：会标签字段
# ===========================

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

print(f"  目标分布: {pd.Series(merged_df['target']).value_counts().to_dict()}")

# ===========================
# 第四个单元：特征工程
# ===========================

print("\n[Step 6] 特征工程...")

formula_cols = formula_df.columns.tolist()
print(f"  原始公式: {len(formula_cols)}")

# 滞后特征
for col in formula_cols:
    for lag in [1, 2, 3, 5]:
        merged_df[f'{col}_lag{lag}'] = merged_df[col].shift(lag)

# 滚动统计
for col in formula_cols:
    merged_df[f'{col}_ma5'] = merged_df[col].rolling(5).mean()
    merged_df[f'{col}_std5'] = merged_df[col].rolling(5).std()
    merged_df[f'{col}_ma20'] = merged_df[col].rolling(20).mean()

print(f"  总特征数: {merged_df.shape[1] - 2}")

# ===========================
# 第五个单元：数据清理
# ===========================

print("\n[Step 7] 数据清理...")

initial_len = len(merged_df)
merged_df = merged_df.dropna()
final_len = len(merged_df)

print(f"  离际: {initial_len - final_len}")
print(f"  最终: {final_len}")

# ===========================
# 第六个单元：特征提取
# ===========================

print("\n[Step 8] 提取特征...")

exclude_cols = ['timestamp', 'close', 'future_return', 'target']
feature_cols = [c for c in merged_df.columns if c not in exclude_cols]

X = merged_df[feature_cols].values
y = merged_df['target'].values

print(f"  特征数: {len(feature_cols)}")
print(f"  样本数: {len(X)}")

# ===========================
# 第七个单元：标准化
# ===========================

print("\n[Step 9] 特征标准化...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  标准化完成")

# ===========================
# 第八个单元：数据分割
# ===========================

print("\n[Step 10] 数据分割...")

split_idx = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")
print(f"  训练会标: {pd.Series(y_train).value_counts().to_dict()}")
print(f"  测试会标: {pd.Series(y_test).value_counts().to_dict()}")

# ===========================
# 第九个单元：保存
# ===========================

print("\n[Step 11] 保存数据集...")

# pickle
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

# JSON
with open('/tmp/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("  ✓ feature_names.json")

# CSV
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df['target'] = y_train
train_df.to_csv('/tmp/train_dataset.csv', index=False)

test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df['target'] = y_test
test_df.to_csv('/tmp/test_dataset.csv', index=False)

print("  ✓ train_dataset.csv")
print("  ✓ test_dataset.csv")

# ===========================
# 第十个单元：信息保存
# ===========================

print("\n[Step 12] 保存数据信息...")

dataset_info = {
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'feature_count': len(feature_cols),
    'formula_count': len(formula_cols),
    'total_features': len(feature_cols),
    'target_classes': 3,
    'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
    'class_distribution_test': pd.Series(y_test).value_counts().to_dict(),
    'feature_scaling_method': 'StandardScaler',
    'timestamp': datetime.now().isoformat()
}

with open('/tmp/dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2, default=str)

print("  ✓ dataset_info.json")

# ===========================
# 第十一个单元：成功总结
# ===========================

print("\n" + "="*80)
print("✓ ML数据集构建完成")
print("="*80)

print(f"""
输出文件:
  1. ml_dataset.pkl          - 完整数据集 (pickle)
  2. train_dataset.csv       - 训练集 ({len(X_train)} rows x {len(feature_cols)+1} cols)
  3. test_dataset.csv        - 测试集 ({len(X_test)} rows x {len(feature_cols)+1} cols)
  4. feature_names.json      - 特征名称
  5. dataset_info.json       - 数据信息

数据集统计:
  总特征: {len(feature_cols)}
  - 原始公式: {len(formula_cols)}
  - 滞后特征 (1,2,3,5): {len(formula_cols) * 4}
  - 滚动统计 (MA5, Std5, MA20): {len(formula_cols) * 3}
  
  训练样本: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)
  测试样本: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)
  
  目标类别: 3 (DOWN, FLAT, UP)
  训练会标分布:
    DOWN:  {(y_train == -1).sum():6d} ({(y_train == -1).sum()/len(y_train)*100:5.2f}%)
    FLAT:  {(y_train == 0).sum():6d} ({(y_train == 0).sum()/len(y_train)*100:5.2f}%)
    UP:    {(y_train == 1).sum():6d} ({(y_train == 1).sum()/len(y_train)*100:5.2f}%)
  
  测试会标分布:
    DOWN:  {(y_test == -1).sum():6d} ({(y_test == -1).sum()/len(y_test)*100:5.2f}%)
    FLAT:  {(y_test == 0).sum():6d} ({(y_test == 0).sum()/len(y_test)*100:5.2f}%)
    UP:    {(y_test == 1).sum():6d} ({(y_test == 1).sum()/len(y_test)*100:5.2f}%)

下一步:
  第三步 - 机器学习模型训练
  选择模型算法:
    - LightGBM (官方推荐)
    - XGBoost (高精度)
    - Neural Networks (深度学习)
""")

print("="*80)
print("\n✓ 数据污准备好进行模型训练")
