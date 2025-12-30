"""
第四步 - 直接使用測試集標籤作為實數據

重點：
1. 直接使用 y_test 作為實際指標值
2. 用 y_pred 與 y_test 相比，並顯示預測輔助線
3. BB 上下軌正常顯示
4. RSI 正常僸動
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("第四步 - 实旦可視化（使用測試集数据）")
print("="*80)

# ====================================================================
# Step 1: 加載模型和数据
# ====================================================================

print("\n[Step 1] 加載模型和数据...")

with open('/tmp/model_final.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/tmp/ml_dataset_v3.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_test = dataset['X_test']
y_test = dataset['y_test']  # 標准化的標籤
 ntarget_names = dataset['target_names']

print(f"  ✓ 模型已加載")
print(f"  ✓ 測試集: {X_test.shape}")
print(f"  ✓ 目標: {target_names}")

# ====================================================================
# Step 2: 模型推理
# ====================================================================

print("\n[Step 2] 進行模型推理...")

def forward(X, model):
    Z1 = np.dot(X, model['W1']) + model['b1']
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, model['W2']) + model['b2']
    return Z2

y_pred = forward(X_test, model)
print(f"  ✓ 预測完成: {y_pred.shape}")

# ====================================================================
# Step 3: 整理数据
# ====================================================================

print("\n[Step 3] 整理数据...")

# 提取每个指標
y_index_map = {name: i for i, name in enumerate(target_names)}

# 实际值（標准化）
BB_Upper_actual = y_test[:, y_index_map['BB_Upper']]
BB_Lower_actual = y_test[:, y_index_map['BB_Lower']]
BB_Pct_actual = y_test[:, y_index_map['BB_Pct']]
RSI_actual = y_test[:, y_index_map['RSI']]
MACD_actual = y_test[:, y_index_map['MACD']]
MACD_Signal_actual = y_test[:, y_index_map['MACD_Signal']]
Support_actual = y_test[:, y_index_map['Support']]
Resistance_actual = y_test[:, y_index_map['Resistance']]

# 预測值（標准化）
BB_Upper_pred = y_pred[:, y_index_map['BB_Upper']]
BB_Lower_pred = y_pred[:, y_index_map['BB_Lower']]
BB_Pct_pred = y_pred[:, y_index_map['BB_Pct']]
RSI_pred = y_pred[:, y_index_map['RSI']]
MACD_pred = y_pred[:, y_index_map['MACD']]
MACD_Signal_pred = y_pred[:, y_index_map['MACD_Signal']]
Support_pred = y_pred[:, y_index_map['Support']]
Resistance_pred = y_pred[:, y_index_map['Resistance']]

print(f"  ✓ BB_Upper: actual [{BB_Upper_actual.min():.4f}, {BB_Upper_actual.max():.4f}], pred [{BB_Upper_pred.min():.4f}, {BB_Upper_pred.max():.4f}]")
print(f"  ✓ RSI: actual [{RSI_actual.min():.4f}, {RSI_actual.max():.4f}], pred [{RSI_pred.min():.4f}, {RSI_pred.max():.4f}]")
print(f"  ✓ Support: actual [{Support_actual.min():.4f}, {Support_actual.max():.4f}], pred [{Support_pred.min():.4f}, {Support_pred.max():.4f}]")

# ====================================================================
# Step 4: 准备可视化数据
# ====================================================================

print("\n[Step 4] 准备可视化数据...")

now = datetime.now()
timestamps = [now - timedelta(minutes=15*i) for i in range(len(X_test)-1, -1, -1)]

df = pd.DataFrame({
    'timestamp': timestamps,
    'BB_Upper_actual': BB_Upper_actual,
    'BB_Lower_actual': BB_Lower_actual,
    'BB_Pct_actual': BB_Pct_actual,
    'RSI_actual': RSI_actual,
    'MACD_actual': MACD_actual,
    'MACD_Signal_actual': MACD_Signal_actual,
    'Support_actual': Support_actual,
    'Resistance_actual': Resistance_actual,
    
    'BB_Upper_pred': BB_Upper_pred,
    'BB_Lower_pred': BB_Lower_pred,
    'BB_Pct_pred': BB_Pct_pred,
    'RSI_pred': RSI_pred,
    'MACD_pred': MACD_pred,
    'MACD_Signal_pred': MACD_Signal_pred,
    'Support_pred': Support_pred,
    'Resistance_pred': Resistance_pred,
})

print(f"  ✓ 数据框: {df.shape}")

# ====================================================================
# Step 5: 创建可视化
# ====================================================================

print("\n[Step 5] 生成可视化图表...")

n_display = 500
df_display = df.tail(n_display).reset_index(drop=True)

fig, axes = plt.subplots(4, 1, figsize=(18, 14))
fig.suptitle('BTC 15分鐘 - 模型预測对比可视化仓轣板', fontsize=16, fontweight='bold')

# 颜色配置
color_actual = 'black'
color_pred = 'blue'
alpha_actual = 1.0
alpha_pred = 0.7

# 1. BB通道
ax1 = axes[0]
ax1.plot(df_display.index, df_display['BB_Upper_actual'], '-', color=color_actual, linewidth=2.5, label='BB上軌(实际)', alpha=alpha_actual, zorder=5)
ax1.plot(df_display.index, df_display['BB_Lower_actual'], '-', color=color_actual, linewidth=2.5, label='BB下軌(实际)', alpha=alpha_actual, zorder=5)
ax1.fill_between(df_display.index, df_display['BB_Upper_actual'], df_display['BB_Lower_actual'], 
                   alpha=0.1, color='black', label='BB通道(实际)')

ax1.plot(df_display.index, df_display['BB_Upper_pred'], '--', color=color_pred, linewidth=1.5, label='BB上軌(预測)', alpha=alpha_pred)
ax1.plot(df_display.index, df_display['BB_Lower_pred'], '--', color=color_pred, linewidth=1.5, label='BB下軌(预測)', alpha=alpha_pred)
ax1.fill_between(df_display.index, df_display['BB_Upper_pred'], df_display['BB_Lower_pred'], 
                   alpha=0.1, color='blue')

ax1.set_ylabel('标准化值', fontsize=11, fontweight='bold')
ax1.set_title('📊 Bollinger Band 通道 - 预測对比实际', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9, ncol=3)
ax1.grid(True, alpha=0.3)

# 2. Support/Resistance
ax2 = axes[1]
ax2.plot(df_display.index, df_display['Support_actual'], '-', color='lime', linewidth=2.5, label='支撉(实际)', alpha=alpha_actual, zorder=5)
ax2.plot(df_display.index, df_display['Resistance_actual'], '-', color='red', linewidth=2.5, label='阻力(实际)', alpha=alpha_actual, zorder=5)
ax2.fill_between(df_display.index, df_display['Support_actual'], df_display['Resistance_actual'], 
                   alpha=0.1, color='gray')

ax2.plot(df_display.index, df_display['Support_pred'], '--', color='lime', linewidth=1.5, label='支撉(预測)', alpha=alpha_pred)
ax2.plot(df_display.index, df_display['Resistance_pred'], '--', color='red', linewidth=1.5, label='阻力(预測)', alpha=alpha_pred)

ax2.set_ylabel('标准化值', fontsize=11, fontweight='bold')
ax2.set_title('🎯 支撉/阻力位 - 预測对比实际', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, ncol=3)
ax2.grid(True, alpha=0.3)

# 3. RSI
ax3 = axes[2]
ax3.plot(df_display.index, df_display['RSI_actual'], '-', color=color_actual, linewidth=2.5, label='RSI(实际)', alpha=alpha_actual, zorder=5)
ax3.plot(df_display.index, df_display['RSI_pred'], '--', color=color_pred, linewidth=2, label='RSI(预測)', alpha=alpha_pred)

ax3.set_ylabel('标准化值', fontsize=11, fontweight='bold')
ax3.set_title('📈 RSI 相对强弱指数 - 预測对比实际', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9, ncol=3)
ax3.grid(True, alpha=0.3)

# 4. MACD
ax4 = axes[3]
ax4.plot(df_display.index, df_display['MACD_actual'], '-', color=color_actual, linewidth=2, label='MACD(实际)', alpha=alpha_actual, zorder=5)
ax4.plot(df_display.index, df_display['MACD_Signal_actual'], '-', color='orange', linewidth=2, label='信號线(实际)', alpha=alpha_actual, zorder=5)
ax4.plot(df_display.index, df_display['MACD_pred'], '--', color=color_pred, linewidth=1.5, label='MACD(预測)', alpha=alpha_pred)
ax4.plot(df_display.index, df_display['MACD_Signal_pred'], '--', color='darkorange', linewidth=1.5, label='信號线(预測)', alpha=alpha_pred)
ax4.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

ax4.set_ylabel('标准化值', fontsize=11, fontweight='bold')
ax4.set_xlabel('時间 (15分鐘K线)', fontsize=11, fontweight='bold')
ax4.set_title('🔄 MACD 动量指標 - 预測对比实际', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9, ncol=3)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/model_final_visualization.png', dpi=150, bbox_inches='tight')
print("  ✓ 圖表已保存")
plt.show()

# ====================================================================
# Step 6: 预測与实际的比较
# ====================================================================

print("\n[Step 6] 预測泛化能力分析...")
print("="*80)

for i, name in enumerate(target_names):
    actual = y_test[:, i]
    pred = y_pred[:, i]
    
    mse = np.mean((actual - pred) ** 2)
    mae = np.mean(np.abs(actual - pred))
    r2 = 1 - (np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    print(f"\n{name:20s}:")
    print(f"  MSE: {mse:.6f}  |  MAE: {mae:.6f}  |  R²: {r2:.6f}")
    print(f"  实际: [{actual.min():.4f}, {actual.max():.4f}]  |  预測: [{pred.min():.4f}, {pred.max():.4f}]")

# ====================================================================
# Step 7: 最新预測信息
# ====================================================================

print("\n[Step 7] 最新预測值 (最后一根K线)...")
print("="*80)

latest = df.iloc[-1]

print(f"\n📊 Bollinger Band 通道:")
print(f"  上軌: 实际={latest['BB_Upper_actual']:.4f}, 预測={latest['BB_Upper_pred']:.4f}")
print(f"  下軌: 实际={latest['BB_Lower_actual']:.4f}, 预測={latest['BB_Lower_pred']:.4f}")
print(f"  元件: 实际={latest['BB_Pct_actual']:.4f}, 预測={latest['BB_Pct_pred']:.4f}")

print(f"\n🎯 支撉/阻力位:")
print(f"  支撉: 实际={latest['Support_actual']:.4f}, 预測={latest['Support_pred']:.4f}")
print(f"  阻力: 实际={latest['Resistance_actual']:.4f}, 预測={latest['Resistance_pred']:.4f}")

print(f"\n📈 RSI:")
print(f"  实际={latest['RSI_actual']:.4f} (转探: {latest['RSI_actual']*50+50:.2f})")
print(f"  预測={latest['RSI_pred']:.4f} (转探: {latest['RSI_pred']*50+50:.2f})")

print(f"\n🔄 MACD:")
print(f"  MACD: 实际={latest['MACD_actual']:.6f}, 预測={latest['MACD_pred']:.6f}")
print(f"  信號: 实际={latest['MACD_Signal_actual']:.6f}, 预測={latest['MACD_Signal_pred']:.6f}")

print("\n" + "="*80)
print("✓ 可视化完成！模型预測与实际值对比")
print("="*80)
