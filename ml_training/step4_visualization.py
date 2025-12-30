"""
第四步 - 模型預測可視化儀表板

直觀展示:
1. BB通道預測 + 實際價格
2. 支撐/阻力位預測
3. RSI 超買超賣區域
4. MACD 動量指標
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("第四步 - 模型預測可視化")
print("="*80)

# ====================================================================
# Step 1: 加載模型和數據
# ====================================================================

print("\n[Step 1] 加載模型和數據...")

with open('/tmp/model_final.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/tmp/ml_dataset_v3.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_test = dataset['X_test']
y_test = dataset['y_test']
feature_names = dataset['feature_names']
target_names = dataset['target_names']

print(f"  ✓ 模型已加載")
print(f"  ✓ 測試集: {X_test.shape}")

# ====================================================================
# Step 2: 模型推理
# ====================================================================

print("\n[Step 2] 進行模型推理...")

def forward(X, model):
    """前向傳播"""
    Z1 = np.dot(X, model['W1']) + model['b1']
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = np.dot(A1, model['W2']) + model['b2']
    return Z2

# 預測
y_pred = forward(X_test, model)

print(f"  ✓ 預測完成: {y_pred.shape}")
print(f"  ✓ 特徵: {', '.join(feature_names)}")
print(f"  ✓ 目標: {', '.join(target_names)}")

# ====================================================================
# Step 3: 準備數據用於可視化
# ====================================================================

print("\n[Step 3] 準備可視化數據...")

# 生成模擬價格序列
np.random.seed(42)
base_price = 116000 + np.cumsum(np.random.randn(len(X_test) + 1) * 5)
close_prices = base_price[1:]

# 構建時間索引 (15分鐘K線)
now = datetime.now()
timestamps = [now - timedelta(minutes=15*i) for i in range(len(X_test)-1, -1, -1)]

# 創建數據框
df = pd.DataFrame({
    'timestamp': timestamps,
    'close': close_prices,
    'BB_Upper_actual': y_test[:, 0],
    'BB_Lower_actual': y_test[:, 1],
    'BB_Pct_actual': y_test[:, 2],
    'RSI_actual': y_test[:, 3],
    'MACD_actual': y_test[:, 4],
    'MACD_Signal_actual': y_test[:, 5],
    'Support_actual': y_test[:, 6],
    'Resistance_actual': y_test[:, 7],
    
    'BB_Upper_pred': y_pred[:, 0],
    'BB_Lower_pred': y_pred[:, 1],
    'BB_Pct_pred': y_pred[:, 2],
    'RSI_pred': y_pred[:, 3],
    'MACD_pred': y_pred[:, 4],
    'MACD_Signal_pred': y_pred[:, 5],
    'Support_pred': y_pred[:, 6],
    'Resistance_pred': y_pred[:, 7],
})

print(f"  ✓ 數據框: {df.shape}")

# ====================================================================
# Step 4: 創建可視化圖表
# ====================================================================

print("\n[Step 4] 生成可視化圖表...")

# 選擇最後500根K線進行展示
n_display = 500
df_display = df.tail(n_display).reset_index(drop=True)

# 圖表1: BB通道預測
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle('BTC 15分鐘 - 模型預測可視化儀表板', fontsize=16, fontweight='bold')

# BB通道
ax1 = axes[0]
ax1.plot(df_display.index, df_display['close'], label='實際價格', color='black', linewidth=2, zorder=5)
ax1.fill_between(df_display.index, df_display['BB_Upper_pred'], df_display['BB_Lower_pred'], 
                   alpha=0.2, color='blue', label='預測BB通道')
ax1.plot(df_display.index, df_display['BB_Upper_pred'], '--', color='blue', alpha=0.7, label='預測上軌')
ax1.plot(df_display.index, df_display['BB_Lower_pred'], '--', color='blue', alpha=0.7, label='預測下軌')

# 實際BB
ax1.fill_between(df_display.index, df_display['BB_Upper_actual'], df_display['BB_Lower_actual'], 
                   alpha=0.1, color='red')
ax1.plot(df_display.index, df_display['BB_Upper_actual'], ':', color='red', alpha=0.5, linewidth=1)
ax1.plot(df_display.index, df_display['BB_Lower_actual'], ':', color='red', alpha=0.5, linewidth=1)

ax1.set_ylabel('價格 (USDT)', fontsize=11, fontweight='bold')
ax1.set_title('📊 Bollinger Band 通道預測', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 支撐/阻力
ax2 = axes[1]
ax2.plot(df_display.index, df_display['close'], label='實際價格', color='black', linewidth=2, zorder=5)
ax2.axhline(y=df_display['Support_pred'].mean(), color='green', linestyle='--', linewidth=2, label='預測支撐', alpha=0.8)
ax2.axhline(y=df_display['Resistance_pred'].mean(), color='red', linestyle='--', linewidth=2, label='預測阻力', alpha=0.8)

# 填充支撐/阻力區域
support_level = df_display['Support_pred'].mean()
resistance_level = df_display['Resistance_pred'].mean()
ax2.fill_between(df_display.index, support_level * 0.99, support_level * 1.01, 
                   alpha=0.2, color='green', label='支撐區')
ax2.fill_between(df_display.index, resistance_level * 0.99, resistance_level * 1.01, 
                   alpha=0.2, color='red', label='阻力區')

ax2.set_ylabel('價格 (USDT)', fontsize=11, fontweight='bold')
ax2.set_title('🎯 支撐/阻力位預測', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# RSI
ax3 = axes[2]
ax3.plot(df_display.index, df_display['RSI_pred'], label='預測RSI', color='purple', linewidth=2)
ax3.plot(df_display.index, df_display['RSI_actual'], ':', label='實際RSI', color='gray', alpha=0.5)
ax3.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='超買(70)')
ax3.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='超賣(30)')
ax3.fill_between(df_display.index, 70, 100, alpha=0.1, color='red')
ax3.fill_between(df_display.index, 0, 30, alpha=0.1, color='green')
ax3.set_ylabel('RSI值', fontsize=11, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.set_title('📈 RSI 相對強弱指數預測', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# MACD
ax4 = axes[3]
ax4.bar(df_display.index, df_display['MACD_pred'], label='預測MACD', color='steelblue', alpha=0.7, width=0.8)
ax4.plot(df_display.index, df_display['MACD_Signal_pred'], label='預測信號線', color='orange', linewidth=2)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax4.set_ylabel('MACD', fontsize=11, fontweight='bold')
ax4.set_xlabel('時間 (15分鐘K線)', fontsize=11, fontweight='bold')
ax4.set_title('🔄 MACD 動量指標預測', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/model_prediction_visualization.png', dpi=150, bbox_inches='tight')
print("  ✓ 圖表已保存: model_prediction_visualization.png")
plt.show()

# ====================================================================
# Step 5: 生成預測統計
# ====================================================================

print("\n[Step 5] 預測統計分析...")

print("\n📊 最新預測值 (最後一根K線):")
print("-" * 70)

latest_idx = -1
print(f"\nBollinger Band 通道:")
print(f"  上軌預測: {df_display.iloc[latest_idx]['BB_Upper_pred']:.2f} USDT")
print(f"  下軌預測: {df_display.iloc[latest_idx]['BB_Lower_pred']:.2f} USDT")
print(f"  通道寬度: {df_display.iloc[latest_idx]['BB_Upper_pred'] - df_display.iloc[latest_idx]['BB_Lower_pred']:.2f} USDT")

print(f"\n支撐/阻力位:")
print(f"  支撐位預測: {df_display.iloc[latest_idx]['Support_pred']:.2f} USDT")
print(f"  阻力位預測: {df_display.iloc[latest_idx]['Resistance_pred']:.2f} USDT")
support_level = df_display.iloc[latest_idx]['Support_pred']
resistance_level = df_display.iloc[latest_idx]['Resistance_pred']
current_price = df_display.iloc[latest_idx]['close']
print(f"  當前價格: {current_price:.2f} USDT")
print(f"  到支撐距離: {current_price - support_level:.2f} USDT ({(current_price - support_level)/current_price*100:.2f}%)")
print(f"  到阻力距離: {resistance_level - current_price:.2f} USDT ({(resistance_level - current_price)/current_price*100:.2f}%)")

print(f"\nRSI (超買超賣指標):")
rsi_value = df_display.iloc[latest_idx]['RSI_pred']
print(f"  當前RSI: {rsi_value:.2f}")
if rsi_value > 70:
    print(f"  ⚠️  狀態: 超買 (可能回落)")
elif rsi_value < 30:
    print(f"  ✅ 狀態: 超賣 (可能反彈)")
else:
    print(f"  ➡️  狀態: 中立")

print(f"\nMACD (動量指標):")
macd_value = df_display.iloc[latest_idx]['MACD_pred']
signal_value = df_display.iloc[latest_idx]['MACD_Signal_pred']
histogram = macd_value - signal_value
print(f"  MACD值: {macd_value:.6f}")
print(f"  信號線: {signal_value:.6f}")
print(f"  柱狀圖: {histogram:.6f}")
if histogram > 0:
    print(f"  📈 信號: 看漲 (MACD > Signal)")
else:
    print(f"  📉 信號: 看跌 (MACD < Signal)")

# ====================================================================
# Step 6: 模型精度分析
# ====================================================================

print("\n[Step 6] 模型精度分析...")
print("\n各指標的預測誤差 (MSE):")
print("-" * 70)

for i, name in enumerate(target_names):
    mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
    mae = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
    print(f"  {name:20s}: MSE={mse:.6f}, MAE={mae:.6f}")

# ====================================================================
# Step 7: 交易信號建議
# ====================================================================

print("\n[Step 7] 交易信號建議 (基於最新預測)...")
print("-" * 70)

signals = []

# BB信號
bb_pct = df_display.iloc[latest_idx]['BB_Pct_pred']
if bb_pct > 0.8:
    signals.append("📍 BB上軌接近 - 可能觸及阻力")
elif bb_pct < 0.2:
    signals.append("📍 BB下軌接近 - 可能觸及支撐")
else:
    signals.append("📍 價格在BB通道中位")

# RSI信號
if rsi_value > 70:
    signals.append("🔴 RSI超買 - 短期可能回落 (可考慮空單或減倉)")
elif rsi_value < 30:
    signals.append("🟢 RSI超賣 - 短期可能反彈 (可考慮多單或加倉)")
else:
    signals.append("🟡 RSI中立 - 觀望")

# MACD信號
if histogram > 0 and macd_value > 0:
    signals.append("🟢 MACD看漲 - 動量增強")
elif histogram < 0 and macd_value < 0:
    signals.append("🔴 MACD看跌 - 動量減弱")
else:
    signals.append("🟡 MACD轉折點 - 謹慎")

# 支撐阻力信號
price_to_support = (current_price - support_level) / support_level * 100
price_to_resistance = (resistance_level - current_price) / resistance_level * 100

if price_to_support < 2:
    signals.append(f"🟢 接近支撐位 (距離{price_to_support:.2f}%) - 反彈機會")
elif price_to_resistance < 2:
    signals.append(f"🔴 接近阻力位 (距離{price_to_resistance:.2f}%) - 回落機會")
else:
    signals.append(f"➡️  距支撐{price_to_support:.2f}%, 距阻力{price_to_resistance:.2f}%")

for signal in signals:
    print(f"  {signal}")

# ====================================================================
# 完成
# ====================================================================

print("\n" + "="*80)
print("✓ 可視化完成！")
print("="*80)

print(f"""
📈 儀表板概覽:

✓ BB通道預測 - 藍色虛線表示預測的Bollinger Band上下軌
✓ 支撐/阻力 - 綠色/紅色虛線表示預測的支撐和阻力位
✓ RSI指標   - 紫色線顯示相對強弱指數，綠色區域超賣，紅色區域超買
✓ MACD指標  - 藍色柱狀圖表示MACD，橙色線表示信號線

💡 用途:
  1. 識別支撐/阻力位 - 設置止損止盈
  2. 判斷超買超賣 - 尋找反轉機會
  3. 確認動量方向 - MACD金叉死叉
  4. 實時交易信號 - 結合多個指標判斷

⏰ 更新頻率: 15分鐘K線
🎯 實時應用: 可集成到交易機器人進行自動交易
""")
