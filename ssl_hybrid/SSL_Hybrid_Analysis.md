# SSL Hybrid V6 指標邏輯分析

## 📊 指標的核心架構

### 三層結構

```
第1層: BASELINE (HMA/EMA 60)
       ├─ 主趨勢線
       ├─ 上通道 (baseline + channel multiplier)
       └─ 下通道 (baseline - channel multiplier)

第2層: SSL1 (HMA/EMA 60)
       ├─ 根據 HIGH/LOW 高低點
       └─ 當 close > emaHigh 時為上升，反之為下降

第3層: SSL2 (JMA 5)
       ├─ 快速趨勢確認
       ├─ ATR continuation criteria 0.9
       └─ 用於進場信號

第4層: EXIT (HMA 15)
       └─ 出場點判定
```

---

## 🎯 買賣信號的產生

### 箭頭信號 (Arrow Signals)

```python
# 出場箭頭 (Exit Arrows) - codiff
base_cross_Long = ta.crossover(close, sslExit)   # 價格穿越 EXIT 線上方 = LONG 出場
base_cross_Short = ta.crossover(sslExit, close)  # 價格穿越 EXIT 線下方 = SHORT 出場
codiff = base_cross_Long ? 1 : base_cross_Short ? -1 : na

# 這些是出場信號，不是進場信號！
# 但箭頭顯示的是趨勢變化點
```

### SSL2 進場信號 (真正的買賣點)

```python
# BUY 條件
buy_inatr = lower_half < sslDown2              # SSL2 在下半部分
buy_cont = close > BBMC and close > sslDown2   # 價格在 baseline 上方且在 SSL2 上方
buy_atr = buy_inatr and buy_cont              # 同時滿足

# SELL 條件
sell_inatr = upper_half > sslDown2            # SSL2 在上半部分
sell_cont = close < BBMC and close < sslDown2 # 價格在 baseline 下方且在 SSL2 下方
sell_atr = sell_inatr and sell_cont           # 同時滿足

# 信號觸發
ssl2_buy_signal = buy_atr and not buy_atr[1]   # buy_atr 變為 true
ssl2_sell_signal = sell_atr and not sell_atr[1] # sell_atr 變為 true
```

### 警告信號 (假信號指示)

```python
# 假突破警告 (False Breakout Warning)
difference = math.abs(close - open)
atr_violation = difference > atr_slen          # 蠟燭大於 1 ATR
InRange = upper_band > BBMC and lower_band < BBMC # baseline 在 ATR band 內
candlesize_violation = atr_violation and InRange   # 兩個條件同時

# 風險等級
risk_level = atr_percentile > 75 ? "High" : atr_percentile < 25 ? "Low" : "Normal"

# 進場距離
distance_from_baseline = math.abs(close - BBMC) / atr_slen
entry_distance = distance_from_baseline < 1 ? "Near" : distance_from_baseline < 2 ? "Extended" : "Far"
```

---

## ⚠️ 假信號的典型特徵

### 導致假信號的因素

| 特徵 | 說明 | 假信號機率 |
|------|------|----------|
| **極端波動** | atr_percentile > 75 | 35-45% |
| **遠距離進場** | distance > 2 ATR | 40-50% |
| **蠟燭大小異常** | 大蠟燭穿過通道 | 30-40% |
| **低風險環境** | atr_percentile < 25 | 45-55% |
| **方向不確定** | SSL1 vs SSL2 衝突 | 25-35% |
| **超短時間框架** | 信號持續 < 2 candles | 50-60% |

### 真實信號的特徵

| 特徵 | 說明 | 真實機率 |
|------|------|----------|
| **正常波動** | 25 < atr_percentile < 75 | 65-75% |
| **近距離進場** | distance < 1 ATR | 70-80% |
| **正常蠟燭** | candle size < 1 ATR | 75-85% |
| **信號一致** | SSL1 = SSL2 方向 | 80-90% |
| **持續確認** | 信號持續 > 3 candles | 75-85% |
| **成交量確認** | volume > avg volume | 70-80% |

---

## 🔄 訓練框架

### 第一步：信號提取

```python
# 記錄所有信號點
signals = []

for i in range(len(close)):
    signal = None
    
    # BUY 信號
    if ssl2_buy_signal[i]:
        signal = {
            'type': 'BUY',
            'index': i,
            'price': close[i],
            'atr_percentile': atr_percentile[i],
            'distance_from_baseline': distance_from_baseline[i],
            'atr_slen': atr_slen[i],
            'volume': volume[i],
            'volume_ratio': volume[i] / volume_sma[i],
            'candlesize': abs(close[i] - open[i]),
            'atr_violation': atr_violation[i],
            'risk_level': risk_level[i]
        }
    
    # SELL 信號
    elif ssl2_sell_signal[i]:
        signal = {
            'type': 'SELL',
            'index': i,
            'price': close[i],
            'atr_percentile': atr_percentile[i],
            'distance_from_baseline': distance_from_baseline[i],
            'atr_slen': atr_slen[i],
            'volume': volume[i],
            'volume_ratio': volume[i] / volume_sma[i],
            'candlesize': abs(close[i] - open[i]),
            'atr_violation': atr_violation[i],
            'risk_level': risk_level[i]
        }
    
    if signal:
        signals.append(signal)
```

### 第二步：標籤生成

```python
# 判斷信號是真是假
for signal in signals:
    idx = signal['index']
    sig_type = signal['type']
    
    # 看未來 5 根蠟燭的表現
    lookforward = 5
    if idx + lookforward >= len(close):
        continue
    
    future_close = close[idx + lookforward]
    signal_price = close[idx]
    future_return = (future_close - signal_price) / signal_price
    
    # 判定標準
    if sig_type == 'BUY':
        # BUY 信號正確：5 candles 後價格上升 > 0.5%
        signal['is_true'] = future_return > 0.005
        signal['actual_return'] = future_return
    
    elif sig_type == 'SELL':
        # SELL 信號正確：5 candles 後價格下跌 < -0.5%
        signal['is_true'] = future_return < -0.005
        signal['actual_return'] = future_return
```

### 第三步：特徵工程

```python
# 為每個信號提取上下文特徵
for signal in signals:
    idx = signal['index']
    lookback = 40  # 往前看 40 根蠟燭
    
    # 價格結構
    hist_close = close[max(0, idx-lookback):idx]
    signal['price_position'] = (close[idx] - np.min(hist_close)) / (np.max(hist_close) - np.min(hist_close))
    signal['price_momentum_5'] = (close[idx] - close[idx-5]) / close[idx-5]
    signal['price_momentum_20'] = (close[idx] - close[idx-20]) / close[idx-20]
    
    # 波動率
    signal['volatility_20'] = np.std(close[max(0, idx-20):idx]) / np.mean(close[max(0, idx-20):idx])
    signal['atr_ratio'] = atr_slen[idx] / np.mean(atr_slen[max(0, idx-20):idx])
    
    # 成交量
    signal['volume_spike'] = volume[idx] / np.mean(volume[max(0, idx-20):idx])
    
    # 趨勢強度
    signal['days_above_baseline'] = sum(1 for j in range(max(0, idx-20), idx) if close[j] > BBMC[j]) / 20
    signal['ssl_alignment'] = 1 if (close[idx] > sslDown[idx] and signal['type'] == 'BUY') else (-1 if (close[idx] < sslDown[idx] and signal['type'] == 'SELL') else 0)
```

### 第四步：數據集構建

```python
# 分割真假信號
true_signals = [s for s in signals if s['is_true']]
false_signals = [s for s in signals if not s['is_true']]

print(f"總信號數：{len(signals)}")
print(f"真實信號：{len(true_signals)} ({100*len(true_signals)/len(signals):.1f}%)")
print(f"假信號：{len(false_signals)} ({100*len(false_signals)/len(signals):.1f}%)")

# 特徵矩陣
feature_names = [
    'atr_percentile',
    'distance_from_baseline',
    'volume_ratio',
    'atr_violation',
    'price_position',
    'price_momentum_5',
    'price_momentum_20',
    'volatility_20',
    'atr_ratio',
    'volume_spike',
    'days_above_baseline',
    'ssl_alignment'
]

X = np.array([[s[fname] for fname in feature_names] for s in signals])
y = np.array([s['is_true'] for s in signals])
```

---

## 🎓 模型訓練策略

### 模型架構

```python
# 簡單但有效的二元分類
model = keras.Sequential([
    keras.layers.Input(shape=(12,)),  # 12 個特徵
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(1, activation='sigmoid')  # 0-1 置信度
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)
```

### 損失函數（應對不平衡）

```python
# 如果假信號多於真信號，使用 class weight
class_weight = {
    0: len(y) / (2 * (y == 0).sum()),  # 假信號權重
    1: len(y) / (2 * (y == 1).sum())   # 真信號權重
}
```

---

## 📈 預期成果

### 基線
- SSL Hybrid 原始準確率：60-65%
- 假信號率：35-40%

### 目標
- 模型過濾假信號後準確率：75-85%
- 過濾掉 70%+ 的假信號
- 保留 90%+ 的真信號

### 指標

```
Accuracy:  真實信號判對 + 假信號判對 / 總數
Precision: 模型說"真"，實際是真 / 模型說"真"的數量
Recall:     模型說"真"，實際是真 / 實際真的數量

目標：
  Accuracy  >= 80%
  Precision >= 85% (重要！寧可漏掉，也不想做假信號)
  Recall    >= 75% (保留大多數真信號)
```

---

## 🚀 下一步

1. 實現 Pine Script → Python 的完整轉換
2. 在 BTC 1h 數據上提取所有信號
3. 標籤真假信號
4. 訓練篩選模型
5. 評估性能：
   - 原始準確率
   - 過濾後準確率
   - 過濾掉多少假信號
   - 保留多少真信號
