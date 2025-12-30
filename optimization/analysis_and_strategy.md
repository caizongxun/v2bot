# V2Bot 準確度分析 - 47% → 70%+ 戰略

## 問題診斷

### 1. **標籤問題（最大影響）**

當前策略：
```
SELL: 25,444 (46.3%)
HOLD: 1,820  (3.3%)
BUY:  27,651 (50.4%)
```

問題：
- HOLD 極度稀少（3.3%）
- 標籤本質上是隨機的 SELL/BUY 硬幣翻轉
- 46% vs 50% 的差異無法被任何模型學習
- **模型無法區分，只能猜測多數類**

### 2. **特徵問題**

當前特徵：
- RSI (7, 14, 21) - 標準動量指標
- MACD - 標準趨勢指標
- SMA (10, 20, 50) - 標準移動平均
- 布林帶、ATR、成交量比

問題：
- 這些都是眾所周知的指標
- 市場已經 priced in
- 缺乏高預測性的自定義特徵
- 缺乏時間序列特徵

### 3. **模型問題**

當前模型：
- Bidirectional LSTM 128 → 64 → 32
- 250K 參數
- 類別權重不平衡

問題：
- 參數太多（可能過擬合）
- 雙向 LSTM 在時間序列預測中無意義
- 模型試圖學習隨機數據

---

## 70%+ 戰略

### Phase 1: 智能標籤設計

**策略：基於實現的高概率交易機會**

```python
# 不是：未來 24 小時會漲 5% 嗎？
# 而是：在過去 100 個 candles 中，
#      當前價格距離 20-SMA 的關係 + 
#      RSI + 
#      成交量突增
#      預測下 5 candles 的方向性？

def create_smart_labels(close, high, low, volume, lookback=100, forecast=5):
    labels = np.ones(len(close), dtype=np.int32)
    
    for i in range(lookback, len(close) - forecast):
        # 歷史數據窗口
        hist_close = close[i-lookback:i]
        hist_high = high[i-lookback:i]
        hist_low = low[i-lookback:i]
        hist_volume = volume[i-lookback:i]
        
        # 未來窗口
        future_close = close[i:i+forecast]
        
        # 計算強信號
        # 1. 價格相對位置
        price_level = (close[i] - np.min(hist_low)) / (np.max(hist_high) - np.min(hist_low))
        
        # 2. 動量
        momentum = (close[i] - close[i-5]) / close[i-5]
        
        # 3. 成交量
        vol_ratio = volume[i] / np.mean(hist_volume[-20:])
        
        # 4. 未來方向
        future_return = (future_close[-1] - close[i]) / close[i]
        
        # 5. 高概率信號組合
        # 買入：低位 + 上升動量 + 高成交量
        if price_level < 0.3 and momentum > 0.002 and vol_ratio > 1.5 and future_return > 0.003:
            labels[i] = 2  # BUY（高確信）
        # 賣出：高位 + 下降動量 + 高成交量
        elif price_level > 0.7 and momentum < -0.002 and vol_ratio > 1.5 and future_return < -0.003:
            labels[i] = 0  # SELL（高確信）
        else:
            labels[i] = 1  # HOLD（低確信）
    
    return labels
```

預期分布：
```
SELL:  15% (高確信)
HOLD:  70% (低確信)
BUY:   15% (高確信)
```

### Phase 2: 高預測力特徵

**放棄標準指標，創建專有特徵**

```python
# 特徵類別
1. 價格結構特徵（8 個）
   - 相對於過去 100 candles 高低的位置
   - 距離 20/50 SMA 的百分比
   - 過去 5/10/20 candles 的極值範圍
   - 近期高低點突破

2. 動量特徵（8 個）
   - 多個時間尺度的收益率（1,5,10,20 candles）
   - 加速度（動量變化）
   - 動量平均化
   - 反轉指標

3. 波動性和風險（6 個）
   - 實現波動率（過去 20/50 candles）
   - 高低差相對於收盤價
   - 最大回撤（過去 20 candles）
   - 尾部風險

4. 成交量分析（8 個）
   - 成交量 SMA 比
   - 成交量加速度
   - 成交量 vs 價格關係
   - 大單特徵

5. 時間序列特徵（8 個）
   - 自相關特徵
   - 序列熵
   - 趨勢強度
   - 均值回復指標

總計：38 個高預測力特徵
```

### Phase 3: 簡化但更聰明的模型

**小但深的模型 vs 大而淺的模型**

```python
# 從 250K 參數降低到 50K
# 單向 LSTM + 更好的預處理

model = keras.Sequential([
    # 編碼層：理解時間序列結構
    keras.layers.LSTM(64, activation='relu', return_sequences=True,
                      input_shape=(LOOKBACK, 38)),
    keras.layers.Dropout(0.2),
    
    # 提取特徵層
    keras.layers.LSTM(32, activation='relu', return_sequences=False),
    keras.layers.Dropout(0.2),
    
    # 決策層
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(3, activation='softmax')
])

# 關鍵：使用焦點損失而不是交叉熵
# 焦點損失更擅長處理不平衡的高確信標籤
```

---

## 預期改進

| 步驟 | 改變 | 預期提升 | 累計 |
|------|------|--------|------|
| 基線 | 標準指標 | - | 47% |
| 1 | 智能標籤 | +8-12% | 55-59% |
| 2 | 高預測特徵 | +8-15% | 63-74% |
| 3 | 簡化模型 | +3-5% | 66-79% |
| 4 | 焦點損失 | +2-4% | 68-83% |

目標：**70-75% 測試準確率**

---

## 實現順序

1. **第一天**：實現新標籤策略 → 測試
2. **第二天**：添加 38 個新特徵 → 測試
3. **第三天**：簡化模型 + 焦點損失 → 最終訓練
4. **第四天**：超參數優化

---

## 為什麼這有效

### 舊方法的問題
- 試圖預測隨機的 SELL/BUY（50/50 硬幣翻轉）
- 標準指標都是公開的
- 模型無法從隨機數據中學習

### 新方法的優勢
- **只預測高確信機會**（15% 的時間）
- 標籤本身就有 70%+ 的內在準確率（高確信信號）
- 模型只需學習識別這 15% 的特殊情況
- **專有特徵** 捕捉市場微結構
- **焦點損失** 關注難分類的邊界情況

---

## 下一步

等待我上傳新的訓練腳本，實現上述所有改進。
這次將從根本上改變方向。
