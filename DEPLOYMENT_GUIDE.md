# V2Bot LSTM Trading System - 完整部署指南

## 概覽

V2Bot 是一個全自動的 LSTM 虛擬貨幣交易系統，包含三個主要模組：

1. **Colab 訓練模組** - 使用 54,915 根 BTC 1h K 棒訓練 LSTM 模型
2. **實時推理引擎** - 對新 K 棒進行推理並生成交易信號
3. **交易執行系統** - 與交易所 API 對接進行自動交易

---

## 第一部分：模型訓練（Colab）

### 1.1 開始訓練

**在 Google Colab 中複製貼上以下代碼：**

```python
SYMBOL = 'BTC'
INTERVAL = '1h'
EPOCHS = 20

import requests

url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_ready.py'
script = requests.get(url, timeout=60).text

exec(script, {
    'SYMBOL': SYMBOL,
    'INTERVAL': INTERVAL,
    'EPOCHS': EPOCHS
})
```

**預期耗時：** 30-45 分鐘（GPU T4）

### 1.2 訓練過程

訓練腳本自動執行以下步驟：

```
✓ Step 1: 導入核心庫 (NumPy 2.0.2, Pandas 2.2.2, TensorFlow 2.19.0)
✓ Step 2: 加載 54,915 根 BTC 1h K 棒
✓ Step 3: 計算 30+ 技術指標（RSI, MACD, ATR, Bollinger Bands 等）
✓ Step 4: 應用 5 套交易公式（混合指標）
✓ Step 5: 創建訓練標籤（SELL/HOLD/BUY）
✓ Step 6: 數據分割（70% train, 15% val, 15% test）
✓ Step 7: 創建 30-bar 序列
✓ Step 8: 構建雙層 LSTM 模型（31,299 參數）
✓ Step 9: 訓練 20 epochs（含早停機制）
✓ Step 10: 評估性能
✓ Step 11: 保存模型和配置
```

### 1.3 預期性能指標

```
Train Accuracy:  ~42%
Val Accuracy:    ~41%
Test Accuracy:   ~38-40%
```

**注意：** 虛擬貨幣市場複雜性高，40% 的準確率優於隨機猜測（33.3%）。通過組合多個信號和風險管理，可以實現正期望值。

---

## 第二部分：下載模型文件

### 2.1 Colab 中下載

**訓練完成後，在 Colab 中執行：**

```python
from google.colab import files

files.download('formula_lstm_model_BTC_1h.keras')
files.download('scaler_config_BTC_1h.json')
files.download('discovered_formulas_BTC_1h.json')

print("✓ 三個文件已下載！")
```

或使用助手腳本：

```python
import requests

url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_downloader.py'
script = requests.get(url).text
exec(script)
```

### 2.2 下載的文件

| 文件 | 說明 | 大小 |
|------|------|------|
| `formula_lstm_model_BTC_1h.keras` | 訓練好的 LSTM 模型 | ~500 KB |
| `scaler_config_BTC_1h.json` | 特徵縮放配置 | ~1 KB |
| `discovered_formulas_BTC_1h.json` | 5 套交易公式定義 | ~1 KB |

---

## 第三部分：本地部署

### 3.1 環境設置

**安裝依賴：**

```bash
pip install tensorflow==2.19.0
pip install numpy==2.0.2
pip install pandas==2.2.2
pip install ccxt  # 用於交易所 API
```

**目錄結構：**

```
trading_project/
├── formula_lstm_model_BTC_1h.keras
├── scaler_config_BTC_1h.json
├── discovered_formulas_BTC_1h.json
├── real_time_predictor_v2.py     # 下載自 GitHub
├── live_trader_example.py         # 下載自 GitHub
└── main.py                        # 你的主程序
```

### 3.2 實時推理示例

**簡單推理（獲取信號）：**

```python
from real_time_predictor_v2 import RealTimeFormulaPredictor

# 初始化預測器
predictor = RealTimeFormulaPredictor(
    model_path='formula_lstm_model_BTC_1h.keras',
    scaler_path='scaler_config_BTC_1h.json',
    formula_file='discovered_formulas_BTC_1h.json'
)

# 當新 K 棒到達時
kline_dict = {
    'timestamp': '2025-12-30 21:00:00',
    'open': 42150.0,
    'high': 42250.0,
    'low': 42100.0,
    'close': 42200.0,
    'volume': 1200,
    'rsi_7': 45.2,
    'rsi_14': 48.5,
    'macd_diff': 0.0025,
    'sma_20': 42050.0,
    'bb_width': 150.0,
    'atr_14': 75.5,
    'volume_ratio': 1.05
}

result = predictor.process_new_kline(kline_dict)

print(f"信號: {result['signal']}")                  # BUY / HOLD / SELL
print(f"信心度: {result['confidence']:.1%}")        # 0-100%
print(f"概率: {result['probabilities']}")           # 三類概率
print(f"公式值: {result['formula_values']}")        # 5 個公式的值
```

**輸出示例：**

```json
{
  "status": "READY",
  "timestamp": "2025-12-30 21:00:00",
  "kline_count": 150,
  "signal": "BUY",
  "action": "Enter Long / Exit Short",
  "confidence": 0.6245,
  "probabilities": {
    "SELL": 0.2156,
    "HOLD": 0.1599,
    "BUY": 0.6245
  },
  "formula_values": {
    "f1_rsi_macd_blend": 42184.5,
    "f2_vol_atr_log": 0.8234,
    "f3_bb_rsi_ratio": 3.3186,
    "f4_macd_atr_div": 0.0000333,
    "f5_vol_sma_inter": 44314.7
  }
}
```

### 3.3 完整的交易系統

**與 Binance 對接：**

```python
from live_trader_example import LiveTrader

trader = LiveTrader(
    exchange_name='binance',
    api_key='YOUR_BINANCE_API_KEY',
    api_secret='YOUR_BINANCE_API_SECRET',
    symbol='BTC/USDT',
    interval='1h',
    model_path='formula_lstm_model_BTC_1h.keras',
    scaler_path='scaler_config_BTC_1h.json',
    formula_file='discovered_formulas_BTC_1h.json',
    risk_per_trade=0.02  # 每筆交易風險 2%
)

# 開始實時交易
trader.start_monitoring(check_interval=60)  # 每 60 秒檢查一次
```

**程式會自動：**
- 每小時獲取最新 K 棒
- 計算技術指標
- 應用 5 套交易公式
- 通過 LSTM 生成信號
- 根據信心度和風險管理執行交易
- 記錄交易歷史和性能

---

## 第四部分：交易策略

### 4.1 5 套交易公式

**公式 1：RSI-MACD 混合**
```
f1 = rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3
```
衡量價格動量和趨勢

**公式 2：成交量-ATR 對數**
```
f2 = log(abs(atr_14 * volume_ratio) + 1e-8)
```
衡量波動率和成交量相互作用

**公式 3：布林帶-RSI 比值**
```
f3 = bb_width / (rsi_7 + 1e-8)
```
衡量波動帶相對於短期動量

**公式 4：MACD-ATR 發散**
```
f4 = macd_diff / (atr_14 + 1e-8)
```
衡量趨勢相對於波動

**公式 5：成交量-SMA 交互**
```
f5 = tanh(volume_ratio) * sma_20
```
衡量成交量驅動的價格水平

### 4.2 信號判斷

**高信心度購買 (Confidence > 65%)：**
```
if signal == 'BUY' and confidence > 0.65:
    execute_long_entry()
```

**高信心度賣出 (Confidence > 65%)：**
```
if signal == 'SELL' and confidence > 0.65:
    execute_short_entry()
```

**低信心度 (< 60%)：**
```
if confidence < 0.60:
    hold_current_position()
```

### 4.3 風險管理

**每筆交易的風險：** 2% 的投資組合價值

```python
risk_per_trade = 0.02  # 2%
position_size = (portfolio_value * risk_per_trade) / current_price
```

**止損：** 根據 ATR（平均真實波幅）設置

```python
stop_loss = entry_price - (atr_14 * 2)  # 2x ATR
take_profit = entry_price + (atr_14 * 4)  # 4x ATR
```

---

## 第五部分：監控和優化

### 5.1 性能指標追蹤

**主要指標：**
- 訊號勝率（% 正確的交易）
- 盈利因子（總利潤 / 總虧損）
- 最大回撤
- Sharpe 比率
- 年化收益率

### 5.2 模型再訓練

**定期重新訓練（建議每月）：**

1. 收集最新 K 棒數據
2. 重新運行 Colab 訓練腳本
3. 評估新模型性能
4. 如果性能提升 > 2%，部署新模型

### 5.3 超參數調整

可在 colab_ready.py 中調整：

```python
EPOCHS = 30              # 增加訓練輪數
LOOKBACK = 40            # 改變序列長度
LSTM_UNITS = [128, 64]  # 調整 LSTM 層大小
DROPOUT = 0.4            # 調整 Dropout 比率
```

---

## 第六部分：故障排除

### 常見問題

**Q: 準確率只有 38-40% 可以盈利嗎？**

A: 是的。只要 Win% - Loss% * RiskReward > 0，就能盈利。例如：
- 40% 勝率 × 2 倍盈利 = 40% 虧損，淨收益 40%

**Q: 模型過擬合了嗎？**

A: 不完全是。驗證集和測試集準確率相近（41% vs 38%），表明泛化良好。低準確率是市場內在複雜性。

**Q: 如何改進模型？**

A: 嘗試以下方法：
1. 增加訓練數據（使用更長的歷史時間段）
2. 添加更多技術指標
3. 調整公式的權重
4. 使用集合學習（多個模型的投票）
5. 加入市場狀態分類（牛市/熊市/盤整）

**Q: API 連接失敗？**

A: 檢查：
1. API Key/Secret 是否正確
2. 網絡連接
3. Exchange 是否支持該交易對
4. Rate Limit 是否超出

---

## 第七部分：實盤交易前檢查清單

- [ ] 模型訓練完成且下載
- [ ] 本地環境已設置（依賴安裝）
- [ ] API Key/Secret 配置正確
- [ ] Backtesting 通過（勝率 > 52%）
- [ ] 紙交易（模擬交易）驗證 1 週
- [ ] 風險管理設置正確（止損/獲利）
- [ ] 初始資金充足（建議 > $1000）
- [ ] 監控系統正常運行
- [ ] 應急停止機制已準備

---

## 聯繫和支援

**GitHub：** https://github.com/caizongxun/v2bot

**文件位置：**
- 訓練代碼：`strategy_design/colab_ready.py`
- 實時推理：`strategy_design/real_time_predictor_v2.py`
- 交易系統：`trading_system/live_trader_example.py`

---

## 免責聲明

本系統僅供教育和研究目的。虛擬貨幣交易具有高風險，過往性能不代表未來結果。在真實交易前，請進行充分的測試和評估，並了解所有相關風險。

**永遠不要投資超過你能承受的虧損金額。**

---

最後更新：2025-12-30
版本：1.0
