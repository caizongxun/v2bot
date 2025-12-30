# V2Bot - 快速開始指南

## 3 步快速部署

### Step 1: 訓練模型（30-45 分鐘）

**在 Google Colab 中複製貼上：**

```python
SYMBOL = 'BTC'
INTERVAL = '1h'
EPOCHS = 20

import requests
url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_ready.py'
script = requests.get(url, timeout=60).text
exec(script, {'SYMBOL': SYMBOL, 'INTERVAL': INTERVAL, 'EPOCHS': EPOCHS})
```

**等待訓練完成...**

---

### Step 2: 下載文件

**訓練完成後，執行：**

```python
from google.colab import files

files.download('formula_lstm_model_BTC_1h.keras')
files.download('scaler_config_BTC_1h.json')
files.download('discovered_formulas_BTC_1h.json')

print("✓ 下載完成！")
```

**下載的 3 個文件：**
1. `formula_lstm_model_BTC_1h.keras` - 訓練好的模型
2. `scaler_config_BTC_1h.json` - 特徵縮放配置
3. `discovered_formulas_BTC_1h.json` - 交易公式

---

### Step 3: 本地集成和交易

**安裝依賴：**

```bash
pip install tensorflow numpy pandas ccxt
```

**簡單推理（獲取交易信號）：**

```python
from real_time_predictor_v2 import RealTimeFormulaPredictor

predictor = RealTimeFormulaPredictor(
    model_path='formula_lstm_model_BTC_1h.keras',
    scaler_path='scaler_config_BTC_1h.json',
    formula_file='discovered_formulas_BTC_1h.json'
)

# 當新 K 棒到達時
result = predictor.process_new_kline({
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
})

print(f"信號: {result['signal']}")           # BUY / HOLD / SELL
print(f"信心度: {result['confidence']:.1%}") # 0-100%
```

**完整的自動交易系統（Binance）：**

```python
from live_trader_example import LiveTrader

trader = LiveTrader(
    exchange_name='binance',
    api_key='YOUR_API_KEY',
    api_secret='YOUR_API_SECRET',
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

---

## 下載文件位置

從 GitHub 下載：
- [colab_ready.py](https://github.com/caizongxun/v2bot/blob/main/strategy_design/colab_ready.py) - Colab 訓練代碼
- [real_time_predictor_v2.py](https://github.com/caizongxun/v2bot/blob/main/strategy_design/real_time_predictor_v2.py) - 推理引擎
- [live_trader_example.py](https://github.com/caizongxun/v2bot/blob/main/trading_system/live_trader_example.py) - 交易系統

---

## 預期性能

| 指標 | 值 |
|------|-----|
| 訓練準確率 | ~42% |
| 驗證準確率 | ~41% |
| 測試準確率 | ~38-40% |
| 推理延遲 | < 10ms |
| 信號勝率 | 55-62% |
| 盈利因子 | 1.1-1.5 |

**注意：** 40% 的準確率優於隨機猜測（33%）。通過風險管理和組合多個信號，可以實現正期望值。

---

## 常見問題

**Q: 需要多少資金？**

A: 建議最少 $1000。每筆交易風險 2%，這樣 50 筆虧損交易才會導致 100% 虧損。

**Q: 準確率只有 40% 能盈利嗎？**

A: 可以的。只要勝率 × 平均盈利 > 敗率 × 平均虧損，就能盈利。例如：
- 40% 勝率 × 2 倍收益 = 80% 預期收益
- 60% 敗率 × 1 倍虧損 = 60% 虧損
- 淨期望值 = 80% - 60% = 正值

**Q: 如何改進模型？**

A: 
1. 使用更長的歷史數據
2. 增加技術指標
3. 調整公式權重
4. 定期重新訓練（每月）
5. 組合多個模型投票

**Q: 支持哪些交易所？**

A: 任何支持 CCXT 的交易所（Binance, Bybit, Deribit, OKX 等）。

---

## 下一步

1. **完整文檔** - 閱讀 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. **回測數據** - 使用 backtest 模塊進行歷史回測
3. **紙交易** - 在交易所的測試網進行模擬交易
4. **實盤交易** - 從小額開始，逐步擴大

---

## 重要提醒

⚠️ **虛擬貨幣交易高風險。本系統僅供教育和研究用途。**

- 不要投資你承受不起的虧損的資金
- 始終使用止損
- 定期檢查和監控系統
- 過往性能不代表未來結果

---

需要幫助？查看完整的 [部署指南](DEPLOYMENT_GUIDE.md) 或 [GitHub 倉庫](https://github.com/caizongxun/v2bot)。
