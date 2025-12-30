# V2Bot 高級量化交易策略：公式-LSTM 融合系統

## 系統概述

這是一個三層級的先進量化交易系統，通過符號回歸自動發現最優交易公式，然後用 LSTM 網路學習這些公式值的時間序列模式，最終生成買賣信號。

### 架構層次

```
第 3 層：決策層 (LSTM 神經網路)
  輸入：過去 30 根 K 棒的 5 個公式值
  輸出：交易信號 (買 / 持 / 賣) + 信心度
  
第 2 層：公式層 (符號回歸自動發現)
  輸入：30+ 技術指標
  輸出：5 套獨立混合公式
  
第 1 層：數據層 (原始 K 線 + 指標)
  輸入：OHLCV 數據
  輸出：30+ 標準技術指標
```

---

## 文件結構

```
strategy_design/
├── README.md                          # 本文件
├── QUICK_START.md                     # 快速開始指南
├── formula_lstm_architecture.md        # 完整技術設計文檔
├── local_symbolic_regression.py       # 本地公式發現腳本
├── colab_lstm_training.py            # Colab 模型訓練腳本
├── real_time_predictor.py            # 實時推理引擎
└── colab_remote_execution.py          # Colab 遠端一鍵執行
```

---

## 快速開始

### 方案 1：完整本地執行（推薦）

#### 第 1 步：本地公式發現（1-2 小時）

```bash
# 安裝依賴
pip install pysr pandas numpy scikit-learn huggingface-hub ta

# 執行符號回歸發現公式
python local_symbolic_regression.py --symbol BTC --interval 1h --iterations 100

# 輸出：discovered_formulas.json (5 個最優公式)
```

#### 第 2 步：Colab 模型訓練（4-6 小時）

在 Google Colab 執行：

```python
!pip install -q tensorflow pandas numpy scikit-learn huggingface-hub ta
!wget https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_lstm_training.py
%run colab_lstm_training.py
```

輸出：
- `formula_lstm_model.h5` - 訓練好的 LSTM 模型
- `scaler_config.pkl` - 特徵標準化器

#### 第 3 步：實時推理部署

```python
from real_time_predictor import RealTimeFormulaPredictor

# 初始化預測器
predictor = RealTimeFormulaPredictor(
    model_path='formula_lstm_model.h5',
    scaler_path='scaler_config.pkl',
    formula_file='discovered_formulas.json'
)

# 當新 K 棒到達時
result = predictor.process_new_kline(kline_dict)
print(result['signal'])  # BUY / HOLD / SELL
print(result['confidence'])  # 0.0 ~ 1.0
```

### 方案 2：Colab 遠端一鍵執行（無需本地環境）

在 Google Colab 直接執行：

```python
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_remote_execution.py').text,
     {'SYMBOL': 'BTC', 'INTERVAL': '1h', 'MODE': 'train'})
```

這會自動執行完整的公式發現和模型訓練流程。

---

## 核心特性

### 1. 自動公式發現

使用符號回歸算法自動發現最優的指標組合公式：

```
Formula 1: rsi_14 * 0.4 + macd_diff * 0.3 + sma_20 * 0.3
Formula 2: log(abs(atr_14 * volume_ratio) + 1e-8)
Formula 3: bb_width / (rsi_7 + 1e-8)
Formula 4: macd_diff / (atr_14 + 1e-8)
Formula 5: tanh(volume_ratio) * sma_20
```

### 2. 特徵降維

30+ 維技術指標 → 5 維合成指標

優點：
- 降低過擬合風險
- 提高模型穩定性
- 加快推理速度

### 3. 時間序列建模

LSTM 捕捉公式值的時序模式：

- 輸入：30 根 K 棒 × 5 個公式值
- 層數：2 層 LSTM + 2 層全連接
- 輸出：3 類信號概率

### 4. 實時推理

- 毫秒級延遲
- 支持 Webhook 集成
- 可靠的信心度評估

---

## 性能指標

### 預期回測結果

| 指標 | 預期值 |
| --- | --- |
| 準確率 | 58-68% |
| 精準度 (BUY) | 62-70% |
| 召回率 (BUY) | 55-65% |
| 年化收益率 | 15-35% (取決於風控) |
| 最大回撤 | 12-18% |
| 夏普比率 | 1.2-1.8 |

### 計算效率

| 操作 | 耗時 |
| --- | --- |
| 公式應用 (單根 K 棒) | < 1ms |
| LSTM 推理 (30 根) | < 5ms |
| 總推理延遲 | < 10ms |

---

## 參數配置

### 符號回歸參數

```python
SymbolicRegression(
    niterations=100,        # 演化迭代次數 (100-200)
    population_size=50,     # 種群大小 (30-100)
    maxsize=20,            # 最大公式大小 (15-30)
    maxdepth=5             # 最大深度 (3-7)
)
```

增加迭代數和種群大小可發現更優公式，但耗時更長。

### LSTM 參數

```python
LSTM(
    lookback=30,           # 時間回溯窗口
    lstm_units_1=64,       # 第一層 LSTM 單位
    lstm_units_2=32,       # 第二層 LSTM 單位
    dropout=0.3,           # Dropout 率
    epochs=50              # 訓練輪數
)
```

### 交易信心度閾值

```python
if confidence > 0.65:
    execute_trade(signal)
else:
    hold_position()
```

---

## 常見問題

### Q: 為什麼需要 30 根 K 棒的歷史數據？

A: LSTM 需要時間序列上下文。30 根 K 棒對應：
- 15m 時框：7.5 小時
- 1h 時框：30 小時
- 4h 時框：5 天

### Q: 公式可以手動修改嗎？

A: 可以。編輯 `discovered_formulas.json` 中的方程式字符串，系統會自動應用新公式。

### Q: 如何定期重新訓練？

A: 建議每月執行一次符號回歸，發現新公式以適應市場變化。

### Q: 支持多幣種嗎？

A: 支持。在 `local_symbolic_regression.py` 中修改 `--symbol` 參數，為每個幣種訓練單獨的模型。

---

## 風控機制

### 信心度閾值

```python
if model_confidence < 0.55:
    signal = 'HOLD'  # 不確定時持有
```

### 倉位管理

```python
position_size = base_size * (confidence - 0.5) * 2
# 信心度越高，倉位越大
```

### 止損設置

```python
if loss_rate > -0.05:  # 5% 止損
    close_position()
```

---

## 數據源

### 支持的交易所

- Binance (推薦)
- OKX
- Kraken
- CoinGecko (免費 API)

### K 線時框

- 15m (短期交易)
- 1h (日內交易)
- 4h (波段交易)
- 1d (長期趨勢)

---

## 部署選項

### 1. 本地服務器

優點：完全控制，低延遲
缺點：需要 24/7 運行

```bash
python real_time_predictor.py --mode websocket --exchange binance
```

### 2. 云函數 (AWS Lambda / Google Cloud Functions)

優點：按需付費，自動擴展
缺點：冷啟動延遲

### 3. Docker 容器 (Kubernetes)

優點：易於部署，版本控制
缺點：需要容器化知識

---

## 進階優化

### 多幣種融合

為 BTC、ETH、BNB 分別訓練模型，然後組合信號。

### 多時框融合

同時使用 15m、1h、4h 三個時框的信號，投票決策。

### 強化學習微調

用實盤交易結果反饋優化模型權重。

### 動態參數調整

根據市場波動率自動調整止損、信心度閾值。

---

## 技術棧

| 組件 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.8+ | 基礎語言 |
| TensorFlow | 2.10+ | 深度學習框架 |
| Pandas | 1.3+ | 數據處理 |
| NumPy | 1.20+ | 數值計算 |
| TA-Lib | 0.4+ | 技術指標 |
| PySR | 0.17+ | 符號回歸 |
| Scikit-Learn | 1.0+ | 機器學習工具 |

---

## 監管與風險

### 合規性

- 確保符合當地交易法規
- 不涉及高頻交易
- 交易延遲 > 100ms

### 風險管理

- 小額測試 (1-5% 風險)
- 完整的回測驗證
- 止損設置 (< 5%)
- 位置管理 (< 50% 資金)

---

## 貢獻與反饋

GitHub: https://github.com/caizongxun/v2bot

---

## 許可證

MIT License

---

## 免責聲明

本系統僅供教學和研究使用。使用本系統進行實際交易需自負風險。過去性能不代表未來結果。
