# v2 虛擬貨幣因子發現與交易信號系統

完全自動化的 BTC 因子發現、優化和交易信號生成系統。基於 219K+ 條真實 K 線數據。

## 📊 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: 因子發現系統 (run_colab_v2.py)                        │
├─────────────────────────────────────────────────────────────────┤
│ 輸入: BTC 15m K線 (219,643 條)                                  │
│ 處理:                                                            │
│  ├─ 數據清洗 (移除異常值)                                        │
│  ├─ 技術指標生成 (12 個標準指標)                                │
│  └─ 符號回歸發現新公式 (遺傳算法)                               │
│ 輸出: btc_factors.pkl (12 個因子)                               │
│       btc_analysis.json (因子統計)                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: 因子回測與優化 (backtest_and_optimize.py)             │
├─────────────────────────────────────────────────────────────────┤
│ 輸入: 12 個因子                                                  │
│ 處理:                                                            │
│  ├─ 單因子策略評估                                              │
│  │  ├─ Sharpe 比 (Best: RSI_7 = 5.77)                          │
│  │  ├─ 年化收益率                                               │
│  │  ├─ 最大回撤                                                 │
│  │  └─ 勝率                                                     │
│  └─ 多因子優化 (遺傳算法)                                       │
│     ├─ 30 代進化                                                │
│     ├─ 50 個種群                                                │
│     └─ 目標: 最大化 Sharpe 比                                   │
│ 輸出: optimization_results.json                                 │
│       最優 Sharpe: 6.7697 (+17% vs 最佳單因子)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: 實時監控與信號生成 (realtime_monitoring.py)           │
├─────────────────────────────────────────────────────────────────┤
│ 輸入: 優化結果 + 因子數據                                       │
│ 處理:                                                            │
│  ├─ 實時風險指標計算                                            │
│  ├─ 信號質量評估                                                │
│  ├─ 因子貢獻分析                                                │
│  ├─ 交易信號生成 (LONG/FLAT)                                    │
│  ├─ 警告系統                                                    │
│  └─ 頭寸規模動態調整                                            │
│ 輸出: trading_dashboard.json (實時信號)                         │
│       Stop Loss / Take Profit 水位                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 關鍵結果

### 因子發現
- **數據量**: 219,643 條 BTC 15m K線
- **發現因子**: 12 個技術指標 + 2 個符號回歸公式
- **最優單因子**: RSI_7 (Sharpe = 5.77)

### 優化結果

| 指標 | 值 |
|------|-----|
| 最優組合 Sharpe | 6.7697 |
| 提升幅度 | +17.4% vs RSI_7 |
| 迭代代數 | 30 |
| 種群規模 | 50 |

### 最優因子權重

| 排名 | 因子 | 權重 | 貢獻度 |
|------|------|------|--------|
| 1 | RSI_7 | 67.61% | 主導因子 |
| 2 | price_change | 14.06% | 輔助因子 |
| 3 | SMA_5 | 7.35% | 微調 |
| 4 | RSI_14 | 6.43% | 微調 |
| 5+ | Others | 4.56% | 噪聲過濾 |

### 實時信號表現 (最近 100 bars)

| 期間 | 占比 | 收益 |
|------|------|------|
| LONG | 36% | -0.2150 |
| FLAT | 64% | +0.2755 |
| **總計** | **100%** | **+0.0605** |

## 🚀 使用方法

### 在 Google Colab 中運行

#### Step 1: 因子發現
```python
import requests

url = "https://raw.githubusercontent.com/caizongxun/v2bot/main/factor_discovery/run_colab_v2.py"
code = requests.get(url).text
exec(code)
```

輸出:
- `/tmp/btc_factors.pkl` - 因子數據
- `/tmp/btc_analysis.json` - 因子質量指標

#### Step 2: 回測與優化
```python
import requests

url = "https://raw.githubusercontent.com/caizongxun/v2bot/main/factor_discovery/backtest_and_optimize.py"
code = requests.get(url).text
exec(code)
```

輸出:
- `/tmp/optimization_results.json` - 優化後的因子權重和信號

#### Step 3: 實時監控
```python
import requests

url = "https://raw.githubusercontent.com/caizongxun/v2bot/main/factor_discovery/realtime_monitoring.py"
code = requests.get(url).text
exec(code)
```

輸出:
- `/tmp/trading_dashboard.json` - 實時交易面板

## 📈 因子詳解

### 生成的技術指標

1. **price_change** - 收盤相對開盤變化率
2. **high_low_ratio** - 高低價差相對收盤比
3. **SMA_5/10/20/50** - 簡單移動平均線 (5/10/20/50 週期)
4. **RSI_7/14/21** - 相對強度指標 (7/14/21 週期)
5. **MACD** - 指數平滑移動平均線差
6. **log_volume** - 成交量對數值
7. **volume_ratio** - 成交量相對均值比

### 符號回歸發現的公式
- Formula_1: `(open - (log(high) + (close - open)))`
- Formula_2: `(abs(abs(close)) + volume)`

## ⚠️ 風險管理

系統包含三層風險管理:

### 1. 信號質量警告
- 低信心度 (< 30%)
- 頻繁信號變化 (抽鞭風險)
- 弱信號強度 (近零值)
- 信號剛改變 (使用謹慎)

### 2. 動態頭寸規模
```
Position Size = Signal Confidence × 100%
              = (Persistence × 0.4 + Strength × 0.6) × 100%
```

### 3. 自動 Stop Loss / Take Profit
```
Stop Loss  = Current Score - 1.0
Take Profit = Current Score + 1.5
```

## 📊 當前交易信號

```
時間戳: 2025-12-30 08:56:32

信號: ⚫ FLAT
信心度: 28.10%
持續期: 3 根 K 線
頭寸規模: 28.10% 資本

投資組合得分: -0.1424
波動率: 0.6360

風險警告:
  ⚠ 信號信心度低 (< 30%)
  ⚠ 頻繁信號變化 (抽鞭風險)
  ⚠ 信號強度弱 (近零)

建議:
  1. 保持觀望或減少敞口
  2. 等待信號信心度提升至 50% 以上
  3. 監控最近 4-5 根 K 線的轉折點
```

## 🔄 工作流程

### 一、完整重新計算 (推薦用於定期更新)

按順序執行三個腳本:

1. `run_colab_v2.py` - 從頭生成因子
2. `backtest_and_optimize.py` - 優化因子權重
3. `realtime_monitoring.py` - 生成實時信號

時間: ~5-10 分鐘

### 二、快速刷新信號 (推薦用於日常監控)

只執行:

```python
# 直接加載已優化的因子和權重，生成最新信號
import requests

url = "https://raw.githubusercontent.com/caizongxun/v2bot/main/factor_discovery/realtime_monitoring.py"
code = requests.get(url).text
exec(code)
```

時間: <1 分鐘

## 📁 輸出文件說明

### `btc_factors.pkl`
Pickle 格式，包含:
- `price_change`: numpy array (218011,)
- `high_low_ratio`: numpy array (218011,)
- `SMA_5`: numpy array (218011,)
- ...
- `volume_ratio`: numpy array (218011,)

### `btc_analysis.json`
```json
{
  "price_change": {
    "corr": 0.0007,
    "vol": 0.002889,
    "sharpe": -0.0000
  },
  ...
}
```

### `optimization_results.json`
```json
{
  "best_weights": {
    "RSI_7": 0.6761,
    "price_change": 0.1406,
    ...
  },
  "best_sharpe": 6.7697,
  "backtest_results": {...},
  "signals": [1, 0, 1, ...],
  "portfolio_scores": [0.5, -0.2, ...]
}
```

### `trading_dashboard.json`
```json
{
  "timestamp": "2025-12-30T08:56:32.123456",
  "current_signal": "FLAT",
  "metrics": {
    "signal_confidence": 0.2810,
    "portfolio_score": -0.1424,
    "score_volatility": 0.6360,
    ...
  },
  "recommendation": {
    "action": "⚫ FLAT",
    "confidence": 0.2810,
    "position_size": 0.2810,
    "stop_loss": -1.1424,
    "take_profit": 1.3576
  },
  "alerts": ["⚠ Low signal confidence...", ...]
}
```

## 🔍 監控指標詳解

| 指標 | 定義 | 理想範圍 | 當前值 |
|------|------|---------|--------|
| Signal Confidence | 信號持續性 + 強度 | > 50% | 28.10% |
| Signal Persistence | 連續相同信號根數 | > 10 bars | 3 bars |
| Score Strength | 投資組合得分相對最大值 | > 50% | 6.84% |
| Signal Frequency | 每根 K 線的信號變化率 | < 5% | 33.33% |
| Score Volatility | 組合得分波動率 | 0.3-0.7 | 0.6360 |
| Max Drawdown (Long) | 做多期間的最大回撤 | > -20% | -13.36% |

## 💡 使用建議

### 何時開倉
1. **信心度 > 60%** 且**連續 > 5 根 K 線** - 開滿倉
2. **信心度 40-60%** - 開半倉
3. **信心度 < 30%** - 只觀望，不開倉

### 風險管理
1. **設定止損** 在 `Stop Loss` 水位
2. **分批止盈** 在 `Take Profit` 水位
3. **動態調整頭寸** 根據 `Position Size` 建議
4. **監控警告** 如出現 3+ 個警告，考慮縮小頭寸

### 最佳交易時間
- 優先在信號剛剛出現時 (< 5 根 K 線)
- 避免在高頻信號變化期間交易 (频率 > 10%)
- 等待信心度穩定在 50% 以上

## 🛠️ 自定義參數

編輯各腳本的以下部分:

### `run_colab_v2.py`
```python
# 符號回歸參數
syreg = SymReg(data_dict, pop_size=40, gens=40)  # 調整種群和代數
```

### `backtest_and_optimize.py`
```python
# 遺傳算法參數
optimizer = GeneticOptimizer(factors, prices, pop_size=50, generations=30)
```

### `realtime_monitoring.py`
```python
# 頭寸規模計算
position_size = overall_confidence  # 可調整權重
```

## 📞 故障排除

### 問題: "IndexError: index out of bounds"
- **原因**: 因子長度不一致
- **解決**: 確保所有因子都經過清洗，長度相同

### 問題: "NameError: name 'xxx' is not defined"
- **原因**: 變數前向參考
- **解決**: 檢查變數定義順序

### 問題: Sharpe 比異常高
- **原因**: 過度擬合或資料泄漏
- **解決**: 檢查因子中是否使用了未來數據

## 📈 性能基準

### 系統執行時間

| 階段 | 耗時 | CPU | RAM |
|------|------|-----|-----|
| Stage 1 (發現) | 2-3 分鐘 | 中 | 2-3 GB |
| Stage 2 (優化) | 1-2 分鐘 | 高 | 1-2 GB |
| Stage 3 (監控) | 10-30 秒 | 低 | <500 MB |
| **總計** | **4-5 分鐘** | - | - |

### 數據規模

| 指標 | 值 |
|------|-----|
| K 線數量 | 219,643 |
| 清洗後 | 218,011 |
| 因子數量 | 12 |
| 優化代數 | 30 |
| 種群規模 | 50 |
| 總評估次數 | 1,500+ |

## 🔮 未來改進方向

1. **多時間框架分析** - 添加 1h, 4h, 1d 級別
2. **機器學習因子** - 用 LSTM/XGBoost 替代符號回歸
3. **市場制度檢測** - 自動區分趨勢/震蕩市
4. **實時執行** - 接入交易所 API
5. **組合管理** - 同時監控多個交易對

## 📝 更新日誌

### v2.0 (2025-12-30)
- ✅ 完整的因子發現系統
- ✅ 遺傳算法優化
- ✅ 實時監控面板
- ✅ 風險管理系統
- ✅ 完整文檔

---

**作者**: zongowo111  
**最後更新**: 2025-12-30  
**License**: MIT
