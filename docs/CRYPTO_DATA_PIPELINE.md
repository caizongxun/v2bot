# V2 加密貨幣數據管道文檔

## 概述

本文檔詳細說明了 v2bot 的加密貨幣數據爬取、處理和上傳管道。

## 架構

```
┌─────────────────────┐
│  Binance API        │
│  (OHLCV 歷史數據)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ crypto_fetcher      │
│ (Colab 優化版)      │
│ - 並行獲取多幣種    │
│ - 實時進度顯示      │
│ - 自動重試邏輯      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ CSV 文件生成        │
│ 46 個文件           │
│ (23幣種 x 2時框)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ HF 上傳器           │
│ (自動創建 Repo)    │
│ - 驗證 Token        │
│ - 創建數據集 Repo   │
│ - 生成 README       │
│ - 批量上傳文件      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hugging Face        │
│ zongowo111/         │
│ v2-crypto-ohlcv...  │
│ (公開可下載)        │
└─────────────────────┘
```

## 完整流程

### 1. 數據爬取 (Colab)

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_colab.py'
).text, {'TARGET_KLINES': 80000})
```

**功能:**
- 從 Binance API 爬取 23 種加密貨幣
- 支持自訂時間框 (15分鐘、1小時)
- 支持自訂 K 線數量 (1000-500000)
- 實時進度顯示
- 自動生成 CSV 文件
- 生成 JSON 格式報告

**參數説明:**

| 參數 | 說明 | 預設值 | 範圍 |
|------|------|--------|------|
| `TARGET_KLINES` | 每個幣種/時框的 K 線數 | 50000 | 1000-500000 |
| `CRYPTO_SYMBOLS` | 爬取的幣種列表 | None (全部) | None 或列表 |
| `INTERVALS` | 時間框列表 | None (全部) | None 或 ['15m', '1h'] |
| `OUTPUT_DIR` | 輸出目錄 | './crypto_data_cache' | 字符串路徑 |

**輸出文件:**
- `{SYMBOL}_{INTERVAL}.csv` - 歷史 K 線數據
- `fetch_report.json` - 爬取統計報告
- `README.md` - 數據文檔

**數據統計 (已完成):**
- 總文件數: 46
- 總數據行數: 2,765,875
- 總大小: 162 MB
- 爬取耗時: 16.5 分鐘

### 2. 數據上傳到 Hugging Face

```python
!pip install -q huggingface-hub
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/hf_uploader_auto_create.py'
).text)
```

**步驟:**

1. **Token 驗證** - 輸入 Hugging Face 寫入權限 Token
2. **Repo 名稱** - 指定或使用預設名稱 (v2-crypto-ohlcv-data)
3. **數據目錄** - 選擇包含 CSV 文件的目錄
4. **自動創建** - 腳本自動在 HF 上創建公開數據集 Repo
5. **上傳** - 批量上傳所有文件和元數據

**Hugging Face Repo:**
- 連結: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- 狀態: 公開
- 大小: 162 MB
- 文件數: 47 (46 CSV + 1 README)

## 數據規格

### CSV 文件結構

每個 CSV 文件包含以下欄位:

```
timestamp,symbol,open,high,low,close,volume
2025-01-01 00:00:00,BTCUSDT,42500.50,42600.00,42400.25,42550.75,1234.56
```

| 欄位 | 類型 | 說明 |
|------|------|------|
| timestamp | datetime | UTC 時間戳 |
| symbol | string | 交易對 (如 BTCUSDT) |
| open | float | 開盤價 |
| high | float | 最高價 |
| low | float | 最低價 |
| close | float | 收盤價 |
| volume | float | 交易量 |

### 包含的幣種 (23 種)

```
BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOT, LINK, MATIC, LTC, UNI,
BCH, ETC, FIL, DOGE, ALGO, ATOM, NEAR, ARB, OP, AAVE, SHIB
```

### 時間框 (2 種)

- `15m` - 15分鐘 K 線 (一般 80000 根)
- `1h` - 1小時 K 線 (一般 35000-55000 根)

## 使用方式

### 方式 1: 使用 Hugging Face Datasets 库

```python
from datasets import load_dataset

# 加載整個數據集
ds = load_dataset('zongowo111/v2-crypto-ohlcv-data')

# 查看所有文件
print(ds.keys())

# 轉換為 pandas
df = ds['train'].to_pandas()
```

### 方式 2: 直接下載

```python
from huggingface_hub import snapshot_download

# 下載整個數據集到本地
data_dir = snapshot_download(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    repo_type='dataset'
)
print(f"Data downloaded to: {data_dir}")
```

### 方式 3: 讀取單個 CSV 文件

```python
import pandas as pd

# 加載比特幣 1 小時數據
btc_1h = pd.read_csv('BTC_1h.csv')
print(f"Bitcoin data shape: {btc_1h.shape}")
print(btc_1h.head())

# 加載以太坊 15 分鐘數據
eth_15m = pd.read_csv('ETH_15m.csv')
```

### 方式 4: 批量加載所有數據

```python
import pandas as pd
from pathlib import Path

def load_all_crypto_data(data_dir='crypto_data_cache'):
    """加載所有 CSV 文件"""
    data = {}
    for csv_file in Path(data_dir).glob('*.csv'):
        symbol_interval = csv_file.stem
        data[symbol_interval] = pd.read_csv(csv_file)
    return data

# 使用
all_data = load_all_crypto_data()
print(f"Loaded {len(all_data)} datasets")

# 查看特定數據
for key, df in all_data.items():
    print(f"{key}: {len(df)} rows")
```

## 數據質量

- **時間完整性** - 無遺漏的時間戳
- **字段完整性** - 所有 OHLCV 字段完整
- **數據排序** - 按時間戳遞增排序
- **數據類型** - 正確的數據類型轉換
- **時區** - 所有時間戳均為 UTC
- **缺失值** - 無 NaN 值

## 常見場景

### 場景 1: 回測加密貨幣策略

```python
import pandas as pd
from datetime import datetime, timedelta

# 加載數據
btc_1h = pd.read_csv('BTC_1h.csv')
eth_1h = pd.read_csv('ETH_1h.csv')

# 轉換時間戳
btc_1h['timestamp'] = pd.to_datetime(btc_1h['timestamp'])

# 計算技術指標
btc_1h['sma_20'] = btc_1h['close'].rolling(20).mean()
btc_1h['rsi'] = calculate_rsi(btc_1h['close'])

# 篩選時間範圍
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
mask = (btc_1h['timestamp'] >= start_date) & (btc_1h['timestamp'] <= end_date)
test_data = btc_1h[mask]
```

### 場景 2: 機器學習特徵工程

```python
import pandas as pd
import numpy as np

# 加載數據
data = pd.read_csv('BTC_15m.csv')

# 特徵工程
data['returns'] = data['close'].pct_change()
data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
data['volatility'] = data['returns'].rolling(20).std()
data['price_range'] = (data['high'] - data['low']) / data['open']
data['volume_ma'] = data['volume'].rolling(20).mean()

# 移除 NaN
data = data.dropna()

# 準備訓練數據
features = ['returns', 'volatility', 'price_range', 'volume_ma']
X = data[features].values
y = (data['close'].shift(-1) > data['close']).astype(int).values
```

### 場景 3: 實時監控多幣種

```python
import pandas as pd
from pathlib import Path

# 加載所有 1 小時數據
data_dir = 'crypto_data_cache'
crpto_data = {}

for csv_file in Path(data_dir).glob('*_1h.csv'):
    symbol = csv_file.stem.split('_')[0]
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 獲取最新數據
    latest = df.iloc[-1]
    crypto_data[symbol] = {
        'price': latest['close'],
        'volume': latest['volume'],
        'high': latest['high'],
        'low': latest['low']
    }

# 顯示監控面板
for symbol, data in sorted(crypto_data.items()):
    print(f"{symbol}: ${data['price']:.2f} (Vol: {data['volume']:.2f})")
```

## 更新流程

### 定期更新數據

```bash
# 1. 在 Colab 中運行爬取腳本
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_colab.py'
).text, {'TARGET_KLINES': 80000})

# 2. 運行上傳腳本
!pip install -q huggingface-hub
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/hf_uploader_auto_create.py'
).text)
# 選擇覆蓋現有 repo
```

## 文件位置

### GitHub Repository
- 爬取腳本: `data_fetcher/crypto_fetcher_colab.py`
- 上傳腳本: `data_fetcher/hf_uploader_auto_create.py`
- 文檔: `docs/CRYPTO_DATA_PIPELINE.md`

### Hugging Face
- 數據集: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data

## 常見問題

### Q1: 如何只下載特定幣種的數據？

```python
from huggingface_hub import list_repo_files

# 列出所有文件
files = list_repo_files(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    repo_type='dataset'
)

# 過濾
btc_files = [f for f in files if 'BTC' in f]
eth_files = [f for f in files if 'ETH' in f]
```

### Q2: 如何增加 K 線數量？

修改爬取腳本的參數：
```python
exec(requests.get(...).text, {'TARGET_KLINES': 200000})  # 增加到 20 萬根
```

### Q3: 數據多久更新一次？

目前手動更新。如需自動更新，可使用 GitHub Actions 或雲函數定時觸發。

### Q4: 如何計算技術指標？

```python
import pandas as pd
import numpy as np

def calculate_sma(prices, period=20):
    return prices.rolling(window=period).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df = pd.read_csv('BTC_1h.csv')
df['sma_20'] = calculate_sma(df['close'], 20)
df['rsi'] = calculate_rsi(df['close'], 14)
```

## 授權與免責

- **授權**: MIT License - 自由使用於研究和教育目的
- **數據來源**: Binance REST API (binance.us)
- **免責聲明**: 過去表現不代表未來結果。交易前進行充分盡職調查。

## 聯絡方式

- GitHub: https://github.com/caizongxun/v2bot
- Hugging Face: https://huggingface.co/zongowo111

## 版本歷史

### v1.0 (2025-12-30)
- 初始版本
- 23 種加密貨幣
- 2 種時間框 (15m, 1h)
- 80000 K 線/幣種/時框
- 自動上傳到 Hugging Face
