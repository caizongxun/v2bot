# 快速開始指南 (5 分鐘上手)

## 最短路徑：在 Colab 上執行

### 步驟 1: 打開 Google Colab (30 秒)

1. 訪問 [Google Colab](https://colab.research.google.com/)
2. 點擊「新增筆記本」
3. 複製下方代碼到第一個 Cell

### 步驟 2: 執行數據抓取 (複製並運行)

在 Colab Cell 中執行：

```python
# 一鍵執行：安裝依賴 + 克隆倉庫 + 開始抓取
!pip install -q requests pandas yfinance huggingface-hub
!git clone https://github.com/caizongxun/v2bot.git /content/v2bot

import sys
sys.path.insert(0, '/content/v2bot')

from data_fetcher.crypto_historical_data_fetcher import CryptoDataFetcher

fetcher = CryptoDataFetcher(output_dir='/content/crypto_data_cache')
results = fetcher.fetch_all_cryptos_parallel(max_workers=5)
fetcher.generate_summary_report(results)
```

**⏱ 等待時間: 20-40 分鐘** (取決於網速)

### 步驟 3: 上傳到 Hugging Face (可選)

1. 訪問 https://huggingface.co/settings/tokens 獲取 token
2. 在新 Cell 中執行：

```python
from huggingface_hub import HfApi

HF_TOKEN = input('Enter your HuggingFace token: ')
HF_REPO = input('Enter repository name (e.g., v2-crypto-data): ')

api = HfApi(token=HF_TOKEN)
user = api.whoami()['name']
repo_id = f'{user}/{HF_REPO}'

print(f'Uploading to {repo_id}...')
api.upload_folder(
    folder_path='/content/crypto_data_cache',
    repo_id=repo_id,
    repo_type='dataset',
    multi_commit=True
)
print(f'✓ Done! https://huggingface.co/datasets/{repo_id}')
```

**⏱ 上傳時間: 5-15 分鐘**

---

## 配置速查表

### 如果網速慢？

減少目標 K 線數：
```python
from data_fetcher.crypto_historical_data_fetcher import FetcherConfig
FetcherConfig.TARGET_KLINES = 10000  # 改為 1 分鐘內完成
```

### 如果想要更多數據？

增加目標 K 線數：
```python
FetcherConfig.TARGET_KLINES = 100000  # 完整歷史數據
```

### 如果只想要特定幣種？

修改幣種列表：
```python
fetcher.CRYPTO_SYMBOLS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'BNB': 'BNBUSDT',  # 只下載這 3 種
}
```

### 如果 Colab 超時？

分批執行：
```python
# 第一批
symbols_batch_1 = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
fetcher.CRYPTO_SYMBOLS = symbols_batch_1
results_1 = fetcher.fetch_all_cryptos_parallel()

# 第二批
symbols_batch_2 = {'BNB': 'BNBUSDT', 'SOL': 'SOLUSDT'}
fetcher.CRYPTO_SYMBOLS = symbols_batch_2
results_2 = fetcher.fetch_all_cryptos_parallel()
```

---

## 輸出文件檢查

執行完成後，查看数据统计：

```python
# 檢查生成的文件
import os
from pathlib import Path

data_dir = Path('/content/crypto_data_cache')
csv_files = list(data_dir.glob('*.csv'))

print(f'總文件數: {len(csv_files)}')
print('\n示例文件:')
for f in csv_files[:5]:
    size_mb = f.stat().st_size / (1024**2)
    print(f'  {f.name} ({size_mb:.2f} MB)')
```

預期輸出：
```
總文件數: 46
示例文件:
  BTC_15m.csv (45.32 MB)
  BTC_1h.csv (22.67 MB)
  ETH_15m.csv (38.19 MB)
  ETH_1h.csv (19.08 MB)
  BNB_15m.csv (28.45 MB)
```

---

## 故障排除

### 問題 1: "ModuleNotFoundError: No module named 'requests'"

**解決**: 重新執行安裝命令
```python
!pip install -q requests pandas yfinance huggingface-hub
```

### 問題 2: "API rate limit exceeded"

**解決**: 減少並行度
```python
results = fetcher.fetch_all_cryptos_parallel(max_workers=2)  # 改為 2
```

### 問題 3: "Connection timeout"

**解決**: 使用備用方案
```python
# 重新嘗試連接
import time
time.sleep(60)  # 等待 60 秒
results = fetcher.fetch_all_cryptos_parallel()
```

### 問題 4: Colab 連接中斷

**解決**: 檢查 Colab 設定
- 點擊右上角 "⚙️ 設定"
- 啟用 "在後台連續運行" (需 Colab Pro)
- 或定期點擊屏幕保持連接

---

## 數據驗證

執行完成後驗證數據完整性：

```python
import pandas as pd

# 檢查單個文件
df = pd.read_csv('/content/crypto_data_cache/BTC_1h.csv')
print(f'BTC 1h 數據:')
print(f'  總行數: {len(df)}')
print(f'  時間範圍: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
print(f'  缺失值: {df.isnull().sum().sum()}')
print(f'  數據點: {df[["open", "high", "low", "close", "volume"]]}')
```

---

## 下載本地（可選）

如果想將數據下載到本地：

```python
# Colab 中執行
from google.colab import files
import shutil

shutil.make_archive('crypto_data', 'zip', '/content/crypto_data_cache')
files.download('crypto_data.zip')
print('✓ 下載完成')
```

---

## 下一步

✓ 數據已準備好！現在可以開始：

1. **特徵工程** - 計算技術指標
2. **模型訓練** - LSTM/CNN 預測
3. **回測** - 驗證交易策略
4. **實盤** - 紙交易驗證

參考完整文檔：[README.md](./README.md)

---

## 常用命令速查

| 任務 | 代碼 |
|------|------|
| 查看数据統計 | `fetcher.get_data_statistics()` |
| 生成摘要報告 | `fetcher.generate_summary_report(results)` |
| 只下載特定幣種 | `fetcher.CRYPTO_SYMBOLS = {...}` |
| 減少 K 線數 | `FetcherConfig.TARGET_KLINES = 10000` |
| 增加並行線程 | `fetcher.fetch_all_cryptos_parallel(max_workers=8)` |
| 上傳到 HF | `api.upload_folder(...)` |

---

**總耗時**: 25-55 分鐘 (包含上傳)

**下一步**: 進行特徵工程與模型訓練 🚀
