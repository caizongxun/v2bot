# Colab 快速執行指南（無需克隆倉庫）

## 方式 1：直接遠端執行 + 預設參數（推薦新手）

在 Colab Cell 中執行以下代碼：

```python
!pip install -q requests pandas

import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text)
```

**執行結果**：
- 下載全部 23 種幣種
- 15m 和 1h 時框
- 各 50,000 根 K 線
- 耗時：20-40 分鐘

---

## 方式 2：自訂參數 - 只下載特定幣種

```python
!pip install -q requests pandas

import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {'CRYPTO_SYMBOLS': ['BTC', 'ETH', 'BNB']})
```

**執行結果**：
- 僅下載 BTC、ETH、BNB
- 15m 和 1h 時框
- 各 50,000 根 K 線
- 耗時：5-10 分鐘

---

## 方式 3：自訂參數 - 只下載特定時框

```python
!pip install -q requests pandas

import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {'INTERVALS': ['1h']})
```

**執行結果**：
- 全部 23 種幣種
- 僅 1h 時框
- 各 50,000 根 K 線
- 耗時：10-20 分鐘

---

## 方式 4：自訂參數 - 減少 K 線數量（快速測試）

```python
!pip install -q requests pandas

import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {'TARGET_KLINES': 10000})
```

**執行結果**：
- 全部 23 種幣種
- 15m 和 1h 時框
- 各 10,000 根 K 線（而非 50,000）
- 耗時：4-8 分鐘

---

## 方式 5：全部參數組合自訂（進階）

```python
!pip install -q requests pandas

import requests

# 定義參數
params = {
    'CRYPTO_SYMBOLS': ['BTC', 'ETH'],        # 只要 BTC 和 ETH
    'INTERVALS': ['1h'],                      # 只要 1 小時
    'TARGET_KLINES': 20000,                   # 20,000 根 K 線
    'MAX_WORKERS': 3,                         # 3 個並行線程
    'OUTPUT_DIR': './my_crypto_data'          # 自訂輸出目錄
}

exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, params)
```

**執行結果**：
- 下載 BTC 和 ETH
- 1h 時框
- 各 20,000 根 K 線
- 耗時：2-4 分鐘

---

## 參數詳解

### CRYPTO_SYMBOLS
可用幣種列表（預設 = None，表示全部）：

```python
# 可選值
'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT',
'LINK', 'MATIC', 'LTC', 'UNI', 'BCH', 'ETC', 'FIL', 'DOGE',
'ALGO', 'ATOM', 'NEAR', 'ARB', 'OP', 'AAVE', 'SHIB'

# 使用方式
'CRYPTO_SYMBOLS': ['BTC', 'ETH']              # 2 種幣種
'CRYPTO_SYMBOLS': ['BTC']                     # 1 種幣種
'CRYPTO_SYMBOLS': None                        # 全部 23 種
```

### INTERVALS
時框選擇（預設 = None，表示全部）：

```python
# 可選值
'15m'   # 15 分鐘
'1h'    # 1 小時

# 使用方式
'INTERVALS': ['15m', '1h']       # 兩個時框
'INTERVALS': ['1h']              # 只要 1 小時
'INTERVALS': ['15m']             # 只要 15 分鐘
'INTERVALS': None                # 全部時框
```

### TARGET_KLINES
K 線數量（預設 = 50000）：

```python
# 推薦值
10000   # 快速測試 (4-8 分鐘)
20000   # 平衡版 (8-15 分鐘)
50000   # 完整版 (20-40 分鐘) ← 推薦
100000  # 超完整 (60+ 分鐘，可能超時)

# 使用方式
'TARGET_KLINES': 10000
```

### MAX_WORKERS
並行線程數（預設 = 5）：

```python
# 推薦值
2       # 超穩定 (Colab 免費版)
3       # 穩定版
5       # 平衡版 ← 推薦
10      # 高速版 (可能超時)

# 使用方式
'MAX_WORKERS': 3
```

### OUTPUT_DIR
數據存儲位置（預設 = './crypto_data_cache'）：

```python
# 使用方式
'OUTPUT_DIR': './crypto_data_cache'  # 預設位置
'OUTPUT_DIR': '/content/my_data'     # 自訂位置
```

---

## 常用場景速查

### 場景 1：快速測試（2-5 分鐘）

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {
    'CRYPTO_SYMBOLS': ['BTC'],
    'INTERVALS': ['1h'],
    'TARGET_KLINES': 5000
})
```

### 場景 2：單幣種完整歷史（5-10 分鐘）

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {
    'CRYPTO_SYMBOLS': ['ETH'],
    'INTERVALS': ['15m', '1h'],
    'TARGET_KLINES': 50000
})
```

### 場景 3：Top 5 幣種（10-20 分鐘）

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {
    'CRYPTO_SYMBOLS': ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'],
    'INTERVALS': ['1h'],
    'TARGET_KLINES': 50000
})
```

### 場景 4：全部幣種（預設，20-40 分鐘）

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text)
```

### 場景 5：完全自訂

```python
!pip install -q requests pandas
import requests
exec(requests.get(
    'https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py'
).text, {
    'CRYPTO_SYMBOLS': ['BTC', 'ETH', 'BNB'],
    'INTERVALS': ['1h'],
    'TARGET_KLINES': 10000,
    'MAX_WORKERS': 2
})
```

---

## 執行流程

```
複製代碼到 Colab Cell
    ↓
點擊運行 (▶)
    ↓
自動安裝 pandas
    ↓
遠端下載腳本
    ↓
開始並行下載
    ↓
每 10-20 秒顯示進度
    ↓
完成！生成 CSV 文件
    ↓
自動生成 fetch_report.json
```

---

## 查看結果

### 查看文件列表

```python
# 在新 Cell 中執行
import os
from pathlib import Path

data_dir = Path('./crypto_data_cache')
csv_files = list(data_dir.glob('*.csv'))

print(f'總文件數: {len(csv_files)}')
for f in sorted(csv_files):
    print(f'  {f.name}')
```

### 查看數據內容

```python
import pandas as pd

df = pd.read_csv('./crypto_data_cache/BTC_1h.csv')
print(f'BTC 1h 統計:')
print(f'  行數: {len(df)}')
print(f'  時間範圍: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
print(f'\n前 5 行:')
print(df.head())
```

### 查看執行報告

```python
import json

with open('./crypto_data_cache/fetch_report.json') as f:
    report = json.load(f)
    print(json.dumps(report, indent=2))
```

---

## 錯誤排除

### 錯誤 1："No module named 'pandas'"

確保執行了安裝命令：
```python
!pip install -q requests pandas
```

### 錯誤 2："Request timeout"

減少並行線程數：
```python
'MAX_WORKERS': 2
```

### 錯誤 3："API rate limit"

等待幾分鐘後重試，或減少幣種數量

### 錯誤 4："Unknown symbol"

檢查幣種名稱是否正確（區分大小寫）：
```python
'CRYPTO_SYMBOLS': ['BTC', 'ETH']  # 正確
'CRYPTO_SYMBOLS': ['btc', 'eth']  # 錯誤
```

---

## 下一步：上傳到 Hugging Face

完成下載後，可選上傳到 Hugging Face：

```python
!pip install -q huggingface-hub

from huggingface_hub import HfApi

token = input('Enter HF token: ')
repo_name = input('Enter repo name: ')

api = HfApi(token=token)
user = api.whoami()['name']
repo_id = f'{user}/{repo_name}'

print(f'Uploading to {repo_id}...')
api.upload_folder(
    folder_path='./crypto_data_cache',
    repo_id=repo_id,
    repo_type='dataset',
    multi_commit=True
)
print(f'✓ Done! https://huggingface.co/datasets/{repo_id}')
```

---

## 提示 & 技巧

1. **分批下載**：如果擔心超時，分多個 Cell 執行不同的幣種
2. **定期保存**：Colab 會話超過 12 小時自動斷開，最好定期下載
3. **檢查進度**：留意日誌輸出，每個幣種會顯示進度
4. **避免重複**：同一目錄已有的文件會被覆蓋
5. **GPU 加速**：啟用 GPU 不能加速此任務（受 I/O 限制），但可用於後續模型訓練

---

**最後提醒**：所有數據僅供研究用途。交易風險自負！
