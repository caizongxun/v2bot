# Colab 地下志 - 最粗壘简介

## 二選一：複製前 直接適用

### 預設 - 全部 23 幣種、兩時框、50K K 線

```python
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text)
```
‣ **耗時**: 20-40 分鐘

---

### 変式 1 - 只要 BTC, ETH, BNB

```python
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text, 
     {'CRYPTO_SYMBOLS': ['BTC', 'ETH', 'BNB']})
```
‣ **耗時**: 5-10 分鐘

---

### 変式 2 - 只要 1h K 線

```python
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text,
     {'INTERVALS': ['1h']})
```
‣ **耗時**: 10-20 分鐘

---

### 変式 3 - 快速測試（只要 10K K 線）

```python
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text,
     {'TARGET_KLINES': 10000})
```
‣ **耗時**: 4-8 分鐘

---

### 変式 4 - 全部自訂 (BTC 1h 10K)

```python
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text,
     {'CRYPTO_SYMBOLS': ['BTC'], 'INTERVALS': ['1h'], 'TARGET_KLINES': 10000})
```
‣ **耗時**: 1-2 分鐘

---

## 參數速查

| 參數 | 可選值 | 預設 | 效果 |
|--------|-----------|--------|------|
| `CRYPTO_SYMBOLS` | `['BTC', 'ETH', ...]` | `None` (全部 23 種) | 針對性選擇幣種 |
| `INTERVALS` | `['15m']` 或 `['1h']` | `None` (兩個) | 針對性選擇時框 |
| `TARGET_KLINES` | 10000 - 100000 | 50000 | 這每種時框每幣種的 K 線數 |
| `MAX_WORKERS` | 1 - 10 | 5 | 並行線程 (超穩定用 2，常規用 5) |
| `OUTPUT_DIR` | 任意位置 | `./crypto_data_cache` | 數據存放位置 |

---

## 流行上傳到 Hugging Face

### 第 1 步: 獲取 Token
訪問: https://huggingface.co/settings/tokens → 建立新 Token → 合佐 "Write"

### 第 2 步: 執行上傳代碼

```python
!pip install -q huggingface-hub
from huggingface_hub import HfApi

token = input('HF Token: ')
repo = input('Repo name: ')
api = HfApi(token=token)
user = api.whoami()['name']
api.upload_folder(
    folder_path='./crypto_data_cache',
    repo_id=f'{user}/{repo}',
    repo_type='dataset'
)
```

---

## 常用抈但

### 所有幣種名稱

```
BTC ETH BNB SOL XRP ADA AVAX DOT LINK MATIC LTC UNI BCH ETC FIL DOGE ALGO ATOM NEAR ARB OP AAVE SHIB
```

### 查看輸出

```python
import pandas as pd
from pathlib import Path

data_dir = Path('./crypto_data_cache')
csv_files = list(data_dir.glob('*.csv'))
print(f'\u7e3d文件數: {len(csv_files)}')
for f in sorted(csv_files):
    df = pd.read_csv(f)
    print(f'{f.name}: {len(df)} rows')
```

### 查看報告

```python
import json
with open('./crypto_data_cache/fetch_report.json') as f:
    print(json.dumps(json.load(f), indent=2))
```

---

## 故障排除

| 問題 | 解決 |
|------|------|
| ModuleNotFoundError: pandas | 重新執行 `!pip install -q requests pandas` |
| API rate limit | 減少 `MAX_WORKERS` 或減少幣種 |
| Connection timeout | 等待幾分鐘 或減少 `TARGET_KLINES` |
| Unknown symbol | 幣種名稱大寫写（正碼：BTC，ERROR：btc） |

---

## 非常寶賯

1. 只有第一個 Cell 需要 `!pip install` ，後續 Cell 可沒有
2. 使用 `exec()` 待緩步技巧：先捕程式碼、再詳行修改、最後下下讋
3. 不需要克隆整個倉庫，囟偏お直接遠端執行腳本
4. 紙交易削减費用：Colab 會話超過 12 小時自動斷開，提削這個時間以上执行

---

**上傳 Hugging Face 重點：**倗但複製整個目錄，不需要一個檔一個上傳
