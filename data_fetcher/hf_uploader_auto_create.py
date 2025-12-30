"""
Hugging Face 數據集上傳器 - 一次性上傳整個資料夾

功能:
- 自動創建數據集 repo
- CSV 轉換為 Parquet
- 按幣種分類組織
- 一次性上傳整個資料夾（減少 API 限制風險）

使用方式:
!pip install -q huggingface-hub pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/hf_uploader_auto_create.py').text)
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from huggingface_hub import HfApi, HfFolder, repo_exists, create_repo
    import pandas as pd
except ImportError:
    print("[Installing required packages...]")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub", "pandas"])
    from huggingface_hub import HfApi, HfFolder, repo_exists, create_repo
    import pandas as pd
    print("Packages installed successfully\n")

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{timestamp}]"
    
    if level == "INFO":
        print(f"{prefix} {message}")
    elif level == "WARNING":
        print(f"{prefix} WARNING: {message}")
    elif level == "ERROR":
        print(f"{prefix} ERROR: {message}")
    elif level == "SUCCESS":
        print(f"{prefix} SUCCESS: {message}")

def get_hf_token():
    """獲取 Hugging Face token"""
    print("\n" + "="*70)
    print("STEP 1: HUGGING FACE TOKEN")
    print("="*70)
    
    # 檢查是否已經設置
    saved_token = HfFolder.get_token()
    if saved_token:
        print(f"Found saved token: {saved_token[:10]}...")
        use_saved = input("Use saved token? (y/n, default=y): ").strip().lower()
        if use_saved != 'n':
            return saved_token
    
    print("\nHow to get your token:")
    print("1. Visit: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Choose name (e.g., 'colab-crypto-data')")
    print("4. Select 'Write' permission")
    print("5. Copy the token")
    
    token = input("\nPaste your token here: ").strip()
    
    if not token:
        log("No token provided. Aborted.", "ERROR")
        sys.exit(1)
    
    # 測試 token 有效性
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        log(f"Token verified. Username: {username}", "SUCCESS")
        return token
    except Exception as e:
        log(f"Invalid token: {str(e)}", "ERROR")
        sys.exit(1)

def get_repo_name():
    """獲取數據集名稱"""
    print("\n" + "="*70)
    print("STEP 2: REPOSITORY NAME")
    print("="*70)
    
    default_name = "v2-crypto-ohlcv-data"
    print(f"\nDefault name: {default_name}")
    print("Requirements:")
    print("  - Only lowercase letters, numbers, hyphens")
    print("  - Must be unique in your account")
    print("  - Example: v2-crypto-data, my-crypto-dataset, etc.")
    
    repo_name = input(f"\nEnter repo name (default={default_name}): ").strip()
    
    if not repo_name:
        repo_name = default_name
    
    # 驗證名稱
    if not repo_name.replace('-', '').replace('_', '').isalnum():
        log("Invalid repo name. Use only alphanumeric and hyphens.", "ERROR")
        sys.exit(1)
    
    return repo_name.lower()

def get_data_directory():
    """獲取數據目錄"""
    print("\n" + "="*70)
    print("STEP 3: DATA DIRECTORY")
    print("="*70)
    
    default_dir = "crypto_data_cache"
    print(f"\nDefault location: {default_dir}")
    print(f"Current directory: {os.getcwd()}")
    
    data_dir = input(f"\nEnter data directory (default={default_dir}): ").strip()
    
    if not data_dir:
        data_dir = default_dir
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        log(f"Directory not found: {data_path}", "ERROR")
        sys.exit(1)
    
    # 檢查 CSV 文件
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        log(f"No CSV files found in {data_path}", "ERROR")
        sys.exit(1)
    
    log(f"Found {len(csv_files)} CSV files", "SUCCESS")
    return str(data_path.resolve())

def parse_symbol_and_interval(filename):
    """
    從檔名解析幣種和時間框架
    例: BTC_15m.csv -> (BTC, BTCUSDT, 15m)
    """
    # 幣種映射表
    SYMBOL_MAP = {
        'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'SOL': 'SOLUSDT',
        'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT',
        'LINK': 'LINKUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT', 'UNI': 'UNIUSDT',
        'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'FIL': 'FILUSDT', 'DOGE': 'DOGEUSDT',
        'ALGO': 'ALGOUSDT', 'ATOM': 'ATOMUSDT', 'NEAR': 'NEARUSDT', 'ARB': 'ARBUSDT',
        'OP': 'OPUSDT', 'AAVE': 'AAVEUSDT', 'SHIB': 'SHIBUSD',
    }
    
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 2:
        return None, None, None
    
    symbol_short = parts[0]
    interval = parts[1]
    
    if symbol_short not in SYMBOL_MAP:
        return None, None, None
    
    symbol_full = SYMBOL_MAP[symbol_short]
    return symbol_short, symbol_full, interval

def organize_files_by_symbol(data_dir):
    """
    掃描 CSV 文件，轉換為 Parquet，並按幣種組織成一個絆構化目錄
    建立以下結構:
    organized_data/
    ├── README.md
    ├── klines/
    │   ├── BTCUSDT/
    │   │   ├── BTC_15m.parquet
    │   │   └── BTC_1h.parquet
    │   ├── ETHUSDT/
    │   └── ...
    """
    print("\n" + "="*70)
    print("STEP 3B: ORGANIZE AND CONVERT FILES")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # 創建絆構化目錄
    organized_dir = data_path / "organized_data"
    if organized_dir.exists():
        log(f"Removing existing organized data directory", "INFO")
        shutil.rmtree(organized_dir)
    
    organized_dir.mkdir(parents=True, exist_ok=True)
    klines_dir = organized_dir / "klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(data_path.glob('*.csv'))
    organized_files = {}
    total_rows = 0
    total_size = 0
    
    print(f"\nConverting and organizing {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        symbol_short, symbol_full, interval = parse_symbol_and_interval(csv_file.name)
        
        if not symbol_full:
            log(f"Skipping unknown symbol: {csv_file.name}", "WARNING")
            continue
        
        try:
            # 讀取 CSV
            df = pd.read_csv(csv_file)
            
            # 轉換數據類型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'trades' in df.columns:
                df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            
            # 建立幣種目錄
            symbol_dir = klines_dir / symbol_full
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存為 Parquet
            parquet_filename = csv_file.stem + ".parquet"
            parquet_path = symbol_dir / parquet_filename
            
            df.to_parquet(parquet_path, compression='snappy', index=False)
            
            if symbol_full not in organized_files:
                organized_files[symbol_full] = []
            organized_files[symbol_full].append(parquet_path)
            
            size_mb = parquet_path.stat().st_size / (1024 * 1024)
            total_rows += len(df)
            total_size += parquet_path.stat().st_size
            
            log(f"{symbol_short:5} {interval:3} -> klines/{symbol_full}/{parquet_filename:30} ({len(df):7,} rows, {size_mb:6.1f} MB)", "INFO")
            
        except Exception as e:
            log(f"Failed to convert {csv_file.name}: {str(e)}", "ERROR")
            continue
    
    return organized_dir, organized_files, total_rows, total_size

def create_readme(repo_name, organized_files, total_rows, total_size):
    """產生 README.md"""
    
    symbols = sorted(organized_files.keys())
    all_intervals = set()
    for files in organized_files.values():
        for f in files:
            if '15m' in f.name:
                all_intervals.add('15m')
            elif '1h' in f.name:
                all_intervals.add('1h')
    
    total_files = sum(len(v) for v in organized_files.values())
    
    readme_content = f"""---
license: mit
dataset_info:
  features:
  - name: open_time
    dtype: int64
  - name: open
    dtype: float64
  - name: high
    dtype: float64
  - name: low
    dtype: float64
  - name: close
    dtype: float64
  - name: volume
    dtype: float64
  - name: quote_volume
    dtype: float64
  - name: trades
    dtype: int64
  - name: taker_buy_base
    dtype: float64
  - name: taker_buy_quote
    dtype: float64
  - name: close_time
    dtype: int64
  splits:
  - name: train
    num_bytes: {int(total_size)}
    num_examples: {total_rows}
  download_size: {int(total_size)}
  dataset_size: {int(total_size)}
---

# {repo_name}

Cryptocurrency OHLCV (Open, High, Low, Close, Volume) historical data from Binance API.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Info

- Total Files: {total_files}
- Total Data Points: {total_rows:,}
- Total Size: {total_size / (1024**2):.2f} MB
- Symbols: {len(symbols)} cryptocurrencies
- Timeframes: {', '.join(sorted(all_intervals))}
- Data Source: Binance REST API

## Directory Structure

```
klines/
├── ADAUSDT/
│   ├── ADA_15m.parquet
│   └── ADA_1h.parquet
├── AAVEUSDT/
│   ├── AAVE_15m.parquet
│   └── AAVE_1h.parquet
├── ALGOUSDT/
├── ARBUSDT/
├── ATOMUSDT/
├── AVAXUSDT/
├── BCHUSDT/
├── BNBUSDT/
├── BTCUSDT/
├── DOGEUSDT/
├── DOTUSDT/
├── ETCUSDT/
├── ETHUSDT/
├── FILUSDT/
├── LINKUSDT/
├── LTCUSDT/
├── MATICUSDT/
├── NEARUSDT/
├── OPUSDT/
├── SHIBUSD/
├── SOLUSDT/
├── UNIUSDT/
└── XRPUSDT/
```

## Symbols Included

{', '.join(symbols)}

## Usage

### Load Specific Symbol

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# Download BTC 15m data
path = hf_hub_download(
    "{{USERNAME}}/{repo_name}",
    "klines/BTCUSDT/BTC_15m.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
print(f"Loaded {{len(df)}} BTC K-lines")
```

### Load Multiple Symbols

```python
import pandas as pd
from huggingface_hub import hf_hub_download

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
interval = '15m'

data = {{}}
for symbol in symbols:
    short_name = symbol.replace('USDT', '').replace('USD', '')
    path = hf_hub_download(
        "{{USERNAME}}/{repo_name}",
        f"klines/{{symbol}}/{{short_name}}_{{interval}}.parquet",
        repo_type="dataset"
    )
    data[symbol] = pd.read_parquet(path)
    print(f"{{symbol}}: {{len(data[symbol])}} K-lines")
```

## Data Columns

Each file contains:
- `open_time` - Candle open time (Unix timestamp)
- `open` - Opening price
- `high` - Highest price in candle
- `low` - Lowest price in candle
- `close` - Closing price
- `volume` - Trading volume in base currency
- `quote_volume` - Volume in quote currency
- `trades` - Number of trades
- `taker_buy_base` - Taker buy base asset volume
- `taker_buy_quote` - Taker buy quote asset volume
- `close_time` - Candle close time (Unix timestamp)

## License

MIT License

## Disclaimer

For research purposes only. Not financial advice.
"""
    
    return readme_content

def upload_to_huggingface(token, username, repo_name, organized_dir, readme_content):
    """
    一次性上傳整個絆構化目錄到 HuggingFace
    使用 upload_folder 据作 - 突凘操作數 最少
    """
    print("\n" + "="*70)
    print("STEP 4: UPLOAD TO HUGGINGFACE")
    print("="*70)
    
    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"
    
    print(f"\nRepository ID: {repo_id}")
    print(f"Upload method: Single folder upload (minimize API calls)")
    print(f"Source directory: {organized_dir}")
    
    # 檢查 repo 是否存在
    print(f"\nChecking if repo exists...")
    if repo_exists(repo_id, repo_type="dataset", token=token):
        log(f"Repository already exists", "WARNING")
        overwrite = input("Overwrite existing repo? (y/n, default=n): ").strip().lower()
        if overwrite != 'y':
            log("Upload cancelled.", "WARNING")
            return None
    else:
        log(f"Repository does not exist. Creating new one.", "INFO")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                private=False
            )
            log(f"Repository created successfully", "SUCCESS")
            time.sleep(2)
        except Exception as e:
            log(f"Failed to create repository: {str(e)}", "ERROR")
            return None
    
    # 切存 README.md 到絆構化目錄
    readme_path = organized_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    log(f"README.md created", "SUCCESS")
    
    # 一次性上傳整個目錄
    print(f"\n" + "-"*70)
    print(f"Uploading entire dataset folder...")
    print(f"This may take 30-90 minutes depending on your connection.")
    print(f"You can monitor progress at: https://huggingface.co/datasets/{repo_id}")
    print("-"*70)
    
    try:
        start_time = time.time()
        
        # 一次性上傳整個絆構化目錄
        api.upload_folder(
            folder_path=str(organized_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial dataset upload - Crypto OHLCV data from Binance",
            ignore_patterns=[
                "*.csv",  # 忽略原始 CSV
                "*.pyc",
                "__pycache__",
                ".git*",
                "organized_data/.gitignore"
            ],
            multi_commit=True,  # 使用多個提交以优化
            multi_commit_pr=False  # 不建立 PR
        )
        
        elapsed = time.time() - start_time
        log(f"Upload completed successfully in {elapsed/60:.1f} minutes", "SUCCESS")
        return repo_id
        
    except Exception as e:
        log(f"Upload failed: {str(e)}", "ERROR")
        print(f"\nFull error: {type(e).__name__}: {e}")
        return None

def main():
    """主執行函數"""
    
    print("\n" + "="*70)
    print("HUGGING FACE DATASET UPLOADER")
    print("Bulk Upload Version (Single Folder Upload)")
    print("="*70)
    
    # 步驟 1: 獲取 token
    token = get_hf_token()
    api = HfApi(token=token)
    username = api.whoami()['name']
    
    # 步驟 2: 獲取 repo 名稱
    repo_name = get_repo_name()
    
    # 步驟 3: 組織數據並轉換
    data_dir = get_data_directory()
    organized_dir, organized_files, total_rows, total_size = organize_files_by_symbol(data_dir)
    
    if not organized_files:
        log("No files to upload. Aborted.", "ERROR")
        return
    
    total_files = sum(len(v) for v in organized_files.values())
    log(f"Conversion complete: {len(organized_files)} symbols, {total_files} files, {total_size/(1024**2):.2f} MB", "SUCCESS")
    
    # 步驟 4: 產生 README
    readme_content = create_readme(repo_name, organized_files, total_rows, total_size)
    
    # 步驟 5: 上傳整個整織化目錄
    repo_id = upload_to_huggingface(token, username, repo_name, organized_dir, readme_content)
    
    if repo_id:
        print("\n" + "="*70)
        print("UPLOAD COMPLETE")
        print("="*70)
        print(f"\nDataset successfully uploaded!")
        print(f"\nDataset URL:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print(f"\nDirectory structure:")
        print(f"https://huggingface.co/datasets/{repo_id}/tree/main/klines")
        print(f"\nLoad dataset with:")
        print(f"import pandas as pd")
        print(f"from huggingface_hub import hf_hub_download")
        print(f"path = hf_hub_download('{repo_id}', 'klines/BTCUSDT/BTC_15m.parquet', repo_type='dataset')")
        print(f"df = pd.read_parquet(path)")
        print("="*70 + "\n")
    else:
        print("\nUpload failed. Please try again.\n")

if __name__ == "__main__":
    main()
else:
    main()
