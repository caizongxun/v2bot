"""
Hugging Face 數據集上傳器 - Colab 優化版

功能:
- 自動建立數據集 repo
- 批量上傳整個目錄
- 進度顯示
- 達成上傳自動產生上傳、下載、查看數據集的代碼

使用方式:
!pip install -q huggingface-hub
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/hf_uploader_colab.py').text)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, HfFolder, repo_exists
except ImportError:
    print("[Installing huggingface-hub...]")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
    from huggingface_hub import HfApi, HfFolder, repo_exists
    print("huggingface-hub installed successfully\n")

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"[{timestamp}] {message}")
    elif level == "WARNING":
        print(f"[{timestamp}] WARNING: {message}")
    elif level == "ERROR":
        print(f"[{timestamp}] ERROR: {message}")
    elif level == "SUCCESS":
        print(f"[{timestamp}] SUCCESS: {message}")

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

def create_readme(repo_name, data_dir, api, username):
    """產生 README.md"""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('*.csv'))
    
    # 統計信息
    total_rows = 0
    total_size = 0
    for csv_file in csv_files:
        import pandas as pd
        try:
            df = pd.read_csv(csv_file)
            total_rows += len(df)
            total_size += csv_file.stat().st_size
        except:
            pass
    
    symbols = sorted(set([f.stem.rsplit('_', 1)[0] for f in csv_files]))
    intervals = sorted(set([f.stem.rsplit('_', 1)[1] for f in csv_files]))
    
    readme_content = f"""# {repo_name}

## Description

Cryptocurrency OHLCV (Open, High, Low, Close, Volume) historical data fetched from Binance API.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Info

- Total Files: {len(csv_files)}
- Total Data Points: {total_rows:,}
- Total Size: {total_size / (1024**2):.2f} MB
- Symbols: {len(symbols)} cryptocurrencies
- Timeframes: {', '.join(intervals)}
- Data Source: Binance REST API (binance.us)

## Symbols Included

{', '.join(symbols)}

## File Structure

Each CSV file contains:
- `timestamp` - UTC time of the candle
- `symbol` - Trading pair (e.g., BTCUSDT)
- `open` - Opening price
- `high` - Highest price in candle
- `low` - Lowest price in candle
- `close` - Closing price
- `volume` - Trading volume in base currency

## Naming Convention

- `{{SYMBOL}}_{{INTERVAL}}.csv`
- Example: `BTC_1h.csv`, `ETH_15m.csv`

## Usage

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

# Load the entire dataset
ds = load_dataset('{username}/{repo_name}')

# Access a specific file
df = ds['train'].to_pandas()
```

### Load Specific CSV Files

```python
import pandas as pd

# Load Bitcoin 1-hour data
btc_1h = pd.read_csv('BTC_1h.csv')

# Load Ethereum 15-minute data
eth_15m = pd.read_csv('ETH_15m.csv')
```

## Data Quality

- All timestamps are in UTC
- Data is sorted chronologically (oldest to newest)
- No missing values in OHLCV columns
- Prices are in USDT (or USD for some pairs)

## License

Public Domain - Free to use for research and educational purposes.

## Disclaimer

This data is provided for research purposes only. Past performance does not guarantee future results.
Always conduct your own due diligence before making trading decisions.

## Generated By

v2bot Crypto Data Fetcher
Repository: https://github.com/caizongxun/v2bot
"""
    
    return readme_content

def upload_to_huggingface(token, username, repo_name, data_dir):
    """上傳整個數據集到 Hugging Face"""
    print("\n" + "="*70)
    print("STEP 4: UPLOAD TO HUGGING FACE")
    print("="*70)
    
    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"
    
    print(f"\nRepository ID: {repo_id}")
    print(f"Data directory: {data_dir}")
    
    # 檢查 repo 是否存在
    print(f"\nChecking if repo exists...")
    if repo_exists(repo_id, repo_type="dataset"):
        log(f"Repository already exists", "WARNING")
        overwrite = input("Overwrite existing repo? (y/n, default=n): ").strip().lower()
        if overwrite != 'y':
            log("Upload cancelled.", "WARNING")
            return None
    else:
        log(f"Repository does not exist. Will create new one.", "INFO")
    
    # 產生 README
    log(f"Generating README.md...", "INFO")
    readme_content = create_readme(repo_name, data_dir, api, username)
    readme_path = Path(data_dir) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    log(f"README.md created", "SUCCESS")
    
    # 上傳整個目錄
    print(f"\nStarting upload...")
    print(f"This may take 10-30 minutes depending on connection speed.")
    print(f"(You can monitor progress in Hugging Face repo)\n")
    
    try:
        start_time = time.time()
        
        api.upload_folder(
            folder_path=data_dir,
            repo_id=repo_id,
            repo_type="dataset",
            multi_commit=True,
            multi_commit_nb_threads=4,
            commit_message="Initial dataset upload"
        )
        
        elapsed = time.time() - start_time
        
        log(f"Upload completed successfully", "SUCCESS")
        log(f"Time elapsed: {elapsed/60:.1f} minutes", "INFO")
        
        return repo_id
        
    except Exception as e:
        log(f"Upload failed: {str(e)}", "ERROR")
        return None

def main():
    """主執行函數"""
    
    print("\n" + "="*70)
    print("HUGGING FACE DATASET UPLOADER")
    print("="*70)
    
    # 步驟 1: 獲取 token
    token = get_hf_token()
    api = HfApi(token=token)
    username = api.whoami()['name']
    
    # 步驟 2: 獲取 repo 名稱
    repo_name = get_repo_name()
    
    # 步驟 3: 獲取數據目錄
    data_dir = get_data_directory()
    
    # 步驟 4: 上傳
    repo_id = upload_to_huggingface(token, username, repo_name, data_dir)
    
    if repo_id:
        print("\n" + "="*70)
        print("UPLOAD COMPLETE")
        print("="*70)
        print(f"\nDataset URL:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print(f"\nLoad dataset with:")
        print(f"from datasets import load_dataset")
        print(f"ds = load_dataset('{repo_id}')")
        print(f"\nView on Hub:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print("="*70 + "\n")
    else:
        print("\nUpload failed. Please try again.\n")

if __name__ == "__main__":
    main()
else:
    main()
