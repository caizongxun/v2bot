"""
Hugging Face 数据集上传器 - 一次性上传整个资料夾

功能:
- 自动创建数据集 repo
- CSV 转换为 Parquet
- 按币种分类组织
- 一次性上传整个资料夾（減少 API 限制風險）

使用方式:
!pip install -q huggingface-hub pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/hf_uploader_auto_create.py').text)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

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
    """获取 Hugging Face token"""
    print("\n" + "="*70)
    print("STEP 1: HUGGING FACE TOKEN")
    print("="*70)
    
    # 检查是否已经设置
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
    
    # 测试 token 有效性
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
    """获取数据集名称"""
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
    
    # 验证名称
    if not repo_name.replace('-', '').replace('_', '').isalnum():
        log("Invalid repo name. Use only alphanumeric and hyphens.", "ERROR")
        sys.exit(1)
    
    return repo_name.lower()

def get_data_directory():
    """获取数据目录"""
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
    
    # 检查 CSV 文件
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        log(f"No CSV files found in {data_path}", "ERROR")
        sys.exit(1)
    
    log(f"Found {len(csv_files)} CSV files", "SUCCESS")
    return str(data_path.resolve())

def parse_symbol_and_interval(filename):
    """
    介文件名分析币种和时间框架
    例: BTC_15m.csv -> (BTC, BTCUSDT, 15m)
    """
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
    扫描 CSV 文件，转换为 Parquet，按币种组织
    返回组织后的目录路径
    """
    print("\n" + "="*70)
    print("STEP 3B: ORGANIZE AND CONVERT FILES")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # 创建组织化目录
    organized_dir = data_path / "organized_data"
    if organized_dir.exists():
        log(f"Removing existing organized data directory", "INFO")
        shutil.rmtree(organized_dir)
    
    organized_dir.mkdir(parents=True, exist_ok=True)
    klines_dir = organized_dir / "klines"
    klines_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(data_path.glob('*.csv'))
    total_rows = 0
    total_size = 0
    file_count = 0
    
    print(f"\nConverting and organizing {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        symbol_short, symbol_full, interval = parse_symbol_and_interval(csv_file.name)
        
        if not symbol_full:
            log(f"Skipping unknown symbol: {csv_file.name}", "WARNING")
            continue
        
        try:
            # 读取 CSV
            df = pd.read_csv(csv_file)
            
            # 转换数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'trades' in df.columns:
                df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            
            # 创建币种目录
            symbol_dir = klines_dir / symbol_full
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为 Parquet
            parquet_filename = csv_file.stem + ".parquet"
            parquet_path = symbol_dir / parquet_filename
            
            df.to_parquet(parquet_path, compression='snappy', index=False)
            
            size_mb = parquet_path.stat().st_size / (1024 * 1024)
            total_rows += len(df)
            total_size += parquet_path.stat().st_size
            file_count += 1
            
            log(f"{symbol_short:5} {interval:3} -> klines/{symbol_full}/{parquet_filename:30} ({len(df):7,} rows, {size_mb:6.1f} MB)", "INFO")
            
        except Exception as e:
            log(f"Failed to convert {csv_file.name}: {str(e)}", "ERROR")
            continue
    
    return organized_dir, file_count, total_rows, total_size

def create_readme(repo_name, file_count, total_rows, total_size):
    """申生 README.md"""
    
    readme_content = f"""---
license: mit
---

# {repo_name}

Cryptocurrency OHLCV (Open, High, Low, Close, Volume) historical data from Binance API.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Info

- Total Files: {file_count}
- Total Data Points: {total_rows:,}
- Total Size: {total_size / (1024**2):.2f} MB
- Data Source: Binance REST API
- Time period: ~2-3 years of historical data

## Directory Structure

```
klines/
├── ADAUSDT/
│   ├── ADA_15m.parquet
│   └── ADA_1h.parquet
├── AAVEUSDT/
│   ├── AAVE_15m.parquet
│   └── AAVE_1h.parquet
├── ... (23 total symbols)
```

## Symbols Included

ADA, AAVE, ALGO, ARB, ATOM, AVAX, BCH, BNB, BTC, DOGE, DOT, ETC, ETH, FIL, LINK, LTC, MATIC, NEAR, OP, SHIB, SOL, UNI, XRP

## Usage

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# Download BTC 15m data
path = hf_hub_download(
    "zongowo111/{repo_name}",
    "klines/BTCUSDT/BTC_15m.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
print(df.head())
```

## Data Columns

- `open_time` - Candle open time (Unix timestamp)
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `volume` - Trading volume
- `quote_volume` - Quote volume
- `trades` - Number of trades
- `taker_buy_base` - Taker buy base volume
- `taker_buy_quote` - Taker buy quote volume
- `close_time` - Candle close time (Unix timestamp)

## License

MIT License - Free for research and educational purposes.
"""
    
    return readme_content

def upload_to_huggingface(token, username, repo_name, organized_dir, readme_content):
    """
    一次性上传整个组织化目录到 HuggingFace
    """
    print("\n" + "="*70)
    print("STEP 4: UPLOAD TO HUGGINGFACE")
    print("="*70)
    
    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"
    
    print(f"\nRepository ID: {repo_id}")
    print(f"Source directory: {organized_dir}")
    
    # 检查 repo 是否存在
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
    
    # 写入 README.md
    readme_path = organized_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    log(f"README.md created", "SUCCESS")
    
    # 上传整个目录
    print(f"\n" + "-"*70)
    print(f"Uploading entire dataset folder...")
    print(f"This may take 30-90 minutes.")
    print(f"Monitor at: https://huggingface.co/datasets/{repo_id}")
    print("-"*70 + "\n")
    
    try:
        start_time = time.time()
        
        # 使用 upload_folder 上传
        # 仅传递基本参数，优先使用默认值
        api.upload_folder(
            folder_path=str(organized_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload cryptocurrency OHLCV data - organized by symbol"
        )
        
        elapsed = time.time() - start_time
        log(f"Upload completed successfully in {elapsed/60:.1f} minutes", "SUCCESS")
        return repo_id
        
    except Exception as e:
        log(f"Upload failed: {str(e)}", "ERROR")
        print(f"\nFull error: {type(e).__name__}: {e}")
        print(f"\nTips:")
        print(f"1. Check your internet connection")
        print(f"2. Ensure HF token has 'Write' permission")
        print(f"3. Try running the script again")
        return None

def main():
    """主执行函数"""
    
    print("\n" + "="*70)
    print("HUGGING FACE DATASET UPLOADER")
    print("Bulk Upload Version (Single Folder Upload)")
    print("="*70)
    
    # 步骥 1: 获取 token
    token = get_hf_token()
    api = HfApi(token=token)
    username = api.whoami()['name']
    
    # 步骥 2: 获取 repo 名称
    repo_name = get_repo_name()
    
    # 步骥 3: 组织数据並转换
    data_dir = get_data_directory()
    organized_dir, file_count, total_rows, total_size = organize_files_by_symbol(data_dir)
    
    if file_count == 0:
        log("No files to upload. Aborted.", "ERROR")
        return
    
    log(f"Conversion complete: {file_count} files, {total_size/(1024**2):.2f} MB", "SUCCESS")
    
    # 步骥 4: 申生 README
    readme_content = create_readme(repo_name, file_count, total_rows, total_size)
    
    # 步骥 5: 上传整个组织化目录
    repo_id = upload_to_huggingface(token, username, repo_name, organized_dir, readme_content)
    
    if repo_id:
        print("\n" + "="*70)
        print("UPLOAD COMPLETE")
        print("="*70)
        print(f"\nDataset successfully uploaded!")
        print(f"\nDataset URL:")
        print(f"https://huggingface.co/datasets/{repo_id}")
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
