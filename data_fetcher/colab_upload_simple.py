"""
最简化版本 - 直接在 Colab 执行，无任何兼容性问题

在 Colab 中执行:
!pip install -q --upgrade huggingface-hub pandas
%cd /content/crypto_data_cache
import subprocess, requests
subprocess.run(['pip', 'install', '-q', '--upgrade', '--no-cache-dir', 'huggingface-hub'], check=True)
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/colab_upload_simple.py').text)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# 清除所有旧的导入
# for module in list(sys.modules.keys()):
#     if 'huggingface' in module:
#         del sys.modules[module]

# 强制重新导入
import pandas as pd
from huggingface_hub import HfApi, repo_exists, create_repo

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level}: {msg}")

print("\n" + "="*70)
print("HUGGING FACE DATASET UPLOADER - Colab Version")
print("="*70)

# ===== STEP 1: Token =====
print("\nSTEP 1: Get HF Token")
print("-"*70)
print("1. Visit: https://huggingface.co/settings/tokens")
print("2. Click 'New token'")
print("3. Name: 'colab-crypto'")
print("4. Permission: 'Write'")
print("5. Copy and paste below\n")

token = input("Paste token: ").strip()
if not token:
    log("No token provided", "ERROR")
    sys.exit(1)

try:
    api = HfApi(token=token)
    user = api.whoami()
    username = user['name']
    log(f"Token verified. User: {username}", "SUCCESS")
except Exception as e:
    log(f"Token invalid: {e}", "ERROR")
    sys.exit(1)

# ===== STEP 2: Repo name =====
print("\nSTEP 2: Repository Name")
print("-"*70)
repo_name = input("Enter repo name (default=v2-crypto-ohlcv-data): ").strip()
if not repo_name:
    repo_name = "v2-crypto-ohlcv-data"

log(f"Repository: {username}/{repo_name}", "INFO")

# ===== STEP 3: Organize data =====
print("\nSTEP 3: Organize Data")
print("-"*70)

data_path = Path("/content/crypto_data_cache")
csv_files = sorted(data_path.glob("*.csv"))
log(f"Found {len(csv_files)} CSV files", "INFO")

# Create organized directory
organized_dir = data_path / "organized_data"
if organized_dir.exists():
    shutil.rmtree(organized_dir)
organized_dir.mkdir()
klines_dir = organized_dir / "klines"
klines_dir.mkdir()

# Symbol mapping
SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT',
    'LINK': 'LINKUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT', 'UNI': 'UNIUSDT',
    'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'FIL': 'FILUSDT', 'DOGE': 'DOGEUSDT',
    'ALGO': 'ALGOUSDT', 'ATOM': 'ATOMUSDT', 'NEAR': 'NEARUSDT', 'ARB': 'ARBUSDT',
    'OP': 'OPUSDT', 'AAVE': 'AAVEUSDT', 'SHIB': 'SHIBUSD',
}

print("\nConverting CSV to Parquet:")
total_rows = 0
total_size = 0
file_count = 0

for csv_file in csv_files:
    stem = csv_file.stem
    parts = stem.split('_')
    if len(parts) < 2:
        continue
    
    symbol_short = parts[0]
    interval = parts[1]
    
    if symbol_short not in SYMBOLS:
        continue
    
    symbol_full = SYMBOLS[symbol_short]
    
    try:
        df = pd.read_csv(csv_file)
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'trades' in df.columns:
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
        
        # Save as parquet
        symbol_dir = klines_dir / symbol_full
        symbol_dir.mkdir(exist_ok=True)
        
        parquet_path = symbol_dir / f"{stem}.parquet"
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        total_rows += len(df)
        total_size += parquet_path.stat().st_size
        file_count += 1
        
        print(f"  {symbol_short:5} {interval:3} -> {symbol_full:10} ({len(df):8,} rows, {size_mb:6.1f} MB)")
        
    except Exception as e:
        log(f"Failed to convert {csv_file.name}: {e}", "ERROR")
        continue

log(f"Organized {file_count} files, {total_size/(1024**2):.2f} MB", "SUCCESS")

# ===== STEP 4: Create README =====
readme_content = f"""---
license: mit
---

# {repo_name}

Cryptocurrency OHLCV data from Binance API.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Info

- Total Files: {file_count}
- Total Data Points: {total_rows:,}
- Total Size: {total_size / (1024**2):.2f} MB
- Symbols: 23 cryptocurrencies
- Timeframes: 15m, 1h

## Usage

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "{username}/{repo_name}",
    "klines/BTCUSDT/BTC_15m.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(path)
print(df.head())
```

## Symbols

ADA, AAVE, ALGO, ARB, ATOM, AVAX, BCH, BNB, BTC, DOGE, DOT, ETC, ETH, FIL, LINK, LTC, MATIC, NEAR, OP, SHIB, SOL, UNI, XRP

## License

MIT License
"""

readme_path = organized_dir / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)
log("README.md created", "INFO")

# ===== STEP 5: Upload =====
print("\nSTEP 4: Upload to HuggingFace")
print("-"*70)

repo_id = f"{username}/{repo_name}"

# Check if repo exists
if repo_exists(repo_id, repo_type="dataset", token=token):
    log(f"Repository exists", "WARNING")
    overwrite = input("Overwrite? (y/n): ").strip().lower()
    if overwrite != 'y':
        log("Cancelled", "ERROR")
        sys.exit(0)
else:
    log("Creating repository", "INFO")
    create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=False)
    time.sleep(2)

print(f"\nUploading to: https://huggingface.co/datasets/{repo_id}")
print(f"This may take 30-90 minutes...\n")

try:
    start = time.time()
    api.upload_folder(
        folder_path=str(organized_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload crypto OHLCV data"
    )
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("UPLOAD SUCCESSFUL!")
    print("="*70)
    print(f"\nDataset URL:")
    print(f"https://huggingface.co/datasets/{repo_id}")
    print(f"\nTime: {elapsed/60:.1f} minutes")
    print(f"\nLoad data:")
    print(f"from huggingface_hub import hf_hub_download")
    print(f"path = hf_hub_download('{repo_id}', 'klines/BTCUSDT/BTC_15m.parquet', repo_type='dataset')")
    print(f"df = pd.read_parquet(path)")
    print("="*70 + "\n")
    
except Exception as e:
    log(f"Upload failed: {e}", "ERROR")
    print(f"\nError: {type(e).__name__}: {e}")
    sys.exit(1)
