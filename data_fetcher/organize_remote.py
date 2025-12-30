"""
遠端數據集組織考幀脚本 - Colab 優化版

功能:
- 從 Hugging Face 下載整個數據集
- 組織成按幣種分組的結構
- 上傳新結構回 Hugging Face
- 不需要剩余的本地空間

使用方式:
!pip install -q huggingface-hub pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/organize_remote.py').text)
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("[Installing pandas...]")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("[Installing huggingface-hub...]")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
    from huggingface_hub import HfApi, snapshot_download

def log(message, level="INFO"):
    """統一日誌函數"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"[{timestamp}] {message}")
    elif level == "WARNING":
        print(f"[{timestamp}] WARNING: {message}")
    elif level == "ERROR":
        print(f"[{timestamp}] ERROR: {message}")
    elif level == "SUCCESS":
        print(f"[{timestamp}] SUCCESS: {message}")

def create_symbol_readme(symbol, files_info):
    """創建幣種級 README"""
    readme = f"""# {symbol} Historical Data

Cryptocurrency trading data for {symbol}

## Files

"""
    
    for filename, interval, rows in files_info:
        readme += f"### {filename}\n\n"
        readme += f"- Interval: {interval}\n"
        readme += f"- Rows: {rows:,}\n\n"
    
    readme += """## Data Structure

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | UTC timestamp |
| symbol | string | Trading pair (e.g., BTCUSDT) |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | float | Trading volume |

## Usage

```python
import pandas as pd

# Load data
df = pd.read_csv(f"{symbol}_1h.csv")
print(df.head())
print(df.info())
```
"""
    return readme

def create_combined_readme():
    """創建合併數據集 README"""
    return """# Combined Datasets

Combined OHLCV data for all symbols.

## Files

### all_symbols_15m.csv
- Contains 15-minute data for all symbols
- Sorted by timestamp

### all_symbols_1h.csv
- Contains 1-hour data for all symbols
- Sorted by timestamp

## Data Structure

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | UTC timestamp |
| symbol | string | Trading pair (e.g., BTCUSDT) |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | float | Trading volume |

## Usage

```python
import pandas as pd

# Load all data
df_15m = pd.read_csv('all_symbols_15m.csv')

# Get specific symbol
btc_data = df_15m[df_15m['symbol'] == 'BTCUSDT']

# Get specific time period
df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
recent_data = df_15m[df_15m['timestamp'] > '2025-01-01']
```
"""

def create_main_readme(symbols):
    """創建主 README"""
    symbols = sorted(symbols)
    
    readme = f"""# Datasets

Organized cryptocurrency OHLCV datasets.

## Directory Structure

```
datasets/
"""
    
    for i, symbol in enumerate(symbols):
        is_last = (i == len(symbols) - 1)
        prefix = "└── " if is_last else "├── "
        readme += f"{prefix}{symbol}/\n"
        readme += f"    ├── {symbol}_15m.csv\n"
        readme += f"    ├── {symbol}_1h.csv\n"
        readme += f"    └── README.md\n"
    
    readme += """```

## Symbols ({count})

{symbols_str}

## Usage

### Load Single Symbol Data

```python
import pandas as pd

# Load Bitcoin 1-hour data
df = pd.read_csv('BTC/BTC_1h.csv')
print(f"Rows: {{len(df)}}")
print(df.head())
```

### Load All Symbols

```python
import pandas as pd
from pathlib import Path

def load_all_symbols(data_dir='datasets', interval='1h'):
    all_data = {{}}
    for symbol_dir in Path(data_dir).iterdir():
        if symbol_dir.is_dir() and not symbol_dir.name.startswith('_'):
            csv_file = symbol_dir / f"{{symbol_dir.name}}_{{interval}}.csv"
            if csv_file.exists():
                all_data[symbol_dir.name] = pd.read_csv(csv_file)
    return all_data

data = load_all_symbols(interval='1h')
for symbol, df in data.items():
    print(f"{{symbol}}: {{len(df)}} rows")
```

## Statistics

- Total Symbols: {count}
- Total Timeframes: 2 (15m, 1h)
- Data Source: Binance REST API

## License

MIT License - Free for research and educational use
""".format(
        count=len(symbols),
        symbols_str=", ".join(symbols)
    )
    
    return readme

def organize_and_upload(repo_id, token):
    """下載、組織、上傳整個流程"""
    
    print("\n" + "="*70)
    print("REMOTE CRYPTO DATASET ORGANIZER")
    print("="*70)
    
    api = HfApi(token=token)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: 下載整個數據集
        print("\n" + "="*70)
        print("STEP 1: DOWNLOAD DATASET")
        print("="*70)
        
        log(f"Downloading from {repo_id}...", "INFO")
        source_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=temp_dir
        )
        log(f"Downloaded to: {source_dir}", "SUCCESS")
        
        # 查找所有 CSV 文件
        csv_files = list(Path(source_dir).glob('*.csv'))
        csv_files = [f for f in csv_files if not f.name.startswith('.')]
        
        if not csv_files:
            log("No CSV files found", "ERROR")
            return False
        
        log(f"Found {len(csv_files)} CSV files", "SUCCESS")
        
        # Step 2: 組織數據
        print("\n" + "="*70)
        print("STEP 2: ORGANIZE DATASETS")
        print("="*70)
        
        # 提取幣種和時框
        crypto_data = {}
        for csv_file in csv_files:
            parts = csv_file.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                interval = parts[1]
                
                if symbol not in crypto_data:
                    crypto_data[symbol] = []
                crypto_data[symbol].append((csv_file, interval))
        
        # 創建組織目錄
        organized_dir = Path(temp_dir) / "datasets_organized"
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"Organizing into {len(crypto_data)} symbol folders...", "INFO")
        
        all_data_15m = []
        all_data_1h = []
        
        for symbol in sorted(crypto_data.keys()):
            symbol_dir = organized_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # 複製文件
            files_info = []
            for csv_file, interval in crypto_data[symbol]:
                dest_file = symbol_dir / csv_file.name
                shutil.copy2(csv_file, dest_file)
                
                # 讀取數據統計
                try:
                    df = pd.read_csv(dest_file)
                    rows = len(df)
                    files_info.append((csv_file.name, interval, rows))
                    
                    if interval == '15m':
                        all_data_15m.append(df)
                    elif interval == '1h':
                        all_data_1h.append(df)
                except Exception as e:
                    log(f"Error reading {csv_file.name}: {str(e)}", "WARNING")
            
            # 創建幣種級 README
            symbol_readme = create_symbol_readme(symbol, files_info)
            with open(symbol_dir / "README.md", 'w') as f:
                f.write(symbol_readme)
            
            log(f"Organized {symbol}", "INFO")
        
        # 創建合併數據
        print("\nCreating combined datasets...")
        combined_dir = organized_dir / "_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        if all_data_15m:
            combined_15m = pd.concat(all_data_15m, ignore_index=True)
            combined_15m = combined_15m.sort_values('timestamp').reset_index(drop=True)
            combined_15m.to_csv(combined_dir / "all_symbols_15m.csv", index=False)
            log(f"Created all_symbols_15m.csv ({len(combined_15m)} rows)", "SUCCESS")
        
        if all_data_1h:
            combined_1h = pd.concat(all_data_1h, ignore_index=True)
            combined_1h = combined_1h.sort_values('timestamp').reset_index(drop=True)
            combined_1h.to_csv(combined_dir / "all_symbols_1h.csv", index=False)
            log(f"Created all_symbols_1h.csv ({len(combined_1h)} rows)", "SUCCESS")
        
        # 創建合併目錄 README
        combined_readme = create_combined_readme()
        with open(combined_dir / "README.md", 'w') as f:
            f.write(combined_readme)
        
        # 創建主 README
        main_readme = create_main_readme(crypto_data.keys())
        with open(organized_dir / "README.md", 'w') as f:
            f.write(main_readme)
        
        log(f"Created all README files", "SUCCESS")
        
        # Step 3: 上傳回 Hugging Face
        print("\n" + "="*70)
        print("STEP 3: UPLOAD TO HUGGING FACE")
        print("="*70)
        
        log(f"Uploading organized datasets to {repo_id}...", "INFO")
        print("This may take 10-20 minutes...\n")
        
        api.upload_folder(
            folder_path=str(organized_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Reorganized datasets: grouped by symbol with combined datasets",
            multi_commit=False
        )
        
        log(f"Upload completed successfully", "SUCCESS")
        
        # 顯示結果
        print("\n" + "="*70)
        print("REORGANIZATION COMPLETE")
        print("="*70)
        print(f"\nRepository: {repo_id}")
        print(f"URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\nNew structure:")
        print(f"  datasets/")
        print(f"    ├── BTC/ (BTC_15m.csv, BTC_1h.csv, README.md)")
        print(f"    ├── ETH/ (ETH_15m.csv, ETH_1h.csv, README.md)")
        print(f"    ├── ... ({len(crypto_data)} symbols total)")
        print(f"    ├── _combined/ (all_symbols_15m.csv, all_symbols_1h.csv)")
        print(f"    └── README.md")
        print()
        
        return True
        
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临時文件
        log(f"Cleaning up temporary files...", "INFO")
        shutil.rmtree(temp_dir, ignore_errors=True)
        log(f"Done", "SUCCESS")

def main():
    print("\n" + "="*70)
    print("REMOTE CRYPTO DATASET ORGANIZER FOR COLAB")
    print("="*70)
    
    # 獲取設置
    print("\nConfiguration:")
    repo_id = input("Hugging Face repo ID (format: username/repo-name): ").strip()
    
    if not '/' in repo_id:
        print("Invalid repo ID. Please use format: username/repo-name")
        return
    
    token = input("Hugging Face token (with write permission): ").strip()
    
    if not token or not repo_id:
        print("Missing required information")
        return
    
    # 驗證
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        log(f"Authenticated as: {user_info['name']}", "SUCCESS")
    except Exception as e:
        log(f"Authentication failed: {str(e)}", "ERROR")
        return
    
    # 執行組織
    success = organize_and_upload(repo_id, token)
    
    if success:
        print("\n" + "="*70)
        print("SUCCESS")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("FAILED")
        print("="*70)

if __name__ == "__main__":
    main()
else:
    main()
