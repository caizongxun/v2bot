"""
數據集組織考幀脚本

功能:
- 將爳種數據分組
- 自動創建轉移目錄
- 保教原泊文件
- 產生流程報告

使用方式:
python organize_datasets.py

最終的檔案結構:
.
├─ datasets/
│  ├─ BTC/
│  │  ├─ BTC_15m.csv
│  │  ├─ BTC_1h.csv
│  │  └─ README.md
│  ├─ ETH/
│  │  ├─ ETH_15m.csv
│  │  ├─ ETH_1h.csv
│  │  └─ README.md
│  ├─ ...
│  └─ _combined/  (可選)
│     ├─ all_symbols_15m.csv
│     ├─ all_symbols_1h.csv
│     └─ README.md
├─ models/  (未來模型版本)
├─ crypto_data_cache/  (原始數據)
└─ README.md
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

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

def organize_datasets(source_dir="crypto_data_cache", output_dir="datasets", create_combined=False):
    """組織數據集"""
    
    print("\n" + "="*70)
    print("CRYPTO DATASET ORGANIZER")
    print("="*70)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 驗證源目錄
    if not source_path.exists():
        log(f"Source directory not found: {source_path}", "ERROR")
        return False
    
    # 查找所有 CSV 文件
    csv_files = list(source_path.glob('*.csv'))
    
    if not csv_files:
        log(f"No CSV files found in {source_path}", "ERROR")
        return False
    
    log(f"Found {len(csv_files)} CSV files", "SUCCESS")
    print()
    
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
    
    # 創建輸出目錄
    if output_path.exists():
        log(f"Output directory already exists: {output_path}", "WARNING")
        response = input(f"Overwrite {output_path}? (y/n, default=n): ").strip().lower()
        if response != 'y':
            log("Operation cancelled.", "WARNING")
            return False
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    log(f"Created output directory: {output_path}", "SUCCESS")
    print()
    
    # 組織每個幣種
    log("Organizing by symbol...", "INFO")
    all_data_15m = []
    all_data_1h = []
    symbol_count = 0
    
    for symbol in sorted(crypto_data.keys()):
        symbol_dir = output_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # 複製文件
        file_count = 0
        for csv_file, interval in crypto_data[symbol]:
            dest_file = symbol_dir / csv_file.name
            shutil.copy2(csv_file, dest_file)
            
            # 如果要創建合併數據
            if create_combined:
                try:
                    df = pd.read_csv(dest_file)
                    if interval == '15m':
                        all_data_15m.append(df)
                    elif interval == '1h':
                        all_data_1h.append(df)
                except Exception as e:
                    log(f"Error reading {csv_file}: {str(e)}", "WARNING")
            
            file_count += 1
        
        # 創建幣種級 README
        symbol_readme = create_symbol_readme(symbol, crypto_data[symbol])
        readme_path = symbol_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(symbol_readme)
        
        log(f"Organized {symbol}: {file_count} files", "INFO")
        symbol_count += 1
    
    print()
    log(f"Organized {symbol_count} symbols", "SUCCESS")
    
    # 創建合併數據集（可選）
    if create_combined and (all_data_15m or all_data_1h):
        print()
        log("Creating combined datasets...", "INFO")
        combined_dir = output_path / "_combined"
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
        
        # 創建合併目錄的 README
        combined_readme = create_combined_readme()
        with open(combined_dir / "README.md", 'w') as f:
            f.write(combined_readme)
    
    # 創建主 README
    print()
    log("Creating main README...", "INFO")
    main_readme = create_main_readme(crypto_data)
    with open(output_path / "README.md", 'w') as f:
        f.write(main_readme)
    log(f"Created main README", "SUCCESS")
    
    # 顯示最終結構
    print()
    print("="*70)
    print("FINAL DIRECTORY STRUCTURE")
    print("="*70)
    print_tree(output_path, prefix="", is_root=True)
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Source directory: {source_path}")
    print(f"Output directory: {output_path}")
    print(f"Total symbols: {symbol_count}")
    print(f"Total files: {len(csv_files)}")
    print()
    
    return True

def create_symbol_readme(symbol, files_info):
    """創建幣種級 README"""
    
    # 統計信息
    file_stats = []
    for csv_file, interval in files_info:
        try:
            df = pd.read_csv(csv_file)
            rows = len(df)
            size_mb = csv_file.stat().st_size / (1024**2)
            file_stats.append({
                'interval': interval,
                'rows': rows,
                'size': size_mb,
                'filename': csv_file.name
            })
        except:
            pass
    
    readme = f"""# {symbol} Historical Data

Cryptocurrency trading data for {symbol}

## Files

"""
    
    for stat in file_stats:
        readme += f"### {stat['filename']}\n\n"
        readme += f"- Interval: {stat['interval']}\n"
        readme += f"- Rows: {stat['rows']:,}\n"
        readme += f"- Size: {stat['size']:.2f} MB\n\n"
    
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
import pandas as pd
df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
recent_data = df_15m[df_15m['timestamp'] > '2025-01-01']
```
"""

def create_main_readme(crypto_data):
    """創建主 README"""
    symbols = sorted(crypto_data.keys())
    
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
        if not is_last:
            next_prefix = "    "
        else:
            next_prefix = "    "
        readme += f"{next_prefix}├── {symbol}_15m.csv\n"
        readme += f"{next_prefix}├── {symbol}_1h.csv\n"
        readme += f"{next_prefix}└── README.md\n"
    
    readme += """```

## Symbols ({count})

""".format(count=len(symbols))
    
    # 分組顯示
    symbols_str = ", ".join(symbols)
    readme += f"{symbols_str}\n\n"
    
    readme += """## Usage

### Load Single Symbol Data

```python
import pandas as pd

# Load Bitcoin 1-hour data
df = pd.read_csv('BTC/BTC_1h.csv')
print(f"Rows: {len(df)}")
print(df.head())
```

### Load All Symbols

```python
import pandas as pd
from pathlib import Path

def load_all_symbols(data_dir='datasets', interval='1h'):
    all_data = {}
    for symbol_dir in Path(data_dir).iterdir():
        if symbol_dir.is_dir() and not symbol_dir.name.startswith('_'):
            csv_file = symbol_dir / f"{symbol_dir.name}_{interval}.csv"
            if csv_file.exists():
                all_data[symbol_dir.name] = pd.read_csv(csv_file)
    return all_data

data = load_all_symbols(interval='1h')
for symbol, df in data.items():
    print(f"{symbol}: {len(df)} rows")
```

### Combine All Symbols

```python
import pandas as pd
from pathlib import Path

def combine_all_symbols(data_dir='datasets', interval='1h'):
    dfs = []
    for symbol_dir in Path(data_dir).iterdir():
        if symbol_dir.is_dir() and not symbol_dir.name.startswith('_'):
            csv_file = symbol_dir / f"{symbol_dir.name}_{interval}.csv"
            if csv_file.exists():
                dfs.append(pd.read_csv(csv_file))
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined

df_combined = combine_all_symbols(interval='1h')
print(f"Total rows: {len(df_combined)}")
print(f"Symbols: {df_combined['symbol'].nunique()}")
```

## Statistics

- Total Symbols: {count}
- Total Timeframes: 2 (15m, 1h)
- Data Source: Binance REST API

## License

MIT License - Free for research and educational use
""".format(count=len(symbols))
    
    return readme

def print_tree(path, prefix="", is_root=False, max_depth=3, current_depth=0):
    """美化打印目錄結構"""
    
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(path.iterdir())
        entries = [e for e in entries if not e.name.startswith('.')]
    except PermissionError:
        return
    
    if is_root:
        print(f"{path.name}/")
    
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    
    for i, entry in enumerate(dirs + files):
        is_last = (i == len(dirs + files) - 1)
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{entry.name}")
        
        if entry.is_dir() and current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, next_prefix, False, max_depth, current_depth + 1)

def main():
    print("\n" + "="*70)
    print("CRYPTO DATASET ORGANIZER")
    print("="*70)
    
    # 源目錄
    source_dir = input("Source directory (default=crypto_data_cache): ").strip() or "crypto_data_cache"
    
    # 輸出目錄
    output_dir = input("Output directory (default=datasets): ").strip() or "datasets"
    
    # 是否創建合併數據集
    create_combined = input("Create combined datasets? (y/n, default=n): ").strip().lower() == 'y'
    
    # 執行組織
    success = organize_datasets(source_dir, output_dir, create_combined)
    
    if success:
        print()
        print("="*70)
        log("Organization completed successfully", "SUCCESS")
        print("="*70)
        print()
        print("Next steps:")
        print(f"1. Review the new structure in '{output_dir}' directory")
        print(f"2. Create 'models' directory for future model versions")
        print(f"3. Update Hugging Face dataset if needed")
        print()
    else:
        print()
        print("="*70)
        log("Organization failed", "ERROR")
        print("="*70)
        print()

if __name__ == "__main__":
    main()
