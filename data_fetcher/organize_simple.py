"""
簡化遠端組織脚本 - 簡简根本，無版本問題

步驟:
1. 下載整個數據集
2. 組織成按幣種分組
3. 分載回 HF

使用方式:
!pip install -q huggingface-hub pandas requests
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/organize_simple.py').text)
"""

import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd

try:
    from huggingface_hub import HfApi, snapshot_download
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
    from huggingface_hub import HfApi, snapshot_download

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"[{ts}] {msg}")
    elif level == "SUCCESS":
        print(f"[{ts}] SUCCESS: {msg}")
    elif level == "ERROR":
        print(f"[{ts}] ERROR: {msg}")

def create_symbol_readme(symbol, files):
    readme = f"""# {symbol}

Cryptocurrency: {symbol}

## Files\n"""
    for fname, interval, rows in files:
        readme += f"- {fname}: {interval} ({rows:,} rows)\n"
    readme += """
## Usage

```python
import pandas as pd
df = pd.read_csv('{}_1h.csv')
```
""".format(symbol)
    return readme

def create_combined_readme():
    return """# Combined Data

All symbols combined.

## Files
- all_symbols_15m.csv
- all_symbols_1h.csv
"""

def create_main_readme(symbols):
    syms = sorted(symbols)
    readme = f"""# Datasets

Organized by symbol.

## Symbols ({len(syms)})

{', '.join(syms)}
"""
    return readme

def organize_and_upload(repo_id, token):
    print("\n" + "="*70)
    print("SIMPLE REMOTE ORGANIZER")
    print("="*70)
    
    api = HfApi(token=token)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # STEP 1: DOWNLOAD
        print("\nSTEP 1: Download...")
        log(f"Downloading from {repo_id}...", "INFO")
        src_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token, cache_dir=temp_dir)
        log(f"Downloaded", "SUCCESS")
        
        csvs = list(Path(src_dir).glob('*.csv'))
        csvs = [f for f in csvs if not f.name.startswith('.')]
        log(f"Found {len(csvs)} CSV files", "SUCCESS")
        
        # STEP 2: ORGANIZE
        print("\nSTEP 2: Organize...")
        crypto_data = {}
        for csv in csvs:
            parts = csv.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                interval = parts[1]
                if symbol not in crypto_data:
                    crypto_data[symbol] = []
                crypto_data[symbol].append((csv, interval))
        
        org_dir = Path(temp_dir) / "organized"
        org_dir.mkdir(exist_ok=True)
        
        all_15m = []
        all_1h = []
        
        for symbol in sorted(crypto_data.keys()):
            sym_dir = org_dir / symbol
            sym_dir.mkdir(exist_ok=True)
            
            files_info = []
            for csv_file, interval in crypto_data[symbol]:
                dest = sym_dir / csv_file.name
                shutil.copy2(csv_file, dest)
                
                try:
                    df = pd.read_csv(dest)
                    rows = len(df)
                    files_info.append((csv_file.name, interval, rows))
                    
                    if interval == '15m':
                        all_15m.append(df)
                    elif interval == '1h':
                        all_1h.append(df)
                except:
                    pass
            
            with open(sym_dir / "README.md", 'w') as f:
                f.write(create_symbol_readme(symbol, files_info))
            
            log(f"Organized {symbol}", "INFO")
        
        # STEP 3: COMBINED
        print("\nSTEP 3: Create combined...")
        comb_dir = org_dir / "_combined"
        comb_dir.mkdir(exist_ok=True)
        
        if all_15m:
            df_15m = pd.concat(all_15m, ignore_index=True)
            df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)
            df_15m.to_csv(comb_dir / "all_symbols_15m.csv", index=False)
            log(f"Created all_symbols_15m.csv ({len(df_15m)} rows)", "SUCCESS")
        
        if all_1h:
            df_1h = pd.concat(all_1h, ignore_index=True)
            df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
            df_1h.to_csv(comb_dir / "all_symbols_1h.csv", index=False)
            log(f"Created all_symbols_1h.csv ({len(df_1h)} rows)", "SUCCESS")
        
        with open(comb_dir / "README.md", 'w') as f:
            f.write(create_combined_readme())
        
        with open(org_dir / "README.md", 'w') as f:
            f.write(create_main_readme(crypto_data.keys()))
        
        # STEP 4: UPLOAD
        print("\nSTEP 4: Upload to HF...")
        log("Uploading...", "INFO")
        print("(This may take 10-20 minutes)\n")
        
        # 直接使用最基本的參數
        api.upload_folder(
            folder_path=str(org_dir),
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        log("Upload completed", "SUCCESS")
        
        print("\n" + "="*70)
        print("SUCCESS")
        print("="*70)
        print(f"\nRepository: {repo_id}")
        print(f"URL: https://huggingface.co/datasets/{repo_id}")
        print()
        
        return True
        
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        log("Cleaning up...", "INFO")
        shutil.rmtree(temp_dir, ignore_errors=True)
        log("Done", "SUCCESS")

if __name__ == "__main__" or True:
    print("\n" + "="*70)
    print("REMOTE ORGANIZER")
    print("="*70)
    
    repo_id = input("\nRepo ID (format: username/repo): ").strip()
    if '/' not in repo_id:
        print("Invalid format")
        sys.exit(1)
    
    token = input("HF Token: ").strip()
    if not token:
        print("No token")
        sys.exit(1)
    
    try:
        api = HfApi(token=token)
        user = api.whoami()
        log(f"Authenticated as: {user['name']}", "SUCCESS")
    except Exception as e:
        log(f"Auth failed: {str(e)}", "ERROR")
        sys.exit(1)
    
    success = organize_and_upload(repo_id, token)
    sys.exit(0 if success else 1)
