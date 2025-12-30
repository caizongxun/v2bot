"""
Dataset 結構調整脚本

作用:
- 下載整個數據集
- 列出當前結構
- 創建 klines 賊文件
- 將 datasets 的屬載移到 klines
- 上傳回 HF

使用方式:
!pip install -q huggingface-hub pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/restructure_klines.py').text)
"""

import os
import sys
import shutil
import tempfile
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
    elif level == "WARNING":
        print(f"[{ts}] WARNING: {msg}")

def print_tree(path, prefix="", is_root=False, max_depth=3, current_depth=0, max_items=50):
    """美化打印目錄結構"""
    if current_depth >= max_depth:
        return
    
    try:
        entries = sorted(path.iterdir())
        entries = [e for e in entries if not e.name.startswith('.')]
    except PermissionError:
        return
    
    if is_root:
        print(f"\n{path.name}/")
    
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    
    # 限制顯示項數
    if len(files) > max_items:
        files_display = files[:max_items]
        remaining = len(files) - max_items
        show_remaining = True
    else:
        files_display = files
        show_remaining = False
    
    all_items = dirs + files_display
    
    for i, entry in enumerate(all_items):
        is_last = (i == len(all_items) - 1 and not show_remaining)
        current_prefix = "└── " if is_last else "├── "
        
        if entry.is_dir():
            print(f"{prefix}{current_prefix}{entry.name}/")
            if current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(entry, next_prefix, False, max_depth, current_depth + 1, max_items)
        else:
            size_kb = entry.stat().st_size / 1024
            print(f"{prefix}{current_prefix}{entry.name} ({size_kb:.0f}KB)")
    
    if show_remaining:
        print(f"{prefix}└── ... and {remaining} more files")

def restructure(repo_id, token):
    print("\n" + "="*70)
    print("DATASET RESTRUCTURE - MOVE TO KLINES")
    print("="*70)
    
    api = HfApi(token=token)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # STEP 1: 下載
        print("\nSTEP 1: Download current dataset...")
        log(f"Downloading from {repo_id}...", "INFO")
        src_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=temp_dir
        )
        log(f"Downloaded", "SUCCESS")
        
        src_path = Path(src_dir)
        
        # 顯示當前結構
        print("\nCurrent structure:")
        print_tree(src_path, is_root=True, max_depth=2)
        
        # STEP 2: 創建新結構
        print("\n" + "="*70)
        print("STEP 2: Create new structure...")
        
        new_dir = Path(temp_dir) / "restructured"
        new_dir.mkdir(exist_ok=True)
        
        klines_dir = new_dir / "klines"
        klines_dir.mkdir(exist_ok=True)
        
        # 複製 datasets 賊文件（如果存在）
        datasets_src = src_path / "datasets"
        if datasets_src.exists() and datasets_src.is_dir():
            log(f"Found datasets folder, moving to klines...", "INFO")
            
            for item in datasets_src.iterdir():
                dest = klines_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                    log(f"Copied folder: {item.name}", "SUCCESS")
                elif item.is_file():
                    shutil.copy2(item, dest)
                    log(f"Copied file: {item.name}", "SUCCESS")
        else:
            log(f"datasets folder not found, checking for symbol folders...", "WARNING")
            # 這些幣種列表後來可以嚼適擊需求修改
            symbols = [d.name for d in src_path.iterdir() if d.is_dir() and not d.name.startswith('_')]
            
            for symbol in sorted(symbols):
                src_sym = src_path / symbol
                dest_sym = klines_dir / symbol
                shutil.copytree(src_sym, dest_sym, dirs_exist_ok=True)
                log(f"Moved: {symbol}", "SUCCESS")
        
        # 複製其他根目錄檔案（基本上伺有 README.md, fetch_report.json 等）
        for item in src_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                dest = new_dir / item.name
                shutil.copy2(item, dest)
                log(f"Copied root file: {item.name}", "SUCCESS")
        
        # 顯示新結構
        print("\nNew structure:")
        print_tree(new_dir, is_root=True, max_depth=2)
        
        # STEP 3: 上傳
        print("\n" + "="*70)
        print("STEP 3: Upload to HF...")
        log(f"Uploading new structure to {repo_id}...", "INFO")
        print("This may take 15-25 minutes...\n")
        
        api.upload_folder(
            folder_path=str(new_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Restructure: move datasets to klines folder"
        )
        
        log(f"Upload completed", "SUCCESS")
        
        # 結果
        print("\n" + "="*70)
        print("RESTRUCTURE COMPLETE")
        print("="*70)
        print(f"\nNew structure:")
        print(f"  .")
        print(f"  ├── klines/")
        print(f"  │  ├── AAVE/")
        print(f"  │  ├── ADA/")
        print(f"  │  ├── ... (23 symbols)")
        print(f"  │  ├── _combined/")
        print(f"  │  └── README.md")
        print(f"  ├── models/ (for future versions)")
        print(f"  ├── README.md")
        print(f"  └── fetch_report.json")
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
        log(f"Cleaning up temporary files...", "INFO")
        shutil.rmtree(temp_dir, ignore_errors=True)
        log(f"Done", "SUCCESS")

def main():
    print("\n" + "="*70)
    print("DATASET RESTRUCTURE TOOL")
    print("="*70)
    
    repo_id = input("\nRepo ID (format: username/repo): ").strip()
    if '/' not in repo_id:
        print("Invalid format")
        return
    
    token = input("HF Token: ").strip()
    if not token:
        print("No token")
        return
    
    # 驗證
    try:
        api = HfApi(token=token)
        user = api.whoami()
        log(f"Authenticated as: {user['name']}", "SUCCESS")
    except Exception as e:
        log(f"Auth failed: {str(e)}", "ERROR")
        return
    
    # 執行組織
    success = restructure(repo_id, token)
    
    if success:
        print("Restructure completed successfully!")
    else:
        print("Restructure failed.")

if __name__ == "__main__":
    main()
else:
    main()
