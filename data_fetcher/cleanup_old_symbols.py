"""
刷除根目錄舊的幣種賊文件

作用:
- 下載 HF dataset
- 列出根目錄的所有幣種賊文件
- 删除所有號幣種（AAVE, ADA, BTC, ETH 等）
- 保留 klines 賊文件

使用方式:
!pip install -q huggingface-hub
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/cleanup_old_symbols.py').text)
"""

import sys
from datetime import datetime

try:
    from huggingface_hub import HfApi
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
    from huggingface_hub import HfApi

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

# 根目錄的舊幣種賊文件清单
OLD_SYMBOL_FOLDERS = [
    "AAVE", "ADA", "ALGO", "ARB", "ATOM",
    "AVAX", "BCH", "BNB", "BTC", "DOGE",
    "DOT", "ETC", "ETH", "FIL", "LINK",
    "LTC", "MATIC", "NEAR", "OP", "SHIB",
    "SOL", "UNI", "XRP"
]

def cleanup_old_symbols(repo_id, token):
    print("\n" + "="*70)
    print("CLEANUP OLD ROOT SYMBOL FOLDERS")
    print("="*70)
    
    api = HfApi(token=token)
    
    print(f"\nFolders to delete ({len(OLD_SYMBOL_FOLDERS)}):")
    for folder in sorted(OLD_SYMBOL_FOLDERS):
        print(f"  - {folder}/")
    
    print(f"\nFolders to keep:")
    print(f"  + klines/")
    print(f"  + _combined/")
    print(f"  + Root files (README.md, fetch_report.json, etc.)")
    
    # 確認削除
    response = input(f"\nDelete {len(OLD_SYMBOL_FOLDERS)} old symbol folders? (yes/no): ").strip().lower()
    if response != 'yes':
        log("Cleanup cancelled", "WARNING")
        return False
    
    # 分別削除每個賊文件
    print("\nDeleting folders...")
    deleted_count = 0
    failed_count = 0
    failed_folders = []
    
    for i, folder in enumerate(sorted(OLD_SYMBOL_FOLDERS), 1):
        try:
            api.delete_folder(
                path_in_repo=folder,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove old root symbol folder: {folder}"
            )
            log(f"[{i}/{len(OLD_SYMBOL_FOLDERS)}] Deleted: {folder}/", "SUCCESS")
            deleted_count += 1
        except Exception as e:
            error_msg = str(e)
            # 如果是 404 表示賊文件已不存在，也算成功
            if "404" in error_msg or "not found" in error_msg.lower():
                log(f"[{i}/{len(OLD_SYMBOL_FOLDERS)}] Already deleted: {folder}/", "SUCCESS")
                deleted_count += 1
            else:
                log(f"[{i}/{len(OLD_SYMBOL_FOLDERS)}] Failed to delete {folder}/: {error_msg}", "ERROR")
                failed_count += 1
                failed_folders.append(folder)
    
    # 結果
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"\nDeleted/Removed: {deleted_count}/{len(OLD_SYMBOL_FOLDERS)} folders")
    if failed_count > 0:
        print(f"Failed: {failed_count} folders")
        print(f"\nFailed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")
    
    print(f"\nFinal structure:")
    print(f"  .")
    print(f"  ├── klines/")
    print(f"  │  ├── AAVE/")
    print(f"  │  ├── ADA/")
    print(f"  │  ├── ... (23 symbols)")
    print(f"  │  ├── _combined/")
    print(f"  │  └── README.md")
    print(f"  ├── _combined/ (if exists)")
    print(f"  ├── README.md")
    print(f"  └── fetch_report.json")
    
    print(f"\nRepository: {repo_id}")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")
    print()
    
    return failed_count == 0

def main():
    print("\n" + "="*70)
    print("CLEANUP OLD SYMBOL FOLDERS")
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
    
    # 執行清理
    success = cleanup_old_symbols(repo_id, token)
    
    if success:
        print("All old symbol folders deleted successfully!")
    else:
        print("Some folders failed to delete.")

if __name__ == "__main__":
    main()
else:
    main()
