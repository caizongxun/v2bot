"""
簡接削除根目錄 CSV 脚本

直接削除該知的根目錄 CSV 檔案

使用方式:
!pip install -q huggingface-hub
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/cleanup_direct.py').text)
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

# 根目錄的舊 CSV 檔案清単
ROOT_CSV_FILES = [
    "AAVE_15m.csv", "AAVE_1h.csv",
    "ADA_15m.csv", "ADA_1h.csv",
    "ALGO_15m.csv", "ALGO_1h.csv",
    "ARB_15m.csv", "ARB_1h.csv",
    "ATOM_15m.csv", "ATOM_1h.csv",
    "AVAX_15m.csv", "AVAX_1h.csv",
    "BCH_15m.csv", "BCH_1h.csv",
    "BNB_15m.csv", "BNB_1h.csv",
    "BTC_15m.csv", "BTC_1h.csv",
    "DOGE_15m.csv", "DOGE_1h.csv",
    "DOT_15m.csv", "DOT_1h.csv",
    "ETC_15m.csv", "ETC_1h.csv",
    "ETH_15m.csv", "ETH_1h.csv",
    "FIL_15m.csv", "FIL_1h.csv",
    "LINK_15m.csv", "LINK_1h.csv",
    "LTC_15m.csv", "LTC_1h.csv",
    "MATIC_15m.csv", "MATIC_1h.csv",
    "NEAR_15m.csv", "NEAR_1h.csv",
    "OP_15m.csv", "OP_1h.csv",
    "SHIB_15m.csv", "SHIB_1h.csv",
    "SOL_15m.csv", "SOL_1h.csv",
    "UNI_15m.csv", "UNI_1h.csv",
    "XRP_15m.csv", "XRP_1h.csv",
]

def cleanup_hf(repo_id, token):
    print("\n" + "="*70)
    print("DIRECT CLEANUP - ROOT CSV FILES")
    print("="*70)
    
    api = HfApi(token=token)
    
    print(f"\nFiles to delete ({len(ROOT_CSV_FILES)}):")
    for fname in sorted(ROOT_CSV_FILES):
        print(f"  - {fname}")
    
    # 確認削除
    response = input(f"\nDelete {len(ROOT_CSV_FILES)} files? (yes/no): ").strip().lower()
    if response != 'yes':
        log("Cleanup cancelled", "WARNING")
        return False
    
    # 分別削除每個檔案
    print("\nDeleting files...")
    deleted_count = 0
    failed_count = 0
    failed_files = []
    
    for i, fname in enumerate(sorted(ROOT_CSV_FILES), 1):
        try:
            api.delete_file(
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove old root-level CSV: {fname}"
            )
            log(f"[{i}/{len(ROOT_CSV_FILES)}] Deleted: {fname}", "SUCCESS")
            deleted_count += 1
        except Exception as e:
            error_msg = str(e)
            # 如果是 404 表示檔案已不存在，也算成功
            if "404" in error_msg or "not found" in error_msg.lower():
                log(f"[{i}/{len(ROOT_CSV_FILES)}] Already deleted: {fname}", "SUCCESS")
                deleted_count += 1
            else:
                log(f"[{i}/{len(ROOT_CSV_FILES)}] Failed to delete {fname}: {error_msg}", "ERROR")
                failed_count += 1
                failed_files.append(fname)
    
    # 結果
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"\nDeleted/Removed: {deleted_count}/{len(ROOT_CSV_FILES)} files")
    if failed_count > 0:
        print(f"Failed: {failed_count} files")
        print(f"\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")
    
    print(f"\nRepository: {repo_id}")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")
    print()
    
    return failed_count == 0

def main():
    print("\n" + "="*70)
    print("HF DATASET DIRECT CLEANUP")
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
    success = cleanup_hf(repo_id, token)
    
    if success:
        print("All files deleted successfully!")
    else:
        print("Some files failed to delete.")

if __name__ == "__main__":
    main()
else:
    main()
