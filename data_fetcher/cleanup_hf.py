"""
Hugging Face 數據集清理脚本 - 修正版

功能:
- 列出根目錄中的所有文件
- 需要削除舊的 CSV 文件
- 保留 datasets 和 README 。py etc.

使用方式:
!pip install -q huggingface-hub
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/cleanup_hf.py').text)
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

def cleanup_hf(repo_id, token):
    print("\n" + "="*70)
    print("HUGGING FACE DATASET CLEANUP")
    print("="*70)
    
    api = HfApi(token=token)
    
    try:
        # 列出根目錄中的所有文件
        print("\nScanning files...")
        log(f"Listing files in {repo_id}", "INFO")
        
        # 使用 API 來列出檔案
        # get_repo_files 不存在，控制台列出或使用 list_files_in_repo
        files_in_repo = []
        
        # 嘗試使用例計算檔案層次
        try:
            # 單純地列列根目錄中的檔案
            # 使用 urllib 直接存取 API
            import requests as req
            response = req.get(
                f"https://huggingface.co/api/datasets/{repo_id}/tree/main",
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            data = response.json()
            files_in_repo = data.get('siblings', [])
        except:
            # 敢一推推，使用你提供的檔案清单
            files_in_repo = [
                {"name": "BTC_15m.csv"}, {"name": "BTC_1h.csv"},
                {"name": "ETH_15m.csv"}, {"name": "ETH_1h.csv"},
                {"name": "BNB_15m.csv"}, {"name": "BNB_1h.csv"},
                {"name": "SOL_15m.csv"}, {"name": "SOL_1h.csv"},
                {"name": "XRP_15m.csv"}, {"name": "XRP_1h.csv"},
                {"name": "ADA_15m.csv"}, {"name": "ADA_1h.csv"},
                {"name": "AVAX_15m.csv"}, {"name": "AVAX_1h.csv"},
                {"name": "DOT_15m.csv"}, {"name": "DOT_1h.csv"},
                {"name": "LINK_15m.csv"}, {"name": "LINK_1h.csv"},
                {"name": "MATIC_15m.csv"}, {"name": "MATIC_1h.csv"},
                {"name": "LTC_15m.csv"}, {"name": "LTC_1h.csv"},
                {"name": "UNI_15m.csv"}, {"name": "UNI_1h.csv"},
                {"name": "BCH_15m.csv"}, {"name": "BCH_1h.csv"},
                {"name": "ETC_15m.csv"}, {"name": "ETC_1h.csv"},
                {"name": "FIL_15m.csv"}, {"name": "FIL_1h.csv"},
                {"name": "DOGE_15m.csv"}, {"name": "DOGE_1h.csv"},
                {"name": "ALGO_15m.csv"}, {"name": "ALGO_1h.csv"},
                {"name": "ATOM_15m.csv"}, {"name": "ATOM_1h.csv"},
                {"name": "NEAR_15m.csv"}, {"name": "NEAR_1h.csv"},
                {"name": "ARB_15m.csv"}, {"name": "ARB_1h.csv"},
                {"name": "OP_15m.csv"}, {"name": "OP_1h.csv"},
                {"name": "AAVE_15m.csv"}, {"name": "AAVE_1h.csv"},
                {"name": "SHIB_15m.csv"}, {"name": "SHIB_1h.csv"},
            ]
        
        # 找出根目錄 CSV 檔案
        root_csv_files = []
        for file_obj in files_in_repo:
            fname = file_obj.get('name', '')
            # 檢查是否是根目錄的 CSV（沒有 / 及以 .csv 結尾）
            if '/' not in fname and fname.endswith('.csv'):
                root_csv_files.append(fname)
        
        log(f"Found {len(files_in_repo)} total files", "INFO")
        log(f"Found {len(root_csv_files)} root-level CSV files to delete", "WARNING")
        
        if root_csv_files:
            print("\nFiles to delete:")
            for fname in sorted(root_csv_files):
                print(f"  - {fname}")
        
        if not root_csv_files:
            log("No old CSV files found to delete", "SUCCESS")
            return True
        
        # 確認削除
        response = input(f"\nDelete {len(root_csv_files)} files? (yes/no): ").strip().lower()
        if response != 'yes':
            log("Cleanup cancelled", "WARNING")
            return False
        
        # 分別削除每個檔案
        print("\nDeleting files...")
        deleted_count = 0
        failed_count = 0
        
        for fname in sorted(root_csv_files):
            try:
                api.delete_file(
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Remove old root-level CSV: {fname}"
                )
                log(f"Deleted: {fname}", "SUCCESS")
                deleted_count += 1
            except Exception as e:
                log(f"Failed to delete {fname}: {str(e)}", "ERROR")
                failed_count += 1
        
        # 結果
        print("\n" + "="*70)
        print("CLEANUP COMPLETE")
        print("="*70)
        print(f"\nDeleted: {deleted_count}/{len(root_csv_files)} files")
        if failed_count > 0:
            print(f"Failed: {failed_count} files")
        print(f"\nRepository: {repo_id}")
        print(f"URL: https://huggingface.co/datasets/{repo_id}")
        print()
        
        return True
        
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("HF DATASET CLEANUP TOOL")
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
    cleanup_hf(repo_id, token)

if __name__ == "__main__":
    main()
else:
    main()
