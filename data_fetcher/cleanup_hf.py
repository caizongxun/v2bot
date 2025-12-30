"""
Hugging Face 數據集清理脚本

功能:
- 列出根目錄中的所有文件
- 該改剚除舊的 CSV 文件
- 保留 datasets 和 README 。py etc.

使用方式:
!pip install -q huggingface-hub
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/cleanup_hf.py').text)
"""

import sys
from datetime import datetime

try:
    from huggingface_hub import HfApi, list_repo_tree
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
    from huggingface_hub import HfApi, list_repo_tree

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
        
        tree = list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        # 找出需要削除的根目錄 CSV 文件
        files_to_delete = []
        root_files = []
        
        for item in tree:
            if item.type == "file":
                # 只看根目錄的檔案（沒有 /）
                if '/' not in item.path:
                    root_files.append(item.path)
                    
                    # 判斷是否是需要削除的舊 CSV
                    if item.path.endswith('.csv'):
                        # 檢查是否是舊格式（不在 datasets 賊文件中）
                        if not item.path.startswith('_'):
                            files_to_delete.append(item.path)
        
        log(f"Found {len(root_files)} files in root directory", "INFO")
        log(f"Found {len(files_to_delete)} old CSV files to delete", "WARNING")
        
        # 顯示需要削除的檔案
        if files_to_delete:
            print("\nFiles to delete:")
            for fname in sorted(files_to_delete):
                print(f"  - {fname}")
        
        # 顯示会保留的檔案
        files_to_keep = [f for f in root_files if f not in files_to_delete]
        if files_to_keep:
            print("\nFiles to keep:")
            for fname in sorted(files_to_keep):
                print(f"  + {fname}")
        
        if not files_to_delete:
            log("No old CSV files found to delete", "SUCCESS")
            return True
        
        # 確認削除
        response = input(f"\nDelete {len(files_to_delete)} files? (yes/no): ").strip().lower()
        if response != 'yes':
            log("Cleanup cancelled", "WARNING")
            return False
        
        # 分別削除每個檔案
        print("\nDeleting files...")
        deleted_count = 0
        
        for fname in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Remove old root-level CSV file: {fname}"
                )
                log(f"Deleted: {fname}", "SUCCESS")
                deleted_count += 1
            except Exception as e:
                log(f"Failed to delete {fname}: {str(e)}", "ERROR")
        
        # 結果
        print("\n" + "="*70)
        print("CLEANUP COMPLETE")
        print("="*70)
        print(f"\nDeleted: {deleted_count}/{len(files_to_delete)} files")
        print(f"Repository: {repo_id}")
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
