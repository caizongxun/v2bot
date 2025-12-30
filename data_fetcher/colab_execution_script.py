"""
Colab 上的執行脚本
直接載入此脚本执行，自動完成：
1. 安裝依賴
2. 抓取數據
3. 上傳到 Hugging Face
"""

import subprocess
import sys
import os

# =====================================================
# 第一步：安裝依賴
# =====================================================
print("[Step 1] Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests", "pandas", "yfinance"])
print("✓ Dependencies installed\n")

# =====================================================
# 第二步：從 GitHub 克隆v2bot
# =====================================================
print("[Step 2] Cloning v2bot repository...")
if not os.path.exists('/content/v2bot'):
    subprocess.check_call(['git', 'clone', 'https://github.com/caizongxun/v2bot.git', '/content/v2bot'])
print("✓ Repository cloned\n")

# =====================================================
# 第三步：匯入抓取模組並抓取數據
# =====================================================
print("[Step 3] Running data fetcher...\n")
sys.path.insert(0, '/content/v2bot')

from data_fetcher.crypto_historical_data_fetcher import CryptoDataFetcher, FetcherConfig

fetcher = CryptoDataFetcher(output_dir='/content/crypto_data_cache')
results = fetcher.fetch_all_cryptos_parallel(max_workers=5)
fetcher.generate_summary_report(results)

data_dir = fetcher.output_dir
print(f"\n✓ Data cached in: {data_dir}")

# =====================================================
# 第四步：上傳到 Hugging Face Datasets
# =====================================================
print("\n[Step 4] Preparing for Hugging Face upload...")

# 安裝 huggingface_hub
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import HfApi, HfFolder
import glob

print("\nHugging Face Upload Configuration:")
print("======================================")
print("""
要上傳數據到 Hugging Face，請按照以下步驟：

1. 造訪 https://huggingface.co/settings/tokens 獲取你的 token

2. 篆鑰語对選選 'write' 權限

3. 解除第下方的評語，然後使用你的 token 執行上載
""")

hf_token = input("Enter your Hugging Face token: ").strip()
hf_repo_name = input("Enter repository name (e.g., 'v2-crypto-ohlcv-data'): ").strip()

if not hf_token or not hf_repo_name:
    print("\n⚠ Token or repo name not provided. Skipping upload.")
    sys.exit(0)

print(f"\n✓ Token set (first 10 chars: {hf_token[:10]}...)")
print(f"✓ Repository: {hf_repo_name}")

# =====================================================
# 第五步：上傳整個數據目錄
# =====================================================
print("\n[Step 5] Uploading data directory to Hugging Face...")

api = HfApi(token=hf_token)

try:
    # 初始化 repo (如果不存在)
    repo_id = f"{api.whoami()['name']}/{hf_repo_name}"
    
    print(f"\nUploading to: https://huggingface.co/datasets/{repo_id}")
    
    # 上傳整個數據目錄
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        multi_commit=True,  # 使用多個提交
        multi_commit_nb_threads=4  # 並行上傳線程
    )
    
    print(f"\n✓ Upload completed!")
    print(f"✓ Dataset URL: https://huggingface.co/datasets/{repo_id}")
    
except Exception as e:
    print(f"\n⚠ Upload failed: {str(e)}")
    print(f"\nManual upload instructions:")
    print(f"1. Visit: https://huggingface.co/new-dataset")
    print(f"2. Create dataset: {hf_repo_name}")
    print(f"3. Upload files from: {data_dir}")

# =====================================================
# 第六步：數據驗證
# =====================================================
print("\n[Step 6] Data Verification...")

stats = fetcher.get_data_statistics()
print(f"\nData Statistics:")
print(f"  Total Files: {stats['total_files']}")
print(f"  Total Rows: {stats['total_rows']:,}")
print(f"  15m K-lines: {stats['files_by_interval']['15m']} files")
print(f"  1h K-lines: {stats['files_by_interval']['1h']} files")
print(f"\nFiles in cache:")
for filename, details in list(stats['file_details'].items())[:5]:
    print(f"  - {filename}: {details['rows']} rows")
if len(stats['file_details']) > 5:
    print(f"  ... and {len(stats['file_details']) - 5} more files")

print("\n" + "="*60)
print("✓ All steps completed successfully!")
print("="*60)
