#!/usr/bin/env python3
"""
SSL Hybrid v3 - 診斷脚本
棂查 HF 數據集狀況和檔案是否已上傳
"""

import os
import sys

print("""
================================================================================
SSL HYBRID v3 - 診斷棂查
================================================================================
""")

# 棂查 1: 本地檔案
print("\n棂查 1: 本地檔案")
print("="*80)

MODEL_FILES = {
    'ssl_filter_v3.keras': './ssl_filter_v3.keras',
    'ssl_scaler_v3.json': './ssl_scaler_v3.json',
    'ssl_metadata_v3.json': './ssl_metadata_v3.json'
}

local_ok = True
for filename, filepath in MODEL_FILES.items():
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ✅ {filename:30s} ({size_mb:6.1f} MB)")
    else:
        print(f"  ❌ {filename:30s} (找不到)")
        local_ok = False

if not local_ok:
    print("\n⚠️ 本地檔案不完整!")
    sys.exit(1)

print("\n  ✅ 所有本地檔案存在")

# 棂查 2: HF 連接
print("\n棂查 2: HuggingFace 連接")
print("="*80)

try:
    from huggingface_hub import HfApi, list_repo_files
    print("  ✅ huggingface-hub 已安裝")
except ImportError:
    print("  ❌ huggingface-hub 未安裝")
    print("  安裝: pip install huggingface-hub")
    sys.exit(1)

# 棂查 3: 數據集連接
print("\n棂查 3: 數據集是否存在")
print("="*80)

HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
MODEL_SUBFOLDER = "models_v1"

api = HfApi()

try:
    print(f"  查詢: {HF_REPO}")
    files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    print(f"  ✅ 數據集存在 ({len(files)} 個項目)")
    
    # 棂查是否有 models_v1 資料夾
    print(f"\n棂查 4: models_v1 子資料夾")
    print("="*80)
    
    model_files_in_repo = [f for f in files if f.startswith("models_v1/")]
    
    if model_files_in_repo:
        print(f"  ✅ 找到 {len(model_files_in_repo)} 個檔案:")
        for f in model_files_in_repo:
            print(f"    - {f}")
    else:
        print(f"  ❌ 找不到 models_v1 子資料夾")
        print(f"\n數據集中的項目 ({len(files)} 個):")
        for f in files[:10]:
            print(f"    - {f}")
        if len(files) > 10:
            print(f"    ... 以及其他 {len(files) - 10} 個")
    
    # 棂查是否有寫入權限
print("\n棂查 5: HuggingFace Token")
    print("="*80)
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"  ✅ 已登錄")
        print(f"    帳號: {user_info['name']}")
    except:
        print(f"  ❌ 未登錄 HuggingFace")
        print(f"    执行: huggingface-cli login")
        print(f"    或訪問: https://huggingface.co/settings/tokens")
        
except Exception as e:
    print(f"  ❌ 錯誤: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("\n總結:")
print("="*80)

if model_files_in_repo:
    print("\n✅ 檔案已成功上傳")
    print("\n在 Colab 中使用下載代程:")
    print("""
from huggingface_hub import hf_hub_download
import shutil

for filename in ['ssl_filter_v3.keras', 'ssl_scaler_v3.json', 'ssl_metadata_v3.json']:
    path = hf_hub_download(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        filename=f"models_v1/{filename}",
        repo_type="dataset"
    )
    shutil.copy(path, filename)
    print(f"Downloaded {filename}")
    """)
else:
    print("\n❌ 檔案本尚未上傳")
    print("\n需要將檔案上傳到 HuggingFace")
    print("\n執行上傳脚本:")
    print("  python ssl_hybrid/upload_to_hf_simple.py")
    print("\n或手動上傳:")
    print("  1. 訪問: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data")
    print("  2. 點擊 'Add file' -> 'Upload files'")
    print("  3. 選擇 3 個檔案並上傳")

print("\n" + "="*80)
