#!/usr/bin/env python3
"""
SSL Hybrid v3 - 上傳到現有的 HuggingFace 數據集
將模型上傳到 models_v1/ 子資料夾
"""

import os
import sys
from pathlib import Path

print("""
================================================================================
SSL HYBRID v3 - 上傳到現有 HF 數據集
================================================================================
""")

print("\n步驄31: 安裝套件...")
try:
    from huggingface_hub import HfApi, login, list_repo_files
    print("  ✅ huggingface-hub 已安裝")
except ImportError:
    print("  安裝 huggingface-hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"])
    from huggingface_hub import HfApi, login, list_repo_files
    print("  ✅ 安裝完成")

print("""
================================================================================
配置信息
================================================================================
""")

# 配置
HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
MODEL_SUBFOLDER = "models_v1"

# 檔案路徑
MODEL_FILES = {
    'ssl_filter_v3.keras': './ssl_filter_v3.keras',
    'ssl_scaler_v3.json': './ssl_scaler_v3.json',
    'ssl_metadata_v3.json': './ssl_metadata_v3.json'
}

print(f"目標數據集: {HF_REPO}")
print(f"上傳子資料夾: {MODEL_SUBFOLDER}/")
print(f"上傳檔案數: {len(MODEL_FILES)}")

print("""
================================================================================
步驄32: 驗證檔案
================================================================================
""")

# 棂查檔案
def check_files():
    missing = []
    total_size = 0
    
    for filename, filepath in MODEL_FILES.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / 1024 / 1024
            total_size += size
            print(f"  ✅ {filename:30s} ({size_mb:6.1f} MB)")
        else:
            missing.append(filename)
            print(f"  ❌ {filename:30s} (找不到)")
    
    total_mb = total_size / 1024 / 1024
    print(f"\n  總檔案大小: {total_mb:.1f} MB")
    return missing

missing_files = check_files()

if missing_files:
    print(f"""
❌ 找不到以下檔案:
{', '.join(missing_files)}
請確保檔案在當前目錄，或修改 MODEL_FILES 路徑
    """)
    sys.exit(1)

print("""
================================================================================
步驄33: 誊橙 HuggingFace
================================================================================
""")

try:
    print("誊橙中...")
    login()
    print("✅ 誊橙成功")
except Exception as e:
    print(f"❌ 誊橙失敗: {e}")
    print("請先在 https://huggingface.co/settings/tokens 產生 token")
    sys.exit(1)

print("""
================================================================================
步驄34: 窗查現有的數據集結構
================================================================================
""")

api = HfApi()

try:
    print(f"列出 {HF_REPO} 中的檔案...\n")
    
    # 窗查存在的檔案
    try:
        files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        print("現有檔案/資料夾:")
        for f in sorted(files)[:10]:  # 顯示前 10 個
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... 以及其他 {len(files) - 10} 個")
        print(f"\n總計: {len(files)} 個項目\n")
    except:
        print("無法列出檔案，繼續進行\n")
        
except Exception as e:
    print(f"❌ 錯誤: {e}")
    sys.exit(1)

print("""
================================================================================
步驄35: 上傳檔案到 models_v1/ 子資料夾
================================================================================
""")

success_count = 0

for filename, filepath in MODEL_FILES.items():
    # 分後的路徑: models_v1/ssl_filter_v3.keras
    path_in_repo = f"{MODEL_SUBFOLDER}/{filename}"
    
    print(f"\n上傳 {filename}")
    print(f"  準誇位置: {path_in_repo}")
    
    try:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Add SSL Hybrid v3 {filename}"
        )
        print(f"  ✅ 成功")
        success_count += 1
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

print("""
================================================================================
封墨
================================================================================
""")

print(f"
上傳伏機: {success_count}/{len(MODEL_FILES)} 檔案")

if success_count == len(MODEL_FILES):
    print("✅ 所有檔案上傳成功!")
    print(f"
數據集地址:")
    print(f"  https://huggingface.co/datasets/{HF_REPO}")
    
    print(f"
檔案位置:")
    print(f"  models_v1/ssl_filter_v3.keras")
    print(f"  models_v1/ssl_scaler_v3.json")
    print(f"  models_v1/ssl_metadata_v3.json")
    
    print(f"
在 Colab 中下載使用:")
    print(f"""
from huggingface_hub import hf_hub_download
import shutil

repo_id = "{HF_REPO}"
subfolder = "models_v1"

for filename in ['ssl_filter_v3.keras', 'ssl_scaler_v3.json', 'ssl_metadata_v3.json']:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{{subfolder}}/{{filename}}",
        repo_type="dataset"
    )
    shutil.copy(path, filename)
    print(f"Downloaded {{filename}}")
    """)
else:
    print(f"⚠️ 只有 {success_count}/{len(MODEL_FILES)} 檔案上傳成功")
    print("請棂查錯誤消淺並重試")

print()
