#!/usr/bin/env python3
"""
SSL Hybrid v3 Model - Upload to HuggingFace
粀合墨方序上傳脚本
"""

import os
import sys
from pathlib import Path

print("""
================================================================================
SSL HYBRID v3 - HuggingFace 上傳助理
================================================================================
""")

print("步驄33：安裝套件...")
try:
    from huggingface_hub import HfApi, login, create_repo
    print("  ✅ huggingface-hub 已安裝")
except ImportError:
    print("  安裝 huggingface-hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"])
    from huggingface_hub import HfApi, login, create_repo
    print("  ✅ 安裝完成")

print("""
================================================================================
配置信息
================================================================================
""")

# 配置
# ============================================================
# ⬆️ 修改這些值
# ============================================================

HF_USERNAME = input("請輸入你的 HuggingFace 帳號名: ").strip()

if not HF_USERNAME:
    print("❌ HuggingFace 帳號名不能為空")
    sys.exit(1)

DATASET_NAME = "ssl-hybrid-v3-model"

# 檔案路徑
MODEL_FILES = {
    'ssl_filter_v3.keras': './ssl_filter_v3.keras',
    'ssl_scaler_v3.json': './ssl_scaler_v3.json',
    'ssl_metadata_v3.json': './ssl_metadata_v3.json'
}

print(f"\
HuggingFace 帳號: {HF_USERNAME}")
print(f"數據集名稱: {DATASET_NAME}")

print("""
================================================================================
步驄32：驗證檔案
================================================================================
""")

# 棂查檔案
def check_files():
    missing = []
    for filename, filepath in MODEL_FILES.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✅ {filename:30s} ({size:6.1f} MB)")
        else:
            missing.append(filename)
            print(f"  ❌ {filename:30s} (找不到)")
    
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
步驄34：誊橙 HuggingFace
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
步驄35：創建數據集
================================================================================
""")

api = HfApi()

try:
    print(f"創建數據集: {HF_USERNAME}/{DATASET_NAME}")
    repo_url = create_repo(
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    print(f"✅ 數據集已創建: {repo_url}")
except Exception as e:
    print(f"注意: {e}")
    print("數據集可能已存在")

print("""
================================================================================
步驄36：上傳檔案
================================================================================
""")

success_count = 0

for filename, filepath in MODEL_FILES.items():
    print(f"
上傳 {filename}...")
    try:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
            repo_type="dataset",
            commit_message=f"Upload {filename}"
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
上傳伏機: {success_count}/3 檔案")

if success_count == 3:
    print("✅ 所有檔案上傳成功!")
    print(f"
數據集地址:")
    print(f"  https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}")
    
    print(f"
在 Colab 中下載 使用:")
    print(f"""
from huggingface_hub import hf_hub_download
import shutil

for filename in ['ssl_filter_v3.keras', 'ssl_scaler_v3.json', 'ssl_metadata_v3.json']:
    path = hf_hub_download(
        repo_id="{HF_USERNAME}/{DATASET_NAME}",
        filename=filename,
        repo_type="dataset"
    )
    shutil.copy(path, filename)
    print(f"Downloaded {{filename}}")
    """)
else:
    print(f"⚠️ 只有 {success_count}/3 檔案上傳成功")
    print("請棂查錯誤消淺並重試")

print()
