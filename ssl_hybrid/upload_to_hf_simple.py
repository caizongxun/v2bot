#!/usr/bin/env python3
"""
SSL Hybrid v3 - 粀合简化上傳脚本
直接上傳模型檔案到 HF
"""

import os
import sys
from pathlib import Path

print("""
================================================================================
SSL HYBRID v3 - HuggingFace 粀合上傳
================================================================================
""")

# 第 1 步: 安裝套件
print("\n步驄31: 安裝套件...")

try:
    from huggingface_hub import HfApi, login
    print("  ✅ huggingface-hub 已安裝")
except ImportError:
    print("  安裝 huggingface-hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"])
    from huggingface_hub import HfApi, login
    print("  ✅ 安裝完成")

# 第 2 步: 配置
print("""
================================================================================
配置
================================================================================
""")

HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
MODEL_SUBFOLDER = "models_v1"

MODEL_FILES = {
    'ssl_filter_v3.keras': './ssl_filter_v3.keras',
    'ssl_scaler_v3.json': './ssl_scaler_v3.json',
    'ssl_metadata_v3.json': './ssl_metadata_v3.json'
}

print(f"目標數據集: {HF_REPO}")
print(f"上傳子資料夾: {MODEL_SUBFOLDER}/")
print(f"檔案數: {len(MODEL_FILES)}")

# 第 3 步: 棂查檔案
print("""
================================================================================
棂查檔案
================================================================================
""")

missing = []
for filename, filepath in MODEL_FILES.items():
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ✅ {filename:30s} ({size_mb:6.1f} MB)")
    else:
        print(f"  ❌ {filename:30s} (找不到)")
        missing.append(filename)

if missing:
    print(f"""
❌ 找不到檔案:
  {', '.join(missing)}
請將檔案放在當前目錄，或修改 MODEL_FILES 路徑
    """)
    sys.exit(1)

print("  ✅ 所有檔案已找到")

# 第 4 步: 登錄
print("""
================================================================================
登錄 HuggingFace
================================================================================
""")

print("登錄中...")
print("注: 第一次會要求輸入 token")
print("     地址: https://huggingface.co/settings/tokens\n")

try:
    login()
    print("\n✅ 登錄成功")
except Exception as e:
    print(f"❌ 登錄失敗: {e}")
    sys.exit(1)

# 第 5 步: 上傳
print("""
================================================================================
上傳檔案
================================================================================
""")

api = HfApi()
success_count = 0
failed_files = []

for filename, filepath in MODEL_FILES.items():
    path_in_repo = f"{MODEL_SUBFOLDER}/{filename}"
    print(f"\n上傳: {filename}")
    print(f"  路徑: {path_in_repo}")
    print(f"  程進...", end=" ")
    
    try:
        # 上傳檔案
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=path_in_repo,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Add {filename}"
        )
        print("\u2705 成功")
        success_count += 1
    except Exception as e:
        print(f"\u274c 失敗")
        print(f"    錯誤: {str(e)[:100]}")
        failed_files.append(filename)

# 第 6 步: 總結
print("""
================================================================================
總結
================================================================================
""")

print(f"\n上傳結果: {success_count}/{len(MODEL_FILES)}")

if success_count == len(MODEL_FILES):
    print("✅ 所有檔案已成功上傳!")
    print(f"
數據集: https://huggingface.co/datasets/{HF_REPO}")
    print(f"位置: models_v1/")
    
    print(f"
在 Colab 中的使用方式:")
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
    print(f"⚠️ 只有 {success_count}/{len(MODEL_FILES)} 檔案上傳成功")
    if failed_files:
        print(f"失敗檔案: {', '.join(failed_files)}")
        print("請棂查:")
        print("  1. 網絡連接")
        print("  2. Token 是否有效")
        print("  3. 是否有寶藩權隨")
        print("並重試")

print("\n" + "="*80)
