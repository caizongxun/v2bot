# SSL Hybrid Model - HuggingFace 上傳與下載指南

## 🚀 快速開始 (3 步)

### Step 1: 創建 HF 數據集 (2 分鐘)

訪問: https://huggingface.co/new-dataset

填寫:
```
Dataset name: ssl-hybrid-v3-model
License: MIT
Visibility: Public
```

記住你的 HF username: `your_username`

### Step 2: 上傳模型檔案 (5 分鐘)

```python
from huggingface_hub import HfApi, login
import os

# 登錄 HF (首次需要)
login()

HF_USERNAME = "your_username"  # ← 替換為你的 HF 帳號

api = HfApi()

# 上傳 3 個檔案
files = {
    'ssl_filter_v3.keras': '/path/to/ssl_filter_v3.keras',
    'ssl_scaler_v3.json': '/path/to/ssl_scaler_v3.json',
    'ssl_metadata_v3.json': '/path/to/ssl_metadata_v3.json'
}

for filename, filepath in files.items():
    if os.path.exists(filepath):
        print(f"上傳 {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=f"{HF_USERNAME}/ssl-hybrid-v3-model",
            repo_type="dataset"
        )
        print(f"  ✅ {filename} 成功")

print(f"\n✅ 完成! 數據集: https://huggingface.co/datasets/{HF_USERNAME}/ssl-hybrid-v3-model")
```

### Step 3: 驗證上傳 (1 分鐘)

訪問: https://huggingface.co/datasets/your_username/ssl-hybrid-v3-model

確認 3 個檔案都在那裡 ✅

---

## 📥 下載檔案 (在 Colab 使用)

### 自動下載函數

```python
from huggingface_hub import hf_hub_download
import os
import shutil

def download_ssl_model(hf_username, output_dir="."):
    """
    從 HuggingFace 下載 SSL Hybrid v3 模型
    """
    repo_id = f"{hf_username}/ssl-hybrid-v3-model"
    files = ['ssl_filter_v3.keras', 'ssl_scaler_v3.json', 'ssl_metadata_v3.json']
    
    print(f"\n從 {repo_id} 下載模型...\n")
    
    for filename in files:
        print(f"下載 {filename}...", end=" ")
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset"
            )
            shutil.copy(file_path, os.path.join(output_dir, filename))
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
    
    print(f"\n✅ 檔案已保存到: {output_dir}")

# 使用方式
download_ssl_model("your_username")  # ← 替換為你的 HF 帳號
```

---

## 🔄 完整的 Colab 工作流

### 一整個 Colab Cell

```python
# 1. 安裝套件
from huggingface_hub import hf_hub_download
import os
import shutil
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

# 2. 下載模型
HF_USERNAME = "your_username"  # ← 替換為你的 HF 帳號
repo_id = f"{HF_USERNAME}/ssl-hybrid-v3-model"

print("下載模型檔案...\n")

files = {}
for filename in ['ssl_filter_v3.keras', 'ssl_scaler_v3.json', 'ssl_metadata_v3.json']:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        shutil.copy(file_path, filename)
        files[filename] = True
        print(f"✅ {filename}")
    except:
        files[filename] = False
        print(f"❌ {filename}")

# 3. 載入模型
if files['ssl_filter_v3.keras']:
    print("\n載入模型...")
    model = keras.models.load_model('ssl_filter_v3.keras')
    
    with open('ssl_scaler_v3.json') as f:
        scaler_data = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data['mean'])
    scaler.scale_ = np.array(scaler_data['scale'])
    
    with open('ssl_metadata_v3.json') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features']
    print(f"✅ 模型已載入 ({len(feature_names)} 個特徵)")
    
    # 4. 定義預測函數
    def predict_signal(features_dict):
        X = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict(X_scaled, verbose=0)[0][0]
        confidence = max(prob, 1 - prob)
        
        return {
            'signal': 'TRUE' if prob > 0.5 else 'FALSE',
            'probability': float(prob),
            'confidence': float(confidence),
            'recommendation': 'ENTER' if prob > 0.5 and confidence > 0.7 else 'SKIP'
        }
    
    # 5. 測試
    test_features = {
        'atr_ratio': 0.02, 'avg_return_strength': 0.8, 'bb_distance_mid': 0.3,
        'bb_position': 0.2, 'macd_bullish': 1.0, 'macd_hist': 0.025,
        'macd_signal_dist': 0.015, 'momentum_10': 0.008, 'momentum_5': 0.006,
        'multi_tf_confirmations': 1.0, 'price_range_position': 0.15,
        'rsi14': 0.28, 'rsi14_from_neutral': 0.44, 'rsi_trend': 0.08,
        'signal_type': 1.0, 'volatility': 0.18, 'volume_ratio': 2.0
    }
    
    result = predict_signal(test_features)
    print(f"\n預測結果:")
    print(f"  信號: {result['signal']}")
    print(f"  概率: {result['probability']:.2%}")
    print(f"  建議: {result['recommendation']}")
else:
    print("❌ 模型下載失敗")
```

---

## 🛠️ 配置選項

### 隱私設置

```python
# 公開 (所有人都能下載)
private=False

# 私有 (只有你能下載)
private=True
```

### 檔案大小限制

- HuggingFace 免費方案: 無限制
- 建議單檔 < 1GB
- 我們的模型 < 20MB (✅ 完全沒問題)

---

## 📋 檔案清單

你需要上傳到 HF 的 3 個檔案:

| 檔案名 | 大小 | 說明 |
|--------|------|------|
| ssl_filter_v3.keras | ~10 MB | 訓練好的神經網絡 |
| ssl_scaler_v3.json | ~3 KB | 特徵標準化器 |
| ssl_metadata_v3.json | ~2 KB | 模型元數據和特徵名 |

**總計**: ~10 MB

---

## ✅ 常見問題

**Q: 我的 HF username 在哪裡?**
A: 登錄後，在右上角頭像點擊，選擇 "Settings" → "Profile"

**Q: Token 怎麼生成?**
A: https://huggingface.co/settings/tokens → "New token"

**Q: 上傳失敗了怎麼辦?**
A: 檢查:
   1. Token 是否有效
   2. 網絡連接是否正常
   3. 檔案路徑是否正確

**Q: 怎麼更新檔案?**
A: 重新上傳同名檔案自動覆蓋

**Q: 可以共享給別人嗎?**
A: 可以! 設置為公開 (private=False) 後分享 URL 即可

---

## 🎯 下一步

1. ✅ 創建 HF 帳號
2. ✅ 創建數據集
3. ✅ 上傳 3 個檔案
4. ✅ 在 Colab 中使用本指南下載
5. ✅ 開始進行預測!

---

**推薦流程**:

```
1. 上傳到 HF (一次性)
   ↓
2. 在 Colab 下載 (每次使用)
   ↓
3. 使用模型進行預測 (即時)
```

這樣你的模型既安全 (備份在 HF) 又方便 (任何地方都能用)
