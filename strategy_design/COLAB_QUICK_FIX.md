# Colab 執行修正指南

## 問題解決

如果您遇到以下錯誤：

```
ModuleNotFoundError: No module named 'numpy.char'
```

這是由於 Colab 環境中 NumPy 版本不相容造成的。已在最新版本 (v27) 修正。

---

## 解決方案 1：使用修正版本（推薦）

在 Colab 新建空白 Cell，複製貼上以下代碼並執行：

```python
# ===== V2BOT COLAB 遠端執行（已修正）=====

# 參數配置
SYMBOL = 'BTC'        # 可改為 'ETH', 'BNB' 等
INTERVAL = '1h'       # 可改為 '15m', '4h', '1d'
MODE = 'train'        # 'train' or 'inference'
EPOCHS = 20           # 訓練輪數 (快速測試用 20，精準訓練用 50)

# 遠端執行（自動修復依賴）
import requests
import sys

print("\n下載 Colab 遠端訓練腳本...")
COLAB_SCRIPT = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_remote_execution.py'

try:
    script_content = requests.get(COLAB_SCRIPT, timeout=30).text
    exec(script_content, {
        'SYMBOL': SYMBOL,
        'INTERVAL': INTERVAL,
        'MODE': MODE,
        'EPOCHS': EPOCHS
    })
except Exception as e:
    print(f"執行失敗: {e}")
    print("\n嘗試備用方案...")
    sys.exit(1)
```

**執行時間：** 10-60 分鐘（取決於 GPU 可用性）

---

## 解決方案 2：手動環境清理（如果方案 1 失敗）

如果上面的方法仍然失敗，在 Colab 運行以下命令逐步清理環境：

### Step 1：完全卸載衝突的包

```bash
!pip uninstall -y numpy tensorflow pandas scikit-learn
```

### Step 2：清空 pip 緩存

```bash
!pip cache purge
```

### Step 3：重新安裝相容版本

```bash
!pip install --upgrade --force-reinstall numpy==1.24.3
!pip install --upgrade --force-reinstall tensorflow==2.13.0
!pip install pandas==1.5.3
!pip install scikit-learn==1.3.0
!pip install huggingface-hub==0.16.4
!pip install ta==0.10.2
```

### Step 4：驗證安裝

```python
import numpy as np
import tensorflow as tf
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Pandas: {pd.__version__}")
print("\n✓ 所有依賴安裝成功！")
```

### Step 5：運行訓練腳本

清理後，再次運行上面的「方案 1」代碼

---

## 解決方案 3：在本地電腦運行

如果 Colab 持續出現問題，可在本地運行：

```bash
# 1. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. 安裝依賴
pip install tensorflow==2.13.0
pip install pandas==1.5.3
pip install scikit-learn==1.3.0
pip install huggingface-hub==0.16.4
pip install ta==0.10.2

# 3. 運行訓練
python colab_remote_execution.py
```

---

## 常見問題排除

### 問題："ImportError: cannot import name 'np_utils'"

**解決：** TensorFlow 版本不相容

```bash
!pip install --force-reinstall tensorflow==2.13.0
```

### 問題："CUDA out of memory"

**解決：** 減少 epoch 和 batch size

```python
EPOCHS = 10              # 改為 10
batch_size = 16          # 改為 16（源碼中修改）
```

### 問題："Connection timeout from HuggingFace"

**解決：** 重試或手動下載數據

```python
# 重試 3 次
for attempt in range(3):
    try:
        file_path = hf_hub_download(...)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt == 2:
            raise
```

### 問題："Insufficient data"

**解決：** 檢查 HuggingFace 數據集是否可用

```python
# 手動驗證
from huggingface_hub import list_files_in_repo
files = list_files_in_repo(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    repo_type='dataset'
)
print([f for f in files if 'BTC' in f])
```

---

## 性能優化提示

### 快速測試（10 分鐘）

```python
EPOCHS = 5
# 使用最少的迭代進行快速測試
```

### 標準訓練（30-60 分鐘）

```python
EPOCHS = 50
# 平衡的性能和訓練時間
```

### 精準訓練（2-3 小時）

```python
EPOCHS = 100
# 更深入的模型訓練
```

---

## 預期輸出

成功執行後，您應該看到：

```
================================================================================
V2BOT FORMULA-LSTM STRATEGY - COLAB REMOTE EXECUTION (FIXED)
================================================================================

Parameters:
  Symbol: BTC
  Interval: 1h
  Mode: train
  LSTM Epochs: 20

Step 1: Fixing dependencies (NumPy compatibility)...
✓ Dependencies fixed and installed.

Step 2: Importing libraries...
✓ NumPy version: 1.24.3
✓ Pandas version: 1.5.3
✓ TensorFlow version: 2.13.0
✓ All libraries imported successfully

Step 3: Loading BTC 1h data from Hugging Face...
✓ Loaded 80,000 rows of BTC 1h data

Step 4: Validating and cleaning data...
✓ Data validated: 80,000 valid rows

Step 5: Computing technical indicators...
✓ Indicators computed for 80,000 bars

Step 6: Applying formulas to generate synthetic indicators...
  Progress: 8,000/80,000 (10.0%)
  Progress: 16,000/80,000 (20.0%)
  ...
✓ Formula values computed: shape (80000, 5)

Step 7: Creating training labels...
✓ Label distribution:
  SELL (0): 4,320 (5.4%)
  HOLD (1): 62,400 (78.0%)
  BUY  (2): 13,280 (16.6%)

... (更多步驟) ...

Step 11: Training LSTM model (20 epochs)...
Epoch 1/20
1200/1200 [==============================] 45s 38ms/step
...
Epoch 20/20
1200/1200 [==============================] 45s 38ms/step
✓ Training completed

Step 12: Evaluating model on test set...
✓ Test Results:
  Accuracy:  0.6234
  Precision: 0.6142
  Recall:    0.6234
  F1-Score:  0.6188

✓ Files saved:
  - formula_lstm_model_BTC_1h.h5
  - scaler_config_BTC_1h.pkl
  - discovered_formulas_BTC_1h.json

================================================================================
TRAINING COMPLETE FOR BTC 1h
================================================================================

Next Steps:
  1. Download the three .h5, .pkl, .json files above
  2. Use real_time_predictor.py for live trading
  3. Monitor performance and retrain monthly

================================================================================
Ready for deployment!
================================================================================
```

---

## 文件下載

訓練完後，在 Colab 左側文件管理器中找到三個文件並下載：

```python
from google.colab import files

# 下載訓練好的模型
files.download('formula_lstm_model_BTC_1h.h5')
files.download('scaler_config_BTC_1h.pkl')
files.download('discovered_formulas_BTC_1h.json')
```

---

## 技術支持

如果問題仍未解決，請查看：

1. GitHub Issues: https://github.com/caizongxun/v2bot/issues
2. TensorFlow 相容性矩陣: https://www.tensorflow.org/install/source
3. Colab GPU 可用性: https://status.colab.research.google.com/

---

## 版本歷史

- **v27** (2025-12-30): 修正 NumPy 相容性問題
- **v26** (2025-12-29): 初始版本

---

祝訓練順利！
