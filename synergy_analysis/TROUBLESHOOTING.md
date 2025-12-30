# Colab 同步性指標 - 故障排查指南

## 🆘 快速查詢

| 症狀 | 原因 | 解決 |
|-----|------|------|
| ModuleNotFoundError | 代碼未正確複製 | 見【錯誤1】 |
| FileNotFoundError | CSV未上傳 | 見【錯誤2】 |
| 執行超時 | 計算太慢 | 見【錯誤3】 |
| 輸出全NaN | 數據問題 | 見【錯誤4】 |
| 無法下載 | 權限問題 | 見【錯誤5】 |

## 核心故障排查流程

### 1. ModuleNotFoundError: No module named 'synergy_features_implementation'

原因：不能在Colab中使用import方式，需要直接複製粘貼代碼

解決：
```python
# ❌ 錯誤做法
from synergy_features_implementation import calculate_all_synergy_features

# ✅ 正確做法
# 直接複製粘貼所有函數到Colab單元格
```

### 2. FileNotFoundError

檢查：
```python
import os
print(os.listdir())  # 確認CSV文件存在
```

### 3. 執行很慢（> 5分鐘）

正常情況！nQQE計算涉及迴圈，1500行數據需要3-5分鐘

### 4. 結果全是NaN

檢查數據列名和格式：
```python
print(df.columns)
print(df.dtypes)
print(df['close'].describe())
```

## 驗證成功

執行完後應該看到：
```
✓ 計算完成！

同步性得分分佈:
4    15
2    45
1    80
0   240

✓ 已保存: synergy_features_full.csv
```
