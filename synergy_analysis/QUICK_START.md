# 🚀 同步性指標 - 5分鐘快速開始

## ⏰ 只需5分鐘！

---

## 第1分鐘: 打開Colab

```
訪問: colab.research.google.com
點擊: 新建Notebook
```

---

## 第2分鐘: 上傳數據

在第一個單元格執行：

```python
from google.colab import files
files.upload()
```

上傳 `BTCUSDT_15m_binance_us.csv`

---

## 第3-4分鐘: 複製代碼

新建第二個單元格，複製完整代碼...

詳見 COLAB_SYNERGY_IMMEDIATE.md

---

## 第5分鐘: 下載結果

新建第三個單元格：

```python
from google.colab import files
files.download('synergy_features_full.csv')
print("✓ 文件已下載")
```

---

## ✅ 完成！

您應該看到：
- 同步性得分分佈表格
- `synergy_features_full.csv` 已下載

---

## 📊 下一步

1. 用 Excel 或 Python 打開 CSV 文件
2. 查看 `synergy_combined` 列
3. 分析同步性與市場走勢的關係

---

## 🆘 有問題？

查看 `TROUBLESHOOTING.md` 文件
