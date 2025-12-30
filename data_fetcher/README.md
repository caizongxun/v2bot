# Cryptocurrency Historical Data Fetcher

## 概述 (Overview)

這是一個完整的加密貨幣歷史 K 線數據抓取系統，支援多幣種、多時間框架的自動化下載和 Hugging Face 上傳。

### 核心特性

- **多幣種支持**: 23 種主流加密貨幣 (BTC, ETH, BNB, SOL 等)
- **多時間框架**: 15 分鐘和 1 小時級別
- **高效數據量**: 每種幣種、每個時框 50,000 根 K 線
- **並行下載**: 使用 5 個並行線程加速抓取
- **自動上傳**: 一鍵上傳到 Hugging Face Datasets
- **Colab 原生支持**: 無需本地配置，直接在 Colab 運行

---

## 支援的幣種 (Supported Cryptocurrencies)

### Top Tier (主流穩定幣)
| 代碼 | 名稱 | 說明 |
|------|------|------|
| BTC | Bitcoin | 市值第一 |
| ETH | Ethereum | 市值第二，智能合約平台 |
| BNB | Binance Coin | Binance 生態代幣 |
| SOL | Solana | 高速公鏈 |
| XRP | Ripple | 跨境支付 |
| ADA | Cardano | 學術型公鏈 |
| AVAX | Avalanche | DeFi 公鏈 |
| DOT | Polkadot | 跨鏈協議 |
| LINK | Chainlink | 預言機龍頭 |
| MATIC | Polygon | 以太坊二層方案 |

### Tier 2 (熱門幣種)
LTC, UNI, BCH, ETC, FIL, DOGE, ALGO, ATOM, NEAR, ARB, OP, AAVE, SHIB

---

## 快速開始 (Quick Start)

### 方案 A: 在 Google Colab 上運行 (推薦)

1. **打開 Colab**
   ```bash
   https://colab.research.google.com/
   ```

2. **新建筆記本**
   - 點選 "新增" → "Python 筆記本"

3. **複製並執行以下代碼**
   ```python
   # 一行代碼載入並執行 Colab notebook
   !git clone https://github.com/caizongxun/v2bot.git /content/v2bot
   %cd /content/v2bot
   !jupyter nbconvert --to script data_fetcher/colab_notebook.ipynb --stdout | python
   ```

4. **按照提示操作**
   - 等待數據下載完成 (約 20-40 分鐘)
   - 輸入 Hugging Face token
   - 指定儲存庫名稱
   - 開始上傳

### 方案 B: 本地運行

1. **克隆存儲庫**
   ```bash
   git clone https://github.com/caizongxun/v2bot.git
   cd v2bot/data_fetcher
   ```

2. **安裝依賴**
   ```bash
   pip install requests pandas yfinance huggingface-hub
   ```

3. **執行數據抓取**
   ```python
   from crypto_historical_data_fetcher import CryptoDataFetcher
   
   fetcher = CryptoDataFetcher(output_dir='./crypto_data')
   results = fetcher.fetch_all_cryptos_parallel(max_workers=5)
   fetcher.generate_summary_report(results)
   ```

4. **上傳到 Hugging Face**
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi(token='your_hf_token')
   api.upload_folder(
       folder_path='./crypto_data',
       repo_id='your_username/repo_name',
       repo_type='dataset'
   )
   ```

---

## 詳細配置 (Configuration)

### 調整參數

在 `crypto_historical_data_fetcher.py` 中的 `FetcherConfig` 類：

```python
class FetcherConfig:
    TARGET_KLINES = 50000      # 每個時框的 K 線數 (可改為 10000, 20000)
    BATCH_SIZE = 1000          # 單次 API 請求返回數 (固定 1000)
    MAX_WORKERS = 5            # 並行線程數 (1-10)
    OUTPUT_DIR = './crypto_data_cache'
    RETRY_ATTEMPTS = 3         # 失敗重試次數
```

### 性能調優

| 參數 | 建議值 | 優點 | 缺點 |
|------|--------|------|------|
| TARGET_KLINES = 10000 | 快速模式 | 5 分鐘完成 | 數據量少 |
| TARGET_KLINES = 50000 | 平衡模式 | 30 分鐘完成 | **推薦** |
| TARGET_KLINES = 100000 | 完整模式 | 1 小時完成 | API 限制風險 |
| MAX_WORKERS = 10 | 激進模式 | 更快並行 | Colab 可能超時 |
| MAX_WORKERS = 3 | 保守模式 | 穩定性好 | 速度較慢 |

---

## 數據格式 (Data Format)

### CSV 文件結構

每個 CSV 文件包含以下列：

```
timestamp,symbol,open,high,low,close,volume
2025-01-01 00:00:00,BTCUSDT,42500.5,42800.0,42400.0,42650.0,123.45
2025-01-01 01:00:00,BTCUSDT,42650.0,43000.0,42600.0,42950.0,115.23
```

### 文件命名約定

- `BTC_15m.csv` - Bitcoin 15 分鐘級別
- `ETH_1h.csv` - Ethereum 1 小時級別
- `fetch_report.json` - 抓取執行報告

---

## Hugging Face 上傳指南 (HF Upload Guide)

### 第 1 步: 獲取 Token

1. 訪問 [Hugging Face 設定](https://huggingface.co/settings/tokens)
2. 點擊 "新建 Token"
3. 給予 "Write" 權限
4. 複製 token

### 第 2 步: 自動上傳

在 Colab 中執行時會自動提示輸入 token。

### 第 3 步: 手動上傳 (如自動失敗)

```bash
# 安裝 Git LFS (大文件支持)
git lfs install

# 初始化本地倉庫
git init
git add .
git commit -m "Initial crypto data commit"

# 推送到 Hugging Face
git remote add origin https://huggingface.co/datasets/your_username/repo_name
git push -u origin main
```

### 驗證上傳

1. 訪問 `https://huggingface.co/datasets/your_username/repo_name`
2. 確認所有 CSV 文件已上傳
3. 獲取數據集加載代碼：

```python
from datasets import load_dataset
ds = load_dataset('your_username/repo_name')
```

---

## 數據統計 (Data Statistics)

### 典型輸出

```
============================================================
DATA FETCH SUMMARY REPORT
============================================================
✓ BTC: Both 15m and 1h fetched successfully
✓ ETH: Both 15m and 1h fetched successfully
✓ BNB: Both 15m and 1h fetched successfully
...
============================================================
Success Rate: 23/23 (100.0%)
Data cached in: /content/crypto_data_cache
============================================================

Data Statistics:
  Total Files: 46 (23 symbols × 2 intervals)
  Total Rows: 2,300,000+
  15m K-lines: 23 files
  1h K-lines: 23 files
```

---

## 常見問題 (FAQ)

### Q1: 抓取時間太長，如何加速？

**A**: 減少 TARGET_KLINES 或減少幣種數量
```python
FetcherConfig.TARGET_KLINES = 10000  # 從 50000 改為 10000
```

### Q2: 出現 "API rate limit exceeded" 錯誤

**A**: Binance API 有速率限制，請：
- 減少 MAX_WORKERS (改為 2-3)
- 增加延遲時間 (在代碼中加 `time.sleep(1)`)
- 使用 binance.us 而不是 binance.com

### Q3: Colab 連接超時

**A**: 長時間運行時常發生，解決方案：
- 定期點擊屏幕保持連接
- 或分批執行 (5 個幣種為一組)
- 或使用 GPU/TPU (Settings → Runtime type)

### Q4: Hugging Face 上傳失敗

**A**: 檢查以下幾點：
- Token 是否有 "Write" 權限
- Token 是否過期
- 網絡連接是否穩定
- 單個文件大小是否超過 50GB

### Q5: 如何只下載特定幣種？

**A**: 修改 CRYPTO_SYMBOLS 字典
```python
CryptoDataFetcher.CRYPTO_SYMBOLS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    # 只下載 BTC 和 ETH
}
```

---

## 技術細節 (Technical Details)

### API 端點

- **REST API**: `https://api.binance.us/api/v3/klines`
- **速率限制**: 1200 請求/分鐘
- **返回限制**: 最多 1000 根 K 線/請求

### 數據源

| 數據源 | 優勢 | 劣勢 | 選用 |
|--------|------|------|------|
| Binance | 數據全、實時、免費 | 可能洗單 | ✓ |
| CoinGecko | 聚合數據、無偏見 | 延遲、API 限制 | 備用 |
| Kraken | 優質數據 | 需付費 | × |

### 並行策略

使用 `concurrent.futures.ThreadPoolExecutor`：
- 5 個線程同時下載 5 個幣種
- 每個線程順序下載 15m 和 1h 數據
- 避免 API 限制同時間發送大量請求

---

## 下一步 (Next Steps)

1. **特徵工程**
   - 計算技術指標 (RSI, MACD, Bollinger Bands)
   - 創建滯後特徵
   - 標籤構造 (看漲/看跌)

2. **模型訓練**
   - 使用 LSTM/CNN 進行時間序列預測
   - 集成多個模型
   - 超參數調優

3. **回測驗證**
   - 實現回測框架
   - 計算績效指標
   - 風控機制設計

---

## 貢獻指南 (Contributing)

發現 bug 或有改進建議？

1. Fork 本倉庫
2. 建立功能分支 (`git checkout -b feature/improvement`)
3. 提交更改 (`git commit -am 'Add improvement'`)
4. 推送分支 (`git push origin feature/improvement`)
5. 開啟 Pull Request

---

## 許可證 (License)

MIT License - 自由使用、修改、分發

---

## 聯絡方式 (Contact)

- GitHub: [@caizongxun](https://github.com/caizongxun)
- Issues: [Report Bug](https://github.com/caizongxun/v2bot/issues)

---

## 最後提醒 (Disclaimer)

⚠️ **重要提示**

1. 此工具僅用於研究和教育目的
2. 過去表現不代表未來結果
3. 在使用任何預測進行實際交易前，務必進行充分的回測和風險評估
4. 對交易損失概不負責
5. 請遵守當地法律法規

---

**最後更新**: 2025-12-30  
**版本**: 1.0.0
