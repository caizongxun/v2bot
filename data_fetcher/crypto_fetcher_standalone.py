"""
独立加密貨幣歷史數據抓取器 (小程序版本)

特點：
- 可直接從 GitHub 遠端執行
- 支援自訂參數 (幣種、時間捙紙、K 線數)
- 不需要克隆整個倉庫
- 完罗的錯誤處理和日誌

使用方式：
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_standalone.py').text, 
     {'CRYPTO_SYMBOLS': ['BTC', 'ETH'], 'INTERVALS': ['1h'], 'TARGET_KLINES': 10000})
"""

import os
import sys
import time
import requests
import json
from datetime import datetime
from pathlib import Path
import logging

try:
    import pandas as pd
except ImportError:
    print("[Installing pandas...]")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd

# =====================================================
# 配置与參數
# =====================================================

# 預設參數
DEFAULT_CONFIG = {
    # 幣種選選 (None = 全部)
    'CRYPTO_SYMBOLS': None,
    
    # 時間時框 (None = 全部)
    'INTERVALS': None,
    
    # 每種時框的 K 線數
    'TARGET_KLINES': 50000,
    
    # 並行線程數
    'MAX_WORKERS': 5,
    
    # 輸出目錄
    'OUTPUT_DIR': './crypto_data_cache'
}

# 超幅候選値
# 例子：
# CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB']  # 只下載這 3 種
# INTERVALS = ['1h']                      # 只下載 1h
# TARGET_KLINES = 10000                   # 只下載 10000 根

CRYPTO_SYMBOLS = CRYPTO_SYMBOLS if 'CRYPTO_SYMBOLS' in dir() else DEFAULT_CONFIG['CRYPTO_SYMBOLS']
INTERVALS = INTERVALS if 'INTERVALS' in dir() else DEFAULT_CONFIG['INTERVALS']
TARGET_KLINES = TARGET_KLINES if 'TARGET_KLINES' in dir() else DEFAULT_CONFIG['TARGET_KLINES']
MAX_WORKERS = MAX_WORKERS if 'MAX_WORKERS' in dir() else DEFAULT_CONFIG['MAX_WORKERS']
OUTPUT_DIR = OUTPUT_DIR if 'OUTPUT_DIR' in dir() else DEFAULT_CONFIG['OUTPUT_DIR']

# 所有支持的幣種
ALL_CRYPTO_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'BNB': 'BNBUSDT', 'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT', 'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'DOT': 'DOTUSDT',
    'LINK': 'LINKUSDT', 'MATIC': 'MATICUSDT', 'LTC': 'LTCUSDT', 'UNI': 'UNIUSDT',
    'BCH': 'BCHUSDT', 'ETC': 'ETCUSDT', 'FIL': 'FILUSDT', 'DOGE': 'DOGEUSDT',
    'ALGO': 'ALGOUSDT', 'ATOM': 'ATOMUSDT', 'NEAR': 'NEARUSDT', 'ARB': 'ARBUSDT',
    'OP': 'OPUSDT', 'AAVE': 'AAVEUSDT', 'SHIB': 'SHIBUSD'
}

ALL_INTERVALS = ['15m', '1h']
BINANCE_BASE_URL = "https://api.binance.us/api/v3/klines"

# =====================================================
# 日誌設置
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# 幜實函數
# =====================================================

def resolve_parameters():
    """解析和驗證參數"""
    
    # 解析幣種
    if CRYPTO_SYMBOLS is None:
        symbols_dict = ALL_CRYPTO_SYMBOLS.copy()
    elif isinstance(CRYPTO_SYMBOLS, list):
        symbols_dict = {k: ALL_CRYPTO_SYMBOLS[k] for k in CRYPTO_SYMBOLS if k in ALL_CRYPTO_SYMBOLS}
        if len(symbols_dict) != len(CRYPTO_SYMBOLS):
            missing = set(CRYPTO_SYMBOLS) - set(symbols_dict.keys())
            logger.warning(f"Unknown symbols: {missing}, skipped")
    else:
        symbols_dict = ALL_CRYPTO_SYMBOLS.copy()
    
    # 解析時間時框
    if INTERVALS is None:
        intervals = ALL_INTERVALS.copy()
    elif isinstance(INTERVALS, list):
        intervals = [i for i in INTERVALS if i in ALL_INTERVALS]
        if len(intervals) != len(INTERVALS):
            invalid = set(INTERVALS) - set(intervals)
            logger.warning(f"Invalid intervals: {invalid}, skipped")
    else:
        intervals = ALL_INTERVALS.copy()
    
    # 驗證 K 線數
    klines = max(1000, min(TARGET_KLINES, 500000))  # 限制在 1000-500000 範圍
    
    return symbols_dict, intervals, klines

def get_binance_klines(symbol, interval, limit=1000):
    """從 Binance API 獲取 K 線"""
    all_klines = []
    end_time = int(time.time() * 1000)
    num_requests = (TARGET_KLINES + limit - 1) // limit
    
    logger.info(f"Fetching {symbol} {interval} - {num_requests} requests needed")
    
    for i in range(num_requests):
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            response = requests.get(BINANCE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            if not klines:
                logger.warning(f"{symbol} {interval} - No more data")
                break
            
            all_klines.extend(klines)
            end_time = klines[0][0] - 1
            
            current = min(len(all_klines), TARGET_KLINES)
            logger.info(f"{symbol} {interval} - Progress: {current}/{TARGET_KLINES}")
            
            time.sleep(0.2)  # 預防 API 速率限制
            
            if len(all_klines) >= TARGET_KLINES:
                all_klines = all_klines[:TARGET_KLINES]
                break
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {symbol} {interval}: {str(e)}")
            break
    
    return all_klines

def klines_to_dataframe(klines, symbol):
    """轉換為 DataFrame"""
    if not klines:
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df['symbol'] = symbol
    df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def fetch_single_crypto(symbol_code, symbol_binance, intervals):
    """下載单個幣種的所有時框数据"""
    results = {}
    
    for interval in intervals:
        try:
            logger.info(f"Fetching {symbol_code} {interval}...")
            klines = get_binance_klines(symbol_binance, interval)
            
            if not klines:
                logger.warning(f"No data for {symbol_code} {interval}")
                results[interval] = False
                continue
            
            df = klines_to_dataframe(klines, symbol_binance)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol_code} {interval}")
                results[interval] = False
                continue
            
            csv_filename = f"{symbol_code}_{interval}.csv"
            csv_path = Path(OUTPUT_DIR) / csv_filename
            df.to_csv(csv_path, index=False)
            
            logger.info(f"✓ Saved {csv_filename} ({len(df)} rows)")
            results[interval] = True
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing {symbol_code} {interval}: {str(e)}")
            results[interval] = False
    
    return {symbol_code: results}

def main():
    """主執行函數"""
    
    # 解析參數
    symbols_dict, intervals, klines = resolve_parameters()
    
    # 創建輸出目錄
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("CRYPTO DATA FETCHER - STANDALONE VERSION")
    logger.info("="*60)
    logger.info(f"Symbols: {list(symbols_dict.keys())} ({len(symbols_dict)} total)")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"K-lines per symbol/interval: {klines}")
    logger.info(f"Total files to fetch: {len(symbols_dict) * len(intervals)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("="*60 + "\n")
    
    # 順序執行 (安撃 Colab CPU 限制)
    all_results = {}
    start_time = time.time()
    
    for idx, (symbol_code, symbol_binance) in enumerate(symbols_dict.items(), 1):
        logger.info(f"\n[{idx}/{len(symbols_dict)}] Processing {symbol_code}...")
        result = fetch_single_crypto(symbol_code, symbol_binance, intervals)
        all_results.update(result)
    
    elapsed = time.time() - start_time
    
    # 產生報告
    logger.info("\n" + "="*60)
    logger.info("FETCH SUMMARY")
    logger.info("="*60)
    
    success_count = 0
    for symbol, status in all_results.items():
        if isinstance(status, dict) and "error" not in status:
            if all(status.values()):
                logger.info(f"✓ {symbol}: All intervals successful")
                success_count += 1
            else:
                partial = [k for k, v in status.items() if v]
                logger.warning(f"△ {symbol}: Partial ({', '.join(partial)})")
    
    logger.info("="*60)
    logger.info(f"Success: {success_count}/{len(symbols_dict)}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Data location: {output_path}")
    logger.info("="*60 + "\n")
    
    # 統計信息
    csv_files = list(output_path.glob('*.csv'))
    if csv_files:
        total_rows = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            total_rows += len(df)
        
        logger.info(f"\nData Statistics:")
        logger.info(f"  Total files: {len(csv_files)}")
        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Average file size: {sum(f.stat().st_size for f in csv_files) / len(csv_files) / (1024**2):.2f} MB")
    
    # 保存結果到 JSON
    report_path = output_path / "fetch_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'symbols': list(symbols_dict.keys()),
                'intervals': intervals,
                'target_klines': klines
            },
            'results': all_results,
            'statistics': {
                'total_time_minutes': elapsed / 60,
                'success_count': success_count,
                'total_symbols': len(symbols_dict),
                'files_created': len(csv_files)
            }
        }, f, indent=2)
    
    logger.info(f"\n✓ Report saved to: {report_path}")
    logger.info("\n✅ 抓取完成！")
    
    return output_path

if __name__ == "__main__":
    main()
