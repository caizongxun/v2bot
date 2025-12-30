"""
Colab 優化版加密貨幣歷史數據抓取器

特點:
- 完整的 Colab 輸出支持
- 實時進度顯示
- 參數自訂支持
- 無需克隆倉庫

使用方式:
!pip install -q requests pandas
import requests
exec(requests.get('https://raw.githubusercontent.com/caizongxun/v2bot/main/data_fetcher/crypto_fetcher_colab.py').text,
     {'TARGET_KLINES': 50000})
"""

import os
import sys
import time
import requests
import json
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[Step 1/3] Installing pandas...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd
    print("pandas installed successfully\n")

# =====================================================
# 配置和參數
# =====================================================

DEFAULT_CONFIG = {
    'CRYPTO_SYMBOLS': None,
    'INTERVALS': None,
    'TARGET_KLINES': 50000,
    'MAX_WORKERS': 5,
    'OUTPUT_DIR': './crypto_data_cache'
}

# 從全局作用域讀取參數 (exec() 注入)
CRYPTO_SYMBOLS = globals().get('CRYPTO_SYMBOLS', DEFAULT_CONFIG['CRYPTO_SYMBOLS'])
INTERVALS = globals().get('INTERVALS', DEFAULT_CONFIG['INTERVALS'])
TARGET_KLINES = globals().get('TARGET_KLINES', DEFAULT_CONFIG['TARGET_KLINES'])
MAX_WORKERS = globals().get('MAX_WORKERS', DEFAULT_CONFIG['MAX_WORKERS'])
OUTPUT_DIR = globals().get('OUTPUT_DIR', DEFAULT_CONFIG['OUTPUT_DIR'])

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
# 核心函數
# =====================================================

def log(message, level="INFO"):
    """統一日誌函數，確保 Colab 中能顯示"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"[{timestamp}] {message}")
    elif level == "WARNING":
        print(f"[{timestamp}] WARNING: {message}")
    elif level == "ERROR":
        print(f"[{timestamp}] ERROR: {message}")

def resolve_parameters():
    """解析和驗證參數"""
    
    # 解析幣種
    if CRYPTO_SYMBOLS is None:
        symbols_dict = ALL_CRYPTO_SYMBOLS.copy()
    elif isinstance(CRYPTO_SYMBOLS, list):
        symbols_dict = {k: ALL_CRYPTO_SYMBOLS[k] for k in CRYPTO_SYMBOLS if k in ALL_CRYPTO_SYMBOLS}
        if len(symbols_dict) != len(CRYPTO_SYMBOLS):
            missing = set(CRYPTO_SYMBOLS) - set(symbols_dict.keys())
            log(f"Unknown symbols: {missing}, skipped", "WARNING")
    else:
        symbols_dict = ALL_CRYPTO_SYMBOLS.copy()
    
    # 解析時框
    if INTERVALS is None:
        intervals = ALL_INTERVALS.copy()
    elif isinstance(INTERVALS, list):
        intervals = [i for i in INTERVALS if i in ALL_INTERVALS]
        if len(intervals) != len(INTERVALS):
            invalid = set(INTERVALS) - set(intervals)
            log(f"Invalid intervals: {invalid}, skipped", "WARNING")
    else:
        intervals = ALL_INTERVALS.copy()
    
    # 驗證 K 線數
    klines = max(1000, min(TARGET_KLINES, 500000))
    
    return symbols_dict, intervals, klines

def get_binance_klines(symbol, interval, limit=1000):
    """從 Binance API 獲取 K 線"""
    all_klines = []
    end_time = int(time.time() * 1000)
    num_requests = (TARGET_KLINES + limit - 1) // limit
    
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
                break
            
            all_klines.extend(klines)
            end_time = klines[0][0] - 1
            
            current = min(len(all_klines), TARGET_KLINES)
            print(f"  {symbol} {interval}: {current}/{TARGET_KLINES} klines", end='\r')
            
            time.sleep(0.2)
            
            if len(all_klines) >= TARGET_KLINES:
                all_klines = all_klines[:TARGET_KLINES]
                break
                
        except requests.exceptions.RequestException as e:
            log(f"Error fetching {symbol} {interval}: {str(e)}", "ERROR")
            break
    
    print(f"  {symbol} {interval}: Complete ({len(all_klines)} klines)            ")
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
    """下載單個幣種的所有時框數據"""
    results = {}
    
    for interval in intervals:
        try:
            log(f"Fetching {symbol_code} {interval}...")
            klines = get_binance_klines(symbol_binance, interval)
            
            if not klines:
                log(f"No data for {symbol_code} {interval}", "WARNING")
                results[interval] = False
                continue
            
            df = klines_to_dataframe(klines, symbol_binance)
            
            if df.empty:
                log(f"Empty DataFrame for {symbol_code} {interval}", "WARNING")
                results[interval] = False
                continue
            
            csv_filename = f"{symbol_code}_{interval}.csv"
            csv_path = Path(OUTPUT_DIR) / csv_filename
            df.to_csv(csv_path, index=False)
            
            log(f"[SAVED] {csv_filename} ({len(df)} rows)")
            results[interval] = True
            time.sleep(0.5)
            
        except Exception as e:
            log(f"Error processing {symbol_code} {interval}: {str(e)}", "ERROR")
            results[interval] = False
    
    return {symbol_code: results}

def main():
    """主執行函數"""
    
    # 解析參數
    symbols_dict, intervals, klines = resolve_parameters()
    
    # 創建輸出目錄
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print("\n" + "="*70)
    print("CRYPTO DATA FETCHER - COLAB VERSION")
    print("="*70)
    print(f"Symbols: {list(symbols_dict.keys())}")
    print(f"Total symbols: {len(symbols_dict)}")
    print(f"Intervals: {intervals}")
    print(f"K-lines per symbol/interval: {klines}")
    print(f"Total files to fetch: {len(symbols_dict) * len(intervals)}")
    print(f"Output directory: {output_path}")
    print("="*70 + "\n")
    
    # 順序執行 (安全起見)
    all_results = {}
    start_time = time.time()
    
    for idx, (symbol_code, symbol_binance) in enumerate(symbols_dict.items(), 1):
        print(f"\n[{idx}/{len(symbols_dict)}] Processing {symbol_code}...")
        result = fetch_single_crypto(symbol_code, symbol_binance, intervals)
        all_results.update(result)
    
    elapsed = time.time() - start_time
    
    # 生成報告
    print("\n" + "="*70)
    print("FETCH SUMMARY")
    print("="*70)
    
    success_count = 0
    for symbol, status in all_results.items():
        if isinstance(status, dict) and "error" not in status:
            if all(status.values()):
                print(f"OK {symbol}: All intervals successful")
                success_count += 1
            else:
                partial = [k for k, v in status.items() if v]
                print(f"PARTIAL {symbol}: ({', '.join(partial)})")
    
    print("="*70)
    print(f"Success: {success_count}/{len(symbols_dict)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Data location: {output_path}")
    print("="*70)
    
    # 數據統計
    csv_files = list(output_path.glob('*.csv'))
    if csv_files:
        total_rows = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            total_rows += len(df)
        
        print(f"\nData Statistics:")
        print(f"  Total files: {len(csv_files)}")
        print(f"  Total rows: {total_rows:,}")
        avg_size = sum(f.stat().st_size for f in csv_files) / len(csv_files) / (1024**2)
        print(f"  Average file size: {avg_size:.2f} MB")
    
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
    
    print(f"\nReport saved to: {report_path}")
    print("\n[COMPLETE] Data fetching completed successfully!")
    
    return output_path

# =====================================================
# 執行
# =====================================================

if __name__ == "__main__":
    main()
else:
    # 當通過 exec() 執行時
    main()
