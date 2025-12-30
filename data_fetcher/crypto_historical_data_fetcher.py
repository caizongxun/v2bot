"""
Cryptocurrency Historical Data Fetcher
支持多幣種、多時間框架的 K 線數據抓取
支持 Binance REST API 和 yfinance 雙數據源

使用方式：
1. 在 Colab 中運行此腳本
2. 自動並行抓取 20+ 幣種的 15m 和 1h K 線
3. 將數據緩存到本地目錄
4. 批量上傳到 Hugging Face Datasets
"""

import os
import json
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import concurrent.futures
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """加密貨幣歷史數據抓取類"""
    
    # 主流幣種列表 (20+ 種)
    CRYPTO_SYMBOLS = {
        # Top Tier (主流穩定幣)
        'BTC': 'BTCUSDT',      # Bitcoin
        'ETH': 'ETHUSDT',      # Ethereum
        'BNB': 'BNBUSDT',      # Binance Coin
        'SOL': 'SOLUSDT',      # Solana
        'XRP': 'XRPUSDT',      # Ripple
        'ADA': 'ADAUSDT',      # Cardano
        'AVAX': 'AVAXUSDT',    # Avalanche
        'DOT': 'DOTUSDT',      # Polkadot
        'LINK': 'LINKUSDT',    # Chainlink
        'MATIC': 'MATICUSDT',  # Polygon
        
        # Tier 2 (熱門幣種)
        'LTC': 'LTCUSDT',      # Litecoin
        'UNI': 'UNIUSDT',      # Uniswap
        'BCH': 'BCHUSDT',      # Bitcoin Cash
        'ETC': 'ETCUSDT',      # Ethereum Classic
        'FIL': 'FILUSDT',      # Filecoin
        'DOGE': 'DOGEUSDT',    # Dogecoin
        'ALGO': 'ALGOUSDT',    # Algorand
        'ATOM': 'ATOMUSDT',    # Cosmos
        'NEAR': 'NEARUSDT',    # NEAR Protocol
        'ARB': 'ARBUSDT',      # Arbitrum
        'OP': 'OPUSDT',        # Optimism
        'AAVE': 'AAVEUSDT',    # Aave
        'SHIB': 'SHIBUSD',     # Shiba Inu (小數位調整)
    }
    
    # Binance API 配置
    BINANCE_BASE_URL = "https://api.binance.us/api/v3/klines"  # 使用 binance.us
    
    # 時間框架
    INTERVALS = ['15m', '1h']
    
    # K 線數量設置
    KLINES_LIMIT = 1000  # Binance API 單次最大返回數
    TARGET_KLINES = 50000  # 目標 K 線數量
    
    def __init__(self, output_dir: str = './crypto_data_cache'):
        """
        初始化數據抓取器
        
        Args:
            output_dir: 數據緩存目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data cache directory: {self.output_dir}")
    
    @staticmethod
    def get_binance_klines(symbol: str, interval: str, limit: int = 1000) -> List[List]:
        """
        從 Binance API 獲取 K 線數據 (支持最多 50 個請求，獲取 50000 根 K 棒)
        
        Args:
            symbol: 交易對 (例如 BTCUSDT)
            interval: 時間框架 (15m, 1h)
            limit: 單次請求返回數量 (最大 1000)
        
        Returns:
            K 線列表
        """
        all_klines = []
        end_time = int(time.time() * 1000)  # 當前時間（毫秒）
        
        # 計算需要請求的次數
        num_requests = (FetcherConfig.TARGET_KLINES + limit - 1) // limit
        
        logger.info(f"Fetching {symbol} {interval} - Total requests: {num_requests}")
        
        for i in range(num_requests):
            try:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit,
                    'endTime': end_time
                }
                
                response = requests.get(FetcherConfig.BINANCE_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                
                klines = response.json()
                
                if not klines:
                    logger.warning(f"{symbol} {interval} - No more data available")
                    break
                
                all_klines.extend(klines)
                
                # 更新 end_time 為最後一根 K 線的時間戳
                end_time = klines[0][0] - 1
                
                logger.info(f"{symbol} {interval} - Progress: {len(all_klines)}/{FetcherConfig.TARGET_KLINES} klines")
                
                # 延遲以避免 API 速率限制
                time.sleep(0.2)
                
                # 達到目標 K 線數量，停止請求
                if len(all_klines) >= FetcherConfig.TARGET_KLINES:
                    all_klines = all_klines[:FetcherConfig.TARGET_KLINES]
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {symbol} {interval}: {str(e)}")
                break
        
        return all_klines
    
    @staticmethod
    def klines_to_dataframe(klines: List[List], symbol: str) -> pd.DataFrame:
        """
        將 Binance K 線列表轉換為 DataFrame
        
        Args:
            klines: K 線列表
            symbol: 交易對
        
        Returns:
            DataFrame
        """
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 轉換時間戳為日期
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # 轉換數值類型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # 添加交易對列
        df['symbol'] = symbol
        
        # 選擇重要列
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        # 按時間排序 (從舊到新)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_single_crypto(self, symbol_code: str, symbol_binance: str) -> Dict[str, bool]:
        """
        為單個幣種抓取所有時間框架的數據
        
        Args:
            symbol_code: 幣種代碼 (例如 BTC)
            symbol_binance: Binance 交易對 (例如 BTCUSDT)
        
        Returns:
            成功/失敗狀態字典
        """
        results = {}
        
        for interval in self.INTERVALS:
            try:
                logger.info(f"Fetching {symbol_code} {interval}...")
                
                # 獲取 K 線數據
                klines = self.get_binance_klines(symbol_binance, interval)
                
                if not klines:
                    logger.warning(f"No data for {symbol_code} {interval}")
                    results[interval] = False
                    continue
                
                # 轉換為 DataFrame
                df = self.klines_to_dataframe(klines, symbol_binance)
                
                if df.empty:
                    logger.warning(f"Empty DataFrame for {symbol_code} {interval}")
                    results[interval] = False
                    continue
                
                # 保存為 CSV
                csv_filename = f"{symbol_code}_{interval}.csv"
                csv_path = self.output_dir / csv_filename
                df.to_csv(csv_path, index=False)
                
                logger.info(f"✓ Saved {csv_filename} ({len(df)} rows)")
                results[interval] = True
                
                # 短暫延遲，避免 API 限制
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {symbol_code} {interval}: {str(e)}")
                results[interval] = False
        
        return {symbol_code: results}
    
    def fetch_all_cryptos_parallel(self, max_workers: int = 5) -> Dict:
        """
        並行抓取所有幣種的數據
        
        Args:
            max_workers: 並行線程數
        
        Returns:
            所有抓取結果
        """
        logger.info(f"Starting parallel data fetch with {max_workers} workers...")
        
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.fetch_single_crypto, 
                    symbol_code, 
                    symbol_binance
                ): symbol_code 
                for symbol_code, symbol_binance in self.CRYPTO_SYMBOLS.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                symbol_code = futures[future]
                try:
                    result = future.result()
                    all_results.update(result)
                except Exception as e:
                    logger.error(f"Exception for {symbol_code}: {str(e)}")
                    all_results[symbol_code] = {"error": str(e)}
        
        return all_results
    
    def generate_summary_report(self, results: Dict) -> None:
        """
        生成數據抓取摘要報告
        
        Args:
            results: 抓取結果字典
        """
        logger.info("\n" + "="*60)
        logger.info("DATA FETCH SUMMARY REPORT")
        logger.info("="*60)
        
        success_count = 0
        total_count = 0
        
        for symbol, status in results.items():
            if isinstance(status, dict) and "error" not in status:
                total_count += 1
                if all(status.values()):
                    success_count += 1
                    logger.info(f"✓ {symbol}: Both 15m and 1h fetched successfully")
                else:
                    partial = [k for k, v in status.items() if v]
                    logger.warning(f"△ {symbol}: Partial ({', '.join(partial)})")
            else:
                logger.error(f"✗ {symbol}: Error or incomplete")
        
        logger.info("="*60)
        logger.info(f"Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
        logger.info(f"Data cached in: {self.output_dir}")
        logger.info("="*60 + "\n")
        
        # 保存結果為 JSON
        report_path = self.output_dir / "fetch_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Report saved to: {report_path}")
    
    def get_data_statistics(self) -> Dict:
        """
        獲取緩存數據的統計信息
        
        Returns:
            統計信息字典
        """
        stats = {
            'total_files': 0,
            'total_rows': 0,
            'files_by_interval': {'15m': 0, '1h': 0},
            'file_details': {}
        }
        
        for csv_file in self.output_dir.glob('*.csv'):
            if csv_file.name == 'combined_data.csv':
                continue
            
            df = pd.read_csv(csv_file)
            stats['total_files'] += 1
            stats['total_rows'] += len(df)
            
            # 提取時間框架
            for interval in self.INTERVALS:
                if interval in csv_file.name:
                    stats['files_by_interval'][interval] += 1
            
            stats['file_details'][csv_file.name] = {
                'rows': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            }
        
        return stats


class FetcherConfig:
    """配置類 - 可根據需要調整"""
    TARGET_KLINES = 50000  # 目標 K 線數量
    BATCH_SIZE = 1000      # 單次 API 請求返回數
    MAX_WORKERS = 5        # 並行線程數
    OUTPUT_DIR = './crypto_data_cache'
    RETRY_ATTEMPTS = 3     # 失敗重試次數


def main():
    """主執行函數"""
    
    logger.info("Initializing Cryptocurrency Historical Data Fetcher...")
    
    # 初始化抓取器
    fetcher = CryptoDataFetcher(output_dir=FetcherConfig.OUTPUT_DIR)
    
    # 並行抓取所有幣種
    logger.info(f"Target K-lines per symbol: {FetcherConfig.TARGET_KLINES}")
    logger.info(f"Intervals: {', '.join(fetcher.INTERVALS)}")
    logger.info(f"Total symbols: {len(fetcher.CRYPTO_SYMBOLS)}")
    
    results = fetcher.fetch_all_cryptos_parallel(max_workers=FetcherConfig.MAX_WORKERS)
    
    # 生成報告
    fetcher.generate_summary_report(results)
    
    # 統計信息
    stats = fetcher.get_data_statistics()
    logger.info(f"\nData Statistics:")
    logger.info(f"  Total Files: {stats['total_files']}")
    logger.info(f"  Total Rows: {stats['total_rows']:,}")
    logger.info(f"  15m Files: {stats['files_by_interval']['15m']}")
    logger.info(f"  1h Files: {stats['files_by_interval']['1h']}")
    
    return fetcher.output_dir


if __name__ == "__main__":
    data_dir = main()
    print(f"\n✓ Data cached in: {data_dir}")
