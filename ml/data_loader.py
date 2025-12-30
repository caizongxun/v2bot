"""
加載程式 - 好丢數據加載器

正一情此
功能:
- 從 HF 或本地加載 OHLCV 數據
- 數據驗證和清理
- 時間序列分割（train/val/test）
- 正見化（normalization）
- 窗口滢动

使用方式:
```python
from ml.data_loader import DataLoader

loader = DataLoader(
    repo_id='zongowo111/v2-crypto-ohlcv-data',
    symbol='BTC',
    interval='15m',
    token='your_hf_token'  # optional
)

# 加載並清理
df = loader.load_and_clean()
print(df.head())
print(f'Shape: {df.shape}')

# 渀光（測試用）
window_size = 100
windows = loader.create_windows(df, window_size)
print(f'Windows: {len(windows)}')

# 正見化
df_normalized = loader.normalize(df)

# 分割
train, val, test = loader.train_val_test_split(df, train_ratio=0.7, val_ratio=0.15)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
```
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """好丢 OHLCV 數據加載器"""
    
    def __init__(self, 
                 repo_id: str = 'zongowo111/v2-crypto-ohlcv-data',
                 symbol: str = 'BTC',
                 interval: str = '15m',
                 token: Optional[str] = None):
        """
        初始化
        
        Args:
            repo_id: HuggingFace 數據集 ID
            symbol: 幣種（BTC, ETH, 等）
            interval: 間隔（15m, 1h 等）
            token: HF token (可選)
        """
        self.repo_id = repo_id
        self.symbol = symbol
        self.interval = interval
        self.token = token
        self.data = None
        self.scaler_params = {}
        
    def load_data(self) -> pd.DataFrame:
        """加載數據"""
        try:
            # 嘗試從 HF 加載
            from huggingface_hub import hf_hub_download
            
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f'klines/{self.symbol}/{self.symbol}_{self.interval}.csv',
                repo_type='dataset',
                token=self.token
            )
            
            df = pd.read_csv(file_path)
            print(f'[DataLoader] Loaded from HF: {self.symbol}_{self.interval}.csv')
            
        except Exception as e:
            print(f'[DataLoader] HF load failed: {str(e)}')
            print(f'[DataLoader] Trying local path...')
            
            # 嘗試從本地加載
            local_path = Path(f'data/klines/{self.symbol}/{self.symbol}_{self.interval}.csv')
            if local_path.exists():
                df = pd.read_csv(local_path)
                print(f'[DataLoader] Loaded from local: {local_path}')
            else:
                raise FileNotFoundError(f'File not found: {local_path}')
        
        self.data = df
        return df
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """數據驗證和清理"""
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError('No data loaded')
        
        # 複製
        df = df.copy()
        
        # 視接時間戲
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # 刪除重複
        df = df.drop_duplicates(subset=['timestamp'] if 'timestamp' in df.columns else None)
        
        # 刪除 NaN
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        removed = initial_len - len(df)
        if removed > 0:
            print(f'[DataLoader] Removed {removed} rows with missing values')
        
        # 碩鑁一定是正數
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[df[numeric_cols].gt(0).all(axis=1)]
        
        # 重設 index
        df = df.reset_index(drop=True)
        
        print(f'[DataLoader] Cleaned data shape: {df.shape}')
        print(f'[DataLoader] Date range: {df["timestamp"].min()} to {df["timestamp"].max()}' 
              if 'timestamp' in df.columns else '[DataLoader] No timestamp column')
        
        self.data = df
        return df
    
    def load_and_clean(self) -> pd.DataFrame:
        """加載並清理數據"""
        self.load_data()
        return self.clean_data()
    
    def normalize(self, 
                  df: Optional[pd.DataFrame] = None,
                  method: str = 'minmax') -> pd.DataFrame:
        """
        正見化數據
        
        Args:
            df: 輸入 DataFrame
            method: 'minmax' 或 'zscore'
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError('No data to normalize')
        
        df = df.copy()
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                self.scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                self.scaler_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)
        
        print(f'[DataLoader] Normalized using {method}')
        return df
    
    def denormalize(self, 
                    values: np.ndarray, 
                    column: str) -> np.ndarray:
        """反正見化"""
        if column not in self.scaler_params:
            return values
        
        params = self.scaler_params[column]
        
        if params['method'] == 'minmax':
            min_val = params['min']
            max_val = params['max']
            return values * (max_val - min_val) + min_val
        elif params['method'] == 'zscore':
            mean_val = params['mean']
            std_val = params['std']
            return values * std_val + mean_val
        
        return values
    
    def create_windows(self, 
                      df: Optional[pd.DataFrame] = None,
                      window_size: int = 100,
                      step: int = 1) -> List[np.ndarray]:
        """
        窗口滢动
        
        Args:
            df: 輸入 DataFrame
            window_size: 窗口大小
            step: 步長
        
        Returns:
            窗口清單
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError('No data for windowing')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        data_matrix = df[numeric_cols].values
        
        windows = []
        for i in range(0, len(data_matrix) - window_size + 1, step):
            window = data_matrix[i:i+window_size]
            windows.append(window)
        
        print(f'[DataLoader] Created {len(windows)} windows (size={window_size}, step={step})')
        return windows
    
    def train_val_test_split(self,
                            df: Optional[pd.DataFrame] = None,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        時間序列分割 (步長分割，不打亂)
        
        Args:
            df: 輸入 DataFrame
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
        
        Returns:
            (train_df, val_df, test_df)
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError('No data to split')
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = df[:train_end].reset_index(drop=True)
        val = df[train_end:val_end].reset_index(drop=True)
        test = df[val_end:].reset_index(drop=True)
        
        print(f'[DataLoader] Split: Train={len(train)} ({train_ratio*100:.0f}%), '
              f'Val={len(val)} ({val_ratio*100:.0f}%), '
              f'Test={len(test)} ({(1-train_ratio-val_ratio)*100:.0f}%)')
        
        return train, val, test
    
    def get_stats(self, df: Optional[pd.DataFrame] = None) -> dict:
        """取得數據統計信息"""
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError('No data')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        return stats
    
    def summary(self) -> None:
        """打印數據水索"""
        if self.data is None:
            print('[DataLoader] No data loaded')
            return
        
        print("\n" + "="*70)
        print(f"DATA SUMMARY - {self.symbol} {self.interval}")
        print("="*70)
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {self.data.columns.tolist()}")
        print(f"\nData types:\n{self.data.dtypes}")
        
        print(f"\nBasic stats:")
        stats = self.get_stats()
        for col, stat in stats.items():
            print(f"\n{col}:")
            for key, val in stat.items():
                print(f"  {key}: {val:.2f}")
        
        if 'timestamp' in self.data.columns:
            print(f"\nDate range:")
            print(f"  Start: {self.data['timestamp'].min()}")
            print(f"  End: {self.data['timestamp'].max()}")
            print(f"  Duration: {self.data['timestamp'].max() - self.data['timestamp'].min()}")
        
        print("\n" + "="*70)


if __name__ == '__main__':
    # 測試例子
    loader = DataLoader(symbol='BTC', interval='15m')
    
    try:
        df = loader.load_and_clean()
        print(f"\nLoaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # 打印水紡
        loader.summary()
        
    except Exception as e:
        print(f"Error: {str(e)}")
