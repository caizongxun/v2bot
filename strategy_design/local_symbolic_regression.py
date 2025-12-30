#!/usr/bin/env python3
"""
Local Symbolic Regression Engine

Usage:
    python local_symbolic_regression.py --symbol BTC --interval 1h --iterations 100

This script discovers optimal trading formulas from historical K-line data.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download
import ta


class SymbolicRegressionDataPrep:
    """Prepare data for symbolic regression formula discovery"""
    
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        self.df = None
        self.scaler = StandardScaler()
        self.indicator_cols = []
    
    def load_data(self, symbol='BTC', interval='1h', repo_id='zongowo111/v2-crypto-ohlcv-data'):
        """Load data from Hugging Face Dataset"""
        print(f"Loading {symbol} {interval} data from HF...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f'klines/{symbol}USDT/{symbol}_{interval}.parquet',
            repo_type='dataset',
            token=self.hf_token
        )
        self.df = pd.read_parquet(file_path)
        print(f"Loaded {len(self.df)} rows")
        return self.df
    
    def compute_indicators(self):
        """Compute 30+ technical indicators"""
        if self.df is None:
            raise ValueError("Load data first")
        
        df = self.df.copy()
        print("Computing technical indicators...")
        
        # Momentum indicators
        df['rsi_7'] = ta.momentum.rsi(df['close'], 7)
        df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], 21)
        
        # MACD
        df['macd_line'] = ta.trend.macd_line(df['close'], 12, 26)
        df['macd_signal'] = ta.trend.macd_signal_line(df['close'], 12, 26, 9)
        df['macd_diff'] = df['macd_line'] - df['macd_signal']
        
        # Trend indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(12).mean()
        df['ema_26'] = df['close'].ewm(26).mean()
        
        # Volatility indicators
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], 20, 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ema'] = df['obv'].ewm(14).mean()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Store indicator columns
        self.indicator_cols = [col for col in df.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        
        self.df = df
        print(f"Computed {len(self.indicator_cols)} indicators")
        return self.df
    
    def create_labels(self, lookahead=24, threshold=0.005):
        """
        Create training labels
        
        Args:
            lookahead: Look ahead N bars (24h for 1h bars = 1 day)
            threshold: Return rate threshold
        """
        if self.df is None:
            raise ValueError("Compute indicators first")
        
        df = self.df.copy()
        
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Three-class labels
        df['label'] = 1  # HOLD (default)
        df.loc[df['future_return'] > threshold, 'label'] = 2   # BUY
        df.loc[df['future_return'] < -threshold, 'label'] = 0  # SELL
        
        self.df = df
        
        print(f"\nLabel distribution:")
        print(f"  SELL (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
        print(f"  HOLD (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
        print(f"  BUY  (2): {(df['label'] == 2).sum()} ({(df['label'] == 2).mean():.2%})")
        
        return self.df
    
    def prepare(self, symbol='BTC', interval='1h', lookahead=24):
        """Complete preparation pipeline"""
        self.load_data(symbol, interval)
        self.compute_indicators()
        self.create_labels(lookahead=lookahead)
        
        # Remove NaN
        self.df = self.df.dropna()
        
        return self.df
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get X (features) and y (labels)"""
        if self.df is None:
            raise ValueError("Prepare data first")
        
        X = self.df[self.indicator_cols]
        y = self.df['label']
        
        # Normalize
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.indicator_cols,
            index=X.index
        )
        
        return X_scaled, y


class SymbolicFormulaDiscoverer:
    """Discover optimal trading formulas using symbolic regression"""
    
    def __init__(self, niterations=100, population_size=50, maxsize=20):
        """
        Args:
            niterations: Number of evolution iterations
            population_size: Size of population per iteration
            maxsize: Maximum formula size (number of nodes)
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError("Please install PySR: pip install pysr")
        
        self.PySRRegressor = PySRRegressor
        self.niterations = niterations
        self.population_size = population_size
        self.maxsize = maxsize
        self.model = None
    
    def discover(self, X, y, output_file='discovered_formulas.json'):
        """
        Discover optimal formulas
        
        Args:
            X: Features DataFrame
            y: Labels Series
            output_file: Where to save discovered formulas
        """
        print(f"\nStarting symbolic regression...")
        print(f"  Data shape: {X.shape}")
        print(f"  Iterations: {self.niterations}")
        print(f"  Population: {self.population_size}")
        print(f"  Max size: {self.maxsize}")
        print(f"\nThis may take 30-60 minutes...\n")
        
        self.model = self.PySRRegressor(
            niterations=self.niterations,
            population_size=self.population_size,
            ncyclesperiteration=50,
            procs=4,
            binary_operators=['+', '-', '*', '/', '^'],
            unary_operators=['sin', 'cos', 'sqrt', 'exp', 'log', 'abs', 'tanh'],
            complexity_of_operators={
                '+': 1, '-': 1, '*': 2, '/': 3, '^': 4,
                'sin': 3, 'cos': 3, 'sqrt': 3, 'exp': 3, 'log': 3, 'abs': 1, 'tanh': 3
            },
            maxsize=self.maxsize,
            maxdepth=5,
            loss='mse',
            denoise=True,
            verbosity=1
        )
        
        self.model.fit(X, y)
        equations = self.model.equations_
        
        return self._export_top_formulas(equations, n=5, output_file=output_file)
    
    def _export_top_formulas(self, equations_df, n=5, output_file='discovered_formulas.json'):
        """Export top N formulas"""
        print(f"\nExtracting top {n} formulas...")
        
        top_equations = equations_df.nsmallest(n, 'loss')
        
        formulas_dict = {}
        for i, (idx, row) in enumerate(top_equations.iterrows(), 1):
            formulas_dict[f'formula_{i}'] = {
                'equation': row['equation'],
                'loss': float(row['loss']),
                'complexity': int(row['complexity'])
            }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(formulas_dict, f, indent=2)
        
        print(f"\nSaved {n} formulas to {output_file}")
        print("\nDiscovered formulas:")
        for name, info in formulas_dict.items():
            equation = info['equation']
            if len(equation) > 60:
                equation = equation[:57] + "..."
            print(f"  {name}: {equation}")
            print(f"    Loss: {info['loss']:.6f}, Complexity: {info['complexity']}")
        
        return formulas_dict


def main():
    parser = argparse.ArgumentParser(description='Discover trading formulas using symbolic regression')
    parser.add_argument('--symbol', default='BTC', help='Crypto symbol (default: BTC)')
    parser.add_argument('--interval', default='1h', help='K-line interval (default: 1h)')
    parser.add_argument('--iterations', type=int, default=100, help='SR iterations (default: 100)')
    parser.add_argument('--population', type=int, default=50, help='Population size (default: 50)')
    parser.add_argument('--maxsize', type=int, default=20, help='Max formula size (default: 20)')
    parser.add_argument('--output', default='discovered_formulas.json', help='Output file')
    parser.add_argument('--hf-token', help='Hugging Face API token')
    
    args = parser.parse_args()
    
    # Step 1: Prepare data
    print("="*80)
    print("SYMBOLIC REGRESSION FORMULA DISCOVERY")
    print("="*80)
    
    prep = SymbolicRegressionDataPrep(hf_token=args.hf_token)
    df = prep.prepare(symbol=args.symbol, interval=args.interval)
    
    # Step 2: Get training data
    X, y = prep.get_training_data()
    
    # Step 3: Discover formulas
    discoverer = SymbolicFormulaDiscoverer(
        niterations=args.iterations,
        population_size=args.population,
        maxsize=args.maxsize
    )
    
    formulas = discoverer.discover(X, y, output_file=args.output)
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print(f"Formulas saved to {args.output}")
    print(f"Next step: Upload to Colab and train LSTM model")
    print("="*80)


if __name__ == '__main__':
    main()
