"""
簡出簡整化的BTC 15m因子自動發現系統 v2
低依賴性、高屲彈性設計
"""

import subprocess
import sys

print("[Setup] 修正版本不相容...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy==2.1.3",
    "pandas==2.2.2",
    "--upgrade", "--force-reinstall"
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface-hub==0.23.0",
    "pyarrow",
])

print("[Setup] ✓ All ready!\n")

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CRYPTOCURRENCY FACTOR DISCOVERY SYSTEM v2.0")
print("="*80)

# ====================================================================
# STEP 1: 下載BTC 15m數據
# ====================================================================

print("\n[STEP 1] 下載BTC 15m K線數據...")

try:
    from huggingface_hub import hf_hub_download
    
    cache_dir = Path("/tmp/crypto_data")
    cache_dir.mkdir(exist_ok=True)
    
    file_path = hf_hub_download(
        repo_id="zongowo111/v2-crypto-ohlcv-data",
        filename="klines/BTCUSDT/BTC_15m.parquet",
        repo_type="dataset",
        cache_dir=str(cache_dir)
    )
    
    df = pd.read_parquet(file_path)
    print(f"[DataLoader] ✓ Loaded {len(df)} records")
    print(f"[DataLoader] Columns: {list(df.columns)}")
    
except Exception as e:
    print(f"[DataLoader] ✗ Failed: {e}")
    print("[DataLoader] Using sample data...")
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=5000, freq='15min')
    close_price = 30000 + np.cumsum(np.random.randn(5000) * 50)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTCUSDT',
        'open': close_price + np.random.randn(5000) * 10,
        'high': close_price + np.abs(np.random.randn(5000) * 15),
        'low': close_price - np.abs(np.random.randn(5000) * 15),
        'close': close_price,
        'volume': np.abs(np.random.randn(5000) * 100 + 500),
    })
    print(f"[DataLoader] Sample data {len(df)} records")

# ====================================================================
# STEP 2: Minimal Data Cleaning
# ====================================================================

print("\n[STEP 2] Clean data...")

required_cols = ['open', 'high', 'low', 'close', 'volume']
df = df[required_cols].copy()
df = df.astype(float)
df = df[(df > 0).all(axis=1)]

print(f"[DataCleaner] ✓ {len(df)} records after cleaning")

if len(df) < 100:
    print(f"[ERROR] Not enough data")
    raise ValueError("Insufficient data")

print(f"[DataCleaner] Price range: {df['close'].min():.0f} - {df['close'].max():.0f}")

# ====================================================================
# STEP 3: Technical Indicators
# ====================================================================

print("\n[STEP 3] Calculate technical indicators...")

class Indicators:
    @staticmethod
    def sma(s, p):
        return pd.Series(s).rolling(p, min_periods=1).mean().values
    
    @staticmethod
    def ema(s, p):
        return pd.Series(s).ewm(span=p, adjust=False).mean().values
    
    @staticmethod
    def rsi(s, p=14):
        delta = pd.Series(s).diff()
        gain = (delta.where(delta > 0, 0)).rolling(p, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return (100 - 100 / (1 + rs)).values
    
    @staticmethod
    def macd(s, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(s).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(s).ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow).values
    
    @staticmethod
    def bbands(s, p=20, k=2):
        sma = pd.Series(s).rolling(p, min_periods=1).mean()
        std = pd.Series(s).rolling(p, min_periods=1).std()
        upper = sma + k * std
        lower = sma - k * std
        return upper.values, sma.values, lower.values

ind = Indicators()

factors = {
    'price_change': (df['close'] - df['open']) / (df['open'] + 1e-10),
    'high_low_ratio': (df['high'] - df['low']) / (df['close'] + 1e-10),
    
    'SMA_5': ind.sma(df['close'].values, 5),
    'SMA_10': ind.sma(df['close'].values, 10),
    'SMA_20': ind.sma(df['close'].values, 20),
    'SMA_50': ind.sma(df['close'].values, 50),
    
    'RSI_7': ind.rsi(df['close'].values, 7),
    'RSI_14': ind.rsi(df['close'].values, 14),
    'RSI_21': ind.rsi(df['close'].values, 21),
    
    'MACD': ind.macd(df['close'].values, 12, 26, 9),
    
    'log_volume': np.log(df['volume'].values + 1),
    'volume_ratio': df['volume'].values / (df['volume'].rolling(20, min_periods=1).mean().values + 1e-10),
}

print(f"[FactorGenerator] ✓ Generated {len(factors)} factors")

# ====================================================================
# STEP 4: Symbolic Regression - Auto-discover factors
# ====================================================================

print("\n[STEP 4] Symbolic regression - discover factor formulas...")

class Node:
    def __init__(self, op=None, left=None, right=None, var=None):
        self.op = op
        self.left = left
        self.right = right
        self.var = var
    
    def eval(self, data):
        if self.var is not None:
            return data.get(self.var, np.zeros(len(data['close'])))
        
        l = self.left.eval(data)
        r = self.right.eval(data) if self.right else None
        
        if self.op == '+': return l + r
        elif self.op == '-': return l - r
        elif self.op == '*': return l * r
        elif self.op == '/': return l / (r + 1e-10)
        elif self.op == 'log': return np.log(np.abs(l) + 1)
        elif self.op == 'abs': return np.abs(l)
        return l
    
    def __str__(self):
        if self.var: return self.var
        if self.op in ['+', '-', '*', '/']: return f"({self.left} {self.op} {self.right})"
        return f"{self.op}({self.left})"

class SymReg:
    def __init__(self, data, pop_size=40, gens=40):
        self.data = data
        self.pop_size = pop_size
        self.gens = gens
        self.vars = ['close', 'open', 'high', 'low', 'volume']
        self.ops = ['+', '-', '*', '/', 'log', 'abs']
    
    def random_node(self, depth=0, max_d=3):
        if depth >= max_d or np.random.random() < 0.3:
            if np.random.random() < 0.7:
                return Node(var=np.random.choice(self.vars))
            else:
                return Node(var='const')
        
        op = np.random.choice(self.ops)
        left = self.random_node(depth+1, max_d)
        if op in ['+', '-', '*', '/']: 
            right = self.random_node(depth+1, max_d)
        else: 
            right = None
        return Node(op=op, left=left, right=right)
    
    def fitness(self, node):
        try:
            result = node.eval(self.data)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)): return -1e6
            if np.std(result) < 1e-10: return -1e6
            
            result = (result - np.mean(result)) / (np.std(result) + 1e-10)
            price = (self.data['close'] - np.mean(self.data['close'])) / (np.std(self.data['close']) + 1e-10)
            
            corr = np.corrcoef(result, price)[0, 1]
            if np.isnan(corr): corr = 0
            
            vol = np.mean(np.abs(np.diff(result)))
            return abs(corr) * 0.7 + vol * 0.3
        except:
            return -1e6
    
    def evolve(self):
        print("[SymReg] Starting evolution...")
        
        pop = [self.random_node() for _ in range(self.pop_size)]
        best_formulas = []
        
        for gen in range(self.gens):
            scores = [self.fitness(node) for node in pop]
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            if gen % 10 == 0:
                print(f"  Gen {gen:2d}: fitness = {best_score:8.4f}")
            
            if best_score > 0.15:
                formula_str = str(pop[best_idx])
                if formula_str not in [f[0] for f in best_formulas]:
                    best_formulas.append((formula_str, best_score))
            
            elite_idx = np.argsort(scores)[-max(5, self.pop_size // 5):]
            elite = [pop[i] for i in elite_idx]
            
            pop = elite + [self.random_node() for _ in range(self.pop_size - len(elite))]
        
        best_formulas.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in best_formulas[:8]]

data_dict = {
    'close': df['close'].values,
    'open': df['open'].values,
    'high': df['high'].values,
    'low': df['low'].values,
    'volume': df['volume'].values,
}

syreg = SymReg(data_dict, pop_size=40, gens=40)
discovered_formulas = syreg.evolve()

print(f"\n[SymReg] ✓ Discovered {len(discovered_formulas)} formulas:")
for i, f in enumerate(discovered_formulas[:5], 1):
    print(f"  Formula_{i}: {str(f)[:50]}...")

# ====================================================================
# STEP 5: Analyze factor quality
# ====================================================================

print("\n[STEP 5] Analyze factor quality...")
print("\n" + "="*80)
print("FACTOR QUALITY SCORES")
print("="*80)

results = {}
for name, factor in factors.items():
    factor = np.asarray(factor, dtype=float)
    factor = np.nan_to_num(factor, 0)
    
    price = df['close'].values
    
    try:
        corr = np.corrcoef(factor, price)[0, 1]
    except:
        corr = 0
    
    if np.isnan(corr): corr = 0
    
    vol = np.mean(np.abs(np.diff(factor)))
    returns = np.diff(factor)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    
    results[name] = {'corr': corr, 'vol': vol, 'sharpe': sharpe}
    
    print(f"{name:18s} | Corr: {corr:7.4f} | Vol: {vol:8.6f} | Sharpe: {sharpe:8.4f}")

# ====================================================================
# STEP 6: Save results
# ====================================================================

print("\n[STEP 6] Save results...")

with open('/tmp/btc_factors.pkl', 'wb') as f:
    pickle.dump(factors, f)
print(f"[Save] ✓ Factors -> /tmp/btc_factors.pkl")

with open('/tmp/btc_formulas.json', 'w') as f:
    json.dump({f'Formula_{i+1}': formula for i, formula in enumerate(discovered_formulas)}, f, indent=2, ensure_ascii=False)
print(f"[Save] ✓ Formulas -> /tmp/btc_formulas.json")

with open('/tmp/btc_analysis.json', 'w') as f:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, ensure_ascii=False)
print(f"[Save] ✓ Analysis -> /tmp/btc_analysis.json")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("SYSTEM COMPLETE")
print("="*80)
print(f"Input records:          {len(df):6d}")
print(f"Standard factors:       {len(factors):6d}")
print(f"Discovered formulas:    {len(discovered_formulas):6d}")
print(f"\nTop 3 Best Factors (by correlation):")
sorted_factors = sorted(results.items(), key=lambda x: abs(x[1]['corr']), reverse=True)
for i, (name, metrics) in enumerate(sorted_factors[:3], 1):
    print(f"  {i}. {name:18s} | Corr: {metrics['corr']:7.4f}")
print("="*80)
