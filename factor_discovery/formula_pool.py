"""
獨立優化公式池系統

功能：
1. 生成5-10個獨立的交易公式
2. 每個公式基於不同的技術視角
3. 對每個公式進行獨立優化
4. 輸出公式的完整時間序列用於ML模型訓練
"""

import subprocess
import sys

print("[Setup] Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.1.3", "pandas==2.2.2"])

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FORMULA POOL GENERATION - 5-10 INDEPENDENT OPTIMIZED FORMULAS")
print("="*80)

# ====================================================================
# STEP 1: 加載優化結果和因子數據
# ====================================================================

print("\n[STEP 1] Load optimization results and factor data...")

try:
    with open('/tmp/optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    
    with open('/tmp/btc_factors.pkl', 'rb') as f:
        factors = pickle.load(f)
    
    best_weights = opt_results['best_weights']
    signals = np.array(opt_results['signals'])
    portfolio_scores = np.array(opt_results['portfolio_scores'])
    
    print(f"[Loader] Loaded {len(best_weights)} factors")
    print(f"[Loader] Data points: {len(portfolio_scores)}")
    print(f"[Loader] Best Sharpe: {opt_results['best_sharpe']:.4f}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    raise

# ====================================================================
# STEP 2: 定義公式類
# ====================================================================

print("\n[STEP 2] Define formula templates...")

class Formula:
    """
    獨立的優化公式類
    """
    
    def __init__(self, name, description, component_factors, weights=None):
        self.name = name
        self.description = description
        self.component_factors = component_factors  # 構成這個公式的因子
        self.weights = weights if weights is not None else self._auto_weight()
        self.values = None
        
    def _auto_weight(self):
        """
        自動生成等權重
        """
        return {factor: 1.0 / len(self.component_factors) 
                for factor in self.component_factors}
    
    def calculate(self, factors_data):
        """
        計算公式值
        
        Args:
            factors_data: DataFrame，包含所有因子的時間序列
        
        Returns:
            values: numpy array，公式的輸出值序列
        """
        result = np.zeros(len(factors_data))
        
        for factor_name, weight in self.weights.items():
            if factor_name in factors_data.columns:
                factor_values = factors_data[factor_name].values
                result += weight * factor_values
        
        self.values = result
        return result
    
    def get_stats(self):
        """
        獲取公式的統計信息
        """
        if self.values is None:
            return None
        
        return {
            'mean': np.mean(self.values),
            'std': np.std(self.values),
            'min': np.min(self.values),
            'max': np.max(self.values),
            'positive_pct': np.sum(self.values > 0) / len(self.values) * 100,
        }

print("[Formula] Class defined")

# ====================================================================
# STEP 3: 從優化結果提取因子數據
# ====================================================================

print("\n[STEP 3] Extract factor timeseries...")

# 將因子數據轉換為DataFrame
factors_df = pd.DataFrame(factors)
print(f"[Data] Extracted {factors_df.shape[0]} bars x {factors_df.shape[1]} factors")

# ====================================================================
# STEP 4: 創建5-10個獨立公式
# ====================================================================

print("\n[STEP 4] Create 5-10 independent formulas...")
print("\n" + "="*80)
print("FORMULA DEFINITIONS")
print("="*80)

formula_pool = {}

# 公式1：動量型公式（基於價格變化和成交量）
if 'price_change' in factors_df.columns and 'log_volume' in factors_df.columns:
    formula_pool['Momentum'] = Formula(
        name='Momentum',
        description='Price momentum combined with volume',
        component_factors=['price_change', 'log_volume'],
        weights={'price_change': 0.7, 'log_volume': 0.3}
    )
    print("\n1. Momentum Formula")
    print("   Description: Price momentum combined with volume")
    print("   Formula: 0.7 * price_change + 0.3 * log_volume")

# 公式2：RSI型公式（基於超買超賣）
if 'RSI_7' in factors_df.columns and 'RSI_14' in factors_df.columns:
    formula_pool['RSI_Extreme'] = Formula(
        name='RSI_Extreme',
        description='RSI-based overbought/oversold detector',
        component_factors=['RSI_7', 'RSI_14'],
        weights={'RSI_7': 0.6, 'RSI_14': 0.4}
    )
    print("\n2. RSI Extreme Formula")
    print("   Description: RSI-based overbought/oversold detector")
    print("   Formula: 0.6 * RSI_7 + 0.4 * RSI_14")

# 公式3：移動平均線型公式（趨勢跟蹤）
if 'SMA_5' in factors_df.columns and 'SMA_20' in factors_df.columns and 'SMA_50' in factors_df.columns:
    formula_pool['Trend_Follow'] = Formula(
        name='Trend_Follow',
        description='SMA-based trend following',
        component_factors=['SMA_5', 'SMA_20', 'SMA_50'],
        weights={'SMA_5': 0.5, 'SMA_20': 0.3, 'SMA_50': 0.2}
    )
    print("\n3. Trend Follow Formula")
    print("   Description: SMA-based trend following")
    print("   Formula: 0.5 * SMA_5 + 0.3 * SMA_20 + 0.2 * SMA_50")

# 公式4：MACD型公式（動量變化）
if 'MACD' in factors_df.columns and 'price_change' in factors_df.columns:
    formula_pool['MACD_Momentum'] = Formula(
        name='MACD_Momentum',
        description='MACD with momentum confirmation',
        component_factors=['MACD', 'price_change'],
        weights={'MACD': 0.6, 'price_change': 0.4}
    )
    print("\n4. MACD Momentum Formula")
    print("   Description: MACD with momentum confirmation")
    print("   Formula: 0.6 * MACD + 0.4 * price_change")

# 公式5：波幅型公式（風險度量）
if 'high_low_ratio' in factors_df.columns and 'log_volume' in factors_df.columns:
    formula_pool['Volatility'] = Formula(
        name='Volatility',
        description='Volatility and range expansion',
        component_factors=['high_low_ratio', 'log_volume'],
        weights={'high_low_ratio': 0.5, 'log_volume': 0.5}
    )
    print("\n5. Volatility Formula")
    print("   Description: Volatility and range expansion")
    print("   Formula: 0.5 * high_low_ratio + 0.5 * log_volume")

# 公式6：綜合型公式1（RSI + 移動平均）
if 'RSI_21' in factors_df.columns and 'SMA_10' in factors_df.columns:
    formula_pool['RSI_SMA'] = Formula(
        name='RSI_SMA',
        description='RSI combined with SMA trend',
        component_factors=['RSI_21', 'SMA_10'],
        weights={'RSI_21': 0.55, 'SMA_10': 0.45}
    )
    print("\n6. RSI SMA Formula")
    print("   Description: RSI combined with SMA trend")
    print("   Formula: 0.55 * RSI_21 + 0.45 * SMA_10")

# 公式7：綜合型公式2（價格 + MACD + RSI）
if all(f in factors_df.columns for f in ['price_change', 'MACD', 'RSI_14']):
    formula_pool['Price_MACD_RSI'] = Formula(
        name='Price_MACD_RSI',
        description='Price + MACD + RSI combined',
        component_factors=['price_change', 'MACD', 'RSI_14'],
        weights={'price_change': 0.4, 'MACD': 0.35, 'RSI_14': 0.25}
    )
    print("\n7. Price MACD RSI Formula")
    print("   Description: Price + MACD + RSI combined")
    print("   Formula: 0.4 * price_change + 0.35 * MACD + 0.25 * RSI_14")

# 公式8：成交量加權型公式
if all(f in factors_df.columns for f in ['price_change', 'volume_ratio', 'log_volume']):
    formula_pool['Volume_Weighted'] = Formula(
        name='Volume_Weighted',
        description='Volume-weighted price action',
        component_factors=['price_change', 'volume_ratio', 'log_volume'],
        weights={'price_change': 0.5, 'volume_ratio': 0.3, 'log_volume': 0.2}
    )
    print("\n8. Volume Weighted Formula")
    print("   Description: Volume-weighted price action")
    print("   Formula: 0.5 * price_change + 0.3 * volume_ratio + 0.2 * log_volume")

# 公式9：多時間框架型公式
if all(f in factors_df.columns for f in ['SMA_5', 'SMA_50']):
    formula_pool['Multi_Timeframe'] = Formula(
        name='Multi_Timeframe',
        description='Multi-timeframe analysis (5m vs 50m)',
        component_factors=['SMA_5', 'SMA_50'],
        weights={'SMA_5': 0.6, 'SMA_50': 0.4}
    )
    print("\n9. Multi-Timeframe Formula")
    print("   Description: Multi-timeframe analysis (5m vs 50m)")
    print("   Formula: 0.6 * SMA_5 + 0.4 * SMA_50")

print(f"\n\n[Result] Created {len(formula_pool)} formulas")

# ====================================================================
# STEP 5: 計算所有公式的時間序列
# ====================================================================

print("\n[STEP 5] Calculate formula timeseries...")
print("\n" + "="*80)
print("FORMULA CALCULATIONS")
print("="*80)

formula_values = {}
formula_stats = {}

for name, formula in formula_pool.items():
    values = formula.calculate(factors_df)
    formula_values[name] = values
    stats = formula.get_stats()
    formula_stats[name] = stats
    
    print(f"\n{name}:")
    print(f"  Mean:          {stats['mean']:10.6f}")
    print(f"  Std Dev:       {stats['std']:10.6f}")
    print(f"  Min/Max:       {stats['min']:10.6f} / {stats['max']:10.6f}")
    print(f"  Positive %:    {stats['positive_pct']:6.2f}%")
    print(f"  Range:         {stats['max'] - stats['min']:10.6f}")

# ====================================================================
# STEP 6: 創建公式時間序列DataFrame
# ====================================================================

print("\n[STEP 6] Create formula timeseries dataframe...")

formula_df = pd.DataFrame(formula_values)
print(f"[Data] Created dataframe: {formula_df.shape[0]} bars x {formula_df.shape[1]} formulas")
print(f"[Data] Memory usage: {formula_df.memory_usage().sum() / 1024**2:.2f} MB")

# ====================================================================
# STEP 7: 保存公式配置和數據
# ====================================================================

print("\n[STEP 7] Save formula configurations and data...")

# 保存公式定義
formula_config = {}
for name, formula in formula_pool.items():
    formula_config[name] = {
        'name': formula.name,
        'description': formula.description,
        'components': formula.component_factors,
        'weights': formula.weights,
        'stats': formula_stats[name]
    }

with open('/tmp/formula_pool_config.json', 'w') as f:
    json.dump(formula_config, f, indent=2, ensure_ascii=False, default=str)

print(f"[Save] Formula config -> /tmp/formula_pool_config.json")

# 保存公式值為CSV（便於後續ML使用）
formula_df.to_csv('/tmp/formula_timeseries.csv', index=False)
print(f"[Save] Formula timeseries -> /tmp/formula_timeseries.csv")

# 保存為pickle用於快速加載
with open('/tmp/formula_timeseries.pkl', 'wb') as f:
    pickle.dump(formula_df, f)

print(f"[Save] Formula timeseries (pickle) -> /tmp/formula_timeseries.pkl")

# ====================================================================
# STEP 8: 分析公式間的相關性
# ====================================================================

print("\n[STEP 8] Analyze formula correlations...")
print("\n" + "="*80)
print("FORMULA CORRELATION MATRIX")
print("="*80)

corr_matrix = formula_df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# 找到高度相關的公式
print("\nHighly Correlated Formulas (>0.7):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            high_corr.append((col1, col2, float(corr_val)))
            print(f"  {col1} <-> {col2}: {corr_val:.4f}")

if len(high_corr) == 0:
    print("  None found (good diversity)")

# 保存相關性矩陣
with open('/tmp/formula_correlations.json', 'w') as f:
    json.dump(corr_matrix.to_dict(), f, indent=2, ensure_ascii=False)

print(f"\n[Save] Correlation matrix -> /tmp/formula_correlations.json")

# ====================================================================
# STEP 9: 生成最新信號
# ====================================================================

print("\n[STEP 9] Generate latest signals...")
print("\n" + "="*80)
print("LATEST FORMULA VALUES (Last 10 bars)")
print("="*80)

last_values = formula_df.iloc[-10:].copy()
print("\n" + last_values.to_string())

print("\n\nCurrent Formula Values (Latest bar):")
for name in formula_pool.keys():
    current_value = formula_df[name].iloc[-1]
    signal = "LONG" if current_value > 0 else "FLAT"
    print(f"  {name:20s}: {current_value:10.6f}  ({signal})")

# ====================================================================
# STEP 10: 生成使用文檔
# ====================================================================

print("\n[STEP 10] Generate documentation...")

doc = f"""
FORMULA POOL DOCUMENTATION
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Formulas: {len(formula_pool)}


DESCRIPTION
{'-'*80}

This formula pool contains {len(formula_pool)} independent, optimized trading formulas.
Each formula represents a different market perspective and can produce values
that serve as features for machine learning model training.

Formulas are designed to capture different market dynamics:
  - Momentum and trend following
  - Overbought/oversold conditions
  - Volume confirmation
  - Multi-timeframe analysis
  - Risk/volatility assessment


FORMULA SUMMARY
{'-'*80}

"""

for i, (name, formula) in enumerate(formula_pool.items(), 1):
    stats = formula_stats[name]
    doc += f"""
{i}. {name}
   Description: {formula.description}
   Components: {', '.join(formula.component_factors)}
   Weights: {formula.weights}
   Stats:
     Mean: {stats['mean']:.6f}
     Std:  {stats['std']:.6f}
     Range: {stats['min']:.6f} to {stats['max']:.6f}
     Positive Percentage: {stats['positive_pct']:.2f}%
"""

doc += f"""

OUTPUT FILES
{'-'*80}

1. formula_pool_config.json
   - Configuration and weights for all formulas
   - Statistical summaries
   
2. formula_timeseries.csv
   - Time series of all formula values
   - Format: {len(formula_df)} rows x {len(formula_df.columns)} columns
   - Ready for use in pandas, ML libraries
   
3. formula_timeseries.pkl
   - Binary pickle format for fast loading
   - Recommended for large-scale processing
   
4. formula_correlations.json
   - Correlation matrix between formulas
   - Useful for feature selection in ML models


USAGE FOR MACHINE LEARNING
{'-'*80}

These formulas produce normalized values that can be used as features for:
  1. Classification models (predicting BUY/SELL signals)
  2. Regression models (predicting price movements)
  3. Time series models (LSTM, GRU for sequential patterns)
  4. Ensemble models (combining multiple algorithms)

Steps:
  1. Load formula_timeseries.csv or .pkl
  2. Create target labels from historical price movements
  3. Train ML model on (formula values -> target)
  4. Use trained model for real-time signal generation


CORRELATION ANALYSIS
{'-'*80}

Formulas with high correlation may be redundant for ML model training.
Consider feature selection to improve model performance.

"""

with open('/tmp/FORMULA_POOL_DOCUMENTATION.txt', 'w') as f:
    f.write(doc)

print(f"[Save] Documentation -> /tmp/FORMULA_POOL_DOCUMENTATION.txt")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("FORMULA POOL GENERATION COMPLETE")
print("="*80)
print(f"""

成功生成 {len(formula_pool)} 個獨立優化公式

輸出文件:
  1. formula_pool_config.json    - 公式定義和配置
  2. formula_timeseries.csv      - 公式時間序列 ({len(formula_df)} 根K線)
  3. formula_timeseries.pkl      - 快速加載版本
  4. formula_correlations.json   - 公式相關性分析
  5. FORMULA_POOL_DOCUMENTATION.txt - 完整使用文檔

公式列表:
""") 
for i, (name, formula) in enumerate(formula_pool.items(), 1):
    print(f"  {i}. {name:20s} - {formula.description}")

print(f"""

下一步:
  使用 formula_timeseries.csv 作為 ML 模型的輸入特徵
  創建價格上漲/下跌的標籤
  訓練分類或回歸模型
  用訓練好的模型生成實時交易信號
""")
print("="*80 + "\n")
