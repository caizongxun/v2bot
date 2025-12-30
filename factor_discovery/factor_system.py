"""
完整的加密货币因子自动发现系统
从K线数据到生成可交易的指标因子

Flow:
1. 从HuggingFace下载BTC 15m K线数据
2. 数据清洗和预处理
3. 符号回归发现新指标公式
4. 遗传算法优化参数
5. 输出最优因子组合
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Callable, Any
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# PART 1: 数据加载和预处理
# ====================================================================

class DataLoader:
    """从HuggingFace加载K线数据"""
    
    def __init__(self, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        self.repo_id = repo_id
        self.data_cache_dir = Path("/tmp/crypto_data")
        self.data_cache_dir.mkdir(exist_ok=True)
    
    def load_btc_15m(self) -> pd.DataFrame:
        """加载BTC 15分钟K线数据"""
        print("[DataLoader] Loading BTC 15m data from HuggingFace...")
        
        try:
            from huggingface_hub import hf_hub_download
            
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="klines/BTCUSDT/BTC_15m.parquet",
                repo_type="dataset",
                cache_dir=str(self.data_cache_dir)
            )
            
            df = pd.read_parquet(file_path)
            print(f"[DataLoader] SUCCESS: Loaded {len(df)} records")
            print(f"[DataLoader] Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"[DataLoader] ERROR: {e}")
            print("[DataLoader] Using sample data instead...")
            return DataLoader._generate_sample_data()
    
    @staticmethod
    def _generate_sample_data() -> pd.DataFrame:
        """生成测试数据"""
        print("[DataLoader] Generating 5000 records of sample BTC data...")
        
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=5000, freq='15min')
        
        close_price = 30000 + np.cumsum(np.random.randn(5000) * 50)
        
        data = {
            'open_time': dates.astype(int) // 10**9,
            'open': close_price + np.random.randn(5000) * 10,
            'high': close_price + np.abs(np.random.randn(5000) * 15),
            'low': close_price - np.abs(np.random.randn(5000) * 15),
            'close': close_price,
            'volume': np.abs(np.random.randn(5000) * 100 + 500),
            'quote_volume': np.abs(np.random.randn(5000) * 5000000 + 15000000),
            'trades': np.abs(np.random.randn(5000) * 50 + 200).astype(int),
            'taker_buy_base': np.abs(np.random.randn(5000) * 50 + 250),
            'taker_buy_quote': np.abs(np.random.randn(5000) * 2500000 + 7500000),
            'close_time': (dates + timedelta(minutes=15)).astype(int) // 10**9,
        }
        
        df = pd.DataFrame(data)
        return df

class DataCleaner:
    """数据清洗和特征工程"""
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        print("[DataCleaner] Cleaning data...")
        
        df = df.sort_values('open_time').reset_index(drop=True)
        df = df.drop_duplicates(subset=['open_time'])
        
        for col in ['open', 'high', 'low', 'close']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"[DataCleaner] SUCCESS: {len(df)} clean records")
        return df

# ====================================================================
# PART 2: 技术指标库（基础特征）
# ====================================================================

class IndicatorLibrary:
    """标准技术指标库"""
    
    @staticmethod
    def sma(series: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        result = np.full_like(series, np.nan)
        for i in range(period - 1, len(series)):
            result[i] = np.mean(series[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def ema(series: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均"""
        result = np.full_like(series, np.nan)
        alpha = 2 / (period + 1)
        result[period - 1] = np.mean(series[:period])
        for i in range(period, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result
    
    @staticmethod
    def std(series: np.ndarray, period: int) -> np.ndarray:
        """标准差"""
        result = np.full_like(series, np.nan)
        for i in range(period - 1, len(series)):
            result[i] = np.std(series[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
        """相对强弱指数"""
        result = np.full_like(series, np.nan)
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = IndicatorLibrary.ema(gain, period)
        avg_loss = IndicatorLibrary.ema(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-10)
        result[period:] = 100 - (100 / (1 + rs[period:]))
        return result

# ====================================================================
# PART 3: 符号回归 - 自动指标发现
# ====================================================================

class ExpressionNode:
    """表达式树节点"""
    
    def __init__(self, op: str = None, left = None, right = None, value = None):
        self.op = op
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self) -> bool:
        return self.op is None
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """计算表达式值"""
        if self.is_leaf():
            if isinstance(self.value, str):
                return data[self.value].copy()
            else:
                return np.full_like(data['close'], self.value)
        
        left_val = self.left.evaluate(data)
        right_val = self.right.evaluate(data) if self.right else None
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            return np.divide(left_val, right_val + 1e-10)
        elif self.op == 'log':
            return np.log(np.abs(left_val) + 1)
        elif self.op == 'exp':
            return np.exp(np.clip(left_val, -10, 10))
        elif self.op == 'abs':
            return np.abs(left_val)
        elif self.op == 'sqrt':
            return np.sqrt(np.abs(left_val))
        
        return left_val
    
    def to_string(self) -> str:
        """转换为公式字符串"""
        if self.is_leaf():
            return str(self.value)
        
        left_str = self.left.to_string()
        
        if self.op in ['+', '-', '*', '/']:
            right_str = self.right.to_string()
            return f"({left_str} {self.op} {right_str})"
        else:
            return f"{self.op}({left_str})"

class SymbolicRegression:
    """符号回归 - 自动指标发现"""
    
    def __init__(self, data: pd.DataFrame, population_size: int = 30, generations: int = 30):
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.leaf_nodes = ['close', 'open', 'high', 'low', 'volume']
        self.operators = ['+', '-', '*', '/', 'log', 'abs']
        self.best_formulas = []
        
        self.data_dict = {
            'close': data['close'].values,
            'open': data['open'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'volume': data['volume'].values,
        }
    
    def random_tree(self, depth: int = 0, max_depth: int = 3) -> ExpressionNode:
        """生成随机表达式树"""
        if depth >= max_depth or np.random.random() < 0.3:
            if np.random.random() < 0.7:
                return ExpressionNode(value=np.random.choice(self.leaf_nodes))
            else:
                return ExpressionNode(value=np.random.uniform(-2, 2))
        else:
            op = np.random.choice(self.operators)
            left = self.random_tree(depth + 1, max_depth)
            
            if op in ['+', '-', '*', '/']:
                right = self.random_tree(depth + 1, max_depth)
                return ExpressionNode(op=op, left=left, right=right)
            else:
                return ExpressionNode(op=op, left=left)
    
    def fitness(self, tree: ExpressionNode) -> float:
        """计算适应度"""
        try:
            result = tree.evaluate(self.data_dict)
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return -1e6
            
            if np.std(result) < 1e-10:
                return -1e6
            
            result = (result - np.mean(result)) / (np.std(result) + 1e-10)
            
            price = self.data_dict['close']
            price_norm = (price - np.mean(price)) / (np.std(price) + 1e-10)
            
            correlation = np.corrcoef(result, price_norm)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            changes = np.abs(np.diff(result))
            volatility = np.mean(changes)
            
            fitness = abs(correlation) * 0.7 + volatility * 0.3
            
            return fitness
        
        except:
            return -1e6
    
    def evolve(self) -> List[str]:
        """进化过程"""
        print("[SymbolicRegression] Starting evolution...")
        
        population = [self.random_tree() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            fitness_scores = [self.fitness(tree) for tree in population]
            
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            
            if gen % 10 == 0:
                formula = population[best_idx].to_string()
                print(f"  Gen {gen:3d}: Best Fitness = {best_fitness:8.4f}")
            
            elite_indices = np.argsort(fitness_scores)[-max(5, self.population_size // 5):]
            elite = [population[i] for i in elite_indices]
            
            if best_fitness > 0.3:
                formula = population[best_idx].to_string()
                if formula not in [f[0] for f in self.best_formulas]:
                    self.best_formulas.append((formula, best_fitness))
            
            new_pop = elite.copy()
            for _ in range(self.population_size - len(elite)):
                new_pop.append(self.random_tree())
            
            population = new_pop
        
        self.best_formulas.sort(key=lambda x: x[1], reverse=True)
        best_formulas = [f[0] for f in self.best_formulas[:5]]
        
        print(f"[SymbolicRegression] SUCCESS: Discovered {len(best_formulas)} formulas")
        for i, formula in enumerate(best_formulas, 1):
            print(f"  Formula_{i}: {formula[:80]}")
        
        return best_formulas

# ====================================================================
# PART 4: 因子生成和验证
# ====================================================================

class FactorGenerator:
    """因子生成器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.factors = {}
        self.indicator_lib = IndicatorLibrary()
    
    def generate_standard_factors(self) -> Dict[str, np.ndarray]:
        """生成标准技术指标因子"""
        print("[FactorGenerator] Generating standard indicators...")
        
        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values
        volume = self.data['volume'].values
        
        factors = {
            'price_change': (close - self.data['open'].values) / (self.data['open'].values + 1e-10),
            'high_low_ratio': (high - low) / (close + 1e-10),
            
            'SMA_5': self.indicator_lib.sma(close, 5),
            'SMA_10': self.indicator_lib.sma(close, 10),
            'SMA_20': self.indicator_lib.sma(close, 20),
            
            'RSI_14': self.indicator_lib.rsi(close, 14),
            'RSI_7': self.indicator_lib.rsi(close, 7),
        }
        
        print(f"[FactorGenerator] Generated {len(factors)} standard factors")
        self.factors.update(factors)
        return factors
    
    def generate_formula_factors(self, formulas: List[str]) -> Dict[str, np.ndarray]:
        """根据公式生成因子"""
        print("[FactorGenerator] Generating formula factors...")
        
        data_dict = {
            'close': self.data['close'].values,
            'open': self.data['open'].values,
            'high': self.data['high'].values,
            'low': self.data['low'].values,
            'volume': self.data['volume'].values,
        }
        
        formula_factors = {}
        
        for i, formula in enumerate(formulas):
            try:
                node = eval(formula)  
                result = node.evaluate(data_dict)
                
                result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
                
                if np.std(result) > 1e-10:
                    result = (result - np.mean(result)) / (np.std(result) + 1e-10)
                
                formula_factors[f'Formula_{i+1}'] = result
                print(f"  Formula_{i+1}: Generated successfully")
                
            except Exception as e:
                print(f"  Formula_{i+1}: Error - {str(e)[:50]}")
        
        print(f"[FactorGenerator] Generated {len(formula_factors)} formula factors")
        self.factors.update(formula_factors)
        return formula_factors

# ====================================================================
# PART 5: 结果输出和分析
# ====================================================================

class ResultAnalyzer:
    """结果分析器"""
    
    @staticmethod
    def analyze_factors(factors: Dict[str, np.ndarray], prices: np.ndarray):
        """分析因子质量"""
        print("\n" + "="*80)
        print("FACTOR ANALYSIS RESULTS")
        print("="*80)
        
        results = {}
        
        for name, factor in factors.items():
            valid_idx = ~(np.isnan(factor) | np.isnan(prices))
            if np.sum(valid_idx) < 10:
                continue
            
            corr = np.corrcoef(factor[valid_idx], prices[valid_idx])[0, 1]
            if np.isnan(corr):
                corr = 0
            
            changes = np.abs(np.diff(factor))
            volatility = np.nanmean(changes)
            
            returns = np.diff(factor)
            sharpe = np.nanmean(returns) / (np.nanstd(returns) + 1e-10)
            
            results[name] = {
                'correlation': corr,
                'volatility': volatility,
                'sharpe': sharpe,
            }
            
            print(f"{name:20s} | Corr: {corr:7.4f} | Vol: {volatility:8.4f} | Sharpe: {sharpe:7.4f}")
        
        return results
    
    @staticmethod
    def save_factors(factors: Dict[str, np.ndarray], output_path: str = "/tmp/discovered_factors.pkl"):
        """保存因子"""
        print(f"\n[ResultAnalyzer] Saving factors to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(factors, f)
        
        print("[ResultAnalyzer] SUCCESS: Factors saved")
        return output_path
    
    @staticmethod
    def export_formulas(formulas: List[str], output_path: str = "/tmp/discovered_formulas.json"):
        """导出公式"""
        print(f"[ResultAnalyzer] Exporting formulas to {output_path}...")
        
        formulas_dict = {
            f'Formula_{i+1}': formula 
            for i, formula in enumerate(formulas)
        }
        
        with open(output_path, 'w') as f:
            json.dump(formulas_dict, f, indent=2)
        
        print("[ResultAnalyzer] SUCCESS: Formulas exported")
        print("\nDiscovered Formulas:")
        for name, formula in formulas_dict.items():
            print(f"\n{name}:")
            print(f"  {formula[:100]}")
        
        return output_path

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    """主执行流程"""
    
    print("="*80)
    print("CRYPTOCURRENCY FACTOR DISCOVERY SYSTEM")
    print("="*80)
    
    print("\n[STEP 1] Loading data...")
    loader = DataLoader()
    raw_data = loader.load_btc_15m()
    
    print("\n[STEP 2] Cleaning data...")
    cleaner = DataCleaner()
    clean_data = cleaner.clean(raw_data)
    
    print(f"Data shape: {clean_data.shape}")
    
    print("\n[STEP 3] Symbolic Regression - Discovering new indicators...")
    sr = SymbolicRegression(clean_data, population_size=30, generations=30)
    discovered_formulas = sr.evolve()
    
    print("\n[STEP 4] Generating factors...")
    
    fg = FactorGenerator(clean_data)
    standard_factors = fg.generate_standard_factors()
    formula_factors = fg.generate_formula_factors(discovered_formulas)
    
    print("\n[STEP 5] Analyzing factors...")
    analyzer = ResultAnalyzer()
    analysis = analyzer.analyze_factors(
        {**standard_factors, **formula_factors},
        clean_data['close'].values
    )
    
    print("\n[STEP 6] Saving results...")
    factors_path = analyzer.save_factors({**standard_factors, **formula_factors})
    formulas_path = analyzer.export_formulas(discovered_formulas)
    
    print("\n" + "="*80)
    print("SYSTEM SUMMARY")
    print("="*80)
    print(f"Input data: {len(clean_data)} K-line records")
    print(f"Standard factors: {len(standard_factors)}")
    print(f"Discovered formulas: {len(discovered_formulas)}")
    print(f"Total factors: {len(standard_factors) + len(formula_factors)}")
    print(f"\nFactors saved to: {factors_path}")
    print(f"Formulas saved to: {formulas_path}")
    print("="*80)
    
    return {
        'data': clean_data,
        'standard_factors': standard_factors,
        'formula_factors': formula_factors,
        'discovered_formulas': discovered_formulas,
        'analysis': analysis,
    }

if __name__ == "__main__":
    results = main()
