"""
因子回測和遺傳算法優化
基於已發現的因子，進行：
1. 單因子策略回測
2. 多因子組合優化
3. 信號生成
"""

import subprocess
import sys

print("[Setup] Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==2.1.3", "pandas==2.2.2"])

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FACTOR BACKTESTING & OPTIMIZATION SYSTEM")
print("="*80)

# ====================================================================
# PART 1: 讀取因子數據
# ====================================================================

print("\n[STEP 1] Load factor data...")

try:
    with open('/tmp/btc_factors.pkl', 'rb') as f:
        factors = pickle.load(f)
    
    with open('/tmp/btc_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    print(f"[Loader] ✓ Loaded {len(factors)} factors")
    print(f"[Loader] Top factors by correlation:")
    sorted_factors = sorted(analysis.items(), key=lambda x: abs(x[1]['corr']), reverse=True)
    for i, (name, metrics) in enumerate(sorted_factors[:5], 1):
        print(f"  {i}. {name:18s} Corr={metrics['corr']:7.4f}")
except Exception as e:
    print(f"[ERROR] {e}")
    raise

# ====================================================================
# PART 2: 單因子策略回測
# ====================================================================

print("\n[STEP 2] Single-factor backtesting...")
print("\n" + "="*80)
print("SINGLE FACTOR BACKTEST RESULTS")
print("="*80)

class SingleFactorBacktest:
    """單因子策略回測"""
    
    def __init__(self, factor, prices, threshold=0.5):
        self.factor = np.asarray(factor, dtype=float)
        self.factor = np.nan_to_num(self.factor, 0)
        self.prices = np.asarray(prices, dtype=float)
        
        # 確保長度一致
        min_len = min(len(self.factor), len(self.prices) - 1)
        self.factor = self.factor[:min_len]
        self.prices = self.prices[:min_len+1]
        
        self.returns = np.diff(np.log(self.prices))
        self.threshold = threshold
    
    def backtest(self):
        """回測"""
        # 標準化因子
        factor_norm = (self.factor - np.mean(self.factor)) / (np.std(self.factor) + 1e-10)
        
        # 生成信號：因子 > 閾值時做多
        signals = np.where(factor_norm > self.threshold, 1, -1)
        
        # 計算策略收益
        strategy_returns = signals * self.returns
        
        # 統計指標
        total_return = np.sum(strategy_returns)
        annual_return = total_return * 252 / len(strategy_returns) if len(strategy_returns) > 0 else 0
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
        max_dd = self._max_drawdown(strategy_returns)
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
        }
    
    @staticmethod
    def _max_drawdown(returns):
        """最大回撤"""
        cumsum = np.cumsum(returns)
        if len(cumsum) == 0:
            return 0
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / (running_max + 1e-10)
        return np.min(drawdown)

# 取一個可代表价格的因子
if 'close' in factors:
    prices = factors['close']
elif 'SMA_5' in factors:
    prices = factors['SMA_5']
else:
    prices = list(factors.values())[0]

backtest_results = {}
for name, factor in factors.items():
    try:
        bt = SingleFactorBacktest(factor, prices)
        result = bt.backtest()
        backtest_results[name] = result
        
        print(f"{name:18s} | Return: {result['annual_return']:7.4f} | Sharpe: {result['sharpe']:7.4f} | MaxDD: {result['max_dd']:7.4f} | WinRate: {result['win_rate']:.2%}")
    except Exception as e:
        print(f"{name:18s} | Error: {str(e)[:30]}")

# ====================================================================
# PART 3: 多因子組合優化（遺傳算法）
# ====================================================================

print("\n[STEP 3] Multi-factor optimization using genetic algorithm...")

class GeneticOptimizer:
    """遺傳算法優化"""
    
    def __init__(self, factors_dict, prices, pop_size=50, generations=30):
        self.factors_dict = factors_dict
        self.prices = np.asarray(prices, dtype=float)
        self.returns = np.diff(np.log(self.prices))
        self.pop_size = pop_size
        self.generations = generations
        self.factor_names = list(factors_dict.keys())
    
    def create_individual(self):
        """隨機創建個體（權重）"""
        weights = np.random.dirichlet(np.ones(len(self.factor_names)))
        return weights
    
    def fitness(self, weights):
        """計算適應度（Sharpe比）"""
        try:
            # 組合因子
            portfolio = np.zeros(min(len(self.returns), min([len(self.factors_dict[n]) for n in self.factor_names])))
            
            for i, name in enumerate(self.factor_names):
                factor = np.asarray(self.factors_dict[name], dtype=float)[:len(portfolio)]
                factor = np.nan_to_num(factor, 0)
                factor_norm = (factor - np.mean(factor)) / (np.std(factor) + 1e-10)
                portfolio += weights[i] * factor_norm
            
            # 標準化組合
            portfolio = (portfolio - np.mean(portfolio)) / (np.std(portfolio) + 1e-10)
            
            # 生成信號
            signals = np.where(portfolio > 0, 1, -1)
            
            # 計算收益
            strategy_returns = signals * self.returns[:len(signals)]
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
            
            return sharpe
        except:
            return -1e6
    
    def optimize(self):
        """優化"""
        print("[GeneticOptimizer] Starting evolution...")
        
        # 初始種群
        population = [self.create_individual() for _ in range(self.pop_size)]
        best_history = []
        
        for gen in range(self.generations):
            # 計算適應度
            fitness_scores = [self.fitness(ind) for ind in population]
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_history.append(best_fitness)
            
            if gen % 5 == 0:
                print(f"  Gen {gen:2d}: Best Sharpe = {best_fitness:8.4f}")
            
            # 精英選擇
            elite_count = max(5, self.pop_size // 5)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = [population[i] for i in elite_indices]
            
            # 交叉和變異
            new_pop = elite.copy()
            while len(new_pop) < self.pop_size:
                parent1, parent2 = np.random.choice(len(elite), 2, replace=False)
                child = elite[parent1] * 0.5 + elite[parent2] * 0.5
                child = child / (np.sum(child) + 1e-10)  # 標準化為概率分佈
                
                # 變異
                if np.random.random() < 0.1:
                    mutation = np.random.normal(0, 0.05, len(child))
                    child = child + mutation
                    child = np.maximum(child, 0)
                    child = child / (np.sum(child) + 1e-10)
                
                new_pop.append(child)
            
            population = new_pop
        
        # 返回最優解
        final_scores = [self.fitness(ind) for ind in population]
        best_idx = np.argmax(final_scores)
        return population[best_idx], max(final_scores)

optimizer = GeneticOptimizer(factors, prices, pop_size=50, generations=30)
best_weights, best_sharpe = optimizer.optimize()

print(f"\n[GeneticOptimizer] ✓ Optimization complete")
print(f"[GeneticOptimizer] Best Sharpe: {best_sharpe:.4f}")
print(f"\n[Optimal Factor Weights]:")
for i, (name, weight) in enumerate(zip(optimizer.factor_names, best_weights)):
    if weight > 0.01:
        print(f"  {i+1}. {name:18s}: {weight:7.4f} ({weight*100:6.2f}%)")

# ====================================================================
# PART 4: 生成交易信號
# ====================================================================

print("\n[STEP 4] Generate trading signals...")

# 用最優權重組合因子
min_len = min(len(optimizer.returns), min([len(factors[n]) for n in optimizer.factor_names]))
portfolio = np.zeros(min_len)

for i, name in enumerate(optimizer.factor_names):
    factor = np.asarray(factors[name], dtype=float)[:min_len]
    factor = np.nan_to_num(factor, 0)
    factor_norm = (factor - np.mean(factor)) / (np.std(factor) + 1e-10)
    portfolio += best_weights[i] * factor_norm

# 標準化
portfolio = (portfolio - np.mean(portfolio)) / (np.std(portfolio) + 1e-10)

# 生成信號
signals = np.where(portfolio > 0, 1, 0)  # 1=做多, 0=空倉

# 最後100根K線的信號
print(f"\n[Signal Generator] Last 20 signals (1=Long, 0=Flat):")
for i in range(max(0, len(signals)-20), len(signals)):
    print(f"  Bar {i:6d}: {signals[i]} | Portfolio score: {portfolio[i]:7.4f}")

print(f"\n[Signal Generator] Current signal: {'LONG' if signals[-1] == 1 else 'FLAT'}")
print(f"[Signal Generator] Current portfolio score: {portfolio[-1]:7.4f}")

# ====================================================================
# PART 5: 保存結果
# ====================================================================

print("\n[STEP 5] Save results...")

results = {
    'best_weights': {name: float(w) for name, w in zip(optimizer.factor_names, best_weights)},
    'best_sharpe': float(best_sharpe),
    'backtest_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in backtest_results.items()},
    'signals': signals.tolist()[-100:],  # 最後100個信號
    'portfolio_scores': portfolio.tolist()[-100:],
}

with open('/tmp/optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"[Save] ✓ Results -> /tmp/optimization_results.json")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)
print(f"Total factors analyzed:     {len(factors):4d}")
if len(backtest_results) > 0:
    print(f"Best single factor Sharpe:  {max([r['sharpe'] for r in backtest_results.values()]):7.4f}")
print(f"Optimal portfolio Sharpe:    {best_sharpe:7.4f}")
print(f"\nCurrent trading signal: {'LONG' if signals[-1] == 1 else 'FLAT'}")
print(f"Portfolio score: {portfolio[-1]:7.4f}")
print("="*80 + "\n")
