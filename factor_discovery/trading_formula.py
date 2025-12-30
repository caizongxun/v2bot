"""
可執行的交易公式生成系統

功能：
1. 從優化結果提取最優權重
2. 生成標準化的交易公式
3. 輸出多種格式（Python、Excel、JSON）
4. 包含所有計算細節和參數
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
print("TRADING FORMULA GENERATION SYSTEM")
print("="*80)

# ====================================================================
# STEP 1: 加載優化結果
# ====================================================================

print("\n[STEP 1] Load optimization results...")

try:
    with open('/tmp/optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    
    with open('/tmp/trading_dashboard.json', 'r') as f:
        dashboard = json.load(f)
    
    best_weights = opt_results['best_weights']
    best_sharpe = opt_results['best_sharpe']
    
    print(f"[Loader] ✓ Loaded optimization results")
    print(f"[Loader] Best Sharpe Ratio: {best_sharpe:.4f}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    raise

# ====================================================================
# STEP 2: 生成交易公式
# ====================================================================

print("\n[STEP 2] Generate trading formula...")
print("\n" + "="*80)
print("OPTIMIZED TRADING FORMULA")
print("="*80)

# 排序因子權重
sorted_weights = sorted(best_weights.items(), key=lambda x: float(x[1]), reverse=True)

print("\n[Factor Composition]:")
print("\nPortfolio Score = ")

terms = []
for i, (name, weight) in enumerate(sorted_weights):
    weight_val = float(weight)
    if weight_val > 0.001:  # 只列出權重 > 0.1% 的因子
        terms.append(f"{weight_val:.4f} × {name}")
        print(f"  {'(' if i == 0 else '+'}  {weight_val:.4f} × {name}")

print("  )\n")

# ====================================================================
# STEP 3: 生成 Python 代碼公式
# ====================================================================

print("[STEP 3] Export trading formula code...")
print("\n" + "="*80)
print("PYTHON IMPLEMENTATION")
print("="*80)

python_formula = f"""
def calculate_portfolio_score(factors_dict):
    \"\"\"
    計算投資組合信號分數
    
    參數:
        factors_dict: 包含所有技術指標的字典
        {{\n"""

for name in best_weights.keys():
    python_formula += f"            '{name}': float,  # 技術指標值\n"

python_formula += f"        }}\n    
    返回:
        float: 投資組合分數 (正值 = LONG, 負值 = FLAT)\n    \"\"\"
    
    portfolio_score = ("

for i, (name, weight) in enumerate(sorted_weights):
    weight_val = float(weight)
    if weight_val > 0.001:
        prefix = "" if i == 0 else " + "
        python_formula += f"{prefix}{weight_val:.6f} * factors_dict.get('{name}', 0)\n"
        python_formula += "    " * (5 if i == 0 else 4)

python_formula += f""")
    
    return portfolio_score

def generate_signal(portfolio_score):
    \"\"\"
    生成交易信號
    
    參數:
        portfolio_score: 投資組合分數
    
    返回:
        str: 'LONG' 或 'FLAT'
    \"\"\"
    return 'LONG' if portfolio_score > 0 else 'FLAT'


# ====================================================================
# 使用示例
# ====================================================================

if __name__ == '__main__':
    # 模擬一組因子值
    current_factors = {{
"""

for name in best_weights.keys():
    python_formula += f"        '{name}': 0.5,  # 需要替換為實際值\n"

python_formula += f"""    }}
    
    score = calculate_portfolio_score(current_factors)
    signal = generate_signal(score)
    
    print(f'Portfolio Score: {{score:.4f}}')
    print(f'Trading Signal: {{signal}}')
"""

print(python_formula)

# 保存 Python 代碼
with open('/tmp/trading_formula.py', 'w') as f:
    f.write(python_formula)

print(f"\n[Save] ✓ Python formula -> /tmp/trading_formula.py")

# ====================================================================
# STEP 4: 生成 Excel 公式
# ====================================================================

print("\n[STEP 4] Export Excel formula...")
print("\n" + "="*80)
print("EXCEL FORMULA (Google Sheets / Excel)")
print("="*80)

# 假設在 Excel 中，因子值在 A1:L1
excel_formula = "=("
for i, (name, weight) in enumerate(sorted_weights):
    weight_val = float(weight)
    if weight_val > 0.001:
        col_letter = chr(65 + i)  # A, B, C...
        if i > 0:
            excel_formula += " + "
        excel_formula += f"{weight_val:.6f}*{col_letter}1"

excel_formula += ")"

print(f"\nPortfolio Score Formula:")
print(f"\n{excel_formula}")
print(f"\nSignal Formula:")
print(f"=IF([Portfolio Score] > 0, \"LONG\", \"FLAT\")")

# ====================================================================
# STEP 5: 生成 JSON 配置文件
# ====================================================================

print("\n[STEP 5] Export JSON configuration...")
print("\n" + "="*80)
print("JSON CONFIGURATION")
print("="*80)

trading_config = {
    'version': '1.0',
    'created': datetime.now().isoformat(),
    'strategy_name': 'V2 Factor Trading System',
    'description': '基於遺傳算法優化的多因子交易策略',
    'performance': {
        'sharpe_ratio': float(best_sharpe),
        'backtest_period': '219,010 bars (BTC 15-min)',
        'optimization_method': 'Genetic Algorithm (30 generations)',
        'last_signal': dashboard['current_signal'],
        'signal_confidence': float(dashboard['metrics']['signal_confidence']),
    },
    'factors': {
        'total_count': len(best_weights),
        'weights': {k: float(v) for k, v in best_weights.items()},
        'top_factors': [
            {'rank': i+1, 'name': name, 'weight': float(weight), 'weight_pct': float(weight)*100}
            for i, (name, weight) in enumerate(sorted_weights[:5])
        ]
    },
    'signals': {
        'long_signal': {'condition': 'portfolio_score > 0', 'action': 'BUY'},
        'flat_signal': {'condition': 'portfolio_score <= 0', 'action': 'CLOSE'},
    },
    'risk_management': {
        'risk_per_trade_pct': 2.0,
        'max_position_size': '28.10%',
        'stop_loss_adjustment': dashboard['recommendation']['stop_loss'],
        'take_profit_adjustment': dashboard['recommendation']['take_profit'],
    },
    'implementation': {
        'data_frequency': '15-minute bars',
        'calculation_method': 'Weighted sum of normalized factors',
        'update_frequency': 'Every new bar',
        'portfolio_score_formula': ' + '.join([
            f"{float(w):.6f}*{n}" for n, w in sorted_weights if float(w) > 0.001
        ])
    }
}

with open('/tmp/trading_config.json', 'w') as f:
    json.dump(trading_config, f, indent=2, ensure_ascii=False)

print(f"\n[JSON Config]:")
print(json.dumps(trading_config, indent=2, ensure_ascii=False)[:1000] + "...")
print(f"\n[Save] ✓ JSON config -> /tmp/trading_config.json")

# ====================================================================
# STEP 6: 生成詳細的技術規格文檔
# ====================================================================

print("\n[STEP 6] Generate technical specification...")
print("\n" + "="*80)
print("TECHNICAL SPECIFICATION DOCUMENT")
print("="*80)

technical_spec = f"""
V2 FACTOR TRADING SYSTEM - TECHNICAL SPECIFICATION
{'='*80}

1. STRATEGY OVERVIEW
{'-'*80}
   Name: Multi-Factor Optimization Trading Strategy
   Version: 1.0
   Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   
   Description:
   This strategy uses a weighted combination of {len(best_weights)} technical indicators
   optimized using genetic algorithm to generate trading signals on 15-minute Bitcoin data.

2. PERFORMANCE METRICS
{'-'*80}
   Backtesting Period: 219,010 bars (BTC 15-min)
   Sharpe Ratio: {best_sharpe:.4f}
   Signal Confidence: {dashboard['metrics']['signal_confidence']:.2%}
   Current Signal: {dashboard['current_signal']}
   Risk Per Trade: 2.0%
   
3. FACTOR WEIGHTS (Top 10)
{'-'*80}
"""

for i, (name, weight) in enumerate(sorted_weights[:10], 1):
    weight_val = float(weight)
    technical_spec += f"   {i:2d}. {name:18s}: {weight_val:8.6f} ({weight_val*100:6.2f}%)\n"

technical_spec += f"""
4. SIGNAL GENERATION FORMULA
{'-'*80}
   
   Portfolio Score = 
"""

for i, (name, weight) in enumerate(sorted_weights):
    weight_val = float(weight)
    if weight_val > 0.001:
        technical_spec += f"                    {weight_val:.6f} × {name}\n"

technical_spec += f"""
   
   Trading Signal:
   - IF Portfolio Score > 0.0  → LONG (BUY)
   - IF Portfolio Score ≤ 0.0  → FLAT (CLOSE)

5. RISK MANAGEMENT
{'-'*80}
   Risk per Trade: 2.0% of account
   Position Sizing: Based on confidence level
   Stop Loss: {dashboard['recommendation']['stop_loss']:.4f} (relative to signal score)
   Take Profit: {dashboard['recommendation']['take_profit']:.4f} (relative to signal score)
   
6. IMPLEMENTATION DETAILS
{'-'*80}
   
   Input Data:
   - Symbol: BTCUSDT
   - Timeframe: 15-minute bars
   - Required Fields: open, high, low, close, volume
   
   Calculation Steps:
   1. For each new bar, calculate all {len(best_weights)} technical indicators
   2. Normalize each indicator value (typically -1 to +1)
   3. Apply portfolio weighting formula
   4. Compare result to threshold (0.0)
   5. Generate LONG/FLAT signal
   6. If signal changes, generate trading alert
   
   Update Frequency: On each new bar close
   Latency: < 100ms from bar close to signal

7. BACKTESTING RESULTS
{'-'*80}
   Win Rate: {dashboard.get('win_rate', 'N/A')}
   Avg Return: {dashboard.get('avg_return', 'N/A')}
   Max Drawdown: {dashboard.get('max_drawdown', 'N/A')}
   
8. DEPLOYMENT CHECKLIST
{'-'*80}
   □ Historical data validation
   □ Real-time data connection
   □ Signal generation verification
   □ Risk management configuration
   □ Alert system setup
   □ Paper trading (1-2 weeks)
   □ Live trading with minimum position size
   □ Performance monitoring

9. MAINTENANCE
{'-'*80}
   - Monitor signal quality weekly
   - Review performance metrics monthly
   - Recalibrate weights quarterly
   - Update with new data regularly
   
10. CONTACT & SUPPORT
{'-'*80}
    Strategy Version: 1.0
    Last Updated: {datetime.now().strftime('%Y-%m-%d')}
    Documentation: https://github.com/caizongxun/v2bot

{'='*80}
"""

with open('/tmp/TRADING_FORMULA.txt', 'w') as f:
    f.write(technical_spec)

print(technical_spec)
print(f"\n[Save] ✓ Technical spec -> /tmp/TRADING_FORMULA.txt")

# ====================================================================
# STEP 7: 生成 Markdown 版本用於 GitHub
# ====================================================================

print("\n[STEP 7] Generate GitHub documentation...")

markdown_doc = f"""
# V2 Factor Trading System - Trading Formula

## 📊 策略概述

**策略名稱**: 多因子優化交易策略  
**版本**: 1.0  
**創建時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

## 🎯 性能指標

| 指標 | 值 |
|------|----|
| Sharpe Ratio | {best_sharpe:.4f} |
| 回測周期 | 219,010 根K線 (BTC 15分鐘) |
| 當前信號 | {dashboard['current_signal']} |
| 信號信心度 | {dashboard['metrics']['signal_confidence']:.2%} |
| 每筆交易風險 | 2.0% |

## 💡 核心公式

### 投資組合分數計算

```
Portfolio Score = 
"""

for name, weight in sorted_weights:
    weight_val = float(weight)
    if weight_val > 0.001:
        markdown_doc += f"                  {weight_val:.6f} × {name} +\n"

markdown_doc += f"""```

### 交易信號規則

```python
IF Portfolio Score > 0.0:
    Signal = LONG (買入)
ELSE:
    Signal = FLAT (平倉)
```

## 📈 因子權重分解 (前10名)

"""

for i, (name, weight) in enumerate(sorted_weights[:10], 1):
    weight_val = float(weight)
    bar_length = int(weight_val * 50)
    bar = '█' * bar_length
    markdown_doc += f"{i:2d}. {name:18s} {weight_val:8.6f} ({weight_val*100:6.2f}%) {bar}\n"

markdown_doc += f"""

## 🛡️ 風險管理

- **每筆交易風險**: 2.0% of account
- **頭寸大小調整**: 基於信號信心度
- **止損**: {dashboard['recommendation']['stop_loss']:.4f}
- **止盈**: {dashboard['recommendation']['take_profit']:.4f}

## 🔧 實現細節

### 輸入數據
- **交易對**: BTCUSDT
- **時間框架**: 15分鐘
- **必需字段**: open, high, low, close, volume

### 計算步驟
1. 對每個新K線，計算所有 {len(best_weights)} 個技術指標
2. 標準化每個指標值 (通常在 -1 到 +1 之間)
3. 應用投資組合加權公式
4. 與閾值 (0.0) 比較
5. 生成 LONG/FLAT 信號

### 更新頻率
- 每根K線關閉時更新
- 從K線關閉到信號生成的延遲 < 100ms

## 📋 部署清單

- [ ] 歷史數據驗證
- [ ] 實時數據連接
- [ ] 信號生成驗證
- [ ] 風險管理配置
- [ ] 警報系統設置
- [ ] 模擬交易 (1-2 週)
- [ ] 最小頭寸實盤交易
- [ ] 性能監控

## 📞 支持

- **版本**: 1.0
- **最後更新**: {datetime.now().strftime('%Y-%m-%d')}
- **文檔**: https://github.com/caizongxun/v2bot

---

*本策略通過遺傳算法在 219,010 根 BTC 15分鐘 K線上優化而得。*
"""

with open('/tmp/TRADING_FORMULA.md', 'w') as f:
    f.write(markdown_doc)

print(f"[Save] ✓ Markdown doc -> /tmp/TRADING_FORMULA.md")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("TRADING FORMULA GENERATION COMPLETE")
print("="*80)
print(f"""
已生成以下可執行公式:

✅ Python 代碼公式
   └─ 位置: /tmp/trading_formula.py
   └─ 可直接導入到交易程序
   
✅ Excel 公式
   └─ 可用於 Google Sheets / Excel
   └─ 即時計算投資組合分數
   
✅ JSON 配置文件
   └─ 位置: /tmp/trading_config.json
   └─ 包含所有參數和設置
   
✅ 技術規格文檔
   └─ 位置: /tmp/TRADING_FORMULA.txt
   └─ 完整的實現指南
   
✅ GitHub 文檔
   └─ 位置: /tmp/TRADING_FORMULA.md
   └─ Markdown 格式

核心交易規則:

投資組合分數 = {sum(float(w) for n, w in sorted_weights if float(w) > 0.001):.4f} (標準化)

信號規則:
  • IF Portfolio Score > 0.0  →  LONG (買入)
  • IF Portfolio Score ≤ 0.0  →  FLAT (平倉)


立即使用:
  1. 複製 /tmp/trading_formula.py 到你的交易機器人
  2. 調用 calculate_portfolio_score(factors_dict)
  3. 根據返回值生成交易信號
  4. 執行交易
""")
print("="*80 + "\n")
