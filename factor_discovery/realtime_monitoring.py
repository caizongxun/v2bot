"""
實時信號監控和風險管理面板

功能：
1. 實時信號監控
2. 風險指標計算
3. 績效追蹤
4. 信號換向檢測
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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REALTIME TRADING SIGNAL MONITORING & RISK MANAGEMENT")
print("="*80)

# ====================================================================
# STEP 1: 載入優化結果
# ====================================================================

print("\n[STEP 1] Load optimization results...")

try:
    with open('/tmp/optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    
    with open('/tmp/btc_factors.pkl', 'rb') as f:
        factors = pickle.load(f)
    
    best_weights = opt_results['best_weights']
    signals = np.array(opt_results['signals'])
    portfolio_scores = np.array(opt_results['portfolio_scores'])
    
    print(f"[Loader] ✓ Loaded optimization results")
    print(f"[Loader] Best portfolio Sharpe: {opt_results['best_sharpe']:.4f}")
    print(f"[Loader] Current signal: {'LONG' if signals[-1] == 1 else 'FLAT'}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    raise

# ====================================================================
# STEP 2: 計算風險指標
# ====================================================================

print("\n[STEP 2] Calculate risk metrics...")
print("\n" + "="*80)
print("RISK METRICS")
print("="*80)

# 直接計算，不使用類
signals = np.asarray(signals, dtype=float)
portfolio_scores = np.asarray(portfolio_scores, dtype=float)

# 1. 計算信號變化
signal_changes = np.abs(np.diff(signals))
signal_frequency = np.sum(signal_changes) / len(signal_changes) if len(signal_changes) > 0 else 0

# 2. 計算分數波動
score_returns = np.diff(portfolio_scores)
score_volatility = np.std(score_returns) if len(score_returns) > 0 else 0
score_trend = np.mean(score_returns[-20:]) if len(score_returns) >= 20 else 0

# 3. 計算連續信號長度
current_signal = signals[-1]
consecutive_bars = 1
for i in range(len(signals)-2, -1, -1):
    if signals[i] == current_signal:
        consecutive_bars += 1
    else:
        break

# 4. 計算信號持續性信心度
pers_confidence = min(consecutive_bars / 5.0, 1.0)

# 5. 計算分數強度
score_strength = abs(portfolio_scores[-1]) / (np.max(np.abs(portfolio_scores)) + 1e-10)

# 6. 總體信心度
overall_confidence = pers_confidence * 0.4 + score_strength * 0.6

# 7. 頭寸大小
position_size = overall_confidence

# 8. 計算最大連續虧損
long_periods = np.where(signals == 1)[0]
short_periods = np.where(signals == 0)[0]

max_drawdown_long = 0
max_drawdown_short = 0

if len(long_periods) > 1:
    long_returns = score_returns[long_periods[:-1]]
    cumsum = np.cumsum(long_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - running_max) / (np.abs(running_max) + 1e-10)
    max_drawdown_long = np.min(drawdown) if len(drawdown) > 0 else 0

if len(short_periods) > 1:
    short_returns = score_returns[short_periods[:-1]]
    cumsum = np.cumsum(short_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - running_max) / (np.abs(running_max) + 1e-10)
    max_drawdown_short = np.min(drawdown) if len(drawdown) > 0 else 0

# 整理成 dict
metrics = {
    'current_signal': 'LONG' if current_signal == 1 else 'FLAT',
    'consecutive_bars': int(consecutive_bars),
    'signal_frequency': float(signal_frequency),
    'signal_persistence_confidence': float(pers_confidence),
    'score_strength': float(score_strength),
    'overall_confidence': float(overall_confidence),
    'portfolio_score': float(portfolio_scores[-1]),
    'score_volatility': float(score_volatility),
    'score_trend_20b': float(score_trend),
    'position_size': float(position_size),
    'max_drawdown_long': float(max_drawdown_long),
    'max_drawdown_short': float(max_drawdown_short),
}

print(f"Current Signal:              {metrics['current_signal']}")
print(f"Consecutive bars:           {metrics['consecutive_bars']} bars")
print(f"Signal frequency:           {metrics['signal_frequency']:.4f} (changes per bar)")
print(f"\nSignal Confidence:          {metrics['overall_confidence']:.2%}")
print(f"  ├─ Signal persistence:    {metrics['signal_persistence_confidence']:.2%}")
print(f"  └─ Score strength:        {metrics['score_strength']:.2%}")
print(f"\nPortfolio Analysis:")
print(f"  ├─ Current score:         {metrics['portfolio_score']:7.4f}")
print(f"  ├─ Score volatility:      {metrics['score_volatility']:.4f}")
print(f"  └─ 20-bar trend:          {metrics['score_trend_20b']:7.4f}")
print(f"\nRisk Management:")
print(f"  ├─ Position size:         {metrics['position_size']:.2%}")
print(f"  ├─ Max DD (long):         {metrics['max_drawdown_long']:.4f}")
print(f"  └─ Max DD (short):        {metrics['max_drawdown_short']:.4f}")

# ====================================================================
# STEP 3: 因子貢獻分析
# ====================================================================

print("\n[STEP 3] Factor contribution analysis...")
print("\n" + "="*80)
print("FACTOR WEIGHTS & CONTRIBUTIONS")
print("="*80)

sorted_weights = sorted(best_weights.items(), key=lambda x: float(x[1]), reverse=True)
print("\nOptimal factor weights:")
cumulative = 0
for i, (name, weight) in enumerate(sorted_weights, 1):
    weight = float(weight)
    cumulative += weight
    bar_length = int(weight * 50)
    bar = '█' * bar_length
    print(f"  {i:2d}. {name:18s} {weight:7.4f} {weight*100:6.2f}% {bar}")
    if cumulative >= 0.95:
        remaining_factors = len(sorted_weights) - i
        if remaining_factors > 0:
            print(f"      ... {remaining_factors} more factors (combined {1-cumulative:.2%})")
        break

# ====================================================================
# STEP 4: 信號轉換警告
# ====================================================================

print("\n[STEP 4] Signal transition analysis...")
print("\n" + "="*80)
print("SIGNAL TRANSITION ALERTS")
print("="*80)

# 檢測最近的信號變化
last_changes = []
for i in range(min(5, len(signals)-1), 0, -1):
    if signals[-i] != signals[-i-1]:
        signal_before = 'LONG' if signals[-i-1] == 1 else 'FLAT'
        signal_after = 'LONG' if signals[-i] == 1 else 'FLAT'
        score_before = portfolio_scores[-i-1]
        score_after = portfolio_scores[-i]
        last_changes.append({
            'bars_ago': i,
            'from': signal_before,
            'to': signal_after,
            'score_change': float(score_after - score_before)
        })

if len(last_changes) > 0:
    print(f"\nRecent signal transitions:")
    for change in last_changes:
        print(f"  {change['bars_ago']:3d} bars ago: {change['from']:4s} → {change['to']:4s} (Δ score: {change['score_change']:7.4f})")
else:
    print("\nNo recent signal transitions.")

# 警告條件
print(f"\n[Risk Alerts]:")
alerts = []

if metrics['overall_confidence'] < 0.3:
    alerts.append("⚠ Low signal confidence (< 30%)")

if metrics['score_volatility'] > 1.0:
    alerts.append("⚠ High portfolio score volatility")

if metrics['signal_frequency'] > 0.1:
    alerts.append("⚠ Frequent signal changes (whipsaw risk)")

if metrics['consecutive_bars'] == 1:
    alerts.append("⚠ Signal just changed (use caution)")

if abs(metrics['portfolio_score']) < 0.3:
    alerts.append("⚠ Weak signal strength (near zero)")

if len(alerts) == 0:
    print("  ✓ No active alerts")
else:
    for alert in alerts:
        print(f"  {alert}")

# ====================================================================
# STEP 5: 績效追蹤
# ====================================================================

print("\n[STEP 5] Performance tracking...")
print("\n" + "="*80)
print("SIGNAL PERFORMANCE WINDOW (Last 100 bars)")
print("="*80)

# 計算最近100根bar的信號表現
recent_signals = signals[-100:]
recent_scores = portfolio_scores[-100:]

long_periods = np.where(recent_signals == 1)[0]
flat_periods = np.where(recent_signals == 0)[0]

long_return = np.sum(np.diff(recent_scores[long_periods])) if len(long_periods) > 1 else 0
flat_return = np.sum(np.diff(recent_scores[flat_periods])) if len(flat_periods) > 1 else 0

long_periods_pct = len(long_periods) / len(recent_signals) * 100
flat_periods_pct = len(flat_periods) / len(recent_signals) * 100

print(f"\nLast 100 bars performance:")
print(f"  LONG periods:  {long_periods_pct:6.2f}% ({len(long_periods):3d} bars) → Return: {long_return:7.4f}")
print(f"  FLAT periods:  {flat_periods_pct:6.2f}% ({len(flat_periods):3d} bars) → Return: {flat_return:7.4f}")
print(f"\n  Total return:  {long_return + flat_return:7.4f}")
print(f"  Avg per bar:   {(long_return + flat_return) / 100:7.4f}")

# ====================================================================
# STEP 6: 交易建議
# ====================================================================

print("\n[STEP 6] Trading recommendations...")
print("\n" + "="*80)
print("ACTIONABLE TRADING SIGNALS")
print("="*80)

recommendation = {}

if metrics['current_signal'] == 'LONG':
    recommendation['action'] = '🟢 LONG'
    recommendation['description'] = 'Factor composite is bullish'
else:
    recommendation['action'] = '⚫ FLAT'
    recommendation['description'] = 'Reduce exposure or stay on sideline'

recommendation['confidence'] = metrics['overall_confidence']
recommendation['position_size'] = metrics['position_size']
recommendation['stop_loss'] = metrics['portfolio_score'] - 1.0
recommendation['take_profit'] = metrics['portfolio_score'] + 1.5

print(f"\nPrimary Action:  {recommendation['action']}")
print(f"Rationale:       {recommendation['description']}")
print(f"Confidence:      {recommendation['confidence']:.2%}")
print(f"Suggested size:  {recommendation['position_size']:.2%}")
print(f"\nRisk parameters:")
print(f"  Stop Loss:     {recommendation['stop_loss']:.4f}")
print(f"  Take Profit:   {recommendation['take_profit']:.4f}")

# ====================================================================
# STEP 7: 儲存監控面板
# ====================================================================

print("\n[STEP 7] Save monitoring dashboard...")

dashboard = {
    'timestamp': datetime.now().isoformat(),
    'current_signal': metrics['current_signal'],
    'metrics': {
        'signal_confidence': metrics['overall_confidence'],
        'portfolio_score': metrics['portfolio_score'],
        'score_volatility': metrics['score_volatility'],
        'signal_frequency': metrics['signal_frequency'],
        'consecutive_bars': metrics['consecutive_bars'],
    },
    'factor_weights': best_weights,
    'recommendation': {
        'action': recommendation['action'],
        'confidence': recommendation['confidence'],
        'position_size': recommendation['position_size'],
        'stop_loss': recommendation['stop_loss'],
        'take_profit': recommendation['take_profit'],
    },
    'alerts': alerts,
    'recent_performance': {
        'long_return': float(long_return),
        'flat_return': float(flat_return),
        'total_return': float(long_return + flat_return),
    }
}

with open('/tmp/trading_dashboard.json', 'w') as f:
    json.dump(dashboard, f, indent=2, ensure_ascii=False)

print(f"[Save] ✓ Dashboard -> /tmp/trading_dashboard.json")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*80)
print("SYSTEM COMPLETE - READY FOR TRADING")
print("="*80)
print(f"\nTimestamp:                 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Signal:                    {metrics['current_signal']}")
print(f"Confidence:                {metrics['overall_confidence']:.2%}")
print(f"Position sizing:           {metrics['position_size']:.2%} of capital")
print(f"Suggested action:          {recommendation['action']}")
print(f"\nNext steps:")
if len(alerts) > 0:
    print(f"  1. Review alerts: {len(alerts)} active")
else:
    print(f"  1. No alerts - system healthy")
print(f"  2. Monitor signal changes in real-time")
print(f"  3. Adjust position size based on confidence")
print(f"  4. Use suggested stop/take-profit levels")
print("="*80 + "\n")
