#!/usr/bin/env python3
"""
SSL Hybrid V6 - Fixed Implementation
All moving average types implemented and tested
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MA:
    """Simple Moving Average implementations"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i < period:
                result[i] = np.mean(data[:i+1])
            else:
                result[i] = np.mean(data[i-period+1:i+1])
        return result
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        result = np.zeros_like(data)
        alpha = 2 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    @staticmethod
    def dema(data, period):
        """Double EMA"""
        ema1 = MA.ema(data, period)
        ema2 = MA.ema(ema1, period)
        return 2 * ema1 - ema2
    
    @staticmethod
    def tema(data, period):
        """Triple EMA"""
        ema1 = MA.ema(data, period)
        ema2 = MA.ema(ema1, period)
        ema3 = MA.ema(ema2, period)
        return 3 * (ema1 - ema2) + ema3
    
    @staticmethod
    def wma(data, period):
        """Weighted Moving Average"""
        result = np.zeros_like(data)
        weights = np.arange(1, period + 1)
        weight_sum = weights.sum()
        
        for i in range(len(data)):
            if i < period:
                w = weights[-(i+1):]
                result[i] = np.sum(data[:i+1] * w) / w.sum()
            else:
                result[i] = np.sum(data[i-period+1:i+1] * weights) / weight_sum
        return result
    
    @staticmethod
    def hma(data, period):
        """Hull Moving Average"""
        wma1 = MA.wma(data, period // 2)
        wma2 = MA.wma(data, period)
        raw = 2 * wma1 - wma2
        sqrt_period = int(np.sqrt(period))
        return MA.wma(raw, sqrt_period)
    
    @staticmethod
    def jma(data, period, phase=3, power=1):
        """Jurik Moving Average"""
        data = np.asarray(data, dtype=np.float64)
        result = np.zeros_like(data)
        
        phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5
        beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
        alpha = np.power(beta, power)
        
        e0 = data[0]
        e1 = 0.0
        e2 = 0.0
        result[0] = data[0]
        
        for i in range(1, len(data)):
            e0 = (1 - alpha) * data[i] + alpha * e0
            e1 = (data[i] - e0) * (1 - beta) + beta * e1
            e2 = (e0 + phase_ratio * e1 - result[i-1]) * ((1 - alpha) ** 2) + (alpha ** 2) * e2
            result[i] = e2 + result[i-1]
        
        return result
    
    @staticmethod
    def get(ma_type: str, data, period, **kwargs):
        """Get MA by type"""
        data = np.asarray(data, dtype=np.float64)
        
        if ma_type == "SMA":
            return MA.sma(data, period)
        elif ma_type == "EMA":
            return MA.ema(data, period)
        elif ma_type == "DEMA":
            return MA.dema(data, period)
        elif ma_type == "TEMA":
            return MA.tema(data, period)
        elif ma_type == "WMA":
            return MA.wma(data, period)
        elif ma_type == "HMA":
            return MA.hma(data, period)
        elif ma_type == "JMA":
            phase = kwargs.get('phase', 3)
            power = kwargs.get('power', 1)
            return MA.jma(data, period, phase, power)
        else:
            return MA.sma(data, period)


class SSLHybrid:
    """SSL Hybrid V6 Indicator"""
    
    def __init__(self, close, high, low, volume, 
                 baseline_type="HMA", baseline_len=60,
                 ssl2_type="JMA", ssl2_len=5,
                 exit_type="HMA", exit_len=15,
                 atr_len=14, atr_mult=1.0,
                 channel_mult=0.2, atr_crit=0.9):
        
        self.close = np.asarray(close, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.n = len(self.close)
        
        # Calculate ATR
        self._calc_atr(atr_len, atr_mult)
        
        # Baseline
        self.baseline = MA.get(baseline_type, self.close, baseline_len)
        range_ma = MA.ema(self.high - self.low, baseline_len)
        self.upper_channel = self.baseline + range_ma * channel_mult
        self.lower_channel = self.baseline - range_ma * channel_mult
        
        # SSL1
        ema_high = MA.get(baseline_type, self.high, baseline_len)
        ema_low = MA.get(baseline_type, self.low, baseline_len)
        self._calc_ssl(self.close, ema_high, ema_low)
        
        # SSL2
        ma_high = MA.get(ssl2_type, self.high, ssl2_len, phase=3, power=1)
        ma_low = MA.get(ssl2_type, self.low, ssl2_len, phase=3, power=1)
        self._calc_ssl2(ma_high, ma_low, atr_crit)
        
        # Exit
        exit_high = MA.get(exit_type, self.high, exit_len)
        exit_low = MA.get(exit_type, self.low, exit_len)
        self._calc_exit(exit_high, exit_low)
        
        # Signals
        self._calc_signals()
    
    def _calc_atr(self, period, mult):
        """Calculate ATR"""
        tr = np.zeros(self.n)
        for i in range(1, self.n):
            tr1 = self.high[i] - self.low[i]
            tr2 = abs(self.high[i] - self.close[i-1])
            tr3 = abs(self.low[i] - self.close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        self.atr = MA.ema(tr, period)
        self.upper_band = self.atr * mult + self.close
        self.lower_band = self.close - self.atr * mult
    
    def _calc_ssl(self, close, ema_high, ema_low):
        """Calculate SSL1"""
        hlv = np.zeros(self.n)
        ssl = np.zeros(self.n)
        
        for i in range(self.n):
            if close[i] > ema_high[i]:
                hlv[i] = 1
            elif close[i] < ema_low[i]:
                hlv[i] = -1
            else:
                hlv[i] = hlv[i-1] if i > 0 else 0
            
            ssl[i] = ema_high[i] if hlv[i] < 0 else ema_low[i]
        
        self.ssl1 = ssl
        self.ssl1_dir = hlv
    
    def _calc_ssl2(self, ma_high, ma_low, atr_crit):
        """Calculate SSL2 and entry signals"""
        hlv2 = np.zeros(self.n)
        ssl2 = np.zeros(self.n)
        
        for i in range(self.n):
            if self.close[i] > ma_high[i]:
                hlv2[i] = 1
            elif self.close[i] < ma_low[i]:
                hlv2[i] = -1
            else:
                hlv2[i] = hlv2[i-1] if i > 0 else 0
            
            ssl2[i] = ma_high[i] if hlv2[i] < 0 else ma_low[i]
        
        self.ssl2 = ssl2
        self.ssl2_dir = hlv2
        
        # Entry conditions
        upper_half = self.atr * atr_crit + self.close
        lower_half = self.close - self.atr * atr_crit
        
        buy_inatr = lower_half < ssl2
        buy_cont = (self.close > self.baseline) & (self.close > ssl2)
        self.buy_atr = buy_inatr & buy_cont
        
        sell_inatr = upper_half > ssl2
        sell_cont = (self.close < self.baseline) & (self.close < ssl2)
        self.sell_atr = sell_inatr & sell_cont
    
    def _calc_exit(self, exit_high, exit_low):
        """Calculate exit levels"""
        hlv3 = np.zeros(self.n)
        ssl_exit = np.zeros(self.n)
        
        for i in range(self.n):
            if self.close[i] > exit_high[i]:
                hlv3[i] = 1
            elif self.close[i] < exit_low[i]:
                hlv3[i] = -1
            else:
                hlv3[i] = hlv3[i-1] if i > 0 else 0
            
            ssl_exit[i] = exit_high[i] if hlv3[i] < 0 else exit_low[i]
        
        self.ssl_exit = ssl_exit
    
    def _calc_signals(self):
        """Calculate entry signals"""
        self.buy_signal = np.zeros(self.n, dtype=bool)
        self.sell_signal = np.zeros(self.n, dtype=bool)
        
        for i in range(1, self.n):
            if self.buy_atr[i] and not self.buy_atr[i-1]:
                self.buy_signal[i] = True
            if self.sell_atr[i] and not self.sell_atr[i-1]:
                self.sell_signal[i] = True


def extract_signals(close, high, low, volume, indicator, lookforward=5) -> List[Dict]:
    """Extract all signals with labels"""
    signals = []
    n = len(close)
    
    for i in range(n - lookforward):
        # BUY signal
        if indicator.buy_signal[i]:
            future_return = (close[i + lookforward] - close[i]) / close[i]
            is_true = future_return > 0.005
            
            signals.append({
                'index': i,
                'type': 'BUY',
                'price': float(close[i]),
                'is_true': is_true,
                'return': float(future_return),
                'atr': float(indicator.atr[i]),
                'baseline': float(indicator.baseline[i]),
                'ssl2': float(indicator.ssl2[i]),
                'volume': float(volume[i])
            })
        
        # SELL signal
        elif indicator.sell_signal[i]:
            future_return = (close[i + lookforward] - close[i]) / close[i]
            is_true = future_return < -0.005
            
            signals.append({
                'index': i,
                'type': 'SELL',
                'price': float(close[i]),
                'is_true': is_true,
                'return': float(future_return),
                'atr': float(indicator.atr[i]),
                'baseline': float(indicator.baseline[i]),
                'ssl2': float(indicator.ssl2[i]),
                'volume': float(volume[i])
            })
    
    return signals


def extract_features(signal: Dict, close, high, low, volume, indicator, lookback=40) -> Dict:
    """Extract features from signal"""
    idx = signal['index']
    features = {}
    
    # Basic signal context
    features['price'] = signal['price']
    features['type_buy'] = 1.0 if signal['type'] == 'BUY' else 0.0
    
    # Distance from baseline
    dist = abs(signal['price'] - signal['baseline']) / (signal['atr'] + 1e-8)
    features['distance_from_baseline'] = min(dist, 5.0) / 5.0
    
    # Price position relative to SSL2
    ssl_dist = abs(signal['price'] - signal['ssl2']) / (signal['atr'] + 1e-8)
    features['distance_from_ssl2'] = min(ssl_dist, 3.0) / 3.0
    
    # Volume
    vol_ma = np.mean(volume[max(0, idx-20):idx])
    vol_ratio = signal['volume'] / (vol_ma + 1e-8)
    features['volume_ratio'] = min(vol_ratio, 5.0) / 5.0
    
    # Momentum
    if idx >= 5:
        mom5 = (close[idx] - close[idx-5]) / close[idx-5]
        features['momentum_5'] = np.clip(mom5 * 100, -1, 1)
    else:
        features['momentum_5'] = 0.0
    
    if idx >= 20:
        mom20 = (close[idx] - close[idx-20]) / close[idx-20]
        features['momentum_20'] = np.clip(mom20 * 100, -2, 2) / 2
    else:
        features['momentum_20'] = 0.0
    
    # Volatility
    if idx >= 20:
        std20 = np.std(close[idx-20:idx])
        mean20 = np.mean(close[idx-20:idx])
        vol = std20 / (mean20 + 1e-8)
        features['volatility'] = min(vol * 10, 1.0)
    else:
        features['volatility'] = 0.0
    
    # Trend
    if idx >= lookback:
        trend_count = sum(1 for j in range(idx-lookback+1, idx) if close[j] > close[j-1])
        features['trend_strength'] = trend_count / lookback
    else:
        features['trend_strength'] = 0.5
    
    # Price levels
    if idx >= 50:
        min_50 = np.min(low[idx-50:idx])
        max_50 = np.max(high[idx-50:idx])
        if max_50 > min_50:
            features['price_position'] = (close[idx] - min_50) / (max_50 - min_50)
        else:
            features['price_position'] = 0.5
    else:
        features['price_position'] = 0.5
    
    return features


if __name__ == "__main__":
    print("SSL Hybrid Fixed Implementation - Ready for use")
