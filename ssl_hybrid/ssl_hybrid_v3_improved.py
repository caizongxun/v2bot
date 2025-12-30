#!/usr/bin/env python3
"""
SSL Hybrid V3 - Improved Implementation
Fixed array indexing and dimension issues
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        
        # Calculate deltas
        deltas = np.diff(data)
        gain = np.zeros(n)
        loss = np.zeros(n)
        
        gain[1:] = np.where(deltas > 0, deltas, 0)
        loss[1:] = np.where(deltas < 0, -deltas, 0)
        
        # Calculate averages
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # Set initial values to neutral
        
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD"""
        data = np.asarray(data, dtype=np.float64)
        ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        data = np.asarray(data, dtype=np.float64)
        ma = pd.Series(data).rolling(period).mean().values
        std = pd.Series(data).rolling(period).std().values
        
        # Fill NaN values
        ma = np.nan_to_num(ma, nan=data[0])
        std = np.nan_to_num(std, nan=np.std(data[:period]))
        
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        
        return upper, ma, lower
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        n = len(close)
        
        tr = np.zeros(n)
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr


class SSLHybridV3:
    """SSL Hybrid V3 - Improved version"""
    
    def __init__(self, close, high, low, volume, **kwargs):
        self.close = np.asarray(close, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.n = len(self.close)
        
        # Technical indicators
        self.rsi14 = TechnicalIndicators.rsi(self.close, 14)
        self.rsi7 = TechnicalIndicators.rsi(self.close, 7)
        self.macd_line, self.macd_signal, self.macd_hist = TechnicalIndicators.macd(self.close)
        self.bb_upper, self.bb_mid, self.bb_lower = TechnicalIndicators.bollinger_bands(self.close, 20, 2)
        self.atr = TechnicalIndicators.atr(self.high, self.low, self.close, 14)
        
        # Calculate signals
        self._calc_signals()
    
    def _calc_signals(self):
        """Calculate multi-timeframe signals"""
        self.buy_signal = np.zeros(self.n, dtype=bool)
        self.sell_signal = np.zeros(self.n, dtype=bool)
        
        for i in range(20, self.n):
            # Entry conditions
            rsi_oversold = self.rsi14[i] < 35
            rsi_midzone = 45 < self.rsi14[i] < 55
            macd_bullish = self.macd_hist[i] > 0 and self.macd_line[i] > self.macd_signal[i]
            price_above_bb = self.close[i] > self.bb_mid[i]
            
            vol_ma_20 = np.mean(self.volume[max(0, i-20):i])
            volume_spike = self.volume[i] > vol_ma_20 * 1.2
            
            if (rsi_oversold or rsi_midzone) and macd_bullish and volume_spike:
                self.buy_signal[i] = True
            
            rsi_overbought = self.rsi14[i] > 65
            macd_bearish = self.macd_hist[i] < 0 and self.macd_line[i] < self.macd_signal[i]
            price_below_bb = self.close[i] < self.bb_mid[i]
            
            if (rsi_overbought or rsi_midzone) and macd_bearish and volume_spike:
                self.sell_signal[i] = True


def extract_signals_v3(close, high, low, volume, indicator) -> List[Dict]:
    """Extract signals with improved multi-timeframe validation"""
    signals = []
    n = len(close)
    
    for i in range(n - 20):
        # Multi-timeframe validation
        if indicator.buy_signal[i]:
            ret_5 = (close[i+5] - close[i]) / close[i] if i+5 < n else 0
            ret_10 = (close[i+10] - close[i]) / close[i] if i+10 < n else 0
            ret_20 = (close[i+20] - close[i]) / close[i] if i+20 < n else 0
            
            # Composite label: at least 2/3 confirmations
            confirmations = sum([
                ret_5 > 0.003,
                ret_10 > 0.005,
                ret_20 > 0.010
            ])
            
            is_true = confirmations >= 2
            avg_return = np.mean([ret_5, ret_10, ret_20])
            
            signals.append({
                'index': i,
                'type': 'BUY',
                'price': float(close[i]),
                'is_true': is_true,
                'return_5': float(ret_5),
                'return_10': float(ret_10),
                'return_20': float(ret_20),
                'avg_return': float(avg_return),
                'confirmations': confirmations,
                'volume': float(volume[i])
            })
        
        elif indicator.sell_signal[i]:
            ret_5 = (close[i+5] - close[i]) / close[i] if i+5 < n else 0
            ret_10 = (close[i+10] - close[i]) / close[i] if i+10 < n else 0
            ret_20 = (close[i+20] - close[i]) / close[i] if i+20 < n else 0
            
            confirmations = sum([
                ret_5 < -0.003,
                ret_10 < -0.005,
                ret_20 < -0.010
            ])
            
            is_true = confirmations >= 2
            avg_return = np.mean([ret_5, ret_10, ret_20])
            
            signals.append({
                'index': i,
                'type': 'SELL',
                'price': float(close[i]),
                'is_true': is_true,
                'return_5': float(ret_5),
                'return_10': float(ret_10),
                'return_20': float(ret_20),
                'avg_return': float(avg_return),
                'confirmations': confirmations,
                'volume': float(volume[i])
            })
    
    return signals


def extract_features_v3(signal: Dict, close, high, low, volume, indicator) -> Dict:
    """Extract improved features"""
    idx = signal['index']
    features = {}
    
    # Safety check
    if idx < 20 or idx >= len(indicator.rsi14):
        return {}
    
    # 1. RSI features
    features['rsi14'] = float(indicator.rsi14[idx]) / 100.0  # Normalize to 0-1
    features['rsi14_from_neutral'] = abs(float(indicator.rsi14[idx]) - 50) / 50  # Distance from 50
    features['rsi_trend'] = (float(indicator.rsi14[idx]) - float(indicator.rsi14[max(0, idx-5)])) / 100
    
    # 2. MACD features
    macd_price_ratio = float(indicator.macd_hist[idx]) / (signal['price'] + 1e-8)
    features['macd_hist'] = np.clip(macd_price_ratio * 100, -1, 1)
    features['macd_bullish'] = 1.0 if float(indicator.macd_hist[idx]) > 0 else 0.0
    
    macd_signal_ratio = (float(indicator.macd_line[idx]) - float(indicator.macd_signal[idx])) / (abs(float(indicator.macd_signal[idx])) + 1e-8)
    features['macd_signal_dist'] = np.clip(macd_signal_ratio, -1, 1)
    
    # 3. Bollinger Bands features
    bb_position = (signal['price'] - float(indicator.bb_lower[idx])) / (float(indicator.bb_upper[idx]) - float(indicator.bb_lower[idx]) + 1e-8)
    features['bb_position'] = np.clip(bb_position, 0, 1)
    features['bb_distance_mid'] = (signal['price'] - float(indicator.bb_mid[idx])) / (float(indicator.atr[idx]) + 1e-8)
    
    # 4. Volatility features
    vol_20 = np.std(close[max(0, idx-20):idx])
    mean_20 = np.mean(close[max(0, idx-20):idx])
    features['volatility'] = np.clip((vol_20 / (mean_20 + 1e-8)) * 100, 0, 5) / 5
    features['atr_ratio'] = np.clip(float(indicator.atr[idx]) / (mean_20 + 1e-8) * 100, 0, 5) / 5
    
    # 5. Volume features
    vol_ma = np.mean(volume[max(0, idx-20):idx])
    features['volume_ratio'] = np.clip(signal['volume'] / (vol_ma + 1e-8), 0, 5) / 5
    
    # 6. Momentum features
    if idx >= 5:
        mom_5 = (close[idx] - close[idx-5]) / close[idx-5]
        features['momentum_5'] = np.clip(mom_5 * 100, -2, 2) / 2
    else:
        features['momentum_5'] = 0.0
    
    if idx >= 10:
        mom_10 = (close[idx] - close[idx-10]) / close[idx-10]
        features['momentum_10'] = np.clip(mom_10 * 100, -3, 3) / 3
    else:
        features['momentum_10'] = 0.0
    
    # 7. Price action features
    if idx >= 50:
        min_50 = np.min(low[idx-50:idx])
        max_50 = np.max(high[idx-50:idx])
        features['price_range_position'] = (signal['price'] - min_50) / (max_50 - min_50 + 1e-8)
    else:
        features['price_range_position'] = 0.5
    
    # 8. Signal confidence
    features['multi_tf_confirmations'] = signal['confirmations'] / 3.0
    features['avg_return_strength'] = np.clip(abs(signal['avg_return']) * 100, 0, 2) / 2
    features['signal_type'] = 1.0 if signal['type'] == 'BUY' else 0.0
    
    return features


if __name__ == "__main__":
    print("SSL Hybrid V3 Implementation - Ready for use")
