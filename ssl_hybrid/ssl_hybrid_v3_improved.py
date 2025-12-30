#!/usr/bin/env python3
"""
SSL Hybrid V3 - Improved Implementation
Complete rewrite with proper indexing
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
        """Relative Strength Index - Fixed version"""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        rsi_result = np.zeros(n)
        
        if n < period + 1:
            return rsi_result
        
        # Use pandas for simplicity and correctness
        df = pd.DataFrame({'close': data})
        df['delta'] = df['close'].diff()
        df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
        df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)
        
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        
        # Forward fill from the first valid value
        for i in range(period, n):
            if i == period:
                df.loc[i, 'avg_gain'] = df['gain'].iloc[1:period+1].mean()
                df.loc[i, 'avg_loss'] = df['loss'].iloc[1:period+1].mean()
            else:
                df.loc[i, 'avg_gain'] = (df.loc[i-1, 'avg_gain'] * (period - 1) + df.loc[i, 'gain']) / period
                df.loc[i, 'avg_loss'] = (df.loc[i-1, 'avg_loss'] * (period - 1) + df.loc[i, 'loss']) / period
        
        rs = df['avg_gain'] / (df['avg_loss'] + 1e-10)
        rsi_result = 100 - (100 / (1 + rs))
        rsi_result[:period] = 50  # Set initial to neutral
        
        return rsi_result.values
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD"""
        data = np.asarray(data, dtype=np.float64)
        try:
            ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean().values
            ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean().values
            macd_line = ema_fast - ema_slow
            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except:
            n = len(data)
            return np.zeros(n), np.zeros(n), np.zeros(n)
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        data = np.asarray(data, dtype=np.float64)
        try:
            s = pd.Series(data)
            ma = s.rolling(window=period).mean().values
            std = s.rolling(window=period).std().values
            
            # Forward fill NaN
            ma = pd.Series(ma).fillna(method='bfill').fillna(method='ffill').values
            std = pd.Series(std).fillna(std[period] if period < len(std) else 0.1).fillna(0.1).values
            
            upper = ma + std_dev * std
            lower = ma - std_dev * std
            return upper, ma, lower
        except:
            n = len(data)
            return data.copy(), data.copy(), data.copy()
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        n = len(close)
        
        tr = np.zeros(n)
        try:
            for i in range(1, n):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr[i] = max(tr1, tr2, tr3)
            
            atr_result = pd.Series(tr).ewm(span=period, adjust=False).mean().values
            return atr_result
        except:
            return np.full(n, np.mean(high - low))


class SSLHybridV3:
    """SSL Hybrid V3 - Improved version"""
    
    def __init__(self, close, high, low, volume, **kwargs):
        self.close = np.asarray(close, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.n = len(self.close)
        
        print(f"    Calculating RSI...")
        self.rsi14 = TechnicalIndicators.rsi(self.close, 14)
        self.rsi7 = TechnicalIndicators.rsi(self.close, 7)
        
        print(f"    Calculating MACD...")
        self.macd_line, self.macd_signal, self.macd_hist = TechnicalIndicators.macd(self.close)
        
        print(f"    Calculating Bollinger Bands...")
        self.bb_upper, self.bb_mid, self.bb_lower = TechnicalIndicators.bollinger_bands(self.close, 20, 2)
        
        print(f"    Calculating ATR...")
        self.atr = TechnicalIndicators.atr(self.high, self.low, self.close, 14)
        
        print(f"    Calculating signals...")
        # Calculate signals
        self._calc_signals()
        print(f"  ✓ All indicators calculated")
    
    def _calc_signals(self):
        """Calculate multi-timeframe signals"""
        self.buy_signal = np.zeros(self.n, dtype=bool)
        self.sell_signal = np.zeros(self.n, dtype=bool)
        
        for i in range(20, self.n):
            try:
                # Entry conditions
                rsi_val = self.rsi14[i]
                rsi_oversold = rsi_val < 35
                rsi_midzone = 45 < rsi_val < 55
                macd_bullish = self.macd_hist[i] > 0 and self.macd_line[i] > self.macd_signal[i]
                
                vol_ma_20 = np.mean(self.volume[max(0, i-20):i])
                volume_spike = self.volume[i] > vol_ma_20 * 1.2
                
                if (rsi_oversold or rsi_midzone) and macd_bullish and volume_spike:
                    self.buy_signal[i] = True
                
                rsi_overbought = rsi_val > 65
                macd_bearish = self.macd_hist[i] < 0 and self.macd_line[i] < self.macd_signal[i]
                
                if (rsi_overbought or rsi_midzone) and macd_bearish and volume_spike:
                    self.sell_signal[i] = True
            except:
                continue


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
    
    try:
        # Safety check
        if idx < 20 or idx >= len(indicator.rsi14):
            return {}
        
        # 1. RSI features
        rsi_val = float(indicator.rsi14[idx])
        features['rsi14'] = rsi_val / 100.0  # Normalize to 0-1
        features['rsi14_from_neutral'] = abs(rsi_val - 50) / 50  # Distance from 50
        features['rsi_trend'] = (rsi_val - float(indicator.rsi14[max(0, idx-5)])) / 100
        
        # 2. MACD features
        macd_price_ratio = float(indicator.macd_hist[idx]) / (signal['price'] + 1e-8)
        features['macd_hist'] = np.clip(macd_price_ratio * 100, -1, 1)
        features['macd_bullish'] = 1.0 if float(indicator.macd_hist[idx]) > 0 else 0.0
        
        macd_signal_ratio = (float(indicator.macd_line[idx]) - float(indicator.macd_signal[idx])) / (abs(float(indicator.macd_signal[idx])) + 1e-8)
        features['macd_signal_dist'] = np.clip(macd_signal_ratio, -1, 1)
        
        # 3. Bollinger Bands features
        bb_range = float(indicator.bb_upper[idx]) - float(indicator.bb_lower[idx])
        if bb_range > 0:
            bb_position = (signal['price'] - float(indicator.bb_lower[idx])) / bb_range
        else:
            bb_position = 0.5
        features['bb_position'] = np.clip(bb_position, 0, 1)
        features['bb_distance_mid'] = (signal['price'] - float(indicator.bb_mid[idx])) / (float(indicator.atr[idx]) + 1e-8)
        
        # 4. Volatility features
        if idx >= 20:
            vol_20 = np.std(close[idx-20:idx])
            mean_20 = np.mean(close[idx-20:idx])
        else:
            vol_20 = np.std(close[:idx])
            mean_20 = np.mean(close[:idx])
        
        features['volatility'] = np.clip((vol_20 / (mean_20 + 1e-8)) * 100, 0, 5) / 5
        features['atr_ratio'] = np.clip(float(indicator.atr[idx]) / (mean_20 + 1e-8) * 100, 0, 5) / 5
        
        # 5. Volume features
        if idx >= 20:
            vol_ma = np.mean(volume[idx-20:idx])
        else:
            vol_ma = np.mean(volume[:idx])
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
        
    except Exception as e:
        return {}
    
    return features


if __name__ == "__main__":
    print("SSL Hybrid V3 Implementation - Ready for use")
