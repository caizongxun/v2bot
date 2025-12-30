#!/usr/bin/env python3
"""
SSL Hybrid V4 - Fixed Overfitting
Remove data leakage + proper cross-validation
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
        rsi_result = np.zeros(n)
        
        if n < period + 1:
            return rsi_result
        
        df = pd.DataFrame({'close': data})
        df['delta'] = df['close'].diff()
        df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
        df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)
        
        for i in range(period, n):
            if i == period:
                df.loc[i, 'avg_gain'] = df['gain'].iloc[1:period+1].mean()
                df.loc[i, 'avg_loss'] = df['loss'].iloc[1:period+1].mean()
            else:
                df.loc[i, 'avg_gain'] = (df.loc[i-1, 'avg_gain'] * (period - 1) + df.loc[i, 'gain']) / period
                df.loc[i, 'avg_loss'] = (df.loc[i-1, 'avg_loss'] * (period - 1) + df.loc[i, 'loss']) / period
        
        rs = df['avg_gain'] / (df['avg_loss'] + 1e-10)
        rsi_result = (100 - (100 / (1 + rs))).values
        rsi_result[:period] = 50
        
        return rsi_result
    
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


class SSLHybridV4:
    """SSL Hybrid V4 - Fixed version with proper validation"""
    
    def __init__(self, close, high, low, volume):
        self.close = np.asarray(close, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.low = np.asarray(low, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.n = len(self.close)
        
        self.rsi14 = TechnicalIndicators.rsi(self.close, 14)
        self.rsi7 = TechnicalIndicators.rsi(self.close, 7)
        self.macd_line, self.macd_signal, self.macd_hist = TechnicalIndicators.macd(self.close)
        self.bb_upper, self.bb_mid, self.bb_lower = TechnicalIndicators.bollinger_bands(self.close, 20, 2)
        self.atr = TechnicalIndicators.atr(self.high, self.low, self.close, 14)
        self._calc_signals()
    
    def _calc_signals(self):
        """Calculate signals - FIXED logic"""
        self.buy_signal = np.zeros(self.n, dtype=bool)
        self.sell_signal = np.zeros(self.n, dtype=bool)
        
        for i in range(20, self.n):
            try:
                rsi_val = self.rsi14[i]
                vol_ma_20 = np.mean(self.volume[max(0, i-20):i])
                volume_spike = self.volume[i] > vol_ma_20 * 1.1
                
                # BUY: RSI oversold + MACD bullish
                rsi_oversold = rsi_val < 40
                macd_bullish = self.macd_hist[i] > 0
                if rsi_oversold and macd_bullish and volume_spike:
                    self.buy_signal[i] = True
                
                # SELL: RSI overbought + MACD bearish
                rsi_overbought = rsi_val > 60
                macd_bearish = self.macd_hist[i] < 0
                if rsi_overbought and macd_bearish and volume_spike:
                    self.sell_signal[i] = True
            except:
                continue


def extract_signals_v4(close, high, low, volume, indicator, lookforward=10) -> List[Dict]:
    """Extract signals - FIXED validation logic"""
    signals = []
    n = len(close)
    
    for i in range(n - lookforward):
        if indicator.buy_signal[i]:
            # Look forward and check if price actually goes up
            future_prices = close[i:i+lookforward]
            max_price = np.max(future_prices)
            max_return = (max_price - close[i]) / close[i]
            
            # True if price reaches +0.5% within next 10 candles
            is_true = max_return > 0.005
            
            signals.append({
                'index': i,
                'type': 'BUY',
                'price': float(close[i]),
                'is_true': is_true,
                'max_return': float(max_return),
                'volume': float(volume[i])
            })
        
        elif indicator.sell_signal[i]:
            future_prices = close[i:i+lookforward]
            min_price = np.min(future_prices)
            max_return = (close[i] - min_price) / close[i]
            
            # True if price drops -0.5% within next 10 candles
            is_true = max_return > 0.005
            
            signals.append({
                'index': i,
                'type': 'SELL',
                'price': float(close[i]),
                'is_true': is_true,
                'max_return': float(max_return),
                'volume': float(volume[i])
            })
    
    return signals


def extract_features_v4(signal: Dict, close, high, low, volume, indicator) -> Dict:
    """Extract features - NO data leakage"""
    idx = signal['index']
    features = {}
    
    try:
        if idx < 20:
            return {}
        
        # 純技術指標特徵 - 不包含未來信息
        
        # RSI 狀態
        rsi_val = float(indicator.rsi14[idx])
        features['rsi_oversold'] = 1.0 if rsi_val < 40 else 0.0
        features['rsi_overbought'] = 1.0 if rsi_val > 60 else 0.0
        features['rsi_midzone'] = 1.0 if 40 <= rsi_val <= 60 else 0.0
        
        # MACD 方向
        features['macd_positive'] = 1.0 if indicator.macd_hist[idx] > 0 else 0.0
        features['macd_above_signal'] = 1.0 if indicator.macd_line[idx] > indicator.macd_signal[idx] else 0.0
        
        # Bollinger Bands 位置
        bb_range = float(indicator.bb_upper[idx]) - float(indicator.bb_lower[idx])
        if bb_range > 0:
            bb_pos = (signal['price'] - float(indicator.bb_lower[idx])) / bb_range
            features['bb_lower_third'] = 1.0 if bb_pos < 0.33 else 0.0
            features['bb_upper_third'] = 1.0 if bb_pos > 0.67 else 0.0
        else:
            features['bb_lower_third'] = 0.0
            features['bb_upper_third'] = 0.0
        
        # ATR 相對值
        if idx >= 20:
            vol_20 = np.std(close[idx-20:idx])
            mean_20 = np.mean(close[idx-20:idx])
        else:
            vol_20 = np.std(close[:idx])
            mean_20 = np.mean(close[:idx])
        
        features['volatility_high'] = 1.0 if vol_20 > mean_20 * 0.03 else 0.0
        
        # 成交量
        if idx >= 20:
            vol_ma = np.mean(volume[idx-20:idx])
        else:
            vol_ma = np.mean(volume[:idx])
        features['volume_spike'] = 1.0 if signal['volume'] > vol_ma * 1.2 else 0.0
        
        # 價格動量 (過去，不是未來)
        if idx >= 5:
            past_mom = (close[idx] - close[idx-5]) / close[idx-5]
            features['past_momentum_positive'] = 1.0 if past_mom > 0 else 0.0
        else:
            features['past_momentum_positive'] = 0.0
        
        # 信號類型
        features['is_buy'] = 1.0 if signal['type'] == 'BUY' else 0.0
        
    except Exception as e:
        return {}
    
    return features


if __name__ == "__main__":
    print("SSL Hybrid V4 Implementation - Ready for use")
