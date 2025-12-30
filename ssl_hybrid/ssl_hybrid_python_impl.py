#!/usr/bin/env python3
"""
SSL Hybrid V6 Indicator - Python Implementation
Converted from TradingView Pine Script

This module:
1. Implements all SSL Hybrid calculations
2. Extracts BUY/SELL signals
3. Identifies true vs false signals
4. Prepares data for neural network training
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MOVING AVERAGE IMPLEMENTATIONS
# ============================================================================

class MovingAverages:
    """Collection of moving average calculations"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def dema(data, period):
        """Double EMA"""
        e = pd.Series(data).ewm(span=period, adjust=False).mean().values
        return 2 * e - pd.Series(e).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def tema(data, period):
        """Triple EMA"""
        e = pd.Series(data).ewm(span=period, adjust=False).mean().values
        e2 = pd.Series(e).ewm(span=period, adjust=False).mean().values
        e3 = pd.Series(e2).ewm(span=period, adjust=False).mean().values
        return 3 * (e - e2) + e3
    
    @staticmethod
    def wma(data, period):
        """Weighted Moving Average"""
        return pd.Series(data).rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1))
        ).values
    
    @staticmethod
    def hma(data, period):
        """Hull Moving Average"""
        wma1 = MovingAverages.wma(data, int(period / 2))
        wma2 = MovingAverages.wma(data, period)
        raw_hma = 2 * wma1 - wma2
        return MovingAverages.wma(raw_hma, int(np.sqrt(period)))
    
    @staticmethod
    def jma(data, period, phase=3, power=1):
        """Jurik Moving Average"""
        data = np.asarray(data, dtype=np.float64)
        jma = np.zeros_like(data)
        
        phase_ratio = 0.5 if phase < -100 else 2.5 if phase > 100 else phase / 100 + 1.5
        beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
        alpha = np.power(beta, power)
        
        e0 = data[0]
        e1 = 0.0
        e2 = 0.0
        jma[0] = data[0]
        
        for i in range(1, len(data)):
            e0 = (1 - alpha) * data[i] + alpha * e0
            e1 = (data[i] - e0) * (1 - beta) + beta * e1
            e2 = (e0 + phase_ratio * e1 - jma[i-1]) * np.power(1 - alpha, 2) + np.power(alpha, 2) * e2
            jma[i] = e2 + jma[i-1]
        
        return jma
    
    @staticmethod
    def get_ma(ma_type, data, period, phase=3, power=1):
        """Get moving average by type"""
        if ma_type == "SMA":
            return MovingAverages.sma(data, period)
        elif ma_type == "EMA":
            return MovingAverages.ema(data, period)
        elif ma_type == "DEMA":
            return MovingAverages.dema(data, period)
        elif ma_type == "TEMA":
            return MovingAverages.tema(data, period)
        elif ma_type == "WMA":
            return MovingAverages.wma(data, period)
        elif ma_type == "HMA":
            return MovingAverages.hma(data, period)
        elif ma_type == "JMA":
            return MovingAverages.jma(data, period, phase, power)
        else:
            return MovingAverages.sma(data, period)


# ============================================================================
# SSL HYBRID INDICATOR
# ============================================================================

@dataclass
class SSLHybridParams:
    """SSL Hybrid indicator parameters"""
    # Baseline
    baseline_type: str = "HMA"
    baseline_len: int = 60
    channel_mult: float = 0.2
    
    # SSL1
    ssl1_type: str = "HMA"
    ssl1_len: int = 60
    
    # SSL2
    ssl2_type: str = "JMA"
    ssl2_len: int = 5
    atr_crit: float = 0.9
    
    # Exit
    exit_type: str = "HMA"
    exit_len: int = 15
    
    # ATR
    atr_len: int = 14
    atr_mult: float = 1.0
    
    # Risk
    risk_lookback: int = 100
    risk_sensitivity: float = 2.0


class SSLHybridIndicator:
    """SSL Hybrid V6 Indicator Implementation"""
    
    def __init__(self, close, high, low, volume, params: SSLHybridParams = None):
        self.close = np.asarray(close, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.low = np.asarray(low, dtype=np.float32)
        self.volume = np.asarray(volume, dtype=np.float32)
        self.n = len(self.close)
        
        self.params = params or SSLHybridParams()
        
        # Calculate all components
        self._calculate_atr()
        self._calculate_baseline()
        self._calculate_ssl1()
        self._calculate_ssl2()
        self._calculate_exit()
        self._calculate_signals()
    
    def _calculate_atr(self):
        """Calculate ATR"""
        tr = np.zeros(self.n)
        for i in range(1, self.n):
            tr1 = self.high[i] - self.low[i]
            tr2 = abs(self.high[i] - self.close[i-1])
            tr3 = abs(self.low[i] - self.close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        self.atr_slen = MovingAverages.ema(tr, self.params.atr_len)
        self.upper_band = self.atr_slen * self.params.atr_mult + self.close
        self.lower_band = self.close - self.atr_slen * self.params.atr_mult
        
        # Risk percentile
        self.atr_percentile = np.zeros(self.n)
        for i in range(self.params.risk_lookback, self.n):
            lookback_atr = self.atr_slen[i-self.params.risk_lookback:i]
            self.atr_percentile[i] = (self.atr_slen[i] <= np.percentile(lookback_atr, np.linspace(0, 100, 101))) * 100
    
    def _calculate_baseline(self):
        """Calculate baseline and channel"""
        self.baseline = MovingAverages.get_ma(
            self.params.baseline_type, self.close, self.params.baseline_len
        )
        
        # Channel calculation
        range_ma = MovingAverages.ema(self.high - self.low, self.params.baseline_len)
        self.upper_channel = self.baseline + range_ma * self.params.channel_mult
        self.lower_channel = self.baseline - range_ma * self.params.channel_mult
    
    def _calculate_ssl1(self):
        """Calculate SSL1 (main trend)"""
        ema_high = MovingAverages.get_ma(self.params.ssl1_type, self.high, self.params.ssl1_len)
        ema_low = MovingAverages.get_ma(self.params.ssl1_type, self.low, self.params.ssl1_len)
        
        hlv = np.zeros(self.n)
        for i in range(1, self.n):
            if self.close[i] > ema_high[i]:
                hlv[i] = 1
            elif self.close[i] < ema_low[i]:
                hlv[i] = -1
            else:
                hlv[i] = hlv[i-1]
        
        self.ssl1 = np.where(hlv < 0, ema_high, ema_low)
        self.ssl1_direction = hlv
    
    def _calculate_ssl2(self):
        """Calculate SSL2 (fast continuation)"""
        ma_high = MovingAverages.get_ma(self.params.ssl2_type, self.high, self.params.ssl2_len)
        ma_low = MovingAverages.get_ma(self.params.ssl2_type, self.low, self.params.ssl2_len)
        
        hlv2 = np.zeros(self.n)
        for i in range(1, self.n):
            if self.close[i] > ma_high[i]:
                hlv2[i] = 1
            elif self.close[i] < ma_low[i]:
                hlv2[i] = -1
            else:
                hlv2[i] = hlv2[i-1]
        
        self.ssl2 = np.where(hlv2 < 0, ma_high, ma_low)
        self.ssl2_direction = hlv2
        
        # SSL2 continuation conditions
        upper_half = self.atr_slen * self.params.atr_crit + self.close
        lower_half = self.close - self.atr_slen * self.params.atr_crit
        
        self.buy_inatr = lower_half < self.ssl2
        self.sell_inatr = upper_half > self.ssl2
        self.buy_cont = (self.close > self.baseline) & (self.close > self.ssl2)
        self.sell_cont = (self.close < self.baseline) & (self.close < self.ssl2)
        
        self.buy_atr = self.buy_inatr & self.buy_cont
        self.sell_atr = self.sell_inatr & self.sell_cont
    
    def _calculate_exit(self):
        """Calculate exit levels"""
        exit_high = MovingAverages.get_ma(self.params.exit_type, self.high, self.params.exit_len)
        exit_low = MovingAverages.get_ma(self.params.exit_type, self.low, self.params.exit_len)
        
        hlv3 = np.zeros(self.n)
        for i in range(1, self.n):
            if self.close[i] > exit_high[i]:
                hlv3[i] = 1
            elif self.close[i] < exit_low[i]:
                hlv3[i] = -1
            else:
                hlv3[i] = hlv3[i-1]
        
        self.ssl_exit = np.where(hlv3 < 0, exit_high, exit_low)
    
    def _calculate_signals(self):
        """Calculate entry signals"""
        # SSL2 Entry signals
        self.buy_signal = np.zeros(self.n, dtype=bool)
        self.sell_signal = np.zeros(self.n, dtype=bool)
        
        for i in range(1, self.n):
            if self.buy_atr[i] and not self.buy_atr[i-1]:
                self.buy_signal[i] = True
            if self.sell_atr[i] and not self.sell_atr[i-1]:
                self.sell_signal[i] = True
        
        # Exit signals
        self.exit_long = np.zeros(self.n, dtype=bool)
        self.exit_short = np.zeros(self.n, dtype=bool)
        
        for i in range(1, self.n):
            if self.close[i-1] > self.ssl_exit[i-1] and self.close[i] <= self.ssl_exit[i]:
                self.exit_long[i] = True
            if self.close[i-1] < self.ssl_exit[i-1] and self.close[i] >= self.ssl_exit[i]:
                self.exit_short[i] = True
        
        # False breakout warning
        candle_diff = np.abs(self.close - np.roll(self.close, 1))
        atr_violation = candle_diff > self.atr_slen
        in_range = (self.upper_band > self.baseline) & (self.lower_band < self.baseline)
        self.false_breakout_warning = atr_violation & in_range


# ============================================================================
# SIGNAL EXTRACTION & LABELING
# ============================================================================

@dataclass
class Signal:
    """Represents a single trading signal"""
    index: int
    signal_type: str  # 'BUY' or 'SELL'
    price: float
    time: int
    
    # Context features at signal time
    atr_percentile: float
    distance_from_baseline: float
    volume_ratio: float
    candle_size: float
    atr_violation: bool
    risk_level: str
    ssl1_direction: int
    ssl2_direction: int
    
    # Label (true/false)
    is_true: bool = None
    actual_return: float = None


class SignalExtractor:
    """Extract signals from SSL Hybrid and label them"""
    
    def __init__(self, indicator: SSLHybridIndicator, close, volume, lookforward=5):
        self.indicator = indicator
        self.close = np.asarray(close, dtype=np.float32)
        self.volume = np.asarray(volume, dtype=np.float32)
        self.lookforward = lookforward
        self.n = len(close)
    
    def extract_signals(self) -> List[Signal]:
        """Extract all BUY/SELL signals"""
        signals = []
        
        for i in range(self.n):
            if self.indicator.buy_signal[i]:
                signal = self._create_signal(i, 'BUY')
                if signal:
                    signals.append(signal)
            
            elif self.indicator.sell_signal[i]:
                signal = self._create_signal(i, 'SELL')
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _create_signal(self, idx: int, signal_type: str) -> Signal:
        """Create signal object with context"""
        if idx + self.lookforward >= self.n:
            return None
        
        # Distance from baseline
        distance = abs(self.close[idx] - self.indicator.baseline[idx]) / (self.indicator.atr_slen[idx] + 1e-8)
        
        # Volume ratio
        vol_ma = np.mean(self.volume[max(0, idx-20):idx])
        volume_ratio = self.volume[idx] / (vol_ma + 1e-8)
        
        # Candle size
        candle_size = abs(self.close[idx] - self.close[idx-1]) / (self.indicator.atr_slen[idx] + 1e-8)
        
        # Risk level
        risk_pct = self.indicator.atr_percentile[idx]
        if risk_pct > 75:
            risk_level = "High"
        elif risk_pct < 25:
            risk_level = "Low"
        else:
            risk_level = "Normal"
        
        signal = Signal(
            index=idx,
            signal_type=signal_type,
            price=float(self.close[idx]),
            time=idx,
            atr_percentile=float(risk_pct),
            distance_from_baseline=float(distance),
            volume_ratio=float(volume_ratio),
            candle_size=float(candle_size),
            atr_violation=bool(self.indicator.false_breakout_warning[idx]),
            risk_level=risk_level,
            ssl1_direction=int(self.indicator.ssl1_direction[idx]),
            ssl2_direction=int(self.indicator.ssl2_direction[idx])
        )
        
        # Label: is signal true?
        future_close = self.close[idx + self.lookforward]
        future_return = (future_close - self.close[idx]) / (self.close[idx] + 1e-8)
        
        if signal_type == 'BUY':
            signal.is_true = future_return > 0.005  # 0.5% profit
        else:  # SELL
            signal.is_true = future_return < -0.005  # 0.5% profit
        
        signal.actual_return = float(future_return)
        
        return signal


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract features for model training"""
    
    def __init__(self, indicator: SSLHybridIndicator, close, high, low, volume, lookback=40):
        self.indicator = indicator
        self.close = np.asarray(close, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.low = np.asarray(low, dtype=np.float32)
        self.volume = np.asarray(volume, dtype=np.float32)
        self.lookback = lookback
        self.n = len(close)
    
    def extract_features(self, signal: Signal) -> Dict[str, float]:
        """Extract all features for a signal"""
        idx = signal.index
        
        features = {}
        
        # Signal context
        features['atr_percentile'] = signal.atr_percentile / 100  # Normalize to 0-1
        features['distance_from_baseline'] = np.clip(signal.distance_from_baseline / 3, 0, 1)
        features['volume_ratio'] = np.clip(signal.volume_ratio, 0.1, 10) / 10
        features['candle_size'] = np.clip(signal.candle_size, 0, 2) / 2
        features['atr_violation'] = float(signal.atr_violation)
        features['ssl1_direction'] = (signal.ssl1_direction + 1) / 2  # Convert -1,0,1 to 0,0.5,1
        features['ssl2_direction'] = (signal.ssl2_direction + 1) / 2
        
        # Price momentum
        if idx >= 5:
            mom5 = (self.close[idx] - self.close[idx-5]) / (self.close[idx-5] + 1e-8)
            features['momentum_5'] = np.clip(mom5 * 100, -5, 5) / 5  # Normalize to ~-1 to 1
        else:
            features['momentum_5'] = 0
        
        if idx >= 20:
            mom20 = (self.close[idx] - self.close[idx-20]) / (self.close[idx-20] + 1e-8)
            features['momentum_20'] = np.clip(mom20 * 100, -10, 10) / 10
        else:
            features['momentum_20'] = 0
        
        # Volatility
        if idx >= 20:
            std20 = np.std(self.close[idx-20:idx])
            mean20 = np.mean(self.close[idx-20:idx])
            features['volatility_20'] = std20 / (mean20 + 1e-8) * 10  # Scale
            features['volatility_20'] = np.clip(features['volatility_20'], 0, 1)
        else:
            features['volatility_20'] = 0
        
        # Trend strength
        if idx >= self.lookback:
            hist_close = self.close[idx-self.lookback:idx]
            trend_up = sum(self.close[i] > self.close[i-1] for i in range(idx-self.lookback+1, idx))
            features['trend_strength'] = trend_up / self.lookback
        else:
            features['trend_strength'] = 0
        
        # Volume trend
        if idx >= 20:
            vol_avg = np.mean(self.volume[idx-20:idx])
            features['volume_trend'] = self.volume[idx] / (vol_avg + 1e-8)
            features['volume_trend'] = np.clip(features['volume_trend'], 0.1, 10) / 10
        else:
            features['volume_trend'] = 0
        
        # Price position in channel
        features['price_in_channel'] = (
            (self.close[idx] - self.indicator.lower_channel[idx]) / 
            (self.indicator.upper_channel[idx] - self.indicator.lower_channel[idx] + 1e-8)
        )
        features['price_in_channel'] = np.clip(features['price_in_channel'], 0, 1)
        
        return features


if __name__ == "__main__":
    print("SSL Hybrid V6 Implementation Module")
    print("Ready for import and usage")
