#!/usr/bin/env python3
"""
Real-Time Formula Predictor

Usage:
    predictor = RealTimeFormulaPredictor()
    result = predictor.process_new_kline(kline_dict)

This module handles real-time inference using discovered formulas and trained LSTM.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
import pickle
from collections import deque
from typing import Dict, Tuple
import ta


class RealTimeFormulaPredictor:
    """Real-time signal generation using formulas and LSTM"""
    
    def __init__(self, model_path='formula_lstm_model.h5', scaler_path='scaler_config.pkl', 
                 formula_file='discovered_formulas.json'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained LSTM model
            scaler_path: Path to saved StandardScaler
            formula_file: Path to discovered formulas JSON
        """
        print("Initializing Real-Time Predictor...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Loaded scaler from {scaler_path}")
        
        # Load formulas
        with open(formula_file) as f:
            self.formulas_config = json.load(f)
        print(f"Loaded {len(self.formulas_config)} formulas from {formula_file}")
        
        # Formula buffer: store last 30 formula values
        self.formula_buffer = deque(maxlen=30)
        
        # Tracking
        self.last_signal = None
        self.last_confidence = None
        self.last_formula_values = None
        self.kline_counter = 0
    
    def compute_indicators(self, ohlcv_dict: Dict) -> Dict:
        """
        Compute all required indicators from OHLCV
        
        Args:
            ohlcv_dict: {open, high, low, close, volume, ...}
        
        Returns:
            Dictionary of all indicators
        """
        indicators = {}
        
        # Simple indicators from single bar
        indicators['open'] = ohlcv_dict.get('open', 0)
        indicators['high'] = ohlcv_dict.get('high', 0)
        indicators['low'] = ohlcv_dict.get('low', 0)
        indicators['close'] = ohlcv_dict.get('close', 0)
        indicators['volume'] = ohlcv_dict.get('volume', 0)
        
        # Pre-computed indicators (should be provided by data source)
        indicators['rsi_7'] = ohlcv_dict.get('rsi_7', 50)
        indicators['rsi_14'] = ohlcv_dict.get('rsi_14', 50)
        indicators['macd_diff'] = ohlcv_dict.get('macd_diff', 0)
        indicators['sma_20'] = ohlcv_dict.get('sma_20', ohlcv_dict.get('close', 0))
        indicators['bb_width'] = ohlcv_dict.get('bb_width', 0)
        indicators['atr_14'] = ohlcv_dict.get('atr_14', 0)
        indicators['obv'] = ohlcv_dict.get('obv', 0)
        indicators['volume_ratio'] = ohlcv_dict.get('volume_ratio', 1)
        
        return indicators
    
    def apply_formulas(self, indicators: Dict) -> np.ndarray:
        """
        Apply 5 discovered formulas
        
        Args:
            indicators: Dictionary of indicator values
        
        Returns:
            Array of 5 formula values
        """
        formulas = np.zeros(5)
        
        try:
            # Formula 1: RSI-MACD blend
            f1 = (indicators.get('rsi_14', 50) * 0.4 + 
                  indicators.get('macd_diff', 0) * 0.3 + 
                  indicators.get('sma_20', 0) * 0.3)
            formulas[0] = f1
            
            # Formula 2: Volume-ATR logarithmic
            atr = indicators.get('atr_14', 0.1)
            vol_ratio = indicators.get('volume_ratio', 1.0)
            f2 = np.log(abs(atr * vol_ratio) + 1e-8)
            formulas[1] = f2
            
            # Formula 3: Bollinger-RSI ratio
            bb_width = indicators.get('bb_width', 1e-8)
            rsi_7 = indicators.get('rsi_7', 50)
            f3 = bb_width / (rsi_7 + 1e-8)
            formulas[2] = f3
            
            # Formula 4: MACD-ATR divergence
            macd = indicators.get('macd_diff', 0)
            atr = indicators.get('atr_14', 0.1)
            f4 = macd / (atr + 1e-8)
            formulas[3] = f4
            
            # Formula 5: Volume-SMA interaction
            vol_ratio = indicators.get('volume_ratio', 1.0)
            sma_20 = indicators.get('sma_20', 0)
            f5 = np.tanh(vol_ratio) * sma_20
            formulas[4] = f5
            
            # Handle NaN and Inf
            formulas = np.nan_to_num(formulas, nan=0.0, posinf=0.0, neginf=0.0)
            
        except Exception as e:
            print(f"Warning: Error applying formulas: {e}")
            formulas = np.zeros(5)
        
        return formulas
    
    def process_new_kline(self, kline_dict: Dict) -> Dict:
        """
        Process new K-line and generate trading signal
        
        Args:
            kline_dict: {
                'timestamp': '2025-12-30 20:00:00',
                'open': 42150.0,
                'high': 42250.0,
                'low': 42100.0,
                'close': 42200.0,
                'volume': 1200,
                'rsi_7': 45.2,
                'rsi_14': 48.5,
                'macd_diff': 0.0025,
                'sma_20': 42050.0,
                'bb_width': 150.0,
                'atr_14': 75.5,
                'volume_ratio': 1.05
            }
        
        Returns:
            {
                'status': 'READY' or 'WARMING_UP',
                'timestamp': '2025-12-30 20:00:00',
                'signal': 'BUY', 'HOLD', 'SELL',
                'confidence': 0.87,
                'formula_values': [0.542, -0.231, ...],
                'action': 'Buy Signal',
                'probabilities': {'SELL': 0.1, 'HOLD': 0.03, 'BUY': 0.87}
            }
        """
        self.kline_counter += 1
        
        # Compute indicators
        indicators = self.compute_indicators(kline_dict)
        
        # Apply formulas
        formula_values = self.apply_formulas(indicators)
        self.last_formula_values = formula_values
        
        # Add to buffer
        self.formula_buffer.append(formula_values)
        
        # Check if we have enough history
        if len(self.formula_buffer) < 30:
            return {
                'status': 'WARMING_UP',
                'timestamp': kline_dict.get('timestamp'),
                'buffer_size': len(self.formula_buffer),
                'required_size': 30,
                'signal': 'HOLD',
                'confidence': 0.0,
                'message': f'Warming up: {len(self.formula_buffer)}/30 bars loaded'
            }
        
        # Prepare model input
        X = np.array(list(self.formula_buffer)).reshape(1, 30, 5)
        X_scaled = self.scaler.transform(X.reshape(-1, 5)).reshape(1, 30, 5)
        
        # Model inference
        try:
            probabilities = self.model.predict(X_scaled, verbose=0)[0]
        except Exception as e:
            print(f"Warning: Model inference error: {e}")
            return {
                'status': 'ERROR',
                'timestamp': kline_dict.get('timestamp'),
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
        
        # Get signal
        signal_idx = np.argmax(probabilities)
        confidence = float(probabilities[signal_idx])
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[signal_idx]
        
        action_map = {
            'SELL': 'Exit Long / Enter Short',
            'HOLD': 'No Action',
            'BUY': 'Enter Long / Exit Short'
        }
        
        # Store current signal
        self.last_signal = signal
        self.last_confidence = confidence
        
        return {
            'status': 'READY',
            'timestamp': kline_dict.get('timestamp'),
            'kline_count': self.kline_counter,
            'signal': signal,
            'action': action_map[signal],
            'confidence': confidence,
            'probabilities': {
                'SELL': float(probabilities[0]),
                'HOLD': float(probabilities[1]),
                'BUY': float(probabilities[2])
            },
            'formula_values': {
                'f1_rsi_macd_blend': float(formula_values[0]),
                'f2_vol_atr_log': float(formula_values[1]),
                'f3_bb_rsi_ratio': float(formula_values[2]),
                'f4_macd_atr_div': float(formula_values[3]),
                'f5_vol_sma_inter': float(formula_values[4])
            }
        }
    
    def get_portfolio_recommendation(self, current_position: str = 'FLAT') -> str:
        """
        Get trading recommendation based on current position
        
        Args:
            current_position: 'LONG', 'SHORT', 'FLAT'
        
        Returns:
            Trading action: 'BUY', 'SELL', 'HOLD'
        """
        if self.last_signal is None or self.last_confidence < 0.6:
            return 'HOLD'
        
        if current_position == 'FLAT':
            return self.last_signal if self.last_signal in ['BUY', 'SELL'] else 'HOLD'
        elif current_position == 'LONG':
            return 'SELL' if self.last_signal == 'SELL' else 'HOLD'
        elif current_position == 'SHORT':
            return 'BUY' if self.last_signal == 'BUY' else 'HOLD'
        else:
            return 'HOLD'


if __name__ == '__main__':
    # Example usage
    predictor = RealTimeFormulaPredictor()
    
    # Simulate incoming K-lines
    sample_kline = {
        'timestamp': '2025-12-30 20:00:00',
        'open': 42150.0,
        'high': 42250.0,
        'low': 42100.0,
        'close': 42200.0,
        'volume': 1200,
        'rsi_7': 45.2,
        'rsi_14': 48.5,
        'macd_diff': 0.0025,
        'sma_20': 42050.0,
        'bb_width': 150.0,
        'atr_14': 75.5,
        'volume_ratio': 1.05
    }
    
    result = predictor.process_new_kline(sample_kline)
    print(json.dumps(result, indent=2))
