#!/usr/bin/env python3
"""
V2Bot Live Trading System Example

Integrates LSTM predictions with exchange API for real-time trading

Requirements:
    - ccxt (for exchange API)
    - numpy, pandas, tensorflow
    - Trained model files from Colab

Usage:
    trader = LiveTrader(
        exchange_name='binance',
        api_key='...',
        api_secret='...',
        symbol='BTC/USDT',
        interval='1h',
        model_path='formula_lstm_model_BTC_1h.keras'
    )
    
    trader.start_monitoring()
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from collections import deque
import tensorflow as tf


class LiveTrader:
    """Live trading system with LSTM predictions"""
    
    def __init__(self, 
                 exchange_name: str,
                 api_key: str,
                 api_secret: str,
                 symbol: str = 'BTC/USDT',
                 interval: str = '1h',
                 model_path: str = 'formula_lstm_model_BTC_1h.keras',
                 scaler_path: str = 'scaler_config_BTC_1h.json',
                 formula_file: str = 'discovered_formulas_BTC_1h.json',
                 risk_per_trade: float = 0.02):
        """
        Initialize live trading system
        
        Args:
            exchange_name: 'binance', 'bybit', 'deribit', etc.
            api_key: Exchange API key
            api_secret: Exchange API secret
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Chart interval ('1h', '4h', '1d')
            model_path: Path to trained .keras model
            scaler_path: Path to scaler config JSON
            formula_file: Path to formulas JSON
            risk_per_trade: Risk per trade as % of portfolio (default 2%)
        """
        print(f"\nInitializing {exchange_name.upper()} Live Trader...")
        
        self.symbol = symbol
        self.interval = interval
        self.risk_per_trade = risk_per_trade
        
        # Load CCXT exchange
        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
            print(f"  Connected to {exchange_name}")
        except Exception as e:
            raise Exception(f"Failed to connect to {exchange_name}: {e}")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"  Loaded model from {model_path}")
        
        # Load scaler
        with open(scaler_path) as f:
            scaler_data = json.load(f)
        self.mean_ = np.array(scaler_data['mean'], dtype=np.float32)
        self.scale_ = np.array(scaler_data['scale'], dtype=np.float32)
        print(f"  Loaded scaler from {scaler_path}")
        
        # Load formulas
        with open(formula_file) as f:
            self.formulas_config = json.load(f)
        print(f"  Loaded {len(self.formulas_config)} formulas")
        
        # State management
        self.formula_buffer = deque(maxlen=30)
        self.position = None  # 'LONG', 'SHORT', 'FLAT'
        self.entry_price = None
        self.entry_time = None
        self.last_signal = None
        self.trade_history = []
        
        print(f"\n  Ready for trading on {symbol} ({interval})")
        print(f"  Risk per trade: {risk_per_trade*100:.1f}%\n")
    
    def fetch_ohlcv(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.interval, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return None
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with indicators
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        def rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9).mean()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        # Add to dataframe
        df['rsi_7'] = rsi(close, 7)
        df['rsi_14'] = rsi(close, 14)
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_diff'] = macd_line - macd_signal
        df['sma_20'] = sma_20
        df['ema_12'] = ema_fast
        df['atr_14'] = atr
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_upper - bb_lower
        df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma'].replace(0, 1))
        
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def apply_formulas(self, row: pd.Series) -> np.ndarray:
        """
        Apply trading formulas
        
        Args:
            row: Data row with indicators
        
        Returns:
            Formula values array
        """
        formulas = np.zeros(5, dtype=np.float32)
        
        try:
            rsi_14 = float(row.get('rsi_14', 50))
            macd_d = float(row.get('macd_diff', 0))
            sma_20 = float(row.get('sma_20', 0))
            atr_14 = float(row.get('atr_14', 1))
            vol_ratio = float(row.get('volume_ratio', 1))
            rsi_7 = float(row.get('rsi_7', 50))
            bb_width = float(row.get('bb_width', 1))
            
            # 5 formulas
            f1 = rsi_14 * 0.4 + macd_d * 0.3 + sma_20 * 0.3
            f2 = np.log(abs(atr_14 * vol_ratio) + 1e-8)
            f3 = bb_width / (rsi_7 + 1e-8)
            f4 = macd_d / (atr_14 + 1e-8)
            f5 = np.tanh(vol_ratio) * sma_20
            
            formulas = np.array([f1, f2, f3, f4, f5], dtype=np.float32)
            formulas = np.nan_to_num(formulas, nan=0.0, posinf=0.0, neginf=0.0)
            
        except Exception as e:
            print(f"Formula error: {e}")
        
        return formulas
    
    def get_signal(self) -> Optional[Dict]:
        """
        Get trading signal from model
        
        Returns:
            Signal dict or None if insufficient data
        """
        if len(self.formula_buffer) < 30:
            return None
        
        # Prepare input
        X = np.array(list(self.formula_buffer)).reshape(1, 30, 5).astype(np.float32)
        X_scaled = (X.reshape(-1, 5) - self.mean_) / self.scale_
        X_scaled = X_scaled.reshape(1, 30, 5)
        
        # Predict
        probs = self.model.predict(X_scaled, verbose=0)[0]
        signal_idx = np.argmax(probs)
        confidence = float(probs[signal_idx])
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        return {
            'signal': signal_map[signal_idx],
            'confidence': confidence,
            'probabilities': {
                'SELL': float(probs[0]),
                'HOLD': float(probs[1]),
                'BUY': float(probs[2])
            }
        }
    
    def execute_trade(self, signal: Dict, current_price: float) -> bool:
        """
        Execute trade based on signal
        
        Args:
            signal: Signal from model
            current_price: Current market price
        
        Returns:
            True if trade executed
        """
        if signal['confidence'] < 0.6:
            return False  # Low confidence
        
        signal_type = signal['signal']
        base, quote = self.symbol.split('/')
        
        try:
            # Get balance
            balance = self.exchange.fetch_balance()
            quote_balance = balance[quote]['free']
            base_balance = balance[base]['free']
            
            # Calculate position size
            portfolio_value = quote_balance + base_balance * current_price
            position_size = (portfolio_value * self.risk_per_trade) / current_price
            
            if signal_type == 'BUY' and self.position != 'LONG':
                # Close short if exists
                if self.position == 'SHORT':
                    self.exchange.create_market_sell_order(self.symbol, base_balance)
                    print(f"Closed SHORT position at {current_price:.2f}")
                
                # Open long
                self.exchange.create_market_buy_order(self.symbol, position_size)
                self.position = 'LONG'
                self.entry_price = current_price
                self.entry_time = datetime.now()
                print(f"Opened LONG position at {current_price:.2f}")
                return True
            
            elif signal_type == 'SELL' and self.position != 'SHORT':
                # Close long if exists
                if self.position == 'LONG':
                    self.exchange.create_market_sell_order(self.symbol, base_balance)
                    print(f"Closed LONG position at {current_price:.2f}")
                
                # Open short
                self.exchange.create_market_sell_order(self.symbol, position_size)
                self.position = 'SHORT'
                self.entry_price = current_price
                self.entry_time = datetime.now()
                print(f"Opened SHORT position at {current_price:.2f}")
                return True
        
        except Exception as e:
            print(f"Trade execution error: {e}")
            return False
    
    def start_monitoring(self, check_interval: int = 60):
        """
        Start live monitoring and trading
        
        Args:
            check_interval: Check interval in seconds
        """
        print(f"Starting live monitoring...\n")
        
        try:
            while True:
                try:
                    # Fetch data
                    df = self.fetch_ohlcv(limit=100)
                    if df is None or len(df) < 30:
                        print(f"[{datetime.now()}] Waiting for data...")
                        time.sleep(check_interval)
                        continue
                    
                    # Compute indicators
                    df = self.compute_indicators(df)
                    
                    # Apply formulas and add to buffer
                    for _, row in df.iterrows():
                        formula_val = self.apply_formulas(row)
                        self.formula_buffer.append(formula_val)
                    
                    # Get signal
                    signal = self.get_signal()
                    current_price = float(df.iloc[-1]['close'])
                    
                    if signal:
                        print(f"[{datetime.now()}] {self.symbol} @ {current_price:.2f}")
                        print(f"  Signal: {signal['signal']} (Confidence: {signal['confidence']:.1%})")
                        print(f"  Probabilities: BUY={signal['probabilities']['BUY']:.1%}, "
                              f"HOLD={signal['probabilities']['HOLD']:.1%}, "
                              f"SELL={signal['probabilities']['SELL']:.1%}")
                        print(f"  Position: {self.position or 'FLAT'}")
                        
                        # Execute trade
                        self.execute_trade(signal, current_price)
                        self.last_signal = signal
                    
                    # Wait for next check
                    time.sleep(check_interval)
                
                except KeyboardInterrupt:
                    print("\nStopping trader...")
                    break
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                    time.sleep(check_interval)
        
        except Exception as e:
            print(f"Fatal error: {e}")


if __name__ == '__main__':
    # Example usage - replace with your credentials
    trader = LiveTrader(
        exchange_name='binance',  # Change to your exchange
        api_key='YOUR_API_KEY',
        api_secret='YOUR_API_SECRET',
        symbol='BTC/USDT',
        interval='1h',
        model_path='formula_lstm_model_BTC_1h.keras',
        scaler_path='scaler_config_BTC_1h.json',
        formula_file='discovered_formulas_BTC_1h.json',
        risk_per_trade=0.02  # 2% risk per trade
    )
    
    # Start trading
    trader.start_monitoring(check_interval=60)  # Check every 60 seconds
