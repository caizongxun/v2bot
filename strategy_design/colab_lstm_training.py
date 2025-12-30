#!/usr/bin/env python3
"""
Colab LSTM Training Script

Usage in Google Colab:
    !pip install -q tensorflow pandas numpy scikit-learn huggingface-hub ta
    !git clone https://github.com/caizongxun/v2bot.git
    %run v2bot/strategy_design/colab_lstm_training.py

This script trains LSTM model on 5 synthetic indicators generated from formulas.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pickle
import warnings
import ta

warnings.filterwarnings('ignore')


class ColabFormulaLSTMTrainer:
    """Train LSTM model in Colab using discovered formulas"""
    
    def __init__(self, formula_file='discovered_formulas.json'):
        self.formulas_config = None
        self.df = None
        self.formula_values = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.load_formulas(formula_file)
    
    def load_formulas(self, formula_file):
        """Load discovered formulas"""
        print("Loading formulas...")
        try:
            with open(formula_file) as f:
                self.formulas_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Formula file not found: {formula_file}")
        
        print(f"Loaded {len(self.formulas_config)} formulas:")
        for name, info in self.formulas_config.items():
            eq = info['equation']
            if len(eq) > 50:
                eq = eq[:47] + "..."
            print(f"  {name}: {eq}")
    
    def load_data(self, symbol='BTC', interval='1h'):
        """Load data from HF"""
        print(f"\nLoading {symbol} {interval} data from HF...")
        file_path = hf_hub_download(
            repo_id='zongowo111/v2-crypto-ohlcv-data',
            filename=f'klines/{symbol}USDT/{symbol}_{interval}.parquet',
            repo_type='dataset'
        )
        self.df = pd.read_parquet(file_path)
        print(f"Loaded {len(self.df)} rows")
        return self.df
    
    def compute_indicators(self):
        """Compute all required indicators"""
        if self.df is None:
            raise ValueError("Load data first")
        
        print("\nComputing indicators...")
        df = self.df.copy()
        
        # Momentum
        df['rsi_7'] = ta.momentum.rsi(df['close'], 7)
        df['rsi_14'] = ta.momentum.rsi(df['close'], 14)
        
        # MACD
        df['macd_line'] = ta.trend.macd_line(df['close'], 12, 26)
        df['macd_signal'] = ta.trend.macd_signal_line(df['close'], 12, 26, 9)
        df['macd_diff'] = df['macd_line'] - df['macd_signal']
        
        # Trend
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(12).mean()
        
        # Volatility
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], 20, 2)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], 20, 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, 1)
        
        self.df = df
        print("Indicators computed")
        return self.df
    
    def apply_formulas(self):
        """
        Apply discovered formulas to generate 5 synthetic indicators
        
        For simplicity, we use these formula implementations:
        (In real scenario, parse and compile discovered equations)
        """
        if self.df is None:
            raise ValueError("Compute indicators first")
        
        print("\nApplying formulas to generate synthetic indicators...")
        df = self.df
        n = len(df)
        
        # Ensure no NaN before formula application
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        formula_values = np.zeros((n, 5))
        
        for i in range(n):
            if i % 5000 == 0:
                print(f"  Progress: {i}/{n}")
            
            row = df.iloc[i]
            
            # Formula 1: RSI-MACD blend
            f1 = (row['rsi_14'] * 0.4 + row['macd_diff'] * 0.3 + row['sma_20'] * 0.3)
            
            # Formula 2: Volume-ATR logarithmic
            f2 = np.log(abs(row['atr_14'] * row['volume_ratio']) + 1e-8)
            
            # Formula 3: Bollinger-RSI ratio
            f3 = (row['bb_width'] / (row['rsi_7'] + 1e-8))
            
            # Formula 4: MACD-ATR divergence
            f4 = (row['macd_diff'] / (row['atr_14'] + 1e-8))
            
            # Formula 5: Volume-SMA interaction
            f5 = np.tanh(row['volume_ratio']) * row['sma_20']
            
            formula_values[i] = [f1, f2, f3, f4, f5]
        
        # Handle NaN and Inf
        formula_values = np.nan_to_num(formula_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.formula_values = formula_values
        print(f"Formula values shape: {formula_values.shape}")
        return formula_values
    
    def create_labels(self, lookahead=24, threshold=0.005):
        """Create training labels"""
        if self.df is None:
            raise ValueError("Load data first")
        
        print("\nCreating labels...")
        df = self.df.copy()
        
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['label'] = 1  # HOLD
        df.loc[df['future_return'] > threshold, 'label'] = 2   # BUY
        df.loc[df['future_return'] < -threshold, 'label'] = 0  # SELL
        
        y = df['label'].values
        
        print(f"Label distribution:")
        print(f"  SELL (0): {(y == 0).sum()} ({(y == 0).mean():.2%})")
        print(f"  HOLD (1): {(y == 1).sum()} ({(y == 1).mean():.2%})")
        print(f"  BUY (2): {(y == 2).sum()} ({(y == 2).mean():.2%})")
        
        return y
    
    def split_data(self, y_labels):
        """Time-series aware train/val/test split"""
        print("\nSplitting data...")
        
        X = self.formula_values
        n = len(X)
        
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        X_train = X[:train_size]
        y_train = y_labels[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y_labels[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y_labels[train_size+val_size:]
        
        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    
    def create_sequences(self, X, y, lookback=30):
        """Create time-series sequences"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, lookback=30):
        """Build LSTM model"""
        print("\nBuilding LSTM model...")
        
        model = keras.Sequential([
            # First LSTM layer
            keras.layers.LSTM(64, activation='relu', return_sequences=True,
                            input_shape=(lookback, 5)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            
            # Second LSTM layer
            keras.layers.LSTM(32, activation='relu', return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            
            # Dense layers
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),
            
            # Output layer (3-class)
            keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the model"""
        print(f"\nTraining model ({epochs} epochs)...\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\nEvaluating model...")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred_classes):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred_classes, average='weighted', zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred_classes, average='weighted', zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(y_test, y_pred_classes, average='weighted', zero_division=0):.4f}")
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                   target_names=['SELL', 'HOLD', 'BUY'],
                                   zero_division=0))
    
    def save(self, model_path='formula_lstm_model.h5', scaler_path='scaler_config.pkl'):
        """Save model and scaler"""
        print(f"\nSaving model...")
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("COLAB LSTM TRAINING - FORMULA-BASED SIGNAL PREDICTION")
    print("="*80)
    
    # Initialize trainer
    trainer = ColabFormulaLSTMTrainer('discovered_formulas.json')
    
    # Load and prepare data
    trainer.load_data('BTC', '1h')
    trainer.compute_indicators()
    formula_values = trainer.apply_formulas()
    y_labels = trainer.create_labels()
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.split_data(y_labels)
    
    # Create sequences
    print("\nCreating time-series sequences...")
    lookback = 30
    X_train_seq, y_train_seq = trainer.create_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = trainer.create_sequences(X_val, y_val, lookback)
    X_test_seq, y_test_seq = trainer.create_sequences(X_test, y_test, lookback)
    
    # One-hot encode
    y_train_one_hot = to_categorical(y_train_seq, 3)
    y_val_one_hot = to_categorical(y_val_seq, 3)
    y_test_one_hot = to_categorical(y_test_seq, 3)
    
    print(f"Train seq: {X_train_seq.shape}, Val seq: {X_val_seq.shape}, Test seq: {X_test_seq.shape}")
    
    # Build and train
    trainer.build_model(lookback=lookback)
    trainer.train(X_train_seq, y_train_one_hot, X_val_seq, y_val_one_hot, epochs=50)
    
    # Evaluate
    trainer.evaluate(X_test_seq, y_test_one_hot)
    
    # Save
    trainer.save()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("Download formula_lstm_model.h5 and scaler_config.pkl for deployment")
    print("="*80)


if __name__ == '__main__':
    main()
