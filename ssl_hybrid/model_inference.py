#!/usr/bin/env python3
"""
SSL Hybrid Model Inference
How to load and use trained models for prediction
"""

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SSLHybridPredictor:
    """Load and use trained SSL Hybrid models"""
    
    def __init__(self, model_path, scaler_path, metadata_path):
        """
        Initialize predictor with trained model and scaler
        
        Args:
            model_path: Path to .keras model file (e.g., 'ssl_filter_v3.keras')
            scaler_path: Path to scaler JSON (e.g., 'ssl_scaler_v3.json')
            metadata_path: Path to metadata JSON (e.g., 'ssl_metadata_v3.json')
        """
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(scaler_data['mean'])
        self.scaler.scale_ = np.array(scaler_data['scale'])
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['features']
        print(f"Model loaded successfully!")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Accuracy: {self.metadata.get('accuracy', 'N/A')}")
        print(f"  AUC: {self.metadata.get('auc', 'N/A')}")
    
    def predict(self, features_dict):
        """
        Make prediction on a single sample
        
        Args:
            features_dict: Dict with feature names as keys
            
        Returns:
            Dict with predictions and confidence
        """
        # Extract features in correct order
        feature_vector = np.array([features_dict[name] for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)
        
        # Normalize
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction_prob = self.model.predict(feature_scaled, verbose=0)[0][0]
        prediction = int(prediction_prob > 0.5)
        confidence = max(prediction_prob, 1 - prediction_prob)
        
        return {
            'signal': 'TRUE' if prediction == 1 else 'FALSE',
            'confidence': float(confidence),
            'probability': float(prediction_prob),
            'recommendation': 'ENTER' if prediction == 1 and confidence > 0.7 else 'WAIT'
        }
    
    def predict_batch(self, features_list):
        """
        Make predictions on multiple samples
        
        Args:
            features_list: List of feature dicts
            
        Returns:
            List of prediction dicts
        """
        n_samples = len(features_list)
        X = np.zeros((n_samples, len(self.feature_names)))
        
        for i, features in enumerate(features_list):
            X[i] = np.array([features[name] for name in self.feature_names])
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        
        results = []
        for prob in predictions:
            pred = int(prob > 0.5)
            conf = max(prob, 1 - prob)
            results.append({
                'signal': 'TRUE' if pred == 1 else 'FALSE',
                'confidence': float(conf),
                'probability': float(prob),
                'recommendation': 'ENTER' if pred == 1 and conf > 0.7 else 'WAIT'
            })
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Load model
    print("\n" + "="*80)
    print("SSL HYBRID MODEL INFERENCE EXAMPLES")
    print("="*80)
    
    print("\nExample 1: Loading v3 model")
    print("-" * 80)
    
    try:
        predictor = SSLHybridPredictor(
            model_path='ssl_filter_v3.keras',
            scaler_path='ssl_scaler_v3.json',
            metadata_path='ssl_metadata_v3.json'
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model files are in the current directory")
        exit(1)
    
    # Example 2: Single prediction
    print("\nExample 2: Single prediction")
    print("-" * 80)
    
    # Create sample features (all v3 model's 17 features)
    sample_features = {
        'atr_ratio': 0.02,
        'avg_return_strength': 0.5,
        'bb_distance_mid': 0.3,
        'bb_position': 0.7,
        'macd_bullish': 1.0,
        'macd_hist': 0.02,
        'macd_signal_dist': 0.01,
        'momentum_10': 0.005,
        'momentum_5': 0.003,
        'multi_tf_confirmations': 0.67,
        'price_range_position': 0.6,
        'rsi14': 0.35,  # 35 out of 100
        'rsi14_from_neutral': 0.3,
        'rsi_trend': 0.05,
        'signal_type': 1.0,
        'volatility': 0.15,
        'volume_ratio': 1.5
    }
    
    result = predictor.predict(sample_features)
    
    print(f"Features:")
    for key, value in sample_features.items():
        print(f"  {key:25s}: {value:8.4f}")
    
    print(f"\nPrediction Result:")
    print(f"  Signal:         {result['signal']}")
    print(f"  Confidence:     {result['confidence']:.4f}")
    print(f"  Probability:    {result['probability']:.4f}")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Example 3: Batch prediction
    print("\n\nExample 3: Batch prediction (multiple signals)")
    print("-" * 80)
    
    # Create 3 sample signals
    samples = [
        {
            'atr_ratio': 0.02, 'avg_return_strength': 0.5, 'bb_distance_mid': 0.3,
            'bb_position': 0.7, 'macd_bullish': 1.0, 'macd_hist': 0.02,
            'macd_signal_dist': 0.01, 'momentum_10': 0.005, 'momentum_5': 0.003,
            'multi_tf_confirmations': 0.67, 'price_range_position': 0.6,
            'rsi14': 0.35, 'rsi14_from_neutral': 0.3, 'rsi_trend': 0.05,
            'signal_type': 1.0, 'volatility': 0.15, 'volume_ratio': 1.5
        },
        {
            'atr_ratio': 0.018, 'avg_return_strength': 0.3, 'bb_distance_mid': 0.5,
            'bb_position': 0.5, 'macd_bullish': 0.0, 'macd_hist': -0.01,
            'macd_signal_dist': -0.02, 'momentum_10': -0.002, 'momentum_5': -0.001,
            'multi_tf_confirmations': 0.33, 'price_range_position': 0.4,
            'rsi14': 0.55, 'rsi14_from_neutral': 0.05, 'rsi_trend': -0.03,
            'signal_type': 0.0, 'volatility': 0.12, 'volume_ratio': 0.9
        },
        {
            'atr_ratio': 0.025, 'avg_return_strength': 0.7, 'bb_distance_mid': -0.2,
            'bb_position': 0.15, 'macd_bullish': 1.0, 'macd_hist': 0.03,
            'macd_signal_dist': 0.015, 'momentum_10': 0.008, 'momentum_5': 0.006,
            'multi_tf_confirmations': 1.0, 'price_range_position': 0.2,
            'rsi14': 0.25, 'rsi14_from_neutral': 0.5, 'rsi_trend': 0.08,
            'signal_type': 1.0, 'volatility': 0.18, 'volume_ratio': 2.0
        }
    ]
    
    results = predictor.predict_batch(samples)
    
    for i, (sample, result) in enumerate(zip(samples, results), 1):
        print(f"\nSignal {i}:")
        print(f"  Signal:         {result['signal']}")
        print(f"  Confidence:     {result['confidence']:.4f}")
        print(f"  Probability:    {result['probability']:.4f}")
        print(f"  Recommendation: {result['recommendation']}")
    
    # Example 4: How to use in trading bot
    print("\n\nExample 4: Integration with trading bot")
    print("-" * 80)
    print("""
In your trading bot, use the predictor like this:

    # Load model once at startup
    predictor = SSLHybridPredictor(
        'ssl_filter_v3.keras',
        'ssl_scaler_v3.json', 
        'ssl_metadata_v3.json'
    )
    
    # On each signal from SSL Hybrid indicator:
    def on_ssl_signal(close, high, low, volume, indicator):
        # Extract features (same as training)
        features = extract_features_v3(signal, close, high, low, volume, indicator)
        
        # Get prediction
        result = predictor.predict(features)
        
        # Trade logic
        if result['signal'] == 'TRUE' and result['confidence'] > 0.7:
            enter_trade()
        else:
            skip_trade()
    """)
    
    print("\n" + "="*80)
    print("Ready to integrate with your bot!")
    print("="*80)
