#!/usr/bin/env python3
"""
SSL Hybrid v3 - Colab 下載並預測
一份完整的 Colab notebook 細裝脚本
"""

import subprocess
import sys
import os
from pathlib import Path

print("""
================================================================================
SSL HYBRID v3 - COLAB 預測程式
================================================================================
""")

print("\n步驄31: 安裝套件...")

required_packages = ['tensorflow', 'scikit-learn', 'huggingface-hub']

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"  安裝 {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

print("  ✅ 所有套件已安裝")

print("\n步驄32: 從 HuggingFace 下載模型...")

from huggingface_hub import hf_hub_download
import shutil
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

# 配置
HF_USERNAME = "your_username"  # ← 替換為你的 HF 帳號

print(f"\n需要修改的配置:")
print(f"  HF_USERNAME = '{HF_USERNAME}'")
print(f"\n對這一行編輯並替換為你的帳號名")

repo_id = f"{HF_USERNAME}/ssl-hybrid-v3-model"
files_to_download = [
    'ssl_filter_v3.keras',
    'ssl_scaler_v3.json', 
    'ssl_metadata_v3.json'
]

downloaded_files = {}

for filename in files_to_download:
    print(f"\n  下載 {filename}...", end=" ")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        shutil.copy(file_path, filename)
        downloaded_files[filename] = True
        print("✅")
    except Exception as e:
        downloaded_files[filename] = False
        print(f"\n    ❌ 失敗: {e}")

print("\n步驄33: 載入模型...")

if all(downloaded_files.values()):
    try:
        print("  載入 Keras 模型...", end=" ")
        model = keras.models.load_model('ssl_filter_v3.keras')
        print(f"✅ ({model.count_params():,} 參數)")
        
        print("  載入標準化器...", end=" ")
        with open('ssl_scaler_v3.json') as f:
            scaler_data = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_data['mean'])
        scaler.scale_ = np.array(scaler_data['scale'])
        print(f"✅ ({len(scaler.mean_)} 特徵)")
        
        print("  載入元數據...", end=" ")
        with open('ssl_metadata_v3.json') as f:
            metadata = json.load(f)
        feature_names = metadata['features']
        print(f"✅")
        
        use_real_model = True
    except Exception as e:
        print(f"\n  ❌ 檔案載入失敗: {e}")
        use_real_model = False
else:
    print("  ❌ 檔案下載不完整")
    use_real_model = False

if use_real_model:
    print("\n步驄34: 定義預測函數...")
    
    def predict_signal(features_dict, threshold=0.5):
        """
        使用 SSL Hybrid v3 模型預測
        """
        try:
            # 驗證特徵
            missing = set(feature_names) - set(features_dict.keys())
            if missing:
                return {'error': f'缺少特徵: {missing}'}
            
            # 提取特徵向量
            feature_vector = np.array([features_dict[name] for name in feature_names])
            feature_vector = feature_vector.reshape(1, -1)
            
            # 標準化
            feature_scaled = scaler.transform(feature_vector)
            
            # 預測
            prediction_prob = model.predict(feature_scaled, verbose=0)[0][0]
            prediction = int(prediction_prob > threshold)
            confidence = max(prediction_prob, 1 - prediction_prob)
            
            return {
                'signal': 'TRUE' if prediction == 1 else 'FALSE',
                'probability': float(prediction_prob),
                'confidence': float(confidence),
                'recommendation': 'ENTER' if prediction == 1 and confidence > 0.7 else 'SKIP',
                'risk': 'LOW' if confidence > 0.85 else 'MEDIUM' if confidence > 0.65 else 'HIGH'
            }
        except Exception as e:
            return {'error': str(e)}
    
    print("  ✅ 函數已定義")
    
    print("\n步驄35: 測試預測...")
    print("="*80)
    
    # 測試樣本 1: 強勢買入
    print("\n測試 1: 強勢買入信號")
    test_1 = {
        'atr_ratio': 0.02, 'avg_return_strength': 0.8, 'bb_distance_mid': 0.3,
        'bb_position': 0.2, 'macd_bullish': 1.0, 'macd_hist': 0.025,
        'macd_signal_dist': 0.015, 'momentum_10': 0.008, 'momentum_5': 0.006,
        'multi_tf_confirmations': 1.0, 'price_range_position': 0.15,
        'rsi14': 0.28, 'rsi14_from_neutral': 0.44, 'rsi_trend': 0.08,
        'signal_type': 1.0, 'volatility': 0.18, 'volume_ratio': 2.0
    }
    result_1 = predict_signal(test_1)
    if 'error' not in result_1:
        for key, value in result_1.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}" if key == 'probability' else f"  {key:20s}: {value:.2%}")
            else:
                print(f"  {key:20s}: {value}")
    else:
        print(f"  ❌ {result_1['error']}")
    
    # 測試樣本 2: 中立信號
    print("\n測試 2: 中立信號")
    test_2 = {
        'atr_ratio': 0.015, 'avg_return_strength': 0.3, 'bb_distance_mid': 0.0,
        'bb_position': 0.5, 'macd_bullish': 0.5, 'macd_hist': 0.002,
        'macd_signal_dist': 0.001, 'momentum_10': 0.001, 'momentum_5': 0.0,
        'multi_tf_confirmations': 0.33, 'price_range_position': 0.5,
        'rsi14': 0.50, 'rsi14_from_neutral': 0.0, 'rsi_trend': 0.0,
        'signal_type': 1.0, 'volatility': 0.10, 'volume_ratio': 1.0
    }
    result_2 = predict_signal(test_2)
    if 'error' not in result_2:
        for key, value in result_2.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}" if key == 'probability' else f"  {key:20s}: {value:.2%}")
            else:
                print(f"  {key:20s}: {value}")
    else:
        print(f"  ❌ {result_2['error']}")
    
    # 測試樣本 3: 強勢賣出
    print("\n測試 3: 強勢賣出信號")
    test_3 = {
        'atr_ratio': 0.022, 'avg_return_strength': 0.7, 'bb_distance_mid': -0.3,
        'bb_position': 0.85, 'macd_bullish': 0.0, 'macd_hist': -0.02,
        'macd_signal_dist': -0.012, 'momentum_10': -0.006, 'momentum_5': -0.004,
        'multi_tf_confirmations': 1.0, 'price_range_position': 0.85,
        'rsi14': 0.72, 'rsi14_from_neutral': 0.44, 'rsi_trend': -0.08,
        'signal_type': 0.0, 'volatility': 0.16, 'volume_ratio': 1.8
    }
    result_3 = predict_signal(test_3)
    if 'error' not in result_3:
        for key, value in result_3.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}" if key == 'probability' else f"  {key:20s}: {value:.2%}")
            else:
                print(f"  {key:20s}: {value}")
    else:
        print(f"  ❌ {result_3['error']}")
    
    print("\n" + "="*80)
    print("\n✅ 預測完成!")
    print("\n檔案:")
    print(f"  Model params: {model.count_params():,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Scaler: 標準化器")
    
else:
    print("❌ 檔案載入失敗。事項:")
    print("  1. 確認你已經修改 HF_USERNAME")
    print("  2. 確認檔案已上傳到 HuggingFace")
    print("  3. 確認網絡連接")
    print(f"\n槛上訪問:")
    print(f"  https://huggingface.co/datasets/{HF_USERNAME}/ssl-hybrid-v3-model")

print("\n" + "="*80)
print("完成!")
print("="*80)
