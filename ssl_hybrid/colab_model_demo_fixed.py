#!/usr/bin/env python3
"""
SSL Hybrid Model - Live Demo (FIXED)
Proper file handling and error recovery
"""

print("\n" + "="*80)
print("SSL HYBRID MODEL - LIVE DEMO (FIXED)")
print("="*80)

print("\nStep 1: 安裝必要套件")
import subprocess
import sys

try:
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    import requests
    print("  OK - 所有套件都已安裝")
except ImportError:
    print("  安裝中...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "-q"])
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    import requests
    print("  OK")

print("\nStep 2: 從 GitHub 下載 v3 模型檔案")
import os
import time

files_to_download = [
    ('ssl_filter_v3.keras', 'https://github.com/caizongxun/v2bot/raw/main/ssl_hybrid/ssl_filter_v3.keras'),
    ('ssl_scaler_v3.json', 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_scaler_v3.json'),
    ('ssl_metadata_v3.json', 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_metadata_v3.json')
]

downloaded_files = {}

for filename, url in files_to_download:
    print(f"  下載 {filename}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'wb' if filename.endswith('.keras') else 'w') as f:
            if filename.endswith('.keras'):
                f.write(response.content)
            else:
                f.write(response.text)
        
        downloaded_files[filename] = True
        file_size = os.path.getsize(filename) / 1024
        print(f"    OK ({file_size:.1f} KB)")
        time.sleep(0.5)  # 避免速率限制
    except Exception as e:
        downloaded_files[filename] = False
        print(f"    警告: {e}")

print("\nStep 3: 載入模型")

# 檢查檔案是否存在
if not downloaded_files.get('ssl_filter_v3.keras', False):
    print("  ❌ ssl_filter_v3.keras 下載失敗")
    print("  注意: 模型檔案可能尚未上傳到 GitHub")
    print("  使用預設特徵進行演示...\n")
    use_demo_mode = True
else:
    use_demo_mode = False

if not use_demo_mode:
    try:
        print("  載入 Keras 模型...")
        model = keras.models.load_model('ssl_filter_v3.keras')
        print(f"    模型參數: {model.count_params():,}")
        
        print("  載入 scaler...")
        with open('ssl_scaler_v3.json', 'r') as f:
            scaler_data = json.load(f)
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_data['mean'])
        scaler.scale_ = np.array(scaler_data['scale'])
        print(f"    特徵數: {len(scaler.mean_)}")
        
        print("  載入元數據...")
        with open('ssl_metadata_v3.json', 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata['features']
        print(f"  特徵列表: {len(feature_names)} 個")
        print("  OK - 模型已成功載入")
        
    except Exception as e:
        print(f"  ❌ 模型載入失敗: {e}")
        print("  切換到演示模式...\n")
        use_demo_mode = True
else:
    feature_names = [
        'atr_ratio', 'avg_return_strength', 'bb_distance_mid', 'bb_position',
        'macd_bullish', 'macd_hist', 'macd_signal_dist', 'momentum_10',
        'momentum_5', 'multi_tf_confirmations', 'price_range_position', 'rsi14',
        'rsi14_from_neutral', 'rsi_trend', 'signal_type', 'volatility', 'volume_ratio'
    ]
    model = None
    scaler = None
    metadata = None

print("\nStep 4: 定義預測函數")

if use_demo_mode:
    print("  (演示模式 - 使用隨機權重)")
    
    # 演示模式: 使用簡單啟發式規則
    def predict_signal(features_dict):
        """
        演示模式預測函數
        基於特徵啟發式規則進行預測
        """
        try:
            # 計算信號強度
            rsi_strength = 0
            if 'rsi14' in features_dict:
                rsi_val = features_dict['rsi14'] * 100
                if rsi_val < 40:
                    rsi_strength = (40 - rsi_val) / 40  # 0-1
                elif rsi_val > 60:
                    rsi_strength = (rsi_val - 60) / 40  # 0-1
            
            # MACD 信號
            macd_strength = 0
            if 'macd_bullish' in features_dict:
                macd_strength = features_dict['macd_bullish']
            
            # 成交量確認
            volume_strength = 0
            if 'volume_ratio' in features_dict:
                volume_strength = min(1.0, features_dict['volume_ratio'] / 2.0)
            
            # 綜合信號
            total_score = (rsi_strength * 0.4 + macd_strength * 0.3 + volume_strength * 0.3)
            
            # 轉換為概率
            probability = 0.5 + total_score * 0.3  # 50% - 80% 範圍
            probability = np.clip(probability, 0.0, 1.0)
            
            prediction = int(probability > 0.5)
            confidence = max(probability, 1 - probability)
            
            return {
                'signal': 'TRUE' if prediction == 1 else 'FALSE',
                'probability': float(probability),
                'confidence': float(confidence),
                'recommendation': 'ENTER' if prediction == 1 and confidence > 0.65 else 'SKIP',
                'risk': 'LOW' if confidence > 0.8 else 'MEDIUM' if confidence > 0.65 else 'HIGH',
                'mode': 'DEMO'
            }
        except Exception as e:
            return {'error': str(e)}
else:
    # 真實模式: 使用訓練的模型
    def predict_signal(features_dict, threshold=0.5, confidence_threshold=0.65):
        """
        使用訓練模型的預測函數
        """
        try:
            # 驗證特徵
            missing = set(feature_names) - set(features_dict.keys())
            if missing:
                return {'error': f'缺少特徵: {missing}'}
            
            # 提取特徵
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
                'recommendation': 'ENTER' if prediction == 1 and confidence > confidence_threshold else 'SKIP',
                'risk': 'LOW' if confidence > 0.85 else 'MEDIUM' if confidence > 0.65 else 'HIGH',
                'mode': 'REAL'
            }
        except Exception as e:
            return {'error': str(e)}

print("  OK")

print("\nStep 5: 測試預測")
print("-" * 80)

# 測試樣本 1: 強勢買入信號
print("\n測試樣本 1: 強勢買入信號")
test_signal_1 = {
    'atr_ratio': 0.02,
    'avg_return_strength': 0.8,
    'bb_distance_mid': 0.3,
    'bb_position': 0.2,
    'macd_bullish': 1.0,
    'macd_hist': 0.025,
    'macd_signal_dist': 0.015,
    'momentum_10': 0.008,
    'momentum_5': 0.006,
    'multi_tf_confirmations': 1.0,
    'price_range_position': 0.15,
    'rsi14': 0.28,
    'rsi14_from_neutral': 0.44,
    'rsi_trend': 0.08,
    'signal_type': 1.0,
    'volatility': 0.18,
    'volume_ratio': 2.0
}

result_1 = predict_signal(test_signal_1)
if 'error' in result_1:
    print(f"  ❌ 錯誤: {result_1['error']}")
else:
    print(f"  信號: {result_1['signal']}")
    print(f"  概率: {result_1['probability']:.4f}")
    print(f"  置信度: {result_1['confidence']:.2%}")
    print(f"  建議: {result_1['recommendation']}")
    print(f"  風險: {result_1['risk']}")
    print(f"  模式: {result_1['mode']}")

# 測試樣本 2: 中立信號
print("\n測試樣本 2: 中立信號")
test_signal_2 = {
    'atr_ratio': 0.015,
    'avg_return_strength': 0.3,
    'bb_distance_mid': 0.0,
    'bb_position': 0.5,
    'macd_bullish': 0.5,
    'macd_hist': 0.002,
    'macd_signal_dist': 0.001,
    'momentum_10': 0.001,
    'momentum_5': 0.0,
    'multi_tf_confirmations': 0.33,
    'price_range_position': 0.5,
    'rsi14': 0.50,
    'rsi14_from_neutral': 0.0,
    'rsi_trend': 0.0,
    'signal_type': 1.0,
    'volatility': 0.10,
    'volume_ratio': 1.0
}

result_2 = predict_signal(test_signal_2)
if 'error' in result_2:
    print(f"  ❌ 錯誤: {result_2['error']}")
else:
    print(f"  信號: {result_2['signal']}")
    print(f"  概率: {result_2['probability']:.4f}")
    print(f"  置信度: {result_2['confidence']:.2%}")
    print(f"  建議: {result_2['recommendation']}")
    print(f"  風險: {result_2['risk']}")
    print(f"  模式: {result_2['mode']}")

# 測試樣本 3: 強勢賣出信號
print("\n測試樣本 3: 強勢賣出信號")
test_signal_3 = {
    'atr_ratio': 0.022,
    'avg_return_strength': 0.7,
    'bb_distance_mid': -0.3,
    'bb_position': 0.85,
    'macd_bullish': 0.0,
    'macd_hist': -0.02,
    'macd_signal_dist': -0.012,
    'momentum_10': -0.006,
    'momentum_5': -0.004,
    'multi_tf_confirmations': 1.0,
    'price_range_position': 0.85,
    'rsi14': 0.72,
    'rsi14_from_neutral': 0.44,
    'rsi_trend': -0.08,
    'signal_type': 0.0,
    'volatility': 0.16,
    'volume_ratio': 1.8
}

result_3 = predict_signal(test_signal_3)
if 'error' in result_3:
    print(f"  ❌ 錯誤: {result_3['error']}")
else:
    print(f"  信號: {result_3['signal']}")
    print(f"  概率: {result_3['probability']:.4f}")
    print(f"  置信度: {result_3['confidence']:.2%}")
    print(f"  建議: {result_3['recommendation']}")
    print(f"  風險: {result_3['risk']}")
    print(f"  模式: {result_3['mode']}")

print("\n" + "="*80)
print("演示完成")
print("="*80)

if use_demo_mode:
    print("\n⚠️ 注意: 目前運行在演示模式")
    print("   - 使用啟發式規則而非訓練模型")
    print("   - 預測結果僅供參考")
    print("   - 完整的 v3 模型檔案需要上傳到 GitHub")
    print("\n要使用真實模型，請確保以下檔案在 GitHub:")
    print("  - ssl_filter_v3.keras (~10 MB)")
    print("  - ssl_scaler_v3.json")
    print("  - ssl_metadata_v3.json")
else:
    print("\n✅ 使用了真實訓練模型")
    print(f"   準確率: {metadata.get('accuracy', 'N/A')}")
    print(f"   AUC: {metadata.get('auc', 'N/A')}")

print("\n" + "="*80)
