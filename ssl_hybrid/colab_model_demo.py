#!/usr/bin/env python3
"""
SSL Hybrid Model - Live Demo
How to use the trained model for real-time prediction
"""

print("\n" + "="*80)
print("SSL HYBRID MODEL - LIVE DEMO")
print("="*80)

print("\nStep 1: 安裝必要套件")
import subprocess
import sys

try:
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    print("  OK - 所有套件都已安裝")
except ImportError:
    print("  安裝 TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "-q"])
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    print("  OK")

print("\nStep 2: 從 GitHub 下載 v3 模型檔案")
import requests

try:
    print("  下載 ssl_filter_v3.keras...")
    url = 'https://github.com/caizongxun/v2bot/raw/main/ssl_hybrid/ssl_filter_v3.keras'
    r = requests.get(url, timeout=60)
    with open('ssl_filter_v3.keras', 'wb') as f:
        f.write(r.content)
    print(f"    OK ({len(r.content)/1024/1024:.1f} MB)")
    
    print("  下載 ssl_scaler_v3.json...")
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_scaler_v3.json'
    r = requests.get(url, timeout=60)
    with open('ssl_scaler_v3.json', 'w') as f:
        f.write(r.text)
    print("    OK")
    
    print("  下載 ssl_metadata_v3.json...")
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/ssl_hybrid/ssl_metadata_v3.json'
    r = requests.get(url, timeout=60)
    with open('ssl_metadata_v3.json', 'w') as f:
        f.write(r.text)
    print("    OK")
except Exception as e:
    print(f"  錯誤: {e}")
    print("  注意: 模型檔案可能尚未上傳到 GitHub")
    print("  請先手動上傳: ssl_filter_v3.keras, ssl_scaler_v3.json, ssl_metadata_v3.json")
    exit(1)

print("\nStep 3: 載入模型")
try:
    model = keras.models.load_model('ssl_filter_v3.keras')
    print(f"  模型參數: {model.count_params():,}")
    
    with open('ssl_scaler_v3.json', 'r') as f:
        scaler_data = json.load(f)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data['mean'])
    scaler.scale_ = np.array(scaler_data['scale'])
    
    with open('ssl_metadata_v3.json', 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features']
    print(f"  特徵數量: {len(feature_names)}")
    print(f"  準確率: {metadata.get('accuracy', 'N/A')}")
    print(f"  AUC: {metadata.get('auc', 'N/A')}")
    print("  OK")
except Exception as e:
    print(f"  錯誤: {e}")
    exit(1)

print("\nStep 4: 定義預測函數")

def predict_signal(features_dict, threshold=0.5, confidence_threshold=0.7):
    """
    預測 SSL Hybrid 信號
    
    Args:
        features_dict: 包含所有特徵的字典
        threshold: 預測閾值 (預設 0.5)
        confidence_threshold: 進場置信度閾值 (預設 0.7)
        
    Returns:
        dict: 包含預測結果
    """
    try:
        # 提取特徵
        feature_vector = np.array([features_dict[name] for name in feature_names])
        feature_vector = feature_vector.reshape(1, -1)
        
        # 標準化
        feature_scaled = scaler.transform(feature_vector)
        
        # 預測
        prediction_prob = model.predict(feature_scaled, verbose=0)[0][0]
        prediction = int(prediction_prob > threshold)
        confidence = max(prediction_prob, 1 - prediction_prob)
        
        # 建議
        recommendation = 'ENTER' if prediction == 1 and confidence > confidence_threshold else 'SKIP'
        
        return {
            'signal': 'TRUE' if prediction == 1 else 'FALSE',
            'probability': float(prediction_prob),
            'confidence': float(confidence),
            'recommendation': recommendation,
            'risk': 'LOW' if confidence > 0.9 else 'MEDIUM' if confidence > 0.7 else 'HIGH'
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
    'avg_return_strength': 0.8,  # 高
    'bb_distance_mid': 0.3,
    'bb_position': 0.2,           # 靠近下緣 -> 買入
    'macd_bullish': 1.0,          # 正值
    'macd_hist': 0.025,           # 正值
    'macd_signal_dist': 0.015,    # 正值
    'momentum_10': 0.008,         # 正動量
    'momentum_5': 0.006,
    'multi_tf_confirmations': 1.0,  # 完全確認
    'price_range_position': 0.15, # 近期低點
    'rsi14': 0.28,                # 過冷 (28/100)
    'rsi14_from_neutral': 0.44,   # 遠離中立
    'rsi_trend': 0.08,            # 上升趨勢
    'signal_type': 1.0,           # 買入信號
    'volatility': 0.18,
    'volume_ratio': 2.0            # 成交量高
}

result_1 = predict_signal(test_signal_1)
print(f"  信號: {result_1['signal']}")
print(f"  概率: {result_1['probability']:.4f}")
print(f"  置信度: {result_1['confidence']:.2%}")
print(f"  建議: {result_1['recommendation']}")
print(f"  風險: {result_1['risk']}")

# 測試樣本 2: 中立信號
print("\n測試樣本 2: 中立信號")
test_signal_2 = {
    'atr_ratio': 0.015,
    'avg_return_strength': 0.3,   # 低
    'bb_distance_mid': 0.0,
    'bb_position': 0.5,            # 中線
    'macd_bullish': 0.5,           # 接近零
    'macd_hist': 0.002,            # 接近零
    'macd_signal_dist': 0.001,
    'momentum_10': 0.001,
    'momentum_5': 0.0,
    'multi_tf_confirmations': 0.33,  # 低確認
    'price_range_position': 0.5,   # 中點
    'rsi14': 0.50,                 # 中立 (50/100)
    'rsi14_from_neutral': 0.0,
    'rsi_trend': 0.0,
    'signal_type': 1.0,
    'volatility': 0.10,
    'volume_ratio': 1.0            # 正常成交量
}

result_2 = predict_signal(test_signal_2)
print(f"  信號: {result_2['signal']}")
print(f"  概率: {result_2['probability']:.4f}")
print(f"  置信度: {result_2['confidence']:.2%}")
print(f"  建議: {result_2['recommendation']}")
print(f"  風險: {result_2['risk']}")

# 測試樣本 3: 強勢賣出信號
print("\n測試樣本 3: 強勢賣出信號")
test_signal_3 = {
    'atr_ratio': 0.022,
    'avg_return_strength': 0.7,   # 高
    'bb_distance_mid': -0.3,
    'bb_position': 0.85,           # 靠近上緣 -> 賣出
    'macd_bullish': 0.0,           # 負值
    'macd_hist': -0.02,            # 負值
    'macd_signal_dist': -0.012,    # 負值
    'momentum_10': -0.006,         # 負動量
    'momentum_5': -0.004,
    'multi_tf_confirmations': 1.0, # 完全確認
    'price_range_position': 0.85,  # 近期高點
    'rsi14': 0.72,                 # 過熱 (72/100)
    'rsi14_from_neutral': 0.44,    # 遠離中立
    'rsi_trend': -0.08,            # 下降趨勢
    'signal_type': 0.0,            # 賣出信號
    'volatility': 0.16,
    'volume_ratio': 1.8            # 成交量高
}

result_3 = predict_signal(test_signal_3)
print(f"  信號: {result_3['signal']}")
print(f"  概率: {result_3['probability']:.4f}")
print(f"  置信度: {result_3['confidence']:.2%}")
print(f"  建議: {result_3['recommendation']}")
print(f"  風險: {result_3['risk']}")

print("\n" + "="*80)
print("STEP 6: 批量預測")
print("="*80)

print("\n生成 10 個隨機信號並批量預測...")

random_signals = []
np.random.seed(42)

for i in range(10):
    signal = {
        'atr_ratio': np.random.uniform(0.01, 0.03),
        'avg_return_strength': np.random.uniform(0, 1),
        'bb_distance_mid': np.random.uniform(-0.5, 0.5),
        'bb_position': np.random.uniform(0, 1),
        'macd_bullish': float(np.random.choice([0, 1])),
        'macd_hist': np.random.uniform(-0.03, 0.03),
        'macd_signal_dist': np.random.uniform(-0.02, 0.02),
        'momentum_10': np.random.uniform(-0.01, 0.01),
        'momentum_5': np.random.uniform(-0.01, 0.01),
        'multi_tf_confirmations': np.random.uniform(0, 1),
        'price_range_position': np.random.uniform(0, 1),
        'rsi14': np.random.uniform(0, 1),
        'rsi14_from_neutral': np.random.uniform(0, 1),
        'rsi_trend': np.random.uniform(-0.1, 0.1),
        'signal_type': float(np.random.choice([0, 1])),
        'volatility': np.random.uniform(0.05, 0.25),
        'volume_ratio': np.random.uniform(0.5, 3)
    }
    random_signals.append(signal)

# 批量預測
print("\n進行中...")
predictions = []
for signal in random_signals:
    result = predict_signal(signal)
    predictions.append(result)

# 統計
true_count = sum(1 for p in predictions if p['signal'] == 'TRUE')
false_count = sum(1 for p in predictions if p['signal'] == 'FALSE')
enter_count = sum(1 for p in predictions if p['recommendation'] == 'ENTER')
skip_count = sum(1 for p in predictions if p['recommendation'] == 'SKIP')
avg_confidence = np.mean([p['confidence'] for p in predictions])

print(f"\n結果統計:")
print(f"  真實信號: {true_count}/10")
print(f"  虛假信號: {false_count}/10")
print(f"  建議進場: {enter_count}/10")
print(f"  建議跳過: {skip_count}/10")
print(f"  平均置信度: {avg_confidence:.2%}")

print(f"\n詳細結果:")
for i, pred in enumerate(predictions, 1):
    status = "✓" if pred['recommendation'] == 'ENTER' else "✗"
    print(f"  {i:2d}. {pred['signal']:5s} | 置信度 {pred['confidence']:.2%} | {pred['recommendation']:4s} | 風險 {pred['risk']:6s} {status}")

print("\n" + "="*80)
print("模型已準備好用於實時交易!")
print("="*80)
print(f"""
後續步驟:
1. 將此代碼集成到您的交易機器人
2. 在 SSL Hybrid 指標產生信號時調用 predict_signal()
3. 監控置信度和推薦
4. 記錄每筆交易的結果
5. 定期重新訓練模型

關鍵參數:
- 置信度閾值: 0.7 (70%) 以上才進場
- 預測概率: > 0.5 代表真實信號
- 風險等級: HIGH/MEDIUM/LOW

注意: v3 模型可能存在過度擬合,建議使用 v4 模型進行生產環境
""")
