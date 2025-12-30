# SSL Hybrid Signal Filter - 使用指南

## 快速開始

### 1. 載入模型

```python
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import json
import numpy as np

# 載入訓練好的模型
model = keras.models.load_model('ssl_filter_v3.keras')

# 載入 scaler
with open('ssl_scaler_v3.json', 'r') as f:
    scaler_data = json.load(f)

scaler = StandardScaler()
scaler.mean_ = np.array(scaler_data['mean'])
scaler.scale_ = np.array(scaler_data['scale'])

# 載入 metadata
with open('ssl_metadata_v3.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['features']
print(f"模型載入完成: {len(feature_names)} 個特徵")
```

---

## 2. 特徵提取

模型需要以下 17 個特徵 (v3)：

| 特徵名 | 含義 | 範圍 |
|------|------|------|
| rsi14 | RSI(14) 值 | 0-1 |
| rsi14_from_neutral | 距離 50 的距離 | 0-1 |
| rsi_trend | RSI 5 根 K 線趨勢 | -1 to 1 |
| macd_hist | MACD histogram 值 | ±1 |
| macd_bullish | MACD 正值 | 0/1 |
| macd_signal_dist | MACD 與 signal 距離 | -1 to 1 |
| bb_position | 價格在 BB 中的位置 | 0-1 |
| bb_distance_mid | 距離 BB 中線 | ±∞ |
| volatility | 波動率 | 0-1 |
| atr_ratio | ATR 相對值 | 0-1 |
| volume_ratio | 成交量比例 | 0-5 |
| momentum_5 | 5 根 K 線動量 | -1 to 1 |
| momentum_10 | 10 根 K 線動量 | -1 to 1 |
| price_range_position | 價格在 50 根 K 線範圍中的位置 | 0-1 |
| multi_tf_confirmations | 多時間框架確認數 | 0-1 |
| avg_return_strength | 預期回報強度 | 0-1 |
| signal_type | 買入(1) 或 賣出(0) | 0/1 |

---

## 3. 單次預測

```python
def predict_signal(features_dict):
    """
    對單個 SSL 信號進行預測
    
    Args:
        features_dict: 包含所有特徵的字典
        
    Returns:
        prediction: 0 (虛假信號) 或 1 (真實信號)
        confidence: 預測置信度 (0-1)
    """
    # 按正確順序提取特徵
    feature_vector = np.array([features_dict[name] for name in feature_names])
    feature_vector = feature_vector.reshape(1, -1)
    
    # 特徵標準化
    feature_scaled = scaler.transform(feature_vector)
    
    # 模型預測
    prediction_prob = model.predict(feature_scaled, verbose=0)[0][0]
    prediction = int(prediction_prob > 0.5)  # 0 or 1
    confidence = max(prediction_prob, 1 - prediction_prob)
    
    return {
        'signal': 'TRUE' if prediction == 1 else 'FALSE',
        'confidence': confidence,
        'probability': prediction_prob,
        'action': 'ENTER' if prediction == 1 and confidence > 0.7 else 'SKIP'
    }

# 使用範例
sample_features = {
    'rsi14': 0.35,
    'rsi14_from_neutral': 0.30,
    'rsi_trend': 0.05,
    'macd_hist': 0.02,
    'macd_bullish': 1.0,
    'macd_signal_dist': 0.01,
    'bb_position': 0.7,
    'bb_distance_mid': 0.3,
    'volatility': 0.15,
    'atr_ratio': 0.02,
    'volume_ratio': 1.5,
    'momentum_5': 0.003,
    'momentum_10': 0.005,
    'price_range_position': 0.6,
    'multi_tf_confirmations': 0.67,
    'avg_return_strength': 0.5,
    'signal_type': 1.0
}

result = predict_signal(sample_features)
print(f"信號預測: {result['signal']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"建議: {result['action']}")
```

---

## 4. 批量預測

```python
def predict_batch(features_list):
    """
    對多個信號進行預測
    
    Args:
        features_list: 特徵字典列表
        
    Returns:
        predictions: 預測結果列表
    """
    n_samples = len(features_list)
    X = np.zeros((n_samples, len(feature_names)))
    
    # 構建特徵矩陣
    for i, features in enumerate(features_list):
        X[i] = np.array([features[name] for name in feature_names])
    
    # 標準化
    X_scaled = scaler.transform(X)
    
    # 預測
    predictions_prob = model.predict(X_scaled, verbose=0).flatten()
    
    results = []
    for prob in predictions_prob:
        pred = int(prob > 0.5)
        conf = max(prob, 1 - prob)
        results.append({
            'signal': 'TRUE' if pred == 1 else 'FALSE',
            'confidence': conf,
            'probability': prob,
            'action': 'ENTER' if pred == 1 and conf > 0.7 else 'SKIP'
        })
    
    return results

# 使用範例 - 預測 100 個信號
signals = [generate_features() for _ in range(100)]
results = predict_batch(signals)

for i, result in enumerate(results):
    if result['action'] == 'ENTER':
        print(f"信號 {i}: {result['signal']} (置信度 {result['confidence']:.2%})")
```

---

## 5. 交易策略集成

```python
class SSLHybridBot:
    def __init__(self, model_path, scaler_path, metadata_path):
        self.model = keras.models.load_model(model_path)
        # ... 載入 scaler 和 metadata
        
    def on_ssl_signal(self, features):
        """
        當收到 SSL Hybrid 信號時調用
        """
        result = self.predict_signal(features)
        
        if result['signal'] == 'TRUE' and result['confidence'] > 0.7:
            # 進場交易
            self.enter_trade(
                size=self.get_position_size(),
                stop_loss=self.calculate_stop_loss(),
                take_profit=self.calculate_take_profit()
            )
        else:
            # 跳過該信號
            self.skip_signal(result['confidence'])
    
    def predict_signal(self, features):
        feature_vector = np.array([features[name] for name in self.feature_names])
        X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        prob = self.model.predict(X_scaled, verbose=0)[0][0]
        
        return {
            'signal': 'TRUE' if prob > 0.5 else 'FALSE',
            'confidence': max(prob, 1 - prob),
            'probability': prob
        }

# 使用
bot = SSLHybridBot(
    'ssl_filter_v3.keras',
    'ssl_scaler_v3.json',
    'ssl_metadata_v3.json'
)

# 在主要交易循環中
while running:
    signal = ssl_indicator.get_signal()
    if signal:
        features = extract_features(signal)
        bot.on_ssl_signal(features)
```

---

## 6. 模型性能

### v3 模型性能
```
準確率:       99.87%
AUC:          100.0%
F1-Score:     99.78%

敏感度 (TPR):  100.0%  (真實信號保留率)
特異度 (TNR):  99.8%   (虛假信號過濾率)

警告: 這些數字可能過度擬合
      建議使用 v4 模型進行生產環境
```

### v4 模型性能 (等待訓練完成)
```
準確率:       60-70%
AUC:          0.60-0.70
F1-Score:     0.60-0.70

驗證方法:     5-Fold Cross-Validation
特性:        無數據洩露，更可靠
```

---

## 7. 常見問題

### Q: 置信度應該設多少？
A: 建議 0.7 (70%) 以上才進場交易

### Q: 能改特徵數量嗎？
A: 不能，必須和訓練時完全相同

### Q: 可以用其他幣種嗎？
A: 需要重新訓練。當前模型針對 BTC 1h

### Q: 能實時預測嗎？
A: 可以，單次預測 < 10ms

### Q: 預測失敗率多少？
A: 約 0.13% (v3，但可能過度擬合)

---

## 8. 下一步

1. 訓練 v4 模型（更可靠）
2. 在模擬環境測試
3. 小額實盤驗證
4. 監控績效和調整

---

## 參考資料

- `model_inference.py` - 完整推理代碼
- `ssl_hybrid_v3_improved.py` - v3 指標計算
- `ssl_hybrid_v4_fixed.py` - v4 指標計算
