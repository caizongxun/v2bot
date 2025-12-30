#!/usr/bin/env python3
"""
Colab File Downloader - Download trained models easily

Usage in Colab:
    from google.colab import files
    import requests
    
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/strategy_design/colab_downloader.py'
    script = requests.get(url).text
    exec(script)
"""

print("\n" + "="*80)
print("COLAB MODEL DOWNLOADER")
print("="*80)

from google.colab import files
import os

# Parameters - Modify these to match your training
SYMBOL = 'BTC'
INTERVAL = '1h'

print(f"\nDownloading {SYMBOL} {INTERVAL} model files...\n")

files_to_download = [
    f'formula_lstm_model_{SYMBOL}_{INTERVAL}.keras',
    f'scaler_config_{SYMBOL}_{INTERVAL}.json',
    f'discovered_formulas_{SYMBOL}_{INTERVAL}.json'
]

for filename in files_to_download:
    if os.path.exists(filename):
        print(f"Downloading {filename}...")
        files.download(filename)
        print(f"  ✓ {filename} downloaded")
    else:
        print(f"  ✗ {filename} not found (ensure training completed)")

print("\n" + "="*80)
print("All files downloaded successfully!")
print("="*80)

print("\nNext steps:")
print("1. Place downloaded files in your project directory")
print(f"2. Download real_time_predictor_v2.py from GitHub")
print("3. Use RealTimeFormulaPredictor for live trading")
print("\nExample:")
print("""
from real_time_predictor_v2 import RealTimeFormulaPredictor

predictor = RealTimeFormulaPredictor(
    model_path='formula_lstm_model_BTC_1h.keras',
    scaler_path='scaler_config_BTC_1h.json',
    formula_file='discovered_formulas_BTC_1h.json'
)

result = predictor.process_new_kline(kline_dict)
print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']:.1%}")
""")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80 + "\n")
