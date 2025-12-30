#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 預測模型可視化化脚本

使用方式:
    python run_visualization.py [--output OUTPUT_PATH]

功能:
    ✓ 加載訓練好的模型
    ✓ 對測試集進行預測
    ✓ 產生实時可視化化图表
    ✓ 打印性能統計和交易信號
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelVisualizer:
    def __init__(self, model_path='/tmp/model_final.pkl', dataset_path='/tmp/ml_dataset_v3.pkl'):
        """初始化可視化化器"""
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.dataset = None
        self.y_pred = None
        
    def load_model_and_data(self):
        """加載模型和數據"""
        print("\n[Step 1] 加載模型和数据...")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"  ✓ 模型已加載: {self.model_path}")
        except Exception as e:
            print(f"  ❌ 模型加載失败: {e}")
            return False
            
        try:
            with open(self.dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"  ✓ 数据已加載: {self.dataset_path}")
        except Exception as e:
            print(f"  ❌ 数据加載失败: {e}")
            return False
            
        return True
    
    def forward(self, X):
        """前向傳播"""
        Z1 = np.dot(X, self.model['W1']) + self.model['b1']
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = np.dot(A1, self.model['W2']) + self.model['b2']
        return Z2
    
    def predict(self):
        """進行預測"""
        print("\n[Step 2] 進行模型預測...")
        
        X_test = self.dataset['X_test']
        self.y_pred = self.forward(X_test)
        print(f"  ✓ 预測完成: {self.y_pred.shape}")
        
    def prepare_data(self):
        """整理数据"""
        print("\n[Step 3] 整理数据...")
        
        X_test = self.dataset['X_test']
        y_test = self.dataset['y_test']
        target_names = self.dataset['target_names']
        
        y_index_map = {name: i for i, name in enumerate(target_names)}
        
        # 整理数据，提取每个指標
        data_dict = {}
        
        for i, name in enumerate(target_names):
            data_dict[f'{name}_actual'] = y_test[:, i]
            data_dict[f'{name}_pred'] = self.y_pred[:, i]
        
        self.df = pd.DataFrame(data_dict)
        print(f"  ✓ 数据框: {self.df.shape}")
        
        return target_names
    
    def visualize(self, output_path='/tmp/model_visualization.png'):
        """生成可視化化图表"""
        print("\n[Step 4] 生成可視化化图表...")
        
        n_display = 500
        df_display = self.df.tail(n_display).reset_index(drop=True)
        
        fig, axes = plt.subplots(4, 1, figsize=(18, 14))
        fig.suptitle('BTC 15分鐘 - 模型预測对比可視化化仓轣板', 
                     fontsize=16, fontweight='bold')
        
        color_actual = 'black'
        color_pred = 'blue'
        alpha_actual = 1.0
        alpha_pred = 0.7
        
        # 1. BB通道
        ax1 = axes[0]
        ax1.plot(df_display.index, df_display['BB_Upper_actual'], '-', 
                color=color_actual, linewidth=2.5, label='BB上軌(实际)', zorder=5)
        ax1.plot(df_display.index, df_display['BB_Lower_actual'], '-', 
                color=color_actual, linewidth=2.5, label='BB下軌(实际)', zorder=5)
        ax1.fill_between(df_display.index, df_display['BB_Upper_actual'], 
                        df_display['BB_Lower_actual'], alpha=0.1, color='black')
        
        ax1.plot(df_display.index, df_display['BB_Upper_pred'], '--', 
                color=color_pred, linewidth=1.5, label='BB上軌(预測)', alpha=alpha_pred)
        ax1.plot(df_display.index, df_display['BB_Lower_pred'], '--', 
                color=color_pred, linewidth=1.5, label='BB下軌(预測)', alpha=alpha_pred)
        ax1.fill_between(df_display.index, df_display['BB_Upper_pred'], 
                        df_display['BB_Lower_pred'], alpha=0.1, color='blue')
        
        ax1.set_ylabel('标准化值', fontsize=11, fontweight='bold')
        ax1.set_title('📊 Bollinger Band 通道', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9, ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # 2. Support/Resistance
        ax2 = axes[1]
        ax2.plot(df_display.index, df_display['Support_actual'], '-', 
                color='lime', linewidth=2.5, label='支撉(实际)', zorder=5)
        ax2.plot(df_display.index, df_display['Resistance_actual'], '-', 
                color='red', linewidth=2.5, label='阻力(实际)', zorder=5)
        ax2.fill_between(df_display.index, df_display['Support_actual'], 
                        df_display['Resistance_actual'], alpha=0.1, color='gray')
        
        ax2.plot(df_display.index, df_display['Support_pred'], '--', 
                color='lime', linewidth=1.5, label='支撉(预測)', alpha=alpha_pred)
        ax2.plot(df_display.index, df_display['Resistance_pred'], '--', 
                color='red', linewidth=1.5, label='阻力(预測)', alpha=alpha_pred)
        
        ax2.set_ylabel('标准化值', fontsize=11, fontweight='bold')
        ax2.set_title('🎯 支撉/阻力位', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9, ncol=3)
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3 = axes[2]
        ax3.plot(df_display.index, df_display['RSI_actual'], '-', 
                color=color_actual, linewidth=2.5, label='RSI(实际)', zorder=5)
        ax3.plot(df_display.index, df_display['RSI_pred'], '--', 
                color=color_pred, linewidth=2, label='RSI(预測)', alpha=alpha_pred)
        
        ax3.set_ylabel('标准化值', fontsize=11, fontweight='bold')
        ax3.set_title('📈 RSI 相对强弱指数', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9, ncol=3)
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = axes[3]
        ax4.plot(df_display.index, df_display['MACD_actual'], '-', 
                color=color_actual, linewidth=2, label='MACD(实际)', zorder=5)
        ax4.plot(df_display.index, df_display['MACD_Signal_actual'], '-', 
                color='orange', linewidth=2, label='信號线(实际)', zorder=5)
        ax4.plot(df_display.index, df_display['MACD_pred'], '--', 
                color=color_pred, linewidth=1.5, label='MACD(预測)', alpha=alpha_pred)
        ax4.plot(df_display.index, df_display['MACD_Signal_pred'], '--', 
                color='darkorange', linewidth=1.5, label='信號线(预測)', alpha=alpha_pred)
        ax4.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        
        ax4.set_ylabel('标准化值', fontsize=11, fontweight='bold')
        ax4.set_xlabel('時间 (15分鐘K线)', fontsize=11, fontweight='bold')
        ax4.set_title('🔄 MACD 动量指標', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=9, ncol=3)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 图表已保存: {output_path}")
        plt.show()
    
    def evaluate(self):
        """评估模型性能"""
        print("\n[Step 5] 模型性能评估...")
        print("="*80)
        
        y_test = self.dataset['y_test']
        target_names = self.dataset['target_names']
        
        results = {}
        for i, name in enumerate(target_names):
            actual = y_test[:, i]
            pred = self.y_pred[:, i]
            
            mse = np.mean((actual - pred) ** 2)
            mae = np.mean(np.abs(actual - pred))
            r2 = 1 - (np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
            
            results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
            
            print(f"\n{name:20s}:")
            print(f"  MSE: {mse:.6f}  |  MAE: {mae:.6f}  |  R²: {r2:.6f}")
            print(f"  实际: [{actual.min():.4f}, {actual.max():.4f}]  |  预測: [{pred.min():.4f}, {pred.max():.4f}]")
        
        return results
    
    def print_signals(self):
        """打印交易信號"""
        print("\n[Step 6] 最新交易信號 (最后一根K线)...")
        print("="*80)
        
        latest = self.df.iloc[-1]
        
        print(f"\n📊 Bollinger Band 通道:")
        print(f"  上軌: 实际={latest['BB_Upper_actual']:.4f}, 预測={latest['BB_Upper_pred']:.4f}")
        print(f"  下軌: 实际={latest['BB_Lower_actual']:.4f}, 预測={latest['BB_Lower_pred']:.4f}")
        
        print(f"\n🎯 支撉/阻力位:")
        print(f"  支撉: 实际={latest['Support_actual']:.4f}, 预測={latest['Support_pred']:.4f}")
        print(f"  阻力: 实际={latest['Resistance_actual']:.4f}, 预測={latest['Resistance_pred']:.4f}")
        
        print(f"\n📈 RSI:")
print(f"  实际={latest['RSI_actual']:.4f} (转探: {latest['RSI_actual']*50+50:.2f})")
        print(f"  预測={latest['RSI_pred']:.4f} (转探: {latest['RSI_pred']*50+50:.2f})")
        
        print(f"\n🔄 MACD:")
        print(f"  MACD: 实际={latest['MACD_actual']:.6f}, 预測={latest['MACD_pred']:.6f}")
        print(f"  信號: 实际={latest['MACD_Signal_actual']:.6f}, 预測={latest['MACD_Signal_pred']:.6f}")
    
    def run(self, output_path='/tmp/model_visualization.png'):
        """完整抢示流程"""
        print("="*80)
        print("BTC 預測模型可視化化 - 开始执行")
        print("="*80)
        
        if not self.load_model_and_data():
            return False
        
        self.predict()
        self.prepare_data()
        self.visualize(output_path)
        self.evaluate()
        self.print_signals()
        
        print("\n" + "="*80)
        print("✓ 执行完成！模型预測对比并已绘制可視化化图表")
        print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='BTC 預測模型可視化化脚本')
    parser.add_argument('--model', type=str, default='/tmp/model_final.pkl',
                       help='模型路径')
    parser.add_argument('--dataset', type=str, default='/tmp/ml_dataset_v3.pkl',
                       help='数据路径')
    parser.add_argument('--output', type=str, default='/tmp/model_visualization.png',
                       help='输出图表路径')
    
    args = parser.parse_args()
    
    visualizer = ModelVisualizer(model_path=args.model, dataset_path=args.dataset)
    success = visualizer.run(output_path=args.output)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
