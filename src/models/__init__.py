# src/models — ML signal engines + inference adapters
#
# Sources:
#   StockPredictionBridge  — huseinzol05/Stock-Prediction-Models (ES agent, pure-numpy)
#   FinRLBridge            — AI4Finance-Foundation/FinRL (PPO/A2C/SAC DRL agents)
#   DeepTradingFeatures    — Rachnog/Deep-Trading (12-feature state + volatility)
#   NVIDIATFTAdapter       — NVIDIA/DeepLearningExamples (TFT multi-horizon forecast)
#   KServeAdapter          — kserve/kserve (production model serving layer)
#   MonteCarloBridge       — gist:b16f9d8cd0a9e817fd3baa3ce3cd0194
#                            ARIMA(1,1,1) + Laplacian MC prediction intervals
#   UniverseClassifier     — MODERATE-Project/building-stock-analysis
#                            Top-down (macro regime) vs bottom-up (fundamentals)
#                            XGB ensemble; EPC A→G → QUALITY_BUY/SELL signals
#   ModelEvaluator         — port of building-stock-analysis/models_utils.py
#                            Balanced accuracy, per-class F1, confusion matrix

from .stock_prediction_bridge import StockPredictionBridge
from .finrl_bridge             import FinRLBridge
from .deep_trading_features    import DeepTradingFeatures
from .nvidia_tft_adapter       import NVIDIATFTAdapter
from .kserve_adapter           import KServeAdapter
from .monte_carlo_bridge       import MonteCarloBridge
from .universe_classifier      import (UniverseClassifier,
                                       build_dual_regime_features,
                                       label_from_alpha)
from .model_evaluator          import ModelEvaluator

__all__ = [
    "StockPredictionBridge",
    "FinRLBridge",
    "DeepTradingFeatures",
    "NVIDIATFTAdapter",
    "KServeAdapter",
    "MonteCarloBridge",
    "UniverseClassifier",
    "build_dual_regime_features",
    "label_from_alpha",
    "ModelEvaluator",
]
