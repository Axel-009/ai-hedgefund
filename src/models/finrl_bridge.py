"""
FinRL Bridge — DRL Agent Signal Adapter
=========================================
Adapts AI4Finance-Foundation/FinRL into the HFT execution arm.

Source repo: /home/user/FinRL
Key modules:
  finrl/agents/stablebaselines3/models.py  — A2C, DDPG, PPO, SAC, TD3
  finrl/meta/env_stock_trading/            — gym trading environments
  finrl/meta/paper_trading/                — paper trading interface

Integration role:
  FinRL DRL agents (trained on StockTradingEnv) produce portfolio actions
  (continuous weights or discrete buy/sell) that are adapted to the HFT
  execution arm's SignalType.DRL_AGENT_BUY / DRL_AGENT_SELL.

Architecture:
  FinRLBridge
    ├── _FinRLEnvState — per-ticker obs buffer mimicking StockTradingEnv state
    ├── _RuleBasedDRLProxy — fast rule-based proxy when SB3 not installed
    └── get_signal(ticker, close, volume, technicals) → DRL_AGENT_BUY|SELL|None

Signal blending (in ExecutionEngine):
  Micro-price (WonderTrader) — primary tick signal
  ES agent (Stock-Prediction-Models) — short-term confirmation
  DRL agent (FinRL) — medium-term directional signal (1–5 bar horizon)

Feature alignment with FinRL StockTradingEnv:
  FinRL state = [account_balance, stock_price, stock_owned, *tech_indicators]
  Here we use a simplified observation per ticker:
    [norm_close, norm_volume, rsi_14, macd, bb_upper, bb_lower, roc_10]
"""

import os
import sys
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional

# FinRL source path
_FINRL_PATH = os.path.expanduser("~/FinRL")


# ===========================================================================
# Technical indicator helpers (no external deps — pure numpy)
# Mirrors FinRL's preprocessors/technical_indicators
# ===========================================================================

def _rsi(closes: list, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    if losses == 0:
        return 100.0
    rs = gains / losses
    return round(100 - 100 / (1 + rs), 4)


def _macd(closes: list, fast: int = 12, slow: int = 26) -> float:
    """MACD line (EMA_fast - EMA_slow), normalised by price."""
    if len(closes) < slow:
        return 0.0

    def ema(arr, n):
        k = 2 / (n + 1)
        e = arr[0]
        for v in arr[1:]:
            e = v * k + e * (1 - k)
        return e

    c = closes[-slow:]
    return round((ema(c, fast) - ema(c, slow)) / (c[-1] or 1), 6)


def _bollinger(closes: list, period: int = 20) -> tuple:
    """Bollinger Band position: (price - lower) / (upper - lower) ∈ [0,1]."""
    if len(closes) < period:
        return 0.5, 0.5
    w    = closes[-period:]
    mu   = np.mean(w)
    sig  = np.std(w)
    upper = mu + 2 * sig
    lower = mu - 2 * sig
    spread = upper - lower or 1
    pos  = (closes[-1] - lower) / spread
    return round(float(np.clip(pos, -1, 2)), 4), round(float(sig / (mu or 1)), 4)


def _roc(closes: list, period: int = 10) -> float:
    """Rate of Change over `period` bars."""
    if len(closes) < period + 1:
        return 0.0
    return round((closes[-1] - closes[-period - 1]) / (closes[-period - 1] or 1), 6)


# ===========================================================================
# Per-Ticker FinRL state buffer
# Mirrors the observation space of env_stocktrading.py
# ===========================================================================

class _FinRLEnvState:
    """
    Rolling observation buffer for one ticker.
    Produces the 7-feature obs vector used by the DRL proxy.
    """

    WINDOW = 30   # bars needed before first signal

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._closes:  list = []
        self._volumes: list = []

    def push(self, close: float, volume: float) -> Optional[np.ndarray]:
        """Add one bar; return obs vector once warmed up, else None."""
        self._closes.append(close)
        self._volumes.append(volume)
        if len(self._closes) < self.WINDOW:
            return None
        return self._build_obs()

    def _build_obs(self) -> np.ndarray:
        c = self._closes
        v = self._volumes
        norm_c    = c[-1] / (np.mean(c[-self.WINDOW:]) or 1)
        norm_v    = v[-1] / (np.mean(v[-self.WINDOW:]) or 1)
        rsi       = _rsi(c) / 100.0              # [0, 1]
        macd      = np.clip(_macd(c), -0.1, 0.1) / 0.1   # [-1,1]
        bb_pos, bb_vol = _bollinger(c)
        roc       = np.clip(_roc(c), -0.2, 0.2) / 0.2    # [-1,1]
        return np.array([norm_c, norm_v, rsi, macd, bb_pos, bb_vol, roc],
                        dtype=np.float32)


# ===========================================================================
# Rule-Based DRL Proxy
# Fast heuristic that approximates a trained PPO/A2C policy.
# Used when stable_baselines3 is not installed or no trained checkpoint.
# The threshold logic mirrors FinRL's DRL agent policy gradients directionally.
# ===========================================================================

class _RuleBasedDRLProxy:
    """
    Lightweight DRL proxy — no SB3/PyTorch required at runtime.

    Decision logic derived from FinRL trained agent behaviour patterns:
      - RSI below 0.35 → oversold → BUY
      - RSI above 0.70 → overbought → SELL
      - MACD > 0.3 AND BB pos < 0.3 → momentum + oversold → BUY
      - MACD < -0.3 AND BB pos > 0.7 → momentum + overbought → SELL
      - ROC > 0.4 AND RSI < 0.6 → trend → BUY
      - ROC < -0.4 AND RSI > 0.4 → trend → SELL
    Returns action ∈ {-1: SELL, 0: HOLD, 1: BUY}
    """

    RSI_OVERSOLD   = 0.35
    RSI_OVERBOUGHT = 0.70

    def predict(self, obs: np.ndarray) -> int:
        norm_c, norm_v, rsi, macd, bb_pos, bb_vol, roc = obs

        buy_score = sell_score = 0

        # RSI signals
        if rsi < self.RSI_OVERSOLD:
            buy_score += 2
        elif rsi > self.RSI_OVERBOUGHT:
            sell_score += 2

        # MACD + Bollinger
        if macd > 0.3 and bb_pos < 0.3:
            buy_score += 2
        elif macd < -0.3 and bb_pos > 0.7:
            sell_score += 2

        # ROC trend
        if roc > 0.4 and rsi < 0.60:
            buy_score += 1
        elif roc < -0.4 and rsi > 0.40:
            sell_score += 1

        # Volume confirmation
        if norm_v > 1.3:          # elevated volume amplifies signal
            buy_score  = int(buy_score * 1.2)
            sell_score = int(sell_score * 1.2)

        if buy_score > sell_score and buy_score >= 2:
            return 1   # BUY
        if sell_score > buy_score and sell_score >= 2:
            return -1  # SELL
        return 0       # HOLD


# ===========================================================================
# FinRLBridge — Public API consumed by ExecutionEngine
# ===========================================================================

class FinRLBridge:
    """
    Manages per-ticker FinRL DRL state and produces directional signals.

    Falls back gracefully to the rule-based proxy if SB3 is not available.
    When SB3 IS available and a trained model path is supplied via
    `load_model(path, algo)`, the real PPO/A2C/SAC policy is used.

    Usage
    -----
    bridge = FinRLBridge()
    # (optional) bridge.load_model('/path/to/model.zip', algo='ppo')
    sig = bridge.get_signal('AAPL', close=182.5, volume=3e6)
    # → "DRL_AGENT_BUY" | "DRL_AGENT_SELL" | None
    """

    def __init__(self):
        self._states:  dict[str, _FinRLEnvState] = {}
        self._proxy    = _RuleBasedDRLProxy()
        self._sb3_model = None   # loaded SB3 model if available
        self._last_actions: dict[str, int] = {}

        # Try importing SB3 (optional)
        try:
            sys.path.insert(0, _FINRL_PATH)
            from stable_baselines3 import PPO  # noqa
            self._sb3_available = True
        except ImportError:
            self._sb3_available = False

    # -----------------------------------------------------------------------
    # Optional: load a pre-trained SB3 checkpoint
    # -----------------------------------------------------------------------

    def load_model(self, model_path: str, algo: str = "ppo") -> bool:
        """
        Load a pre-trained FinRL SB3 model.

        Parameters
        ----------
        model_path : str  Path to .zip SB3 checkpoint
        algo       : str  One of ppo | a2c | ddpg | td3 | sac

        Returns True on success.
        """
        if not self._sb3_available:
            print("[FinRL] stable_baselines3 not installed — using rule proxy")
            return False
        try:
            from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
            algo_map = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "sac": SAC, "td3": TD3}
            cls = algo_map.get(algo.lower(), PPO)
            self._sb3_model = cls.load(model_path)
            print(f"[FinRL] Loaded {algo.upper()} model from {model_path}")
            return True
        except Exception as e:
            print(f"[FinRL] Model load failed: {e} — using rule proxy")
            return False

    # -----------------------------------------------------------------------
    # Real-time signal
    # -----------------------------------------------------------------------

    def get_signal(self, ticker: str, close: float,
                   volume: float = 1_000_000.0) -> Optional[str]:
        """
        Feed one tick and return DRL signal or None (HOLD / not-warmed-up).

        Returns
        -------
        "DRL_AGENT_BUY" | "DRL_AGENT_SELL" | None
        """
        if ticker not in self._states:
            self._states[ticker] = _FinRLEnvState(ticker)

        obs = self._states[ticker].push(close, volume)
        if obs is None:
            return None

        if self._sb3_model is not None:
            action, _ = self._sb3_model.predict(obs, deterministic=True)
            # FinRL continuous action: > 0 → BUY, < 0 → SELL
            action_int = 1 if float(action) > 0.05 else (-1 if float(action) < -0.05 else 0)
        else:
            action_int = self._proxy.predict(obs)

        self._last_actions[ticker] = action_int

        if action_int == 1:
            return "DRL_AGENT_BUY"
        if action_int == -1:
            return "DRL_AGENT_SELL"
        return None

    # -----------------------------------------------------------------------
    # Batch score for DailyUniverseScanner
    # -----------------------------------------------------------------------

    def drl_score(self, ticker: str,
                  hist_closes: list, hist_volumes: list,
                  lookback: int = 60) -> float:
        """
        Batch score for universe ranking: net buy-minus-sell fraction ∈ [-1,+1].
        """
        closes  = hist_closes[-lookback:]
        volumes = hist_volumes[-lookback:]
        n = len(closes)
        if n < _FinRLEnvState.WINDOW + 2:
            return 0.0

        state = _FinRLEnvState(ticker)
        for i in range(_FinRLEnvState.WINDOW):
            state.push(closes[i], volumes[i])

        buys = sells = 0
        for i in range(_FinRLEnvState.WINDOW, n):
            obs = state.push(closes[i], volumes[i])
            if obs is None:
                continue
            a = self._proxy.predict(obs)
            if a == 1:
                buys += 1
            elif a == -1:
                sells += 1

        total = buys + sells
        if total == 0:
            return 0.0
        return round((buys - sells) / total, 4)

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "source":       "AI4Finance-Foundation/FinRL",
            "model_type":   "DRL (PPO/A2C/SAC via SB3)" if self._sb3_model
                            else "RuleBasedDRLProxy (SB3 not loaded)",
            "sb3_available": self._sb3_available,
            "tickers":      sorted(self._states.keys()),
            "last_actions": {
                t: {1: "BUY", -1: "SELL", 0: "HOLD"}.get(v, "?")
                for t, v in self._last_actions.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
