"""
Deep-Trading Feature Engine
=============================
Adapts Rachnog/Deep-Trading into the HFT execution arm as a
feature engineering and volatility forecasting layer.

Source repo: /home/user/Deep-Trading
Key modules:
  volatility/volatility.py   — LSTM/Conv volatility forecasting
  volatility/feature_extractor.py — OHLCV feature builder
  strategy/skew.py           — 12-feature multivariate state
                               (OHLCV + vol + skew + kurtosis + MACD + Williams%R + RSI + Ichimoku)
  multivariate/multivariate.py — multi-input LSTM
  bayesian/                   — Bayesian NN (uncertainty estimation)

Integration role in HFT arm:
  1. Feature engineering: builds the 12-feature state vector used by
     StockPredictionBridge and FinRLBridge, replacing their raw [close, volume]
     inputs with richer representations.
  2. Volatility oracle: provides rolling realised vol + skewness/kurtosis
     estimates used by AlphaBetaUnleashed for dynamic position sizing.
  3. Regime detection: rolling skewness sign flip signals regime change.

Architecture:
  DeepTradingFeatures
    ├── build_state(ohlcv_window) → 12-feature np.ndarray (same as strategy/skew.py)
    ├── rolling_volatility(closes, window) → float (annualised realised vol)
    ├── rolling_skew(closes, window) → float (regime indicator)
    ├── rolling_kurtosis(closes, window) → float (tail risk indicator)
    ├── vol_regime(closes) → "LOW_VOL" | "MED_VOL" | "HIGH_VOL" | "STRESS"
    └── enrich_obs(ticker, close, high, low, volume) → 12-feature ndarray
"""

import numpy as np
from typing import Optional
from collections import deque
from datetime import datetime


# ===========================================================================
# Pure-numpy technical indicators
# Translated from Deep-Trading/strategy/skew.py and feature_extractor.py
# ===========================================================================

def _pct_change(data: list) -> list:
    """Convert price series to % changes (mirrors data2change in skew.py)."""
    arr = np.array(data, dtype=float)
    out = np.zeros_like(arr)
    out[1:] = (arr[1:] - arr[:-1]) / np.where(arr[:-1] != 0, arr[:-1], 1)
    return out.tolist()


def _remap(arr: np.ndarray, out_min: float = -1.0, out_max: float = 1.0) -> np.ndarray:
    """Min-max normalise to [out_min, out_max] (mirrors remap() in skew.py)."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo) * (out_max - out_min) + out_min


def _ema(arr: list, span: int) -> list:
    """Exponential moving average (pandas-compatible ewm)."""
    k = 2 / (span + 1)
    result = [arr[0]]
    for v in arr[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _macd(closes: list, fast: int = 6, slow: int = 12) -> list:
    """MACD line series (matches moving_average_convergence in skew.py)."""
    if len(closes) < slow:
        return [0.0] * len(closes)
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    return [f - s for f, s in zip(ema_fast, ema_slow)]


def _williams_r(high: list, low: list, close: list, period: int = 14) -> list:
    """Williams %R — period-wise."""
    out = []
    for i in range(len(close)):
        if i < period - 1:
            out.append(-50.0)
            continue
        h_max = max(high[i - period + 1: i + 1])
        l_min = min(low[i  - period + 1: i + 1])
        if h_max == l_min:
            out.append(-50.0)
        else:
            out.append((h_max - close[i]) / (h_max - l_min) * -100)
    return out


def _rsi(closes: list, period: int = 14) -> list:
    """RSI series."""
    out = [50.0] * len(closes)
    for i in range(period, len(closes)):
        deltas = np.diff(closes[i - period: i + 1])
        gains  = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        if losses == 0:
            out[i] = 100.0
        else:
            out[i] = 100 - 100 / (1 + gains / losses)
    return out


def _ichimoku_conversion(high: list, low: list, period: int = 9) -> list:
    """Ichimoku conversion line (Tenkan-sen)."""
    out = []
    for i in range(len(high)):
        if i < period - 1:
            out.append((high[0] + low[0]) / 2)
            continue
        h_max = max(high[i - period + 1: i + 1])
        l_min = min(low[i  - period + 1: i + 1])
        out.append((h_max + l_min) / 2)
    return out


def _rolling_skew(data: list, window: int = 30) -> list:
    """Rolling skewness series."""
    arr = np.array(data)
    out = np.zeros_like(arr)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        mu = np.mean(w); sig = np.std(w)
        if sig == 0:
            out[i] = 0.0
        else:
            out[i] = np.mean(((w - mu) / sig) ** 3)
    return out.tolist()


def _rolling_kurtosis(data: list, window: int = 30) -> list:
    """Rolling excess kurtosis series."""
    arr = np.array(data)
    out = np.zeros_like(arr)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        mu = np.mean(w); sig = np.std(w)
        if sig == 0:
            out[i] = 0.0
        else:
            out[i] = np.mean(((w - mu) / sig) ** 4) - 3
    return out.tolist()


# ===========================================================================
# DeepTradingFeatures — Public API
# ===========================================================================

class DeepTradingFeatures:
    """
    Builds the 12-feature multivariate state from Deep-Trading/strategy/skew.py
    and provides volatility / regime metrics for the HFT arm.

    Features (index → name):
      0  open_chg    - % change open  (normalised)
      1  high_chg    - % change high
      2  low_chg     - % change low
      3  close_chg   - % change close
      4  volume_chg  - % change volume
      5  volatility  - rolling std of high_chg (window)
      6  skewness    - rolling skewness of close_chg
      7  kurtosis    - rolling kurtosis of close_chg
      8  macd        - MACD(6,12) of close
      9  williams    - Williams %R(14)
      10 rsi         - RSI(14)
      11 ichimoku    - Ichimoku conversion line (normalised)
    """

    WINDOW   = 30   # bars per state window (matches WINDOW=30 in skew.py)
    FEATURES = 12

    def __init__(self):
        self._buffers: dict[str, dict] = {}   # per-ticker OHLCV buffers

    # -----------------------------------------------------------------------
    # Per-ticker streaming update
    # -----------------------------------------------------------------------

    def push(self, ticker: str,
             close: float, high: Optional[float] = None,
             low: Optional[float] = None, open_: Optional[float] = None,
             volume: float = 1_000_000.0) -> Optional[np.ndarray]:
        """
        Feed one bar. Returns 12-feature obs vector once buffer is full.
        Uses close±spread estimates when high/low/open are not available.
        """
        if ticker not in self._buffers:
            self._buffers[ticker] = {k: [] for k in ("o", "h", "l", "c", "v")}

        spread = close * 0.001  # 10bps synthetic spread if OHLC not given
        buf = self._buffers[ticker]
        buf["c"].append(close)
        buf["h"].append(high  if high  is not None else close + spread)
        buf["l"].append(low   if low   is not None else close - spread)
        buf["o"].append(open_ if open_ is not None else close)
        buf["v"].append(volume)

        if len(buf["c"]) < self.WINDOW + 1:
            return None

        return self._build(buf)

    # -----------------------------------------------------------------------
    # Batch build (for DailyUniverseScanner)
    # -----------------------------------------------------------------------

    def build_from_series(self,
                          closes: list, highs: Optional[list] = None,
                          lows: Optional[list] = None,
                          opens: Optional[list] = None,
                          volumes: Optional[list] = None) -> Optional[np.ndarray]:
        """
        Build the 12-feature state from historical series (last WINDOW+1 bars).
        Returns None if insufficient history.
        """
        n = len(closes)
        if n < self.WINDOW + 1:
            return None
        sp = [c * 0.001 for c in closes]
        buf = {
            "c": closes[-self.WINDOW - 1:],
            "h": (highs  or [c + s for c, s in zip(closes, sp)])[-self.WINDOW - 1:],
            "l": (lows   or [c - s for c, s in zip(closes, sp)])[-self.WINDOW - 1:],
            "o": (opens  or closes)[-self.WINDOW - 1:],
            "v": (volumes or [1e6] * n)[-self.WINDOW - 1:],
        }
        return self._build(buf)

    # -----------------------------------------------------------------------
    # Volatility / Regime metrics
    # -----------------------------------------------------------------------

    def rolling_volatility(self, closes: list, window: int = 20,
                           annualise: bool = True) -> float:
        """Annualised realised volatility from close returns."""
        if len(closes) < window + 1:
            return 0.20  # default 20% vol
        rets = np.diff(np.log(np.array(closes[-window - 1:]) + 1e-9))
        vol = float(np.std(rets))
        return round(vol * np.sqrt(252) if annualise else vol, 6)

    def rolling_skew(self, closes: list, window: int = 30) -> float:
        """Latest rolling skewness of close returns."""
        if len(closes) < window + 1:
            return 0.0
        rets = np.diff(np.log(np.array(closes[-window - 1:]) + 1e-9))
        mu = np.mean(rets); sig = np.std(rets)
        if sig == 0:
            return 0.0
        return round(float(np.mean(((rets - mu) / sig) ** 3)), 4)

    def rolling_kurtosis(self, closes: list, window: int = 30) -> float:
        """Latest rolling excess kurtosis of close returns."""
        if len(closes) < window + 1:
            return 0.0
        rets = np.diff(np.log(np.array(closes[-window - 1:]) + 1e-9))
        mu = np.mean(rets); sig = np.std(rets)
        if sig == 0:
            return 0.0
        return round(float(np.mean(((rets - mu) / sig) ** 4) - 3), 4)

    def vol_regime(self, closes: list, window: int = 20) -> str:
        """
        Classify volatility regime (mirrors Deep-Trading vol stratification).
        Returns: "LOW_VOL" | "MED_VOL" | "HIGH_VOL" | "STRESS"
        """
        vol = self.rolling_volatility(closes, window)
        if vol < 0.15:
            return "LOW_VOL"
        if vol < 0.25:
            return "MED_VOL"
        if vol < 0.40:
            return "HIGH_VOL"
        return "STRESS"

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "source":      "Rachnog/Deep-Trading",
            "feature_dim": self.FEATURES,
            "window":      self.WINDOW,
            "tickers_buffered": sorted(self._buffers.keys()),
            "timestamp":   datetime.now().isoformat(),
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _build(self, buf: dict) -> np.ndarray:
        """Build 12-feature vector from a filled buffer dict."""
        o = _pct_change(buf["o"])
        h = _pct_change(buf["h"])
        l = _pct_change(buf["l"])
        c = _pct_change(buf["c"])
        v = _pct_change(buf["v"])

        n = len(c)
        w = self.WINDOW

        # Rolling features (series, then take last w values)
        volat_s = [np.std(h[max(0, i - w):i + 1]) for i in range(n)]
        skew_s  = _rolling_skew(c, w)
        kurt_s  = _rolling_kurtosis(c, w)
        macd_s  = _macd(buf["c"], fast=6, slow=12)
        wpr_s   = _williams_r(buf["h"], buf["l"], buf["c"], period=min(14, w))
        rsi_s   = _rsi(buf["c"], period=min(14, w))
        ichi_s  = _ichimoku_conversion(buf["h"], buf["l"], period=min(9, w))

        # Take last WINDOW points for the state window
        def _tail(lst):
            return np.array(lst[-w:], dtype=float)

        o_w    = _remap(_tail(o))
        h_w    = _remap(_tail(h))
        l_w    = _remap(_tail(l))
        c_w    = _remap(_tail(c))
        v_w    = _remap(_tail(v))
        vol_w  = _remap(_tail(volat_s))
        skew_w = _remap(_tail(skew_s))
        kurt_w = _remap(_tail(kurt_s))
        macd_w = _remap(_tail(macd_s))
        wpr_w  = _remap(_tail(wpr_s))
        rsi_w  = _remap(_tail(rsi_s))
        ichi_w = _remap(_tail(ichi_s))

        # Return the LAST bar's 12 scalar features (not the full window)
        return np.array([
            float(o_w[-1]), float(h_w[-1]), float(l_w[-1]), float(c_w[-1]),
            float(v_w[-1]), float(vol_w[-1]), float(skew_w[-1]), float(kurt_w[-1]),
            float(macd_w[-1]), float(wpr_w[-1]), float(rsi_w[-1]), float(ichi_w[-1]),
        ], dtype=np.float32)
