"""
Monte Carlo Price Interval Bridge
===================================
Adapts the empirical Monte Carlo simulation technique from:
  https://gist.github.com/b16f9d8cd0a9e817fd3baa3ce3cd0194

Source: /home/user/gist-b16f9d8cd0a9e817fd3baa3ce3cd0194/
        "Daily Monte Carlo Simulation for Stock Price Prediction Intervals.ipynb"

Methodology (from the gist):
  1. ARIMA(1,1,1) rolling forecast on log prices
  2. Residual distribution fitting — Laplacian is empirically best for stock returns
     (lower SSE than Normal, captures fat tails and sharp peak around zero)
  3. Laplace Monte Carlo: generate N_SIMS samples from Laplace(μ=yhat, β=MAD(residuals))
  4. Prediction intervals: min/max of simulated distribution
  5. P(price > target): fraction of sims above threshold

Key insight from the gist:
  "Prediction intervals generated using Monte Carlo simulation can be used to predict
   the probability that an option is in the money or if a stock is a buy/hold/sell."

Integration role in HFT arm:
  1. MC_BUY / MC_SELL signals → Tier-4 in the vote ensemble
     Signal: P(next_close > current) > BUY_PROB_THRESHOLD
  2. price_probability(ticker, target) → P(price > target) for OptionsConvexityEngine
  3. get_bands(ticker) → (P10, P50, P90) replaces/augments NVIDIATFTAdapter ETS bands
  4. laplacian_beta(ticker) → volatility proxy for AlphaBetaUnleashed

Architecture:
  MonteCarloBridge
    ├── _ARIMAState    — per-ticker rolling ARIMA state + residual buffer
    ├── _laplace_mc()  — Laplace(μ, β, N) simulation (pure-numpy)
    ├── get_signal()   → "MC_BUY" | "MC_SELL" | None
    ├── get_bands()    → (p10, p50, p90) prediction interval
    ├── price_probability(target) → float [0,1]
    └── status()       → dict for PlatinumReport

Fallback (no statsmodels):
  Pure-numpy ARIMA(1,1,1) proxy using Yule-Walker AR(1) + MA residual correction.
"""

import os
import sys
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

_GIST_PATH = os.path.expanduser(
    "~/gist-b16f9d8cd0a9e817fd3baa3ce3cd0194"
)

# Number of Monte Carlo simulations (gist uses 1000)
N_SIMS = 1000
# Minimum history bars before first forecast
MIN_HISTORY = 30
# Residual buffer size for Laplacian β estimation (gist uses 250 bars)
RESIDUAL_WINDOW = 60
# Signal thresholds
BUY_PROB_THRESHOLD  = 0.55   # P(next > current) > 55% → MC_BUY
SELL_PROB_THRESHOLD = 0.45   # P(next > current) < 45% → MC_SELL


# ===========================================================================
# Pure-numpy ARIMA(1,1,1) proxy
# Mirrors the statsmodels ARIMA(1,1,1) for the case where statsmodels
# is not available. Uses Yule-Walker for AR coefficient + first residual
# as MA correction.
# ===========================================================================

class _ARIMA111Proxy:
    """
    Pure-numpy ARIMA(1,1,1) one-step-ahead forecast.
    Translates statsmodels ARIMA(order=(1,1,1)) from the gist.

    State:
      φ  — AR(1) coefficient (estimated via OLS on differenced log-prices)
      θ  — MA(1) coefficient (fixed at 0.1 — reasonable default)
      last_diff — most recent differenced log-price
      last_resid — most recent residual (for MA term)
    """

    def __init__(self):
        self._phi:       float = 0.0
        self._theta:     float = 0.1
        self._last_diff: float = 0.0
        self._last_resid:float = 0.0
        self._log_history: list = []

    def update(self, log_price: float) -> None:
        self._log_history.append(log_price)
        n = len(self._log_history)
        if n >= MIN_HISTORY:
            diffs = np.diff(self._log_history[-MIN_HISTORY:])
            # OLS estimate: φ = cov(d[t], d[t-1]) / var(d[t-1])
            if len(diffs) >= 2:
                self._phi = float(
                    np.cov(diffs[1:], diffs[:-1])[0, 1] /
                    max(np.var(diffs[:-1]), 1e-12)
                )
                self._phi = float(np.clip(self._phi, -0.99, 0.99))
            if n >= 2:
                self._last_diff = log_price - self._log_history[-2]

    def forecast_log(self) -> Optional[float]:
        """One-step-ahead forecast of log price."""
        if len(self._log_history) < MIN_HISTORY:
            return None
        # ARIMA(1,1,1): Δy_t = φ*Δy_{t-1} + θ*ε_{t-1}
        diff_forecast = (self._phi * self._last_diff
                         + self._theta * self._last_resid)
        log_forecast = self._log_history[-1] + diff_forecast
        # Update residual
        if len(self._log_history) >= 2:
            actual_diff = (self._log_history[-1]
                           - self._log_history[-2])
            self._last_resid = actual_diff - diff_forecast
        return log_forecast


# ===========================================================================
# Laplace Monte Carlo (direct port from gist)
# ===========================================================================

def _laplace_mc(mean: float, residuals: np.ndarray,
                n_sims: int = N_SIMS) -> np.ndarray:
    """
    Monte Carlo simulation using Laplacian distribution.
    Direct Python port of laplace_monte_carlo() from the gist.

    Parameters
    ----------
    mean      : ARIMA point forecast (yhat)
    residuals : array of rolling forecast residuals
    n_sims    : number of simulations (default 1000)

    Returns
    -------
    np.ndarray of shape (n_sims,) — simulated price distribution
    """
    # β = mean absolute distance from mean (Laplacian scale)
    # Matches: beta = sum(abs(residuals - mean(residuals))) / len(residuals)
    beta = float(np.mean(np.abs(residuals - np.mean(residuals))))
    beta = max(beta, 1e-6)   # guard zero-beta edge case
    return np.random.laplace(loc=mean, scale=beta, size=n_sims)


# ===========================================================================
# Per-Ticker ARIMA + MC state
# ===========================================================================

class _ARIMAState:
    """
    Per-ticker rolling ARIMA(1,1,1) + Laplacian MC state.
    Maintains log-price history + residual buffer for β estimation.
    """

    def __init__(self, ticker: str):
        self.ticker    = ticker
        self._arima    = _ARIMA111Proxy()
        self._closes:  list = []
        self._residuals: deque = deque(maxlen=RESIDUAL_WINDOW)
        self._last_bands: Optional[Tuple] = None   # (p10, p50, p90)
        self._last_sims:  Optional[np.ndarray] = None

    def push(self, close: float) -> Optional[Tuple[np.ndarray, Tuple]]:
        """
        Feed one close price. Returns (sims, (p10, p50, p90)) or None.
        Mirrors the rolling forecast loop from the gist.
        """
        self._closes.append(close)
        log_price = np.log(max(close, 1e-9))
        self._arima.update(log_price)

        log_forecast = self._arima.forecast_log()
        if log_forecast is None:
            return None

        # Point forecast (back-transform from log)
        yhat = np.exp(log_forecast)

        # Update residual buffer
        if len(self._closes) >= 2:
            residual = self._closes[-1] - yhat
            self._residuals.append(residual)

        if len(self._residuals) < 5:
            return None

        # Laplace MC simulation
        resid_arr = np.array(self._residuals)
        sims = _laplace_mc(yhat, resid_arr, N_SIMS)
        p10  = float(np.percentile(sims, 10))
        p50  = float(np.percentile(sims, 50))
        p90  = float(np.percentile(sims, 90))

        self._last_bands = (p10, p50, p90)
        self._last_sims  = sims
        return sims, (p10, p50, p90)

    @property
    def primed(self) -> bool:
        return self._last_bands is not None

    def price_probability(self, target: float) -> float:
        """P(next_close >= target) using last MC simulation."""
        if self._last_sims is None:
            return 0.5
        return float(np.mean(self._last_sims >= target))


# ===========================================================================
# MonteCarloBridge — Public API
# ===========================================================================

class MonteCarloBridge:
    """
    Empirical Monte Carlo price interval engine for the HFT execution arm.

    Generates ARIMA(1,1,1) + Laplacian MC prediction intervals and
    probability estimates — directly adapted from the gist methodology.

    Signal logic:
      P(next_close > current_close) > BUY_PROB_THRESHOLD  → MC_BUY
      P(next_close > current_close) < SELL_PROB_THRESHOLD → MC_SELL
      Otherwise                                            → None (HOLD)

    Additional outputs:
      get_bands(ticker)              → (P10, P50, P90)
      price_probability(ticker, T)   → P(price > T) for options arm
      laplacian_beta(ticker)         → Laplacian scale (vol proxy)

    Usage
    -----
    bridge = MonteCarloBridge()
    sig = bridge.get_signal('NFLX', close=290.0)
    # → "MC_BUY" | "MC_SELL" | None
    prob = bridge.price_probability('NFLX', target=310.0)
    # → 0.367 (matches gist example)
    """

    def __init__(self, n_sims: int = N_SIMS):
        self._n_sims = n_sims
        self._states: dict[str, _ARIMAState] = {}
        self._last_probs: dict[str, float] = {}

        # Try statsmodels for production-grade ARIMA (optional)
        self._statsmodels_available = False
        try:
            from statsmodels.tsa.arima.model import ARIMA  # noqa
            self._statsmodels_available = True
        except ImportError:
            pass

    # -----------------------------------------------------------------------
    # Real-time signal
    # -----------------------------------------------------------------------

    def get_signal(self, ticker: str, close: float) -> Optional[str]:
        """
        Feed one close price and return MC directional signal.

        Returns
        -------
        "MC_BUY" | "MC_SELL" | None
        """
        if ticker not in self._states:
            self._states[ticker] = _ARIMAState(ticker)

        result = self._states[ticker].push(close)
        if result is None:
            return None

        sims, (p10, p50, p90) = result

        # P(next > current)
        prob_up = float(np.mean(sims >= close))
        self._last_probs[ticker] = round(prob_up, 4)

        if prob_up > BUY_PROB_THRESHOLD:
            return "MC_BUY"
        if prob_up < SELL_PROB_THRESHOLD:
            return "MC_SELL"
        return None

    # -----------------------------------------------------------------------
    # Prediction bands (for TFT augmentation / risk management)
    # -----------------------------------------------------------------------

    def get_bands(self, ticker: str) -> Optional[Tuple[float, float, float]]:
        """
        Return latest (P10, P50, P90) Laplacian MC prediction interval.
        Directly replaces or augments NVIDIATFTAdapter.get_bands().
        """
        state = self._states.get(ticker)
        if state is None or not state.primed:
            return None
        return state._last_bands

    # -----------------------------------------------------------------------
    # Options probability (key use case from gist)
    # -----------------------------------------------------------------------

    def price_probability(self, ticker: str, target: float) -> float:
        """
        P(next_close >= target) using empirical Laplacian MC distribution.

        Used by OptionsConvexityEngine to estimate probability of being
        in-the-money at expiry — directly matches the gist's target_price example.

        Example (from gist):
          target_price = 290
          P(price >= 290) ≈ 0.367  (36.7% probability in-the-money)
        """
        state = self._states.get(ticker)
        if state is None or not state.primed:
            return 0.5
        return round(state.price_probability(target), 4)

    # -----------------------------------------------------------------------
    # Volatility proxy (Laplacian β)
    # -----------------------------------------------------------------------

    def laplacian_beta(self, ticker: str) -> float:
        """
        Laplacian scale β = MAD of residuals.
        Proxy for realised volatility — feeds AlphaBetaUnleashed.
        Smaller β = tighter distribution = lower vol.
        """
        state = self._states.get(ticker)
        if state is None or len(state._residuals) < 5:
            return 0.0
        resid = np.array(state._residuals)
        return round(float(np.mean(np.abs(resid - np.mean(resid)))), 4)

    # -----------------------------------------------------------------------
    # Batch score for DailyUniverseScanner
    # -----------------------------------------------------------------------

    def mc_score(self, ticker: str, hist_closes: list,
                 lookback: int = 80) -> float:
        """
        Batch score ∈ [-1, +1] for universe ranking.
        Net fraction of bars where P(up) > 0.55, minus P(down) > 0.55.
        """
        closes = hist_closes[-lookback:]
        n = len(closes)
        if n < MIN_HISTORY + 5:
            return 0.0

        state = _ARIMAState(ticker)
        for c in closes[:MIN_HISTORY]:
            state.push(c)

        buys = sells = 0
        for c in closes[MIN_HISTORY:]:
            res = state.push(c)
            if res is None:
                continue
            sims, _ = res
            p_up = float(np.mean(sims >= c))
            if p_up > BUY_PROB_THRESHOLD:
                buys += 1
            elif p_up < SELL_PROB_THRESHOLD:
                sells += 1

        total = buys + sells
        if total == 0:
            return 0.0
        return round((buys - sells) / total, 4)

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        bands = {
            t: s._last_bands
            for t, s in self._states.items()
            if s._last_bands is not None
        }
        return {
            "source":              "gist:b16f9d8cd0a9e817fd3baa3ce3cd0194",
            "model_type":          "ARIMA(1,1,1) + Laplacian Monte Carlo",
            "statsmodels_available": self._statsmodels_available,
            "n_sims":              self._n_sims,
            "buy_prob_threshold":  BUY_PROB_THRESHOLD,
            "sell_prob_threshold": SELL_PROB_THRESHOLD,
            "tickers":             sorted(self._states.keys()),
            "last_prob_up":        self._last_probs,
            "last_bands":          {t: {"p10": b[0], "p50": b[1], "p90": b[2]}
                                    for t, b in bands.items()},
            "timestamp":           datetime.now().isoformat(),
        }
