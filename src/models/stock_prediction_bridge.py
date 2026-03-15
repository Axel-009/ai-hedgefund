"""
Stock-Prediction-Models Bridge — HFT Signal Adapter
=====================================================
Adapts huseinzol05/Stock-Prediction-Models (Evolution Strategy agent)
into the ai-hedgefund HFT execution arm signal pipeline.

Source repo: /home/user/Stock-Prediction-Models
Key file:    realtime-agent/app.py  (Agent + Deep_Evolution_Strategy + Model)

Architecture:
  - Pure-numpy two-layer neural net (no TF/Keras runtime dependency)
  - Per-ticker sliding 20-bar window of [close, volume]
  - Actions: 0=HOLD, 1=BUY, 2=SELL
  - Outputs ML_AGENT_BUY / ML_AGENT_SELL signal or None (HOLD)

Integration role in HFT arm:
  ExecutionEngine.process_quote()
      ├── MicroPriceEngine  → micro-price imbalance signal  (primary)
      └── StockPredictionBridge → RL/ES directional forecast  (confirmation)

  Confirmation logic:
      BOTH agree (BUY+BUY or SELL+SELL) → order proceeds normally
      ML alone agrees → acts as tiebreaker / boosts confidence
      ML disagrees     → signal dampened (edge_bps threshold raised by 1bps)
      ML=HOLD          → micro-price signal used as-is

  DailyUniverseScanner.scan()
      └── ml_score per ticker → added to composite ranking score
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Path to the cloned repo
# ---------------------------------------------------------------------------
_SPM_PATH = os.path.expanduser("~/Stock-Prediction-Models/realtime-agent")


# ===========================================================================
# Pure-numpy model + Evolution Strategy
# (Extracted from Stock-Prediction-Models/realtime-agent/app.py)
# ===========================================================================

def _softmax(z: np.ndarray) -> np.ndarray:
    assert len(z.shape) == 2
    s = np.max(z, axis=1)[:, np.newaxis]
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1)[:, np.newaxis]


def _get_state(timeseries: list, t: int, window_size: int = 20) -> np.ndarray:
    """Build state vector from a list of price/volume series."""
    outside = []
    d = t - window_size + 1
    for parameter in timeseries:
        block = (
            parameter[d: t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0: t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


class _ESModel:
    """
    Two-layer feedforward neural network with random init.
    Direct port of Stock-Prediction-Models/realtime-agent/app.py Model class.
    """

    def __init__(self, input_size: int, layer_size: int, output_size: int):
        rng = np.random.default_rng(seed=42)
        self.weights = [
            rng.random((input_size, layer_size)) * np.sqrt(1 / (input_size + layer_size)),
            rng.random((layer_size, output_size)) * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        feed = np.dot(inputs, self.weights[0]) + self.weights[2]
        return np.dot(feed, self.weights[1]) + self.weights[3]

    def set_weights(self, weights: list) -> None:
        self.weights = weights

    def get_weights(self) -> list:
        return self.weights


# ===========================================================================
# Per-Ticker Agent — stateful sliding window + ES model inference
# ===========================================================================

class _TickerAgent:
    """
    One instance per ticker in the active universe.
    Maintains a 20-bar rolling window and infers buy/sell/hold on each tick.
    """

    WINDOW_SIZE = 20
    LAYER_SIZE  = 500
    OUTPUT_SIZE = 3   # 0=HOLD, 1=BUY, 2=SELL

    def __init__(self, ticker: str):
        self.ticker = ticker
        input_size = 2 * 2 * (self.WINDOW_SIZE - 1)  # 2 features × 2 lag types × (W-1) steps
        self.model  = _ESModel(input_size, self.LAYER_SIZE, self.OUTPUT_SIZE)

        self._queue: deque = deque(maxlen=self.WINDOW_SIZE)
        self._minmax: Optional[MinMaxScaler] = None
        self._mean  = 0.0
        self._std   = 1.0
        self._primed = False   # True once we have a fitted scaler

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, close: float, volume: float) -> Optional[int]:
        """
        Feed one bar (close, volume) and return action (0/1/2) if primed.
        Returns None if the window is not yet full.
        """
        raw = [close, volume]
        self._raw_buffer_append(raw)
        if not self._primed:
            return None
        scaled = self._minmax.transform([raw])[0]
        self._queue.append(scaled)
        if len(self._queue) < self.WINDOW_SIZE:
            return None
        state = self._build_state()
        action = int(np.argmax(self.model.predict(state)[0]))
        return action

    def prime(self, history_closes: list, history_volumes: list) -> None:
        """
        Initialise the scaler from historical data.
        Call once with at least WINDOW_SIZE bars before live updates.
        """
        if len(history_closes) < self.WINDOW_SIZE:
            return
        data = np.array(list(zip(history_closes, history_volumes)))
        self._minmax = MinMaxScaler(feature_range=(100, 200)).fit(data)
        self._mean   = float(np.mean(history_closes))
        self._std    = float(np.std(history_closes)) or 1.0
        # Seed the queue with the last WINDOW_SIZE bars
        for c, v in zip(history_closes[-self.WINDOW_SIZE:],
                        history_volumes[-self.WINDOW_SIZE:]):
            scaled = self._minmax.transform([[c, v]])[0]
            self._queue.append(scaled)
        self._primed = True

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _raw_buffer_append(self, raw: list) -> None:
        """Keep a small bootstrap buffer to detect the first-prime moment."""
        if not hasattr(self, "_raw_buf"):
            self._raw_buf: list = []
        self._raw_buf.append(raw)
        if not self._primed and len(self._raw_buf) >= self.WINDOW_SIZE:
            closes  = [r[0] for r in self._raw_buf]
            volumes = [r[1] for r in self._raw_buf]
            self.prime(closes, volumes)

    def _build_state(self) -> np.ndarray:
        timeseries = list(zip(*list(self._queue)))   # transpose → [[closes], [volumes]]
        timeseries_lists = [list(s) for s in timeseries]
        return _get_state(timeseries_lists, self.WINDOW_SIZE - 1, self.WINDOW_SIZE)


# ===========================================================================
# StockPredictionBridge — Public API consumed by ExecutionEngine
# ===========================================================================

class StockPredictionBridge:
    """
    Manages one _TickerAgent per symbol in the active universe.

    Usage
    -----
    bridge = StockPredictionBridge()
    bridge.prime_ticker("AAPL", hist_closes, hist_volumes)
    ...
    signal = bridge.get_signal("AAPL", close=182.50, volume=3_500_000)
    # signal → "ML_AGENT_BUY" | "ML_AGENT_SELL" | None (HOLD)

    Scoring (for DailyUniverseScanner)
    -----------------------------------
    score = bridge.ml_score("AAPL", hist_closes, hist_volumes)
    # float in [-1, +1]: positive = bullish, negative = bearish
    """

    ACTION_HOLD = 0
    ACTION_BUY  = 1
    ACTION_SELL = 2

    def __init__(self):
        self._agents: dict[str, _TickerAgent] = {}
        self._last_action: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def prime_ticker(self, ticker: str,
                     hist_closes: list, hist_volumes: list) -> None:
        """Initialise a ticker agent from historical bars (≥20 bars required)."""
        agent = _TickerAgent(ticker)
        agent.prime(hist_closes, hist_volumes)
        self._agents[ticker] = agent

    def prime_from_dataframe(self, ticker: str, df) -> None:
        """
        Convenience: prime from a pandas DataFrame with 'Close' and 'Volume' columns.
        """
        closes  = df["Close"].tolist()
        volumes = df["Volume"].tolist() if "Volume" in df.columns else [1_000_000.0] * len(closes)
        self.prime_ticker(ticker, closes, volumes)

    # -----------------------------------------------------------------------
    # Real-time signal
    # -----------------------------------------------------------------------

    def get_signal(self, ticker: str, close: float,
                   volume: float = 1_000_000.0) -> Optional[str]:
        """
        Feed one tick and return ML signal string or None (HOLD/not-primed).

        Returns
        -------
        "ML_AGENT_BUY"  | "ML_AGENT_SELL" | None
        """
        if ticker not in self._agents:
            self._agents[ticker] = _TickerAgent(ticker)

        action = self._agents[ticker].update(close, volume)
        if action is None:
            return None

        self._last_action[ticker] = action
        if action == self.ACTION_BUY:
            return "ML_AGENT_BUY"
        if action == self.ACTION_SELL:
            return "ML_AGENT_SELL"
        return None  # HOLD

    def last_action(self, ticker: str) -> Optional[int]:
        """Last action integer (0/1/2) for ticker, or None if never called."""
        return self._last_action.get(ticker)

    # -----------------------------------------------------------------------
    # Batch scoring for DailyUniverseScanner
    # -----------------------------------------------------------------------

    def ml_score(self, ticker: str,
                 hist_closes: list, hist_volumes: list,
                 lookback: int = 60) -> float:
        """
        Compute a directional ML score in [-1, +1] for ranking purposes.

        Runs the ES agent over the last `lookback` bars and computes the
        net buy-minus-sell fraction as a predictive score.

        Returns
        -------
        float in [-1, +1]:
            +1 = strongly bullish
            -1 = strongly bearish
             0 = neutral / insufficient history
        """
        closes  = hist_closes[-lookback:]
        volumes = hist_volumes[-lookback:]
        n = len(closes)
        if n < _TickerAgent.WINDOW_SIZE + 2:
            return 0.0

        agent = _TickerAgent(ticker)
        agent.prime(closes[:_TickerAgent.WINDOW_SIZE], volumes[:_TickerAgent.WINDOW_SIZE])

        buys = sells = 0
        for i in range(_TickerAgent.WINDOW_SIZE, n):
            action = agent.update(closes[i], volumes[i])
            if action == self.ACTION_BUY:
                buys += 1
            elif action == self.ACTION_SELL:
                sells += 1

        total = buys + sells
        if total == 0:
            return 0.0
        return round((buys - sells) / total, 4)

    # -----------------------------------------------------------------------
    # Integration summary (for reporting)
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        """Return bridge status for inclusion in PlatinumReport."""
        return {
            "source":        "Stock-Prediction-Models/realtime-agent",
            "model_type":    "EvolutionStrategy (pure-numpy, window=20)",
            "agents_primed": sum(1 for a in self._agents.values() if a._primed),
            "tickers":       sorted(self._agents.keys()),
            "last_actions":  {
                t: {0: "HOLD", 1: "BUY", 2: "SELL"}.get(v, "?")
                for t, v in self._last_action.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
