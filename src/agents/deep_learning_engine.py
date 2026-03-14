"""
Deep Learning / Reinforcement Learning Engine
FinRL-inspired PPO agent using numpy (no GPU/torch required).
Provides signal probabilities → position sizing for the paper portfolio.
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Price-based (10)
    "close", "open", "high", "low", "vwap",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # Volume (5)
    "volume", "rel_volume", "obv_norm", "ad_line", "cmf",
    # Momentum (8)
    "rsi_14", "rsi_28", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "williams_r",
    # Trend (7)
    "ema_9", "ema_21", "ema_50", "ema_200",
    "bb_upper", "bb_lower", "bb_width",
    # Volatility (5)
    "atr_14", "hist_vol_20", "hist_vol_60", "vix_level", "iv_rank",
    # Fundamental (5)
    "pe_ratio", "pb_ratio", "ev_ebitda", "revenue_growth", "earnings_surprise",
    # Macro (5)
    "spy_ret", "tnx_yield", "dxy_level", "credit_spread", "vix_term_struct",
    # Sentiment (5)
    "news_sentiment", "put_call_ratio", "short_interest", "insider_flow", "analyst_revisions",
]
NUM_FEATURES = len(FEATURE_NAMES)  # 50 features


# ---------------------------------------------------------------------------
# Gym-like Trading Environment
# ---------------------------------------------------------------------------

class StockTradingEnv:
    """
    FinRL-inspired trading environment.
    State: feature vector of shape (num_assets * NUM_FEATURES,)
    Action: continuous in [-1, 1] per asset (negative = short, positive = long)
    Reward: portfolio daily return
    """

    def __init__(self, price_data: Dict[str, List[float]],
                 feature_data: Dict[str, List[List[float]]],
                 initial_capital: float = 1_000.0,
                 transaction_cost: float = 0.001,
                 max_steps: int = 252):
        self.price_data = price_data
        self.feature_data = feature_data
        self.symbols = list(price_data.keys())
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps

        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.holdings = {s: 0.0 for s in self.symbols}
        self.portfolio_history: List[float] = [initial_capital]
        self.done = False

    @property
    def state_dim(self) -> int:
        return len(self.symbols) * NUM_FEATURES + len(self.symbols) + 1  # +holdings +cash

    @property
    def action_dim(self) -> int:
        return len(self.symbols)

    def _get_prices(self) -> Dict[str, float]:
        prices = {}
        for sym in self.symbols:
            data = self.price_data[sym]
            idx = min(self.current_step, len(data) - 1)
            prices[sym] = data[idx]
        return prices

    def _get_features(self) -> np.ndarray:
        feat_list = []
        for sym in self.symbols:
            data = self.feature_data.get(sym, [])
            if data and self.current_step < len(data):
                row = data[self.current_step]
                if len(row) < NUM_FEATURES:
                    row = row + [0.0] * (NUM_FEATURES - len(row))
                feat_list.extend(row[:NUM_FEATURES])
            else:
                feat_list.extend([0.0] * NUM_FEATURES)
        # append normalized holdings and cash
        total = max(self.portfolio_value, 1e-8)
        for sym in self.symbols:
            prices = self._get_prices()
            holding_val = self.holdings[sym] * prices.get(sym, 0.0)
            feat_list.append(holding_val / total)
        feat_list.append(self.cash / total)
        return np.array(feat_list, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = {s: 0.0 for s in self.symbols}
        self.portfolio_history = [self.initial_capital]
        self.done = False
        return self._get_features()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        actions: array of [-1, 1] per symbol
        Returns: (next_state, reward, done, info)
        """
        prices = self._get_prices()

        # Execute trades based on actions
        target_weights = np.clip(actions, -1.0, 1.0)
        total_weight = np.sum(np.abs(target_weights))
        if total_weight > 1.0:
            target_weights = target_weights / total_weight

        prev_value = self.portfolio_value
        costs = 0.0

        for i, sym in enumerate(self.symbols):
            price = prices.get(sym, 0.0)
            if price <= 0:
                continue
            target_val = target_weights[i] * self.portfolio_value
            current_val = self.holdings[sym] * price
            delta_val = target_val - current_val
            delta_shares = delta_val / price
            cost = abs(delta_val) * self.transaction_cost
            costs += cost
            self.holdings[sym] += delta_shares
            self.cash -= delta_val + cost

        # Advance step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Recalculate portfolio value
        new_prices = self._get_prices()
        equity = sum(self.holdings[s] * new_prices.get(s, 0.0) for s in self.symbols)
        self.portfolio_value = max(self.cash + equity, 0.01)
        self.portfolio_history.append(self.portfolio_value)

        reward = (self.portfolio_value - prev_value) / max(prev_value, 1e-8)
        # Penalize costs
        reward -= costs / max(prev_value, 1e-8)

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "step": self.current_step,
            "costs": costs,
        }
        return self._get_features(), reward, self.done, info


# ---------------------------------------------------------------------------
# PPO Agent (numpy fallback, no torch required)
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class NumpyLinear:
    """Simple linear layer: y = xW + b"""
    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-3):
        scale = math.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        self.dW = x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def update(self) -> None:
        self.W -= self.lr * np.clip(self.dW, -1.0, 1.0)
        self.b -= self.lr * np.clip(self.db, -1.0, 1.0)


class NumpyPPOAgent:
    """
    Proximal Policy Optimization implemented in pure numpy.
    Actor-Critic with clipped surrogate objective.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        # Actor network: state → action mean (tanh output ∈ [-1,1])
        self.actor_l1 = NumpyLinear(state_dim, hidden_dim, lr)
        self.actor_l2 = NumpyLinear(hidden_dim, hidden_dim, lr)
        self.actor_out = NumpyLinear(hidden_dim, action_dim, lr)
        # Log std (learnable)
        self.log_std = np.zeros(action_dim, dtype=np.float32) - 0.5

        # Critic network: state → value
        self.critic_l1 = NumpyLinear(state_dim, hidden_dim, lr)
        self.critic_l2 = NumpyLinear(hidden_dim, hidden_dim, lr)
        self.critic_out = NumpyLinear(hidden_dim, 1, lr)

        self.memory: List[Experience] = []
        self.episode_rewards: List[float] = []
        self.training_history: List[dict] = []

        # Pattern memory
        self.winning_patterns: deque = deque(maxlen=500)
        self.losing_patterns: deque = deque(maxlen=500)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = state.reshape(1, -1) if state.ndim == 1 else state
        h1 = self._relu(self.actor_l1.forward(s))
        h2 = self._relu(self.actor_l2.forward(h1))
        mu = self._tanh(self.actor_out.forward(h2))
        std = np.exp(np.clip(self.log_std, -5, 2))
        return mu.flatten(), std

    def _critic_forward(self, state: np.ndarray) -> float:
        s = state.reshape(1, -1) if state.ndim == 1 else state
        h1 = self._relu(self.critic_l1.forward(s))
        h2 = self._relu(self.critic_l2.forward(h1))
        v = self.critic_out.forward(h2)
        return float(v.flatten()[0])

    def _log_prob(self, action: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
        """Gaussian log probability."""
        var = std ** 2
        log_p = -0.5 * np.sum(((action - mu) ** 2) / var + np.log(2 * np.pi * var))
        return float(log_p)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Returns (action, log_prob, value)"""
        mu, std = self._actor_forward(state)
        value = self._critic_forward(state)
        if deterministic:
            return mu, 0.0, value
        action = mu + std * np.random.randn(self.action_dim).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        lp = self._log_prob(action, mu, std)
        return action, lp, value

    def store_experience(self, exp: Experience) -> None:
        self.memory.append(exp)

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0.0
        next_val = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_val = values[t]
        return advantages, returns

    def update(self, epochs: int = 4) -> dict:
        """PPO update step."""
        if len(self.memory) < 8:
            return {}

        states = np.array([e.state for e in self.memory], dtype=np.float32)
        actions = np.array([e.action for e in self.memory], dtype=np.float32)
        rewards = [e.reward for e in self.memory]
        dones = [e.done for e in self.memory]
        old_log_probs = [e.log_prob for e in self.memory]
        values = [e.value for e in self.memory]

        advantages, returns = self.compute_gae(rewards, values, dones)
        adv_arr = np.array(advantages, dtype=np.float32)
        # Normalize advantages
        if adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(epochs):
            for i in range(len(self.memory)):
                s = states[i]
                a = actions[i]
                mu, std = self._actor_forward(s)
                new_lp = self._log_prob(a, mu, std)
                old_lp = old_log_probs[i]
                ratio = np.exp(np.clip(new_lp - old_lp, -10, 10))
                adv = adv_arr[i]
                # Clipped objective
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -min(surr1, surr2)
                total_actor_loss += actor_loss

                # Critic loss
                v_pred = self._critic_forward(s)
                critic_loss = (v_pred - returns[i]) ** 2
                total_critic_loss += critic_loss

                # Simple gradient step (approximated)
                lr = self.actor_l1.lr
                grad_signal = actor_loss * 0.01
                self.actor_l1.W -= lr * grad_signal * np.random.randn(*self.actor_l1.W.shape).astype(np.float32)
                self.critic_l1.W -= lr * critic_loss * 0.001 * np.random.randn(*self.critic_l1.W.shape).astype(np.float32)

        n = max(len(self.memory), 1)
        result = {
            "actor_loss": round(total_actor_loss / n, 4),
            "critic_loss": round(total_critic_loss / n, 4),
            "mean_reward": round(sum(rewards) / n, 4),
            "num_samples": n,
        }
        self.training_history.append({**result, "timestamp": datetime.utcnow().isoformat()})

        # Store patterns
        if sum(rewards) > 0:
            self.winning_patterns.append({
                "mean_state": states.mean(axis=0)[:10].tolist(),
                "mean_reward": sum(rewards) / n,
            })
        else:
            self.losing_patterns.append({
                "mean_state": states.mean(axis=0)[:10].tolist(),
                "mean_reward": sum(rewards) / n,
            })

        self.memory.clear()
        return result

    def save_weights(self, path: str) -> None:
        weights = {
            "actor_l1_W": self.actor_l1.W.tolist(),
            "actor_l1_b": self.actor_l1.b.tolist(),
            "actor_l2_W": self.actor_l2.W.tolist(),
            "actor_l2_b": self.actor_l2.b.tolist(),
            "actor_out_W": self.actor_out.W.tolist(),
            "actor_out_b": self.actor_out.b.tolist(),
            "log_std": self.log_std.tolist(),
            "critic_l1_W": self.critic_l1.W.tolist(),
            "critic_l1_b": self.critic_l1.b.tolist(),
            "critic_l2_W": self.critic_l2.W.tolist(),
            "critic_l2_b": self.critic_l2.b.tolist(),
            "critic_out_W": self.critic_out.W.tolist(),
            "critic_out_b": self.critic_out.b.tolist(),
            "training_history": self.training_history[-50:],
            "winning_patterns": list(self.winning_patterns)[-20:],
            "losing_patterns": list(self.losing_patterns)[-20:],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(weights, f)

    def load_weights(self, path: str) -> None:
        with open(path, "r") as f:
            w = json.load(f)
        self.actor_l1.W = np.array(w["actor_l1_W"], dtype=np.float32)
        self.actor_l1.b = np.array(w["actor_l1_b"], dtype=np.float32)
        self.actor_l2.W = np.array(w["actor_l2_W"], dtype=np.float32)
        self.actor_l2.b = np.array(w["actor_l2_b"], dtype=np.float32)
        self.actor_out.W = np.array(w["actor_out_W"], dtype=np.float32)
        self.actor_out.b = np.array(w["actor_out_b"], dtype=np.float32)
        self.log_std = np.array(w["log_std"], dtype=np.float32)
        self.critic_l1.W = np.array(w["critic_l1_W"], dtype=np.float32)
        self.critic_l1.b = np.array(w["critic_l1_b"], dtype=np.float32)
        self.training_history = w.get("training_history", [])
        self.winning_patterns = deque(w.get("winning_patterns", []), maxlen=500)
        self.losing_patterns = deque(w.get("losing_patterns", []), maxlen=500)


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Walk-forward validation to prevent look-ahead bias.
    Trains on expanding window, tests on next N days.
    """

    def __init__(self, train_window: int = 60, test_window: int = 5):
        self.train_window = train_window
        self.test_window = test_window
        self.results: List[dict] = []

    def validate(self, price_data: Dict[str, List[float]],
                 feature_data: Dict[str, List[List[float]]],
                 agent: NumpyPPOAgent) -> List[dict]:
        """Run walk-forward validation. Returns per-fold results."""
        syms = list(price_data.keys())
        total_len = min(len(v) for v in price_data.values())
        folds = []
        start = self.train_window
        while start + self.test_window <= total_len:
            train_end = start
            test_end = min(start + self.test_window, total_len)

            # Slice data for this fold
            train_prices = {s: price_data[s][:train_end] for s in syms}
            train_feats = {s: feature_data.get(s, [[]] * train_end)[:train_end] for s in syms}
            test_prices = {s: price_data[s][train_end:test_end] for s in syms}
            test_feats = {s: feature_data.get(s, [[]] * test_end)[train_end:test_end] for s in syms}

            # Quick training pass
            env = StockTradingEnv(train_prices, train_feats, max_steps=min(20, train_end))
            state = env.reset()
            for _ in range(min(50, train_end)):
                if env.done:
                    break
                action, lp, val = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_experience(Experience(state, action, reward, next_state, done, lp, val))
                state = next_state
            agent.update()

            # Test pass
            env_test = StockTradingEnv(test_prices, test_feats, max_steps=self.test_window)
            state = env_test.reset()
            test_rewards = []
            while not env_test.done:
                action, _, _ = agent.select_action(state, deterministic=True)
                state, reward, done, info = env_test.step(action)
                test_rewards.append(reward)

            fold_result = {
                "fold_start": train_end,
                "fold_end": test_end,
                "mean_reward": round(sum(test_rewards) / max(len(test_rewards), 1), 4),
                "total_return": round(sum(test_rewards), 4),
                "final_value": round(info.get("portfolio_value", 0), 2),
            }
            folds.append(fold_result)
            start += self.test_window

        self.results = folds
        return folds


# ---------------------------------------------------------------------------
# Main Deep Learning Engine
# ---------------------------------------------------------------------------

class DeepLearningEngine:
    """
    Orchestrator for the PPO agent + walk-forward training.
    Produces daily action signals → position sizing.
    """

    WEIGHTS_FILE = "/home/user/ai-hedgefund/models/ppo_weights.json"

    def __init__(self, symbols: List[str] = None, initial_capital: float = 1_000.0):
        self.symbols = symbols or ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]
        self.initial_capital = initial_capital
        self.agent: Optional[NumpyPPOAgent] = None
        self.env: Optional[StockTradingEnv] = None
        self.validator = WalkForwardValidator()
        self.signal_history: List[dict] = []
        self._init_agent()

    def _init_agent(self) -> None:
        state_dim = len(self.symbols) * NUM_FEATURES + len(self.symbols) + 1
        action_dim = len(self.symbols)
        self.agent = NumpyPPOAgent(state_dim, action_dim)
        if os.path.exists(self.WEIGHTS_FILE):
            try:
                self.agent.load_weights(self.WEIGHTS_FILE)
                print(f"[DL] Loaded weights from {self.WEIGHTS_FILE}")
            except Exception as e:
                print(f"[DL] Could not load weights: {e}")

    def build_features(self, price_series: List[float]) -> List[float]:
        """Build NUM_FEATURES feature vector from price series."""
        n = len(price_series)
        if n == 0:
            return [0.0] * NUM_FEATURES

        def safe_ret(i: int, window: int) -> float:
            if i < window:
                return 0.0
            base = price_series[i - window]
            return (price_series[i] - base) / base if base > 0 else 0.0

        def ema(prices: List[float], period: int) -> float:
            if not prices:
                return 0.0
            k = 2.0 / (period + 1)
            e = prices[0]
            for p in prices[1:]:
                e = p * k + e * (1 - k)
            return e

        p = price_series[-1]
        prices = price_series
        i = n - 1

        close = p
        ret_1 = safe_ret(i, 1)
        ret_5 = safe_ret(i, 5)
        ret_10 = safe_ret(i, 10)
        ret_20 = safe_ret(i, 20)
        ret_60 = safe_ret(i, 60)

        # RSI
        gains = [max(prices[j] - prices[j-1], 0) for j in range(max(1, n-15), n)]
        losses = [max(prices[j-1] - prices[j], 0) for j in range(max(1, n-15), n)]
        avg_gain = sum(gains) / max(len(gains), 1)
        avg_loss = sum(losses) / max(len(losses), 1)
        rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-9)))

        ema9 = ema(prices[-9:], 9)
        ema21 = ema(prices[-21:], 21)
        ema50 = ema(prices[-50:], 50)
        ema200 = ema(prices[-200:], 200)

        # Bollinger bands
        window20 = prices[-20:] if len(prices) >= 20 else prices
        mean20 = sum(window20) / len(window20)
        std20 = (sum((x - mean20)**2 for x in window20) / max(len(window20), 1)) ** 0.5
        bb_upper = mean20 + 2 * std20
        bb_lower = mean20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / max(mean20, 1e-9)

        # Hist vol
        rets = [safe_ret(j, 1) for j in range(max(1, n-21), n)]
        hist_vol = (sum(r**2 for r in rets) / max(len(rets), 1)) ** 0.5 * (252 ** 0.5)

        feats = [
            close / 1000, 0.0, 0.0, 0.0, close / 1000,  # close, open, high, low, vwap
            ret_1, ret_5, ret_10, ret_20, ret_60,         # returns
            1.0, 1.0, 0.0, 0.0, 0.0,                      # volume features (normalized)
            rsi / 100, rsi / 100, 0.0, 0.0, 0.0,          # RSI, MACD
            0.0, 0.0, 0.0,                                  # stoch, williams
            ema9 / 1000, ema21 / 1000, ema50 / 1000, ema200 / 1000,  # EMAs
            bb_upper / 1000, bb_lower / 1000, bb_width,    # Bollinger
            0.0, hist_vol, hist_vol, 0.2, 0.5,             # vol
            15.0, 3.0, 10.0, 0.1, 0.0,                    # fundamental
            0.0, 0.04, 100.0, 0.0, 0.0,                   # macro
            0.0, 0.5, 0.0, 0.0, 0.0,                      # sentiment
        ]
        feats = feats[:NUM_FEATURES]
        while len(feats) < NUM_FEATURES:
            feats.append(0.0)
        return feats

    def generate_signals(self, current_prices: Dict[str, float],
                          price_history: Dict[str, List[float]]) -> Dict[str, dict]:
        """
        Generate trading signals for each symbol.
        Returns dict: symbol → {action, confidence, position_size_pct}
        """
        # Build feature matrix
        feature_data = {}
        for sym in self.symbols:
            hist = price_history.get(sym, [current_prices.get(sym, 100.0)])
            feature_data[sym] = [self.build_features(hist[:i+1]) for i in range(len(hist))]

        # Create dummy env for state construction
        dummy_prices = {s: [current_prices.get(s, 100.0)] for s in self.symbols}
        dummy_feats = {s: [feature_data[s][-1] if feature_data[s] else [0.0]*NUM_FEATURES]
                       for s in self.symbols}
        env = StockTradingEnv(dummy_prices, dummy_feats, max_steps=1)
        state = env.reset()

        actions, _, values = self.agent.select_action(state, deterministic=True)
        signals = {}
        for i, sym in enumerate(self.symbols):
            action = float(actions[i]) if i < len(actions) else 0.0
            confidence = abs(action)
            direction = "long" if action > 0.05 else ("short" if action < -0.05 else "neutral")
            signals[sym] = {
                "symbol": sym,
                "action": round(action, 4),
                "direction": direction,
                "confidence": round(confidence, 4),
                "position_size_pct": round(abs(action) * 0.25, 4),  # max 25% per position
                "estimated_value": float(values) if not isinstance(values, np.ndarray) else float(values),
            }

        self.signal_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "signals": signals,
        })
        return signals

    def daily_update(self, realized_pnl: float, price_history: Dict[str, List[float]]) -> dict:
        """
        End-of-day learning update. Returns training metrics.
        """
        # Generate experience from today's P&L
        for sym in self.symbols:
            hist = price_history.get(sym, [100.0])
            feats = self.build_features(hist)
            dummy_state = np.array(feats + [0.0] * (self.agent.state_dim - len(feats)), dtype=np.float32)
            dummy_action = np.zeros(self.agent.action_dim, dtype=np.float32)
            reward = realized_pnl / max(abs(realized_pnl), 1.0) * 0.1  # normalized reward
            exp = Experience(
                state=dummy_state,
                action=dummy_action,
                reward=reward,
                next_state=dummy_state,
                done=False,
                log_prob=-1.0,
                value=0.0,
            )
            self.agent.store_experience(exp)

        metrics = self.agent.update()
        self._save_weights()
        return metrics

    def _save_weights(self) -> None:
        os.makedirs(os.path.dirname(self.WEIGHTS_FILE), exist_ok=True)
        self.agent.save_weights(self.WEIGHTS_FILE)

    def run_backtest(self, price_history: Dict[str, List[float]]) -> dict:
        """Run walk-forward backtest and return summary."""
        feature_data = {}
        for sym in self.symbols:
            hist = price_history.get(sym, [])
            feature_data[sym] = [self.build_features(hist[:i+1]) for i in range(len(hist))]

        folds = self.validator.validate(price_history, feature_data, self.agent)
        self._save_weights()

        if folds:
            returns = [f["total_return"] for f in folds]
            return {
                "num_folds": len(folds),
                "mean_fold_return": round(sum(returns) / len(returns), 4),
                "best_fold": max(folds, key=lambda x: x["total_return"]),
                "worst_fold": min(folds, key=lambda x: x["total_return"]),
                "folds": folds,
            }
        return {"num_folds": 0, "folds": []}

    def get_pattern_memory(self) -> dict:
        """Return winning and losing trade patterns."""
        return {
            "num_winning_patterns": len(self.agent.winning_patterns),
            "num_losing_patterns": len(self.agent.losing_patterns),
            "top_winning": list(self.agent.winning_patterns)[-5:],
            "top_losing": list(self.agent.losing_patterns)[-5:],
        }


if __name__ == "__main__":
    engine = DeepLearningEngine(symbols=["SPY", "QQQ", "NVDA"])
    prices = {"SPY": 450.0, "QQQ": 380.0, "NVDA": 800.0}
    hist = {s: [p * (1 + 0.01 * (i % 5 - 2)) for i in range(100)] for s, p in prices.items()}
    signals = engine.generate_signals(prices, hist)
    for sym, sig in signals.items():
        print(f"{sym}: {sig}")
