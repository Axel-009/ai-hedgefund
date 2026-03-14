"""
Alpha-Beta Unleashed — Continuous Exposure Engine
===================================================
Thesis Strategy: Dynamic beta management via annualized log-drift (Rm)
riding the Gamma Corridor (7%-12%) with vol-normalized scaling.

DATA: yfinance (stub for live broker until API connected)
EXECUTION: Paper-trade logger mirroring IB placeOrder interface exactly
           → swap to real broker with one-line change

Author: Platform Init — claude/init-test-repos-oPogr
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# ==============================================================================
# 1. QUANTITATIVE CONFIGURATION (PROFIT MAXIMIZATION MODE)
# ==============================================================================
ALPHA = 0.02                # 2% Secular Alpha headstart (Selection Alpha)
R_LOW, R_HIGH = 0.07, 0.12   # The "Gamma Corridor"
BETA_MAX = 2.0               # Full Throttle Cap
BETA_INV = -0.136            # Strategic Hedge Floor
EXECUTION_MULTIPLIER = 4.7   # Thesis Scaling Factor
MIN_TRADE_THRESHOLD = 0.05   # Friction Control (Gamma Throttle)
CHECK_INTERVAL = 60          # 1-minute execution resolution

# Paper portfolio NLV for simulation (replace with live NLV call when broker live)
PAPER_NLV = 100_000.0
MES_MULTIPLIER = 5           # MES contract: $5 per index point

TRADE_LOG_PATH = Path(__file__).parent / "trade_log.jsonl"


# ==============================================================================
# 2. DATA LAYER — yfinance (production stub until live broker API connected)
# ==============================================================================
class MarketDataYF:
    """
    yfinance-backed market intelligence.
    Interface mirrors what the IB reqHistoricalData layer returned.
    Swap entire class for IBDataLayer when broker is live.
    """

    def __init__(self, ticker: str = "^GSPC", lookback: str = "1y"):
        self.ticker = ticker
        self.lookback = lookback
        self._df: pd.DataFrame | None = None

    def fetch(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            # Try from Financial-Data repo
            sys.path.insert(0, str(Path(__file__).parents[3] / "Financial-Data"))
            import yfinance as yf

        try:
            raw = yf.download(self.ticker, period=self.lookback, progress=False, auto_adjust=True)
        except Exception:
            raw = pd.DataFrame()

        if raw.empty:
            # Offline / sandbox fallback: generate realistic synthetic SPX data
            # Production: this branch is never hit when network is available
            print(f"[DATA] Network unavailable — using synthetic SPX for logic validation")
            raw = self._synthetic_spx()

        df = raw[["Close"]].copy()
        df.columns = ["close"]
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        self._df = df.dropna()
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.fetch()
        return self._df

    def _synthetic_spx(self) -> pd.DataFrame:
        """
        Generates synthetic SPX-like OHLCV data calibrated to realistic params:
        - Annualized drift ~ 9% (mid-corridor, typical bull market)
        - Annualized vol ~ 15% (thesis baseline)
        - Starting price ~ 5700 (approximate SPX level as of 2026-Q1)
        Used ONLY when live network is unavailable (sandbox / CI).
        """
        np.random.seed(42)
        n = 252  # 1 trading year
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        mu_daily = 0.09 / 252
        sigma_daily = 0.15 / np.sqrt(252)
        log_returns = np.random.normal(mu_daily, sigma_daily, n)
        prices = 5700.0 * np.exp(np.cumsum(log_returns))
        df = pd.DataFrame({"Close": prices}, index=dates)
        df.index.name = "Date"
        return df

    @property
    def sigma_m(self) -> float:
        """Realized annualized volatility."""
        return float(self.df["log_ret"].std() * np.sqrt(252))

    @property
    def Rm(self) -> float:
        """Annualized log-drift (sum of daily log returns over lookback)."""
        return float(self.df["log_ret"].sum())

    @property
    def last_price(self) -> float:
        return float(self.df["close"].iloc[-1])


# ==============================================================================
# 3. PAPER EXECUTION LAYER
#    Mirrors IB placeOrder interface exactly — swap for real broker one line
# ==============================================================================
class PaperBroker:
    """
    Paper-trade logger. Exact same interface as the IB execution layer.
    When live broker is ready:
        broker = IBBroker(host, port, clientId)  # ← one-line swap
    """

    def __init__(self, nlv: float = PAPER_NLV):
        self._nlv = nlv
        self._positions: dict[str, int] = {"MES": 0}
        TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def get_nlv(self) -> float:
        """Dynamic NLV — paper uses fixed initial; live uses account summary."""
        return self._nlv

    def get_position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    def place_order(self, symbol: str, action: str, qty: int, price: float) -> dict:
        """
        Paper fills at last price. Live version routes to IB MarketOrder.
        Returns order receipt (mirrors IB trade object structure).
        """
        if qty <= 0:
            return {}

        fill_price = price  # market order → fill at last
        pnl_impact = (1 if action == "BUY" else -1) * qty * fill_price * MES_MULTIPLIER

        if action == "BUY":
            self._positions[symbol] = self._positions.get(symbol, 0) + qty
        else:
            self._positions[symbol] = self._positions.get(symbol, 0) - qty

        receipt = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "fill_price": fill_price,
            "notional": qty * fill_price * MES_MULTIPLIER,
            "new_position": self._positions[symbol],
        }

        # Append to trade log (JSONL for easy streaming/analysis)
        with open(TRADE_LOG_PATH, "a") as f:
            f.write(json.dumps(receipt) + "\n")

        return receipt

    @property
    def positions(self) -> dict:
        return dict(self._positions)


# ==============================================================================
# 4. CONTINUOUS EXPOSURE ENGINE
# ==============================================================================
class AlphaBetaUnleashed:
    """
    The Quantitative Beta Engine.
    Converts rolling annualized market drift (Rm) into target portfolio beta,
    then routes execution via MES futures (paper or live broker).
    """

    def __init__(self, broker: PaperBroker | None = None):
        self.data = MarketDataYF(ticker="^GSPC", lookback="1y")
        self.broker = broker or PaperBroker()
        self.current_beta = 0.0

        # Initial market sync
        self._refresh_market_state()

    # --------------------------------------------------------------------------
    # Market Intelligence
    # --------------------------------------------------------------------------
    def _refresh_market_state(self):
        """Fetch and cache rolling Rm + sigma_m from yfinance."""
        self.data.fetch()
        self.sigma_m = self.data.sigma_m
        self.Rm = self.data.Rm
        self.last_price = self.data.last_price
        print(f"[MARKET STATE] Rm={self.Rm:.2%} | Vol={self.sigma_m:.2%} | SPX={self.last_price:.2f}")

    # --------------------------------------------------------------------------
    # The Smooth Beta Curve  (Section VI of thesis)
    # --------------------------------------------------------------------------
    def calculate_target_beta(self, Rm: float) -> float:
        """
        Converts annualized log-drift into target portfolio beta.

        Below R_LOW  (7%):  base_beta = -0.029  (slight short / hedge bias)
        Above R_HIGH (12%): base_beta =  0.425  (moderate long)
        Between:            linear slope M = 9.08

        Then:  target = base * EXECUTION_MULTIPLIER * vol_adj
               clamped to [BETA_INV, BETA_MAX]
        """
        # 1. Base linear interpolation across the Gamma Corridor
        if Rm <= R_LOW:
            base_beta = -0.029
        elif Rm >= R_HIGH:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (R_HIGH - R_LOW)  # M ≈ 9.08
            base_beta = -0.029 + slope * (Rm - R_LOW)

        # 2. Scale + vol-normalize (thesis standard: 15% target vol)
        #    If realized vol spikes, beta auto-contracts to protect core
        vol_adj = 0.15 / max(self.sigma_m, 0.05)
        target_beta = base_beta * EXECUTION_MULTIPLIER * vol_adj

        return max(BETA_INV, min(BETA_MAX, target_beta))

    # --------------------------------------------------------------------------
    # Execution Router
    # --------------------------------------------------------------------------
    def execute_logic(self, target_beta: float) -> dict:
        """
        Translates target beta → MES contract delta → order (paper or live).
        Fires only when |Δbeta| > MIN_TRADE_THRESHOLD (gamma friction control).
        """
        nlv = self.broker.get_nlv()
        contract_notional = self.last_price * MES_MULTIPLIER

        target_qty = int((target_beta * nlv) / contract_notional)
        current_qty = self.broker.get_position("MES")
        qty_diff = target_qty - current_qty

        beta_drift = abs(target_beta - self.current_beta)

        hold_reason = None
        if beta_drift <= MIN_TRADE_THRESHOLD:
            hold_reason = f"beta_drift={beta_drift:.4f} ≤ threshold={MIN_TRADE_THRESHOLD}"
        elif qty_diff == 0:
            hold_reason = f"qty_diff=0 (NLV too small for 1 contract at this beta)"

        if hold_reason is None:
            action = "BUY" if qty_diff > 0 else "SELL"
            receipt = self.broker.place_order("MES", action, abs(qty_diff), self.last_price)
            self.current_beta = target_beta
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | "
                f"ACTION: {action} {abs(qty_diff)} MES | "
                f"FILL: {self.last_price:.2f} | "
                f"NEW BETA: {target_beta:.3f} | "
                f"POSITION: {self.broker.get_position('MES')} contracts"
            )
            return receipt
        else:
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | "
                f"HOLD | {hold_reason} | "
                f"CURRENT BETA: {self.current_beta:.3f} | TARGET: {target_beta:.3f}"
            )
            return {}

    # --------------------------------------------------------------------------
    # State Snapshot (for portfolio analytics layer)
    # --------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "Rm": self.Rm,
            "sigma_m": self.sigma_m,
            "last_price": self.last_price,
            "current_beta": self.current_beta,
            "target_beta": self.calculate_target_beta(self.Rm),
            "nlv": self.broker.get_nlv(),
            "positions": self.broker.positions,
            "vol_adj": 0.15 / max(self.sigma_m, 0.05),
            "regime": self._classify_regime(),
        }

    def _classify_regime(self) -> str:
        """Quick regime label for analytics dashboard."""
        if self.Rm < R_LOW:
            return "CONTRACTION"
        elif self.Rm > R_HIGH:
            return "EXPANSION"
        else:
            return "GAMMA_CORRIDOR"


# ==============================================================================
# 5. PRODUCTION EXECUTION LOOP
# ==============================================================================
def run_engine(paper_nlv: float = PAPER_NLV, interval: int = CHECK_INTERVAL):
    """
    Main loop. Replace PaperBroker() with IBBroker() when live.
    """
    broker = PaperBroker(nlv=paper_nlv)
    cockpit = AlphaBetaUnleashed(broker=broker)

    print("\n" + "=" * 60)
    print("  ALPHA-BETA UNLEASHED — QUANTITATIVE EXPOSURE ENGINE")
    print("  Mode: PAPER (yfinance data / simulated execution)")
    print("  Broker: PaperBroker → swap IBBroker for live")
    print("=" * 60 + "\n")

    while True:
        try:
            cockpit._refresh_market_state()
            target = cockpit.calculate_target_beta(cockpit.Rm)
            cockpit.execute_logic(target)
            snap = cockpit.snapshot()
            print(f"  REGIME: {snap['regime']} | VOL_ADJ: {snap['vol_adj']:.3f}\n")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Engine stopped by operator.")
            break
        except Exception as e:
            print(f"[ERROR] {e} — recovery in 10s...")
            time.sleep(10)


if __name__ == "__main__":
    run_engine()
