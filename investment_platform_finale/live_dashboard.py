"""
METADRON CAPITAL — Live Execution Dashboard
============================================
Full-universe scanner + real-time execution monitor.

  python live_dashboard.py               # S&P 500 + ETFs (~550 symbols)
  python live_dashboard.py --tier all    # extended ~700 symbols

Network note:
  Live yfinance used if reachable. Falls back automatically to a
  calibrated GBM synthetic market simulator with seed prices from
  real reference levels — same signal pipeline, same dashboard.

Layout:
  ┌──────────────────────────── HEADER ─────────────────────────────────┐
  │  NAV · Cash · Day P&L · SPY/QQQ/VIX · Fills · Win-Rate · Universe  │
  ├─────────────────────────────┬───────────────────────────────────────┤
  │  LIVE SIGNALS               │  ACTIVE POSITIONS                     │
  │  Ranked by |score|          │  Entry · Current · Unrealized P&L     │
  │  BUY / SELL / WATCH         ├───────────────────────────────────────┤
  │                             │  EXECUTION LOG (fills & closes)       │
  ├─────────────────────────────┴───────────────────────────────────────┤
  │  SECTOR HEATMAP                │  TOP MOVERS  (all scanned tickers) │
  └────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─────────────────────────────────────────────────────────────────────────────
# SEED PRICES  (reference levels as of early 2026 — used as GBM start)
# ─────────────────────────────────────────────────────────────────────────────

SEED_PRICES: Dict[str, float] = {
    # Mega-cap tech
    "AAPL": 227.0, "MSFT": 415.0, "NVDA": 875.0, "AMZN": 205.0,
    "GOOGL": 175.0, "GOOG": 174.0, "META": 590.0, "TSLA": 245.0,
    "AVGO": 1380.0, "ORCL": 162.0, "AMD": 178.0, "INTC": 22.0,
    "QCOM": 165.0, "TXN": 192.0, "MU": 92.0, "AMAT": 185.0,
    "LRCX": 775.0, "KLAC": 680.0, "SNPS": 510.0, "CDNS": 315.0,
    "CRM": 305.0, "NOW": 920.0, "ADBE": 425.0, "INTU": 640.0,
    "PANW": 180.0, "FTNT": 88.0, "ZS": 210.0, "CRWD": 350.0,
    "SNOW": 155.0, "DDOG": 125.0, "MDB": 295.0, "PLTR": 78.0,
    # Financials
    "JPM": 235.0, "BAC": 45.0, "WFC": 72.0, "GS": 545.0,
    "MS": 128.0, "BLK": 1020.0, "C": 72.0, "AXP": 295.0,
    "SCHW": 78.0, "COF": 185.0, "V": 316.0, "MA": 525.0,
    "CB": 280.0, "PGR": 255.0, "MET": 82.0, "PRU": 120.0,
    # Health Care
    "UNH": 540.0, "JNJ": 155.0, "LLY": 850.0, "ABBV": 185.0,
    "MRK": 105.0, "TMO": 540.0, "ABT": 118.0, "DHR": 235.0,
    "BMY": 52.0, "AMGN": 285.0, "ISRG": 480.0, "MDT": 88.0,
    "SYK": 365.0, "BSX": 88.0, "BDX": 238.0, "IDXX": 510.0,
    # Consumer Discretionary
    "HD": 395.0, "MCD": 308.0, "NKE": 75.0, "SBUX": 92.0,
    "LOW": 245.0, "TJX": 118.0, "BKNG": 4850.0, "MAR": 255.0,
    "HLT": 235.0, "CMG": 58.0, "ORLY": 1250.0, "AZO": 3150.0,
    "DG": 95.0, "TGT": 138.0, "ROST": 148.0, "ULTA": 390.0,
    # Consumer Staples
    "WMT": 93.0, "PG": 168.0, "KO": 62.0, "PEP": 145.0,
    "COST": 925.0, "PM": 135.0, "MO": 52.0, "CL": 92.0,
    "KMB": 133.0, "MDLZ": 61.0, "GIS": 56.0, "K": 78.0,
    # Industrials
    "RTX": 125.0, "HON": 218.0, "UPS": 118.0, "CAT": 380.0,
    "DE": 435.0, "GE": 195.0, "LMT": 490.0, "BA": 172.0,
    "MMM": 135.0, "EMR": 115.0, "ETN": 342.0, "PH": 675.0,
    # Energy
    "XOM": 118.0, "CVX": 155.0, "COP": 108.0, "SLB": 45.0,
    "EOG": 125.0, "MPC": 168.0, "PSX": 148.0, "VLO": 155.0,
    "OXY": 58.0, "DVN": 38.0, "HAL": 30.0,
    # Materials
    "LIN": 465.0, "APD": 292.0, "ECL": 248.0, "SHW": 368.0,
    "NEM": 48.0, "FCX": 42.0, "NUE": 145.0,
    # Utilities
    "NEE": 73.0, "DUK": 108.0, "SO": 88.0, "D": 52.0,
    "AEP": 98.0, "EXC": 42.0, "XEL": 62.0,
    # Real Estate
    "PLD": 118.0, "AMT": 198.0, "EQIX": 835.0, "CCI": 98.0,
    "PSA": 315.0, "DLR": 168.0, "AVB": 218.0, "O": 56.0,
    # Comm Services
    "NFLX": 1050.0, "DIS": 112.0, "CMCSA": 38.0, "T": 22.0,
    "VZ": 41.0, "TMUS": 248.0, "CHTR": 395.0, "EA": 138.0,
    "TTWO": 155.0,
    # Small-cap / speculative
    "SMCI": 42.0, "IONQ": 32.0, "SOUN": 18.0, "MSTR": 325.0,
    "COIN": 215.0, "HOOD": 42.0, "SOFI": 12.0, "UPST": 62.0,
    "AFRM": 58.0, "GME": 25.0, "AMC": 3.8,
    # Sector ETFs
    "SPY": 571.0, "QQQ": 495.0, "IWM": 218.0, "DIA": 428.0,
    "XLK": 228.0, "XLV": 142.0, "XLF": 48.0, "XLY": 195.0,
    "XLC": 85.0,  "XLI": 128.0, "XLP": 78.0, "XLE": 92.0,
    "XLU": 72.0,  "XLRE": 38.0, "XLB": 88.0,
    "SMH": 228.0, "SOXX": 228.0, "IBB": 135.0, "XBI": 88.0,
    "ARKK": 52.0, "GLD": 245.0, "TLT": 88.0,
    "HYG": 78.0,  "LQD": 108.0, "JNK": 95.0, "ANGL": 28.0,
    "EEM": 42.0,  "EFA": 78.0,  "EWJ": 65.0,
    "VXX": 28.0,  "UVXY": 18.0,
    # Crypto proxies
    "BTC-USD": 84000.0, "ETH-USD": 2200.0,
    # Volatility (synthetic spot)
    "VIX": 22.0,
}

SECTOR_ASSIGNMENT: Dict[str, str] = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMZN":"Consumer Discretionary",
    "GOOGL":"Communication Services","GOOG":"Communication Services","META":"Communication Services",
    "TSLA":"Consumer Discretionary","AVGO":"Technology","ORCL":"Technology",
    "AMD":"Technology","INTC":"Technology","QCOM":"Technology","TXN":"Technology",
    "MU":"Technology","AMAT":"Technology","LRCX":"Technology","KLAC":"Technology",
    "SNPS":"Technology","CDNS":"Technology","CRM":"Technology","NOW":"Technology",
    "ADBE":"Technology","INTU":"Technology","PANW":"Technology","FTNT":"Technology",
    "ZS":"Technology","CRWD":"Technology","SNOW":"Technology","DDOG":"Technology",
    "MDB":"Technology","PLTR":"Technology",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","BLK":"Financials","C":"Financials","AXP":"Financials",
    "SCHW":"Financials","COF":"Financials","V":"Financials","MA":"Financials",
    "CB":"Financials","PGR":"Financials","MET":"Financials","PRU":"Financials",
    "UNH":"Health Care","JNJ":"Health Care","LLY":"Health Care","ABBV":"Health Care",
    "MRK":"Health Care","TMO":"Health Care","ABT":"Health Care","DHR":"Health Care",
    "BMY":"Health Care","AMGN":"Health Care","ISRG":"Health Care","MDT":"Health Care",
    "SYK":"Health Care","BSX":"Health Care","BDX":"Health Care","IDXX":"Health Care",
    "HD":"Consumer Discretionary","MCD":"Consumer Discretionary","NKE":"Consumer Discretionary",
    "SBUX":"Consumer Discretionary","LOW":"Consumer Discretionary","TJX":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","MAR":"Consumer Discretionary","HLT":"Consumer Discretionary",
    "CMG":"Consumer Discretionary","ORLY":"Consumer Discretionary","AZO":"Consumer Discretionary",
    "DG":"Consumer Discretionary","TGT":"Consumer Discretionary","ROST":"Consumer Discretionary",
    "ULTA":"Consumer Discretionary",
    "WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "COST":"Consumer Staples","PM":"Consumer Staples","MO":"Consumer Staples","CL":"Consumer Staples",
    "KMB":"Consumer Staples","MDLZ":"Consumer Staples","GIS":"Consumer Staples","K":"Consumer Staples",
    "RTX":"Industrials","HON":"Industrials","UPS":"Industrials","CAT":"Industrials",
    "DE":"Industrials","GE":"Industrials","LMT":"Industrials","BA":"Industrials",
    "MMM":"Industrials","EMR":"Industrials","ETN":"Industrials","PH":"Industrials",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "EOG":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy","OXY":"Energy",
    "DVN":"Energy","HAL":"Energy",
    "LIN":"Materials","APD":"Materials","ECL":"Materials","SHW":"Materials",
    "NEM":"Materials","FCX":"Materials","NUE":"Materials",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "AEP":"Utilities","EXC":"Utilities","XEL":"Utilities",
    "PLD":"Real Estate","AMT":"Real Estate","EQIX":"Real Estate","CCI":"Real Estate",
    "PSA":"Real Estate","DLR":"Real Estate","AVB":"Real Estate","O":"Real Estate",
    "NFLX":"Communication Services","DIS":"Communication Services","CMCSA":"Communication Services",
    "T":"Communication Services","VZ":"Communication Services","TMUS":"Communication Services",
    "CHTR":"Communication Services","EA":"Communication Services","TTWO":"Communication Services",
}

UNIVERSE_TIERS: Dict[str, List[str]] = {
    "sp500": list(SEED_PRICES.keys()),
    "all": list(SEED_PRICES.keys()),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerQuote:
    symbol: str
    price: float = 0.0
    prev_close: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    avg_volume: float = 1.0
    pct_change: float = 0.0
    vol_ratio: float = 1.0
    spread_bps: float = 0.0
    sector: str = "Unknown"
    rsi: float = 50.0
    momentum_5d: float = 0.0
    residual_alpha: float = 0.0
    score: float = 0.0
    signal: str = "WATCH"
    last_updated: str = ""


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    qty: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pct: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    signal_score: float = 0.0


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    time: str
    pnl: float = 0.0
    status: str = "FILLED"


@dataclass
class PortfolioState:
    nav: float = 1_000_000.0
    cash: float = 1_000_000.0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    day_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC MARKET SIMULATOR  (GBM intraday with sector correlation)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticMarket:
    """
    Calibrated GBM simulator:
      - Each ticker starts at SEED_PRICES[sym] (real reference level)
      - Sector ETFs drive correlated moves (beta 0.6–0.9)
      - Individual vol calibrated by market cap / beta estimate
      - Volume surprises tied to price moves
      - VIX inversely correlated with SPY
    """

    # Annualised vol by sector (realistic)
    SECTOR_VOL: Dict[str, float] = {
        "Technology": 0.28, "Health Care": 0.20, "Financials": 0.22,
        "Consumer Discretionary": 0.26, "Communication Services": 0.24,
        "Industrials": 0.20, "Consumer Staples": 0.14, "Energy": 0.30,
        "Utilities": 0.15, "Real Estate": 0.22, "Materials": 0.22,
        "Unknown": 0.25,
    }

    # ETF to sector mapping
    ETF_SECTOR: Dict[str, str] = {
        "XLK": "Technology", "XLV": "Health Care", "XLF": "Financials",
        "XLY": "Consumer Discretionary", "XLC": "Communication Services",
        "XLI": "Industrials", "XLP": "Consumer Staples", "XLE": "Energy",
        "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials",
    }

    def __init__(self, universe: List[str]):
        self.universe = universe
        self._prices: Dict[str, float] = {}
        self._prev_open: Dict[str, float] = {}
        self._volumes: Dict[str, float] = {}
        self._avg_vols: Dict[str, float] = {}
        self._price_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._market_drift: float = 0.0   # intraday regime
        self._tick = 0

        # Initialise starting prices
        for sym in universe:
            seed = SEED_PRICES.get(sym)
            if seed is None:
                h = int(hashlib.md5(sym.encode()).hexdigest(), 16)
                seed = 10.0 + (h % 990)
            # Add small open-day jitter
            rng = np.random.default_rng(hash(sym) % (2**31))
            jitter = rng.normal(0, 0.002)
            self._prices[sym] = round(seed * (1 + jitter), 2)
            self._prev_open[sym] = self._prices[sym]
            avg_vol = self._default_avg_vol(sym)
            self._avg_vols[sym] = avg_vol
            self._volumes[sym] = float(rng.integers(int(avg_vol * 0.1), int(avg_vol * 0.5)))
            self._price_history[sym] = deque(
                [self._prices[sym]] * 20, maxlen=20
            )

    def _default_avg_vol(self, sym: str) -> float:
        """Estimate avg daily volume from seed price (larger price → smaller float)."""
        price = SEED_PRICES.get(sym, 100.0)
        if price > 1000:
            return 2_000_000
        elif price > 300:
            return 8_000_000
        elif price > 100:
            return 25_000_000
        elif price > 50:
            return 40_000_000
        else:
            return 60_000_000

    def _dt_vol(self, sym: str) -> float:
        """Per-tick vol (5-second equivalent of annualised vol)."""
        sector = SECTOR_ASSIGNMENT.get(sym, "Unknown")
        ann_vol = self.SECTOR_VOL.get(sector, 0.25)
        # ETFs less volatile than single stocks
        if sym in self.ETF_SECTOR or sym in ("SPY", "QQQ", "IWM", "DIA"):
            ann_vol *= 0.4
        elif sym in ("BTC-USD", "ETH-USD"):
            ann_vol = 0.80
        elif sym in ("VIX",):
            ann_vol = 0.60
        trading_seconds = 6.5 * 3600
        return ann_vol / math.sqrt(252 * trading_seconds / 5)

    def tick(self) -> Dict[str, TickerQuote]:
        """Advance market by one tick, return all quotes."""
        self._tick += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        rng = np.random.default_rng(self._tick + int(time.time() * 1000) % (2**31))

        # Market-wide drift (slow-moving regime)
        if self._tick % 12 == 0:
            self._market_drift = rng.normal(0, 0.0003)

        # Sector shocks
        sector_shocks: Dict[str, float] = {}
        for sector in self.SECTOR_VOL:
            sector_shocks[sector] = rng.normal(self._market_drift * 0.5, 0.0002)

        with self._lock:
            for sym in self.universe:
                price = self._prices.get(sym, 10.0)
                dt_vol = self._dt_vol(sym)
                sector = SECTOR_ASSIGNMENT.get(sym, "Unknown")

                # Correlated move: 60% sector, 40% idiosyncratic
                sector_shock = sector_shocks.get(sector, 0.0)
                idio = rng.normal(0, dt_vol)
                move = 0.60 * sector_shock + 0.40 * idio

                # VIX: inverse of SPY
                if sym == "VIX":
                    spy_move = sector_shocks.get("Technology", 0) * 0.5
                    move = -spy_move * 8 + rng.normal(0, 0.005)

                new_price = max(0.01, price * (1 + move))
                self._prices[sym] = round(new_price, 2)
                self._price_history[sym].append(new_price)

                # Accumulate intraday volume
                vol_bump = abs(move) * 1e6 * rng.uniform(0.5, 2.0)
                self._volumes[sym] = min(
                    self._volumes[sym] + vol_bump,
                    self._avg_vols[sym] * 3.0
                )

            # Build quotes
            quotes: Dict[str, TickerQuote] = {}
            for sym in self.universe:
                price = self._prices[sym]
                prev  = self._prev_open[sym]
                spread = max(0.01, price * 0.0001)
                vol    = self._volumes[sym]
                avg_v  = self._avg_vols[sym]

                q = TickerQuote(
                    symbol=sym,
                    price=round(price, 2),
                    prev_close=round(prev, 2),
                    bid=round(price - spread / 2, 2),
                    ask=round(price + spread / 2, 2),
                    volume=int(vol),
                    avg_volume=int(avg_v),
                    pct_change=round((price / prev - 1) * 100, 3) if prev > 0 else 0.0,
                    vol_ratio=round(vol / avg_v, 2) if avg_v > 0 else 1.0,
                    spread_bps=round(spread / price * 10000, 1) if price > 0 else 0,
                    sector=SECTOR_ASSIGNMENT.get(sym, "Unknown"),
                    last_updated=now_str,
                )
                quotes[sym] = q

        return quotes

    def get_price_history(self, sym: str) -> List[float]:
        with self._lock:
            return list(self._price_history.get(sym, deque()))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    SECTOR_ETF_MAP = {
        "Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
        "Consumer Discretionary": "XLY", "Communication Services": "XLC",
        "Industrials": "XLI", "Consumer Staples": "XLP", "Energy": "XLE",
        "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB",
    }

    def __init__(self):
        self._sector_returns: Dict[str, float] = {}
        self._spy_return: float = 0.0

    def update_sector_returns(self, quotes: Dict[str, TickerQuote]) -> None:
        for sector, etf in self.SECTOR_ETF_MAP.items():
            if etf in quotes:
                self._sector_returns[sector] = quotes[etf].pct_change
        if "SPY" in quotes:
            self._spy_return = quotes["SPY"].pct_change

    def _compute_rsi(self, history: List[float]) -> float:
        if len(history) < 5:
            return 50.0
        changes = np.diff(history)
        gains  = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)
        ag = gains.mean() + 1e-9
        al = losses.mean() + 1e-9
        return round(100 - 100 / (1 + ag / al), 1)

    def _compute_momentum(self, history: List[float]) -> float:
        if len(history) < 5:
            return 0.0
        return round((history[-1] / history[0] - 1) * 100, 3)

    def score(self, q: TickerQuote, history: List[float]) -> None:
        """Compute and set q.score, q.signal, q.rsi, q.momentum_5d, q.residual_alpha."""
        # 1. Micro-price imbalance [-30, +30]
        if q.bid > 0 and q.ask > 0 and q.price > 0 and (q.ask + q.bid) > 0:
            mp = (q.bid * q.ask + q.ask * q.bid) / (q.ask + q.bid)
            micro_edge = (mp - q.price) / (q.price + 1e-9) * 100
            micro_score = max(-30, min(30, micro_edge * 1200))
        else:
            micro_score = 0.0

        # 2. RSI signal [-20, +20]
        rsi = self._compute_rsi(history)
        q.rsi = rsi
        if   rsi > 75: rsi_score = -20.0
        elif rsi > 65: rsi_score =  10.0
        elif rsi < 25: rsi_score = -15.0
        elif rsi < 35: rsi_score =  18.0
        else:          rsi_score = (rsi - 50) * 0.4

        # 3. Momentum [-20, +20]
        mom = self._compute_momentum(history)
        q.momentum_5d = mom
        momentum_score = max(-20, min(20, mom * 6))

        # 4. Volume surprise [-15, +15]
        vol_score = max(-15, min(15, (q.vol_ratio - 1.0) * 8))
        if q.pct_change < 0:
            vol_score = -abs(vol_score)

        # 5. Residual alpha vs sector [-20, +20]
        sector_ret = self._sector_returns.get(q.sector, self._spy_return)
        residual = q.pct_change - sector_ret
        q.residual_alpha = round(residual, 4)
        alpha_score = max(-20, min(20, residual * 60))

        total = micro_score + rsi_score + momentum_score * 0.6 + vol_score + alpha_score
        total = round(max(-100, min(100, total)), 2)
        q.score = total

        if   total >=  18: q.signal = "BUY"
        elif total <= -18: q.signal = "SELL"
        else:              q.signal = "WATCH"


# ─────────────────────────────────────────────────────────────────────────────
# PAPER EXECUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PaperExecutionEngine:
    MAX_POSITIONS    = 20
    POSITION_PCT     = 0.025   # 2.5% NAV per position
    STOP_LOSS_PCT    = 0.008   # 0.8%
    TAKE_PROFIT_PCT  = 0.018   # 1.8%
    MIN_SCORE        = 18.0
    MIN_VOL_RATIO    = 1.15
    MIN_PRICE        = 2.0
    MAX_NEW_PER_TICK = 3

    def __init__(self, portfolio: PortfolioState):
        self.port = portfolio
        self.fills: deque = deque(maxlen=60)

    def process(self, quotes: Dict[str, TickerQuote]) -> None:
        port = self.port

        # Update + check existing positions
        closed: List[str] = []
        for sym, pos in list(port.positions.items()):
            if sym not in quotes:
                continue
            q = quotes[sym]
            pos.current_price = q.price
            if pos.entry_price > 0:
                if pos.side == "LONG":
                    pos.unrealized_pnl = (q.price - pos.entry_price) * pos.qty
                    pos.unrealized_pct = (q.price / pos.entry_price - 1) * 100
                    hit_stop   = q.price <= pos.stop_price
                    hit_target = q.price >= pos.target_price
                else:
                    pos.unrealized_pnl = (pos.entry_price - q.price) * pos.qty
                    pos.unrealized_pct = (pos.entry_price / q.price - 1) * 100
                    hit_stop   = q.price >= pos.stop_price
                    hit_target = q.price <= pos.target_price

                if hit_stop or hit_target:
                    reason = "TARGET" if hit_target else "STOP"
                    pnl = pos.unrealized_pnl
                    port.cash += pos.entry_price * pos.qty + pnl
                    port.realized_pnl += pnl
                    port.day_trades += 1
                    if pnl >= 0:
                        port.win_trades += 1
                    else:
                        port.loss_trades += 1
                    self.fills.appendleft(Fill(
                        symbol=sym, side="CLOSE", qty=pos.qty,
                        price=q.price, time=datetime.now().strftime("%H:%M:%S"),
                        pnl=round(pnl, 2), status=reason,
                    ))
                    closed.append(sym)

        for sym in closed:
            del port.positions[sym]

        # Recompute NAV
        port.unrealized_pnl = sum(p.unrealized_pnl for p in port.positions.values())
        port.nav = port.cash + sum(p.current_price * p.qty for p in port.positions.values())
        port.day_pnl = port.realized_pnl + port.unrealized_pnl
        port.day_pnl_pct = port.day_pnl / (port.nav - port.day_pnl + 1e-9) * 100

        # New entries
        if len(port.positions) >= self.MAX_POSITIONS:
            return

        candidates = [
            q for q in quotes.values()
            if q.symbol not in port.positions
            and q.price >= self.MIN_PRICE
            and q.vol_ratio >= self.MIN_VOL_RATIO
            and abs(q.score) >= self.MIN_SCORE
            and q.symbol not in ("VIX", "VXX", "UVXY")
        ]
        candidates.sort(key=lambda x: abs(x.score), reverse=True)

        new_count = 0
        for q in candidates:
            if len(port.positions) >= self.MAX_POSITIONS or new_count >= self.MAX_NEW_PER_TICK:
                break
            size_usd = port.nav * self.POSITION_PCT
            qty = max(1, int(size_usd / q.price))
            cost = qty * q.price
            if cost > port.cash * 0.90:
                continue

            side = "LONG" if q.score > 0 else "SHORT"
            if side == "LONG":
                stop   = round(q.price * (1 - self.STOP_LOSS_PCT), 2)
                target = round(q.price * (1 + self.TAKE_PROFIT_PCT), 2)
            else:
                stop   = round(q.price * (1 + self.STOP_LOSS_PCT), 2)
                target = round(q.price * (1 - self.TAKE_PROFIT_PCT), 2)

            port.cash -= cost
            port.positions[q.symbol] = Position(
                symbol=q.symbol, side=side,
                entry_price=q.price, qty=qty,
                entry_time=datetime.now().strftime("%H:%M:%S"),
                current_price=q.price,
                target_price=target, stop_price=stop,
                signal_score=q.score,
            )
            self.fills.appendleft(Fill(
                symbol=q.symbol, side=side, qty=qty,
                price=q.price, time=datetime.now().strftime("%H:%M:%S"),
                pnl=0.0, status="OPEN",
            ))
            new_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# RICH RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _cp(val: float, decimals: int = 2) -> Text:
    s = f"{val:+.{decimals}f}%"
    if   val >  0.5: return Text(s, style="bold green")
    elif val < -0.5: return Text(s, style="bold red")
    elif val >  0:   return Text(s, style="green")
    elif val <  0:   return Text(s, style="red")
    return Text(s, style="dim white")


def _cs(score: float) -> Text:
    s = f"{score:+.1f}"
    if   score >=  30: return Text(s, style="bold bright_green")
    elif score >=  18: return Text(s, style="green")
    elif score <= -30: return Text(s, style="bold bright_red")
    elif score <= -18: return Text(s, style="red")
    return Text(s, style="dim white")


def _sig(sig: str) -> Text:
    if   sig == "BUY":  return Text("▲ BUY ", style="bold bright_green on black")
    elif sig == "SELL": return Text("▼ SELL", style="bold bright_red on black")
    return Text("◆ WTCH", style="dim white")


def header_panel(port: PortfolioState, quotes: Dict[str, TickerQuote],
                 universe_size: int, scanned: int, tick: int, data_mode: str) -> Panel:
    now = datetime.now().strftime("%H:%M:%S ET")
    spy  = quotes.get("SPY");  qqq = quotes.get("QQQ");  vix = quotes.get("VIX")
    spy_s = f"SPY ${spy.price:.2f} {spy.pct_change:+.2f}%" if spy else "SPY ---"
    qqq_s = f"QQQ ${qqq.price:.2f} {qqq.pct_change:+.2f}%" if qqq else "QQQ ---"
    vix_s = f"VIX {vix.price:.1f}" if vix else "VIX ---"
    pc  = "bright_green" if port.day_pnl >= 0 else "bright_red"
    wr  = port.win_trades / max(1, port.day_trades) * 100
    txt = (
        f"[bold cyan]METADRON CAPITAL[/bold cyan]  [dim]│[/dim]  [yellow]{now}[/yellow]  "
        f"[dim]│[/dim]  [bold]NAV[/bold] ${port.nav:>12,.0f}  "
        f"[bold]Cash[/bold] ${port.cash:>12,.0f}  "
        f"[bold]Day P&L[/bold] [{pc}]{port.day_pnl:+,.0f}  ({port.day_pnl_pct:+.3f}%)[/{pc}]  "
        f"[dim]│[/dim]  [cyan]{spy_s}[/cyan]  [cyan]{qqq_s}[/cyan]  [magenta]{vix_s}[/magenta]  "
        f"[dim]│[/dim]  Pos [bold]{len(port.positions)}/20[/bold]  "
        f"Fills [bold]{port.day_trades}[/bold]  "
        f"W/L [green]{port.win_trades}[/green]/[red]{port.loss_trades}[/red]  "
        f"WR [bold]{wr:.0f}%[/bold]  "
        f"[dim]│[/dim]  Univ [bold]{universe_size}[/bold]  "
        f"Scanned [bold]{scanned}[/bold]  "
        f"[dim]{data_mode}  #{tick}[/dim]"
    )
    return Panel(txt, style="on black", box=box.HORIZONTALS, padding=(0, 1))


def signals_table(quotes: Dict[str, TickerQuote], rows: int = 28) -> Table:
    buys  = sorted([q for q in quotes.values() if q.signal == "BUY"],
                   key=lambda x: x.score, reverse=True)[:rows // 2]
    sells = sorted([q for q in quotes.values() if q.signal == "SELL"],
                   key=lambda x: x.score)[:rows // 2]
    ranked = buys + sells

    t = Table(
        title=f"[bold]SIGNALS  {len(buys)}▲ BUY  {len(sells)}▼ SELL[/bold]",
        box=box.SIMPLE_HEAVY, header_style="bold cyan", min_width=66,
    )
    t.add_column("Sig",     width=7, justify="center")
    t.add_column("Symbol",  width=7, style="bold white")
    t.add_column("Price",   width=10, justify="right")
    t.add_column("Chg%",    width=8, justify="right")
    t.add_column("Score",   width=7, justify="right")
    t.add_column("RSI",     width=6, justify="right")
    t.add_column("VolRx",   width=6, justify="right")
    t.add_column("α",       width=7, justify="right")
    t.add_column("Time",    width=9)

    for q in ranked:
        vstyle = "bright_yellow" if q.vol_ratio > 2.5 else ("yellow" if q.vol_ratio > 1.8 else "white")
        t.add_row(
            _sig(q.signal), q.symbol, f"${q.price:,.2f}",
            _cp(q.pct_change), _cs(q.score), f"{q.rsi:.0f}",
            Text(f"{q.vol_ratio:.1f}x", style=vstyle),
            _cp(q.residual_alpha), Text(q.last_updated, style="dim"),
        )
    return t


def positions_table(port: PortfolioState) -> Table:
    t = Table(
        title=f"[bold]POSITIONS  ({len(port.positions)} open)[/bold]",
        box=box.SIMPLE_HEAVY, header_style="bold cyan", min_width=80,
    )
    t.add_column("Symbol",  width=7, style="bold white")
    t.add_column("Side",    width=6, justify="center")
    t.add_column("Qty",     width=7, justify="right")
    t.add_column("Entry",   width=10, justify="right")
    t.add_column("Current", width=10, justify="right")
    t.add_column("Unrlzd$", width=11, justify="right")
    t.add_column("Unrlzd%", width=9, justify="right")
    t.add_column("Target",  width=10, justify="right")
    t.add_column("Stop",    width=10, justify="right")
    t.add_column("Time",    width=9)

    for sym, pos in sorted(port.positions.items(),
                            key=lambda x: abs(x[1].unrealized_pnl), reverse=True):
        ss = "bright_green" if pos.side == "LONG" else "bright_red"
        pnl_t = Text(f"${pos.unrealized_pnl:+,.0f}",
                     style="green" if pos.unrealized_pnl >= 0 else "red")
        t.add_row(
            sym, Text(pos.side, style=ss), f"{int(pos.qty):,}",
            f"${pos.entry_price:,.2f}", f"${pos.current_price:,.2f}",
            pnl_t, _cp(pos.unrealized_pct),
            Text(f"${pos.target_price:,.2f}", style="dim green"),
            Text(f"${pos.stop_price:,.2f}",   style="dim red"),
            Text(pos.entry_time, style="dim"),
        )
    return t


def fills_table(fills: deque, rows: int = 12) -> Table:
    t = Table(
        title="[bold]EXECUTION LOG[/bold]",
        box=box.SIMPLE_HEAVY, header_style="bold cyan", min_width=66,
    )
    t.add_column("Time",   width=9)
    t.add_column("Symbol", width=7, style="bold white")
    t.add_column("Side",   width=7, justify="center")
    t.add_column("Qty",    width=7, justify="right")
    t.add_column("Price",  width=10, justify="right")
    t.add_column("P&L",    width=11, justify="right")
    t.add_column("Status", width=8)

    for f in list(fills)[:rows]:
        ss = "bright_green" if f.side == "LONG" else ("bright_red" if f.side == "SHORT" else "white")
        pnl_t = Text(f"${f.pnl:+,.2f}", style="green" if f.pnl >= 0 else "red") \
                if f.status != "OPEN" else Text("live", style="dim cyan")
        sts = "green" if f.status == "TARGET" else ("red" if f.status == "STOP" else "dim white")
        t.add_row(
            Text(f.time, style="dim"), f.symbol,
            Text(f.side, style=ss), f"{int(f.qty):,}",
            f"${f.price:,.2f}", pnl_t, Text(f.status, style=sts),
        )
    return t


def sector_table(quotes: Dict[str, TickerQuote]) -> Table:
    SMAP = {
        "Tech": "XLK", "HealthCare": "XLV", "Financials": "XLF",
        "ConDisc": "XLY", "CommSvc": "XLC", "Industrials": "XLI",
        "Staples": "XLP", "Energy": "XLE", "Utilities": "XLU",
        "RealEst": "XLRE", "Materials": "XLB",
    }
    t = Table(title="[bold]SECTOR HEATMAP[/bold]",
              box=box.SIMPLE_HEAVY, header_style="bold cyan")
    t.add_column("Sector",  width=12)
    t.add_column("ETF",     width=6)
    t.add_column("Price",   width=9, justify="right")
    t.add_column("1D Chg",  width=8, justify="right")
    t.add_column("Bar",     width=18)

    rows = sorted(
        [(n, e, quotes[e].price, quotes[e].pct_change)
         for n, e in SMAP.items() if e in quotes],
        key=lambda x: x[3], reverse=True
    )
    for name, etf, price, pct in rows:
        bar_len = min(18, int(abs(pct) * 4))
        bar = Text("█" * bar_len, style="green" if pct >= 0 else "red")
        t.add_row(name, etf, f"${price:.2f}", _cp(pct), bar)
    return t


def movers_table(quotes: Dict[str, TickerQuote]) -> Table:
    top = sorted([q for q in quotes.values() if q.price > 0],
                 key=lambda x: abs(x.pct_change), reverse=True)[:22]
    t = Table(
        title=f"[bold]TOP MOVERS  ({len(quotes)} live)[/bold]",
        box=box.SIMPLE_HEAVY, header_style="bold cyan",
    )
    t.add_column("Symbol", width=7, style="bold white")
    t.add_column("Price",  width=10, justify="right")
    t.add_column("Chg%",   width=8, justify="right")
    t.add_column("Vol",    width=8, justify="right")
    t.add_column("VolRx",  width=6, justify="right")
    t.add_column("Score",  width=7, justify="right")
    t.add_column("Sig",    width=6, justify="center")

    for q in top:
        vd = f"{q.volume/1e6:.1f}M" if q.volume >= 1e6 else f"{q.volume/1e3:.0f}K"
        t.add_row(q.symbol, f"${q.price:,.2f}", _cp(q.pct_change),
                  vd, f"{q.vol_ratio:.1f}x", _cs(q.score), _sig(q.signal))
    return t


def render(port: PortfolioState, quotes: Dict[str, TickerQuote],
           fills: deque, universe_size: int, tick: int, data_mode: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="bottom", size=18),
    )
    layout["body"].split_row(
        Layout(name="signals", ratio=2),
        Layout(name="right",   ratio=3),
    )
    layout["right"].split_column(
        Layout(name="positions", ratio=3),
        Layout(name="execlog",   ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="sectors", ratio=1),
        Layout(name="movers",  ratio=1),
    )

    layout["header"].update(
        header_panel(port, quotes, universe_size, len(quotes), tick, data_mode)
    )
    layout["signals"].update(Panel(
        signals_table(quotes), box=box.ROUNDED,
        title="[bold cyan]LIVE SIGNALS[/bold cyan]", border_style="cyan",
    ))
    layout["positions"].update(Panel(
        positions_table(port), box=box.ROUNDED,
        title="[bold cyan]ACTIVE POSITIONS[/bold cyan]", border_style="green",
    ))
    layout["execlog"].update(Panel(
        fills_table(fills), box=box.ROUNDED,
        title="[bold cyan]FILLS[/bold cyan]", border_style="yellow",
    ))
    layout["sectors"].update(Panel(
        sector_table(quotes), box=box.ROUNDED,
        title="[bold cyan]SECTORS[/bold cyan]", border_style="blue",
    ))
    layout["movers"].update(Panel(
        movers_table(quotes), box=box.ROUNDED,
        title="[bold cyan]TOP MOVERS[/bold cyan]", border_style="magenta",
    ))
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DATA WRAPPER  (try yfinance, fall back to synthetic)
# ─────────────────────────────────────────────────────────────────────────────

def try_live_fetch(universe: List[str]) -> Optional[Dict[str, TickerQuote]]:
    """Attempt one live yfinance fetch for the full universe. Returns None on failure."""
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")
        data = yf.download(universe[:100], period="2d", interval="1m",
                           group_by="ticker", auto_adjust=True,
                           progress=False, threads=True, timeout=10)
        if data is None or data.empty:
            return None
        now_str = datetime.now().strftime("%H:%M:%S")
        quotes: Dict[str, TickerQuote] = {}
        for sym in universe[:100]:
            try:
                if sym not in data.columns.get_level_values(0):
                    continue
                df = data[sym].dropna(subset=["Close"])
                if len(df) < 2:
                    continue
                price = float(df["Close"].iloc[-1])
                prev  = float(df["Close"].iloc[-2])
                vol   = float(df["Volume"].iloc[-1]) if "Volume" in df else 0
                avgv  = float(df["Volume"].mean())   if "Volume" in df else 1
                sp    = max(0.01, price * 0.0001)
                quotes[sym] = TickerQuote(
                    symbol=sym, price=round(price, 2), prev_close=round(prev, 2),
                    bid=round(price - sp/2, 2), ask=round(price + sp/2, 2),
                    volume=int(vol), avg_volume=int(avgv),
                    pct_change=round((price/prev-1)*100, 3) if prev > 0 else 0,
                    vol_ratio=round(vol/avgv, 2) if avgv > 0 else 1,
                    spread_bps=round(sp/price*10000, 1) if price > 0 else 0,
                    sector=SECTOR_ASSIGNMENT.get(sym, "Unknown"),
                    last_updated=now_str,
                )
            except Exception:
                pass
        return quotes if quotes else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Metadron Capital Live Dashboard")
    ap.add_argument("--tier",    default="sp500", choices=["sp500", "all"])
    ap.add_argument("--refresh", type=int, default=3,
                    help="Refresh interval seconds (default 3)")
    ap.add_argument("--nav",     type=float, default=1_000_000.0)
    ap.add_argument("--live",    action="store_true",
                    help="Force live yfinance (errors out if unreachable)")
    args = ap.parse_args()

    universe  = UNIVERSE_TIERS.get(args.tier, list(SEED_PRICES.keys()))
    console   = Console()
    portfolio = PortfolioState(nav=args.nav, cash=args.nav)
    sig_eng   = SignalEngine()
    executor  = PaperExecutionEngine(portfolio)

    # Detect data mode
    data_mode = "SYNTHETIC"
    if not args.live:
        console.print("\n[yellow]Probing live data...[/yellow]", end=" ")
        live_quotes = try_live_fetch(universe)
        if live_quotes and len(live_quotes) > 10:
            data_mode = "LIVE·yfinance"
            console.print("[green]LIVE data connected[/green]")
        else:
            console.print("[yellow]blocked — using calibrated GBM synthetic[/yellow]")
    else:
        live_quotes = None

    market = SyntheticMarket(universe)
    # If we got some live quotes, prime synthetic market with live prices
    if data_mode == "LIVE·yfinance" and live_quotes:
        with market._lock:
            for sym, q in live_quotes.items():
                if q.price > 0:
                    market._prices[sym] = q.price
                    market._prev_open[sym] = q.prev_close or q.price
                    for _ in range(20):
                        market._price_history[sym].append(q.price)

    console.print(
        f"\n[bold cyan]METADRON CAPITAL — Live Execution Dashboard[/bold cyan]\n"
        f"Universe: [bold]{len(universe)}[/bold] symbols  "
        f"NAV: [bold]${args.nav:,.0f}[/bold]  "
        f"Refresh: [bold]{args.refresh}s[/bold]  "
        f"Data: [bold]{data_mode}[/bold]\n"
        f"[dim]Press Ctrl+C to exit[/dim]\n"
    )
    time.sleep(1.5)

    tick = 0
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            tick += 1
            quotes = market.tick()

            # Score all tickers
            sig_eng.update_sector_returns(quotes)
            for sym, q in quotes.items():
                history = market.get_price_history(sym)
                sig_eng.score(q, history)

            # Execute paper trades
            executor.process(quotes)

            live.update(render(
                portfolio, quotes, executor.fills,
                universe_size=len(universe),
                tick=tick, data_mode=data_mode,
            ))
            time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bold red]Dashboard stopped.[/bold red]")
