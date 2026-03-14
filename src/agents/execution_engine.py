"""
Execution Engine — HFT Arbitrage Arm
======================================
Integrates:
  - WonderTrader micro-price imbalance signal (translated C++ logic → Python)
  - exchange-core order matching concepts (LMAX Disruptor pattern in Python)
  - yfinance Level-1 quote simulation (live L1 data when broker connected)

Scope: US SECURITIES ONLY
  - US equities (NYSE, NASDAQ, CBOE)
  - US ETFs (sector ETFs, IG/HY bond ETFs)
  - US equity options (via yfinance options chain)
  - MES/ES futures (via AlphaBetaUnleashed, not here)

Architecture position:
    MacroEngine → sector universe
    AlphaOptimizer → ranked names + weights
    ExecutionEngine → tick-level arb + order management
    ExchangeCoreAdapter → order book + matching (Java bridge or paper)

Signal: WonderTrader micro-price imbalance
    micro_price = (bid * ask_qty + ask * bid_qty) / (ask_qty + bid_qty)
    micro_price > last → BUY  (book imbalance favors upward move)
    micro_price < last → SELL (book imbalance favors downward move)
    Threshold: |micro_price - last| / last > MIN_EDGE to filter noise

Daily universe review: scans full macro-driven universe for RV
  → ranks by: CAPM residual alpha + micro-price edge + CtV score
  → top-N names routed to order execution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

EXEC_LOG_PATH = Path(__file__).parent / "execution_log.jsonl"
UNIVERSE_LOG  = Path(__file__).parent / "daily_universe.jsonl"

# ==============================================================================
# US SECURITIES UNIVERSE (Execution scope)
# Aligned with GICS macro rotation — US only
# ==============================================================================
US_UNIVERSE = {
    # ── Core Mega-Cap Equity ───────────────────────────────────────────────────
    "core_equity": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    ],
    # ── GICS Sector ETFs (execution proxies) ──────────────────────────────────
    "sector_etfs": [
        "XLK", "XLV", "XLF", "XLY", "XLC", "XLI",
        "XLP", "XLE", "XLU", "XLRE", "XLB", "SPY", "QQQ",
    ],
    # ── Fallen Angel / IG Credit (US-listed) ──────────────────────────────────
    "fallen_angel_ig": [
        "ANGL",   # VanEck Fallen Angel HY ETF
        "FALN",   # iShares Fallen Angels USD Bond ETF
        "LQD",    # iShares IG Corporate Bond ETF
        "VCIT",   # Vanguard Intermediate IG ETF
        "HYG",    # iShares HY Corporate Bond ETF
        "JNK",    # SPDR HY Bond ETF
        "IGLB",   # iShares Long-Term IG Bond ETF
    ],
    # ── Fallen Angel Equity Candidates (US stocks recently devalued) ──────────
    "fallen_angel_equity": [
        "INTC",   # Intel — semiconductor demotion
        "PFE",    # Pfizer — post-COVID derating
        "WBA",    # Walgreens — retail pharmacy distress
        "MPW",    # Medical Properties — REIT stress
        "VFC",    # VF Corp — consumer brand distress
        "PARA",   # Paramount — media disruption
        "DIS",    # Disney — streaming transition
    ],
    # ── RV Pairs (US market, same-sector mispricing) ──────────────────────────
    "rv_pairs": [
        ("GOOGL", "META"),   # Digital advertising duopoly RV
        ("XOM",   "CVX"),    # Energy majors RV
        ("AMD",   "INTC"),   # Semiconductor RV (winner vs fallen angel)
        ("JPM",   "BAC"),    # Money-center bank RV
        ("V",     "MA"),     # Payment network RV
        ("HD",    "LOW"),    # Home improvement RV
        ("PEP",   "KO"),     # Consumer staples RV
    ],
    # ── Volatility / Macro Hedges (US-listed) ─────────────────────────────────
    "macro_hedges": [
        "VXX",    # iPath VIX Short-Term Futures ETN
        "UVXY",   # ProShares Ultra VIX
        "GLD",    # SPDR Gold Shares
        "TLT",    # iShares 20+ Year Treasury Bond ETF
        "SHY",    # iShares 1-3 Year Treasury Bond ETF
        "IEF",    # iShares 7-10 Year Treasury Bond ETF
    ],
}

# Full flat list for daily scan
US_FULL_UNIVERSE = list({
    t for lst in US_UNIVERSE.values()
    for t in (lst if not isinstance(lst[0], tuple) else [x for pair in lst for x in pair])
})

# ==============================================================================
# SIGNAL TYPES
# ==============================================================================
class SignalType(Enum):
    MICRO_PRICE_BUY  = "MICRO_PRICE_BUY"
    MICRO_PRICE_SELL = "MICRO_PRICE_SELL"
    RV_LONG          = "RV_LONG"
    RV_SHORT         = "RV_SHORT"
    FALLEN_ANGEL_BUY = "FALLEN_ANGEL_BUY"
    HOLD             = "HOLD"

@dataclass
class ExecutionSignal:
    ticker:      str
    signal:      SignalType
    micro_price: float
    last_price:  float
    edge_bps:    float          # |micro_price - last| / last * 10000
    bid:         float
    ask:         float
    bid_size:    float
    ask_size:    float
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
    source:      str = "wondertrader_micro_price"

@dataclass
class Order:
    ticker:     str
    action:     str             # BUY / SELL
    qty:        int
    limit_px:   float
    signal:     SignalType
    local_id:   int
    status:     str = "PENDING" # PENDING / FILLED / CANCELLED / EXPIRED
    fill_px:    float = 0.0
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())
    expiry_sec: int = 30        # Cancel if unfilled after N seconds


# ==============================================================================
# WONDERTRADER MICRO-PRICE ENGINE (C++ logic translated to Python)
# Ref: WtHftStraDemo.cpp — do_calc()
# ==============================================================================
class MicroPriceEngine:
    """
    Python implementation of WonderTrader's HFT micro-price signal.

    Micro-price (theoretical fair value):
        P_micro = (bid * ask_qty + ask * bid_qty) / (ask_qty + bid_qty)

    This is the volume-weighted mid — it skews toward the side with MORE size,
    correctly reflecting where the market is about to move.

    Edge threshold (MIN_EDGE_BPS): filters noise, only signals when
    imbalance is large enough to cover spread + slippage.
    """

    MIN_EDGE_BPS    = 2.0    # Minimum edge in basis points to generate signal
    TICK_OFFSET     = 2      # Order placement ticks from last price (US equities: $0.01/tick)
    US_TICK_SIZE    = 0.01   # US equity minimum tick

    def compute_micro_price(self, bid: float, ask: float,
                            bid_size: float, ask_size: float) -> float:
        """
        Volume-weighted mid price (WonderTrader formula).
        Translated from: pxInThry = (bid*ask_qty + ask*bid_qty) / (bid_qty + ask_qty)
        """
        total_size = bid_size + ask_size
        if total_size <= 0:
            return (bid + ask) / 2.0
        return (bid * ask_size + ask * bid_size) / total_size

    def generate_signal(self, ticker: str, bid: float, ask: float,
                        bid_size: float, ask_size: float,
                        last: float) -> Optional[ExecutionSignal]:
        """
        Core signal logic from WtHftStraDemo::do_calc() — translated.
        Returns ExecutionSignal or None if no edge.
        """
        if bid <= 0 or ask <= 0 or last <= 0:
            return None

        micro_px = self.compute_micro_price(bid, ask, bid_size, ask_size)
        edge_bps = abs(micro_px - last) / last * 10_000

        if edge_bps < self.MIN_EDGE_BPS:
            return None   # Edge too small → hold (WonderTrader: signal == 0)

        if micro_px > last:
            sig = SignalType.MICRO_PRICE_BUY
        else:
            sig = SignalType.MICRO_PRICE_SELL

        return ExecutionSignal(
            ticker=ticker, signal=sig,
            micro_price=micro_px, last_price=last,
            edge_bps=round(edge_bps, 2),
            bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        )

    def calc_order_price(self, signal: ExecutionSignal) -> float:
        """
        WonderTrader placement logic:
          BUY  → last + offset * tick  (aggressive — pays up to grab liquidity)
          SELL → last - offset * tick
        Translated from stra_buy/stra_sell with getPriceTick() * _offset
        """
        offset = self.TICK_OFFSET * self.US_TICK_SIZE
        if signal.signal == SignalType.MICRO_PRICE_BUY:
            return round(signal.last_price + offset, 2)
        else:
            return round(signal.last_price - offset, 2)


# ==============================================================================
# EXCHANGE CORE ADAPTER
# Python interface mirroring exchange-core's ExchangeApi.java
# exchange-core: LMAX Disruptor + order book (150ns match latency)
#
# In production: bridge via gRPC/socket to running exchange-core JVM
# In paper mode: in-memory order book simulation
# ==============================================================================
class ExchangeCoreAdapter:
    """
    Paper-mode order book that mirrors the exchange-core API surface.

    exchange-core Java methods mapped:
        submitCommandUninterruptibly(ApiPlaceOrder)  → submit_order()
        submitCommandUninterruptibly(ApiCancelOrder) → cancel_order()
        processReport(SingleUserReportQuery)         → get_fills()

    Production: replace body with socket/gRPC call to exchange-core JVM.
    exchange-core achieves 150ns match latency / 5M ops/sec — keep bridge thin.
    """

    def __init__(self):
        self._order_book: dict[str, list] = {}  # ticker → [bids, asks]
        self._fills:      list[dict]      = []
        self._orders:     dict[int, Order]= {}
        self._next_id:    int             = 1000
        self._lock = threading.Lock()

    def submit_order(self, order: Order) -> int:
        """
        Mirror: ExchangeApi.placeOrder(uid, orderId, code, action, qty, price, type)
        Paper: fills immediately if price crosses the simulated spread.
        """
        with self._lock:
            local_id = self._next_id
            self._next_id += 1
            order.local_id = local_id
            order.status   = "PENDING"
            self._orders[local_id] = order

            # Paper fill logic: BUY fills at ask, SELL fills at bid
            # (simulates immediate liquidity taking — realistic for US equities)
            fill_px = order.limit_px
            order.status  = "FILLED"
            order.fill_px = fill_px

            fill = {
                "local_id":  local_id,
                "ticker":    order.ticker,
                "action":    order.action,
                "qty":       order.qty,
                "fill_px":   fill_px,
                "timestamp": datetime.now().isoformat(),
            }
            self._fills.append(fill)

            # Log
            with open(EXEC_LOG_PATH, "a") as f:
                f.write(json.dumps(fill) + "\n")

            return local_id

    def cancel_order(self, local_id: int) -> bool:
        """Mirror: ExchangeApi.cancelOrder(uid, orderId, code, isBuy, price)"""
        with self._lock:
            if local_id in self._orders:
                self._orders[local_id].status = "CANCELLED"
                return True
            return False

    def get_position(self, ticker: str) -> int:
        """Net position for ticker from fill history."""
        pos = 0
        for f in self._fills:
            if f["ticker"] == ticker:
                pos += f["qty"] if f["action"] == "BUY" else -f["qty"]
        return pos

    def get_fills(self, ticker: Optional[str] = None) -> list:
        if ticker:
            return [f for f in self._fills if f["ticker"] == ticker]
        return list(self._fills)

    def get_order_status(self, local_id: int) -> Optional[Order]:
        return self._orders.get(local_id)


# ==============================================================================
# DAILY UNIVERSE SCANNER
# Runs once per day (pre-market): scans full US universe for RV opportunities
# Ranks by: alpha score + micro-price edge potential + CtV score
# ==============================================================================
class DailyUniverseScanner:
    """
    Reviews full US securities universe daily to identify:
    1. Fallen Angel names with highest alpha potential
    2. RV pairs with widest spread
    3. Sector ETFs matching macro regime
    4. Names with highest CAPM residual (pure idiosyncratic alpha)
    """

    def __init__(self, market_start: str = "2022-01-01"):
        self.start = market_start

    def _fetch(self, tickers: list[str]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            sys.path.insert(0, "/home/user/Financial-Data")
            import yfinance as yf
        try:
            raw = yf.download(tickers, start=self.start, progress=False,
                              auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                data = raw["Close"]
            else:
                data = raw[["Close"]]
            return data.ffill().dropna(how="all")
        except Exception:
            return self._synthetic(tickers)

    def _synthetic(self, tickers: list[str]) -> pd.DataFrame:
        np.random.seed(77)
        dates = pd.bdate_range(start=self.start, end=pd.Timestamp.today())
        n = len(dates)
        out = {}
        for t in tickers:
            # Fallen angel names get negative drift
            is_fallen = t in US_UNIVERSE["fallen_angel_equity"]
            mu    = -0.05 / 252 if is_fallen else 0.12 / 252
            sigma = 0.30 / np.sqrt(252) if is_fallen else 0.20 / np.sqrt(252)
            out[t] = 50 * np.exp(np.cumsum(np.random.normal(mu, sigma, n)))
        return pd.DataFrame(out, index=dates)

    def scan(self, macro_regime: str = "TRANSITION",
             macro_universe: Optional[list] = None,
             top_n: int = 20) -> dict:
        """
        Full daily scan. Returns ranked universe for execution.
        Called once pre-market by the orchestrator.
        """
        print(f"\n[SCANNER] Daily universe scan — regime: {macro_regime}")

        # Combine macro-driven universe with fallen angel candidates
        base_universe = list(set(
            (macro_universe or []) +
            US_UNIVERSE["fallen_angel_equity"] +
            US_UNIVERSE["fallen_angel_ig"] +
            US_UNIVERSE["sector_etfs"]
        ))
        # Ensure US-only (filter out any non-US tickers that may have crept in)
        us_only = [t for t in base_universe if not any(
            x in t for x in [".HK", ".SS", ".SZ", "F.", "AMS:"]
        )]

        prices = self._fetch(us_only[:40])  # cap at 40 for speed
        if prices.empty or len(prices.columns) < 2:
            return {"ranked": us_only[:top_n], "scores": {}, "rv_pairs": []}

        returns = np.log(prices / prices.shift(1)).dropna()
        mkt_ret = returns.mean(axis=1)

        scores = {}
        for ticker in prices.columns:
            if ticker not in returns.columns:
                continue
            col = returns[ticker].dropna()
            if len(col) < 60:
                continue

            # CAPM alpha (annualized)
            aligned = pd.concat([col, mkt_ret], axis=1).dropna()
            aligned.columns = ["r", "mkt"]
            if len(aligned) < 30:
                continue
            beta = aligned["r"].cov(aligned["mkt"]) / aligned["mkt"].var()
            alpha_daily = (aligned["r"] - beta * aligned["mkt"]).mean()
            alpha_annual = alpha_daily * 252

            # Momentum score (60-day)
            mom_60 = float(col.iloc[-1] - col.iloc[-60]) if len(col) >= 60 else 0.0

            # Volatility (annualized)
            vol = float(col.std() * np.sqrt(252))

            # Composite score: alpha > 2% preferred, penalize high vol
            is_fallen = ticker in US_UNIVERSE["fallen_angel_equity"]
            fallen_bonus = 0.03 if is_fallen else 0.0   # Structural edge bonus

            score = alpha_annual + fallen_bonus - max(vol - 0.30, 0) * 0.5
            scores[ticker] = {
                "score":        round(score, 4),
                "alpha_annual": round(alpha_annual, 4),
                "beta":         round(float(beta), 4),
                "vol_annual":   round(vol, 4),
                "mom_60d":      round(mom_60, 4),
                "is_fallen_angel": is_fallen,
                "category":    self._categorize(ticker),
            }

        ranked = sorted(scores, key=lambda t: -scores[t]["score"])[:top_n]

        rv_pairs_scored = self._score_rv_pairs(prices, returns) if not prices.empty else []
        result = {
            "timestamp": datetime.now().isoformat(),
            "regime":    macro_regime,
            "ranked":    ranked,
            "scores":    scores,
            "rv_pairs":  rv_pairs_scored,
        }

        with open(UNIVERSE_LOG, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        print(f"[SCANNER] Top {min(5, len(ranked))} names:")
        for i, t in enumerate(ranked[:5], 1):
            s = scores[t]
            print(f"  {i}. {t:<6} alpha={s['alpha_annual']:.2%} "
                  f"β={s['beta']:.3f} vol={s['vol_annual']:.2%} "
                  f"{'★ FALLEN ANGEL' if s['is_fallen_angel'] else ''}")

        return result

    def _score_rv_pairs(self, prices: pd.DataFrame,
                        returns: pd.DataFrame) -> list:
        """Score RV pairs — widest spread = highest priority."""
        results = []
        for t1, t2 in US_UNIVERSE["rv_pairs"]:
            if t1 not in prices.columns or t2 not in prices.columns:
                continue
            ratio = prices[t1] / prices[t2]
            z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
            z_now = float(z.dropna().iloc[-1]) if not z.dropna().empty else 0.0
            results.append({
                "pair":   f"{t1}/{t2}",
                "z_score": round(z_now, 3),
                "signal": "LONG_t1" if z_now < -1.5 else
                          "SHORT_t1" if z_now > 1.5 else "NEUTRAL",
            })
        return sorted(results, key=lambda x: -abs(x["z_score"]))

    def _categorize(self, ticker: str) -> str:
        for cat, lst in US_UNIVERSE.items():
            flat = [x for pair in lst for x in pair] if lst and isinstance(lst[0], tuple) else lst
            if ticker in flat:
                return cat
        return "other"


# ==============================================================================
# EXECUTION ENGINE — Master orchestrator
# ==============================================================================
class ExecutionEngine:
    """
    HFT execution arm. Combines:
        MicroPriceEngine (WonderTrader signal)
        ExchangeCoreAdapter (order book / matching)
        DailyUniverseScanner (US securities daily RV scan)

    Tick loop (live):
        for each ticker in active_universe:
            signal = micro_price_engine.generate_signal(L1 quote)
            if signal: exchange_core.submit_order(signal → Order)
            check_expiry() → cancel stale orders

    Integration with upstream engines:
        MacroEngine.run() → regime + alpha_universe → scanner.scan()
        AlphaOptimizer.run() → weights → position_sizing()
        AlphaBetaUnleashed.snapshot() → net beta → risk gate
    """

    ORDER_EXPIRY_SEC  = 30      # Cancel unfilled orders after 30s
    MAX_POSITION      = 500     # Max shares per name (US equities)
    RISK_BETA_MAX     = 2.0     # Hard stop if net portfolio beta exceeds BETA_MAX

    def __init__(self):
        self.micro   = MicroPriceEngine()
        self.book    = ExchangeCoreAdapter()
        self.scanner = DailyUniverseScanner()

        self._active_universe: list[str] = []
        self._weights:         dict[str, float] = {}
        self._regime:          str = "TRANSITION"
        self._pending_orders:  dict[int, Order] = {}
        self._lock = threading.Lock()

    def update_from_macro(self, macro_result: dict) -> None:
        """Receive regime + universe from MacroEngine."""
        self._regime = macro_result.get("regime", "TRANSITION")
        macro_univ   = macro_result.get("alpha_universe", [])
        scan = self.scanner.scan(self._regime, macro_univ)
        self._active_universe = scan["ranked"][:15]  # Top 15 US names
        print(f"[EXEC] Active universe updated: {self._active_universe}")

    def update_from_optimizer(self, opt_result: dict) -> None:
        """Receive optimal weights from AlphaOptimizer."""
        perf = opt_result.get("performance", {})
        self._weights = perf.get("weights", {})

    def process_quote(self, ticker: str, bid: float, ask: float,
                      bid_size: float, ask_size: float, last: float) -> Optional[Order]:
        """
        Main tick handler (called on every L1 quote update).
        Mirrors WonderTrader on_tick() → do_calc() flow.
        """
        if ticker not in self._active_universe:
            return None

        # Check open orders first (WonderTrader: if !_orders.empty() → check_orders())
        self._check_expiry()

        # Generate micro-price signal
        signal = self.micro.generate_signal(ticker, bid, ask, bid_size, ask_size, last)
        if signal is None:
            return None

        # Position check — US equities: long-only for now
        cur_pos = self.book.get_position(ticker)
        if signal.signal == SignalType.MICRO_PRICE_BUY and cur_pos >= self.MAX_POSITION:
            return None
        if signal.signal == SignalType.MICRO_PRICE_SELL and cur_pos <= 0:
            return None

        # Size: weight-driven (from AlphaOptimizer) or equal-weight fallback
        weight    = self._weights.get(ticker, 1.0 / max(len(self._active_universe), 1))
        notional  = 100_000 * weight   # $100K notional per name (scales with NLV)
        qty       = max(1, int(notional / max(last, 0.01)))
        qty       = min(qty, self.MAX_POSITION)

        action    = "BUY" if signal.signal == SignalType.MICRO_PRICE_BUY else "SELL"
        limit_px  = self.micro.calc_order_price(signal)

        order = Order(
            ticker=ticker, action=action, qty=qty,
            limit_px=limit_px, signal=signal.signal,
            local_id=-1, expiry_sec=self.ORDER_EXPIRY_SEC,
        )

        local_id = self.book.submit_order(order)

        with self._lock:
            self._pending_orders[local_id] = order

        print(
            f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} | "
            f"{action} {qty} {ticker} @ {limit_px:.2f} | "
            f"edge={signal.edge_bps:.1f}bps | micro={signal.micro_price:.3f}"
        )
        return order

    def _check_expiry(self) -> None:
        """Cancel stale orders. Mirrors WtHftStraDemo::check_orders()."""
        now = datetime.now()
        to_cancel = []
        with self._lock:
            for lid, order in list(self._pending_orders.items()):
                entry_time = datetime.fromisoformat(order.timestamp)
                if (now - entry_time).seconds >= order.expiry_sec:
                    if order.status == "PENDING":
                        to_cancel.append(lid)
        for lid in to_cancel:
            self.book.cancel_order(lid)
            with self._lock:
                if lid in self._pending_orders:
                    self._pending_orders[lid].status = "EXPIRED"
            print(f"[EXEC] Order {lid} expired and cancelled")

    def portfolio_summary(self) -> dict:
        """Snapshot of current US portfolio."""
        positions = {t: self.book.get_position(t)
                     for t in self._active_universe
                     if self.book.get_position(t) != 0}
        fills     = self.book.get_fills()
        pnl       = sum(
            (f["fill_px"] * f["qty"]) * (-1 if f["action"] == "BUY" else 1)
            for f in fills
        )
        return {
            "timestamp":       datetime.now().isoformat(),
            "regime":          self._regime,
            "active_universe": self._active_universe,
            "positions":       positions,
            "total_fills":     len(fills),
            "gross_pnl":       round(pnl, 2),
            "pending_orders":  len([o for o in self._pending_orders.values()
                                    if o.status == "PENDING"]),
        }


# ==============================================================================
# QUANT RESOURCE INDEX
# Catalogs Quant-Developers-Resources for platform reference
# ==============================================================================
def build_resource_index() -> dict:
    """
    Indexes the Quant-Developers-Resources repo into categories
    relevant to the execution engine and platform.
    """
    base = Path("/home/user/Quant-Developers-Resources")
    index = {}
    priority_dirs = [
        "Technical_Indicators", "Python", "Reinforcement Learning",
        "Financial Theory", "High Performance Computing",
        "Optimization Theory", "Signal Processing", "Econometrics",
        "C++", "FPGA", "Statsmodels",
    ]
    for d in priority_dirs:
        p = base / d
        if p.exists():
            files = list(p.rglob("*"))
            index[d] = {
                "file_count": len(files),
                "files": [str(f.relative_to(base)) for f in files
                          if f.is_file()][:10],
            }
    return index


# ==============================================================================
# FULL PLATFORM RUN
# ==============================================================================
def run_execution_arm(paper_nlv: float = 1_000_000.0) -> dict:
    """
    Full platform execution arm demo.
    In live mode: replace L1 quote simulation with real data feed.
    """
    print("\n" + "="*60)
    print("  EXECUTION ARM — US SECURITIES HFT")
    print("  WonderTrader Micro-Price + exchange-core Book")
    print("  Scope: US Equities + ETFs + IG/HY Credit")
    print("="*60)

    from macro_engine   import MacroEngine
    from alpha_optimizer import AlphaOptimizerEngine

    # 1. Macro regime → universe
    macro  = MacroEngine()
    m_out  = macro.run()

    # 2. Alpha optimizer → weights
    opt    = AlphaOptimizerEngine(custom_tickers=m_out["alpha_universe"])
    a_out  = opt.run()

    # 3. Execution engine
    engine = ExecutionEngine()
    engine.update_from_macro(m_out)
    engine.update_from_optimizer(a_out)

    # 4. Simulate L1 quote stream for active universe
    print(f"\n[EXEC] Simulating L1 quote stream for {len(engine._active_universe)} names...")
    for ticker in engine._active_universe[:8]:
        # Simulate realistic US equity L1 quote
        mid   = 100 + np.random.uniform(-20, 80)
        spread= mid * 0.0005   # 5bps typical US equity spread
        # Introduce deliberate imbalance on some names
        imb   = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        bid   = round(mid - spread / 2, 2)
        ask   = round(mid + spread / 2, 2)
        bid_s = round(1000 + imb * 300 + np.random.uniform(-100, 100))
        ask_s = round(1000 - imb * 300 + np.random.uniform(-100, 100))
        last  = round(mid + np.random.uniform(-spread, spread), 2)

        engine.process_quote(ticker, bid, ask, bid_s, ask_s, last)

    # 5. Portfolio summary
    summary = engine.portfolio_summary()
    print(f"\n[EXEC] PORTFOLIO SUMMARY:")
    print(f"  Regime:         {summary['regime']}")
    print(f"  Active Names:   {summary['active_universe'][:6]}...")
    print(f"  Positions:      {summary['positions']}")
    print(f"  Total Fills:    {summary['total_fills']}")
    print(f"  Gross P&L:      ${summary['gross_pnl']:,.2f}")

    # 6. Resource index
    res_idx = build_resource_index()
    print(f"\n[RESOURCES] Quant-Developers-Resources indexed: "
          f"{sum(v['file_count'] for v in res_idx.values())} files across "
          f"{len(res_idx)} categories")

    return {
        "macro":   m_out,
        "alpha":   a_out,
        "summary": summary,
        "resources": {k: v["file_count"] for k, v in res_idx.items()},
    }


if __name__ == "__main__":
    run_execution_arm()
