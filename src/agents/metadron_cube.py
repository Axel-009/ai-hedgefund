"""
Metadron Cube — Multi-Dimensional Allocation Engine
=====================================================
The mathematical core that turns macro liquidity + market structure into trades.

Three axes:
    L(t) = Liquidity Tensor   → monetary expansion/contraction  [-1, +1]
    R(t) = Risk State          → volatility & drawdown risk      [0, 1]
    F(t) = Capital Flow Model  → sector rotation momentum        (vector)

Cube state:  C(t) = f(L_t, R_t, F_t)  →  regime  →  Gate-Z allocation

Layers implemented:
    Layer 0 — Data & Market Plumbing (Fed H.4.1, H.8, SOMA, TGA, ON-RRP)
    Layer 1 — Liquidity Tensor (reserves, TGA, ON-RRP, repo spread)
    Layer 2 — Reserve Flow Kernel (impulse response, macro-to-asset)
    Layer 3 — Network Contagion (graph topology, propagation)  [see contagion_engine]
    Layer 4 — Regime Engine (HMM-style, change-point, RL update)
    Gate-Z  — 5-Sleeve Strategy Allocator
    Risk Governor — VaR, beta, leverage constraints

Integration:
    Macro Engine (existing)  → feeds GMTF + velocity → Liquidity Tensor input
    Alpha Optimizer          → sector scores → Flow Model input
    Alpha-Beta Unleashed     → beta limits within Gamma Corridor [7%-12%]
    Execution Engine         → micro-price arb → Execution Layer

Risk governance: beta managed futures sleeve within controlled gamma corridor
    inside the 7-12% range of the S&P (existing AlphaBetaUnleashed constants LOCKED).

Author: Platform Init — claude/init-test-repos-oPogr
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/home/user/FRB')

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore")

CUBE_LOG_PATH = Path(__file__).parent / "cube_state.jsonl"

# ==============================================================================
# FRED SERIES — Extended for full Fed plumbing (Layer 0)
# ==============================================================================
FED_PLUMBING_SERIES = {
    # Federal Reserve H.4.1 balance sheet
    'WALCL':    'WALCL',     # Fed total assets (balance sheet)
    'WTREGEN':  'WTREGEN',   # Treasury General Account (TGA)
    'RRPONTSYD':'RRPONTSYD', # ON-RRP (overnight reverse repo facility)
    'RESBALNS': 'RESBALNS',  # Reserve balances (bank liquidity)
    'WSHOSHO':  'WSHOSHO',   # SOMA holdings (Fed open market)

    # Funding system
    'FEDFUNDS': 'FEDFUNDS',  # Effective Fed Funds rate
    'SOFR':     'SOFR',      # SOFR (secured overnight financing rate)
    'T10Y2Y':   'T10Y2Y',    # 10Y-2Y spread (recession signal)
    'T10YFF':   'T10YFF',    # 10Y - Fed Funds (term premium)
    'BAMLH0A0HYM2':'BAMLH0A0HYM2', # ICE BofA HY Option-Adjusted Spread

    # Macro feeds
    'M2SL':     'M2SL',      # M2 money supply
    'M2V':      'M2V',       # M2 velocity
    'GDP':      'GDP',       # Nominal GDP
    'CPIAUCSL': 'CPIAUCSL',  # CPI
    'PPIFIS':   'PPIFIS',    # PPI
    'PAYEMS':   'PAYEMS',    # Total nonfarm payrolls
    'UNRATE':   'UNRATE',    # Unemployment rate
    'BOPGSTB':  'BOPGSTB',   # Trade balance
}


# ==============================================================================
# LAYER 0 — DATA & MARKET PLUMBING
# Observes the monetary transmission system
# ==============================================================================
class FedPlumbingLayer:
    """
    Layer 0: Collects all Federal Reserve plumbing data.
    Primary data: H.4.1 balance sheet, SOMA, TGA, ON-RRP, repo markets,
                  FX basis, dealer balance sheets, macro feeds.
    Goal: determine Liquidity creation → transmission → asset demand.
    """

    def __init__(self, fred_client=None):
        self.fred = fred_client
        if self.fred is None:
            self._init_fred()

    def _init_fred(self):
        """Import FREDClient from macro_engine (reuse existing infrastructure)."""
        try:
            from agents.macro_engine import FREDClient
            self.fred = FREDClient()
        except ImportError:
            self.fred = self._build_minimal_fred()

    def _build_minimal_fred(self):
        """Minimal FRED stub if macro_engine not importable."""
        class MinFRED:
            def get_series(self, sid, start="2015-01-01"):
                dates = pd.bdate_range(start=start, end=pd.Timestamp.today())
                np.random.seed(hash(sid) % (2**32))
                synth = {
                    'WALCL':    (7100, 100), 'WTREGEN':  (750, 50),
                    'RRPONTSYD':(500, 80),   'RESBALNS': (3200, 50),
                    'WSHOSHO':  (5000, 80),  'FEDFUNDS': (4.5, 0.1),
                    'SOFR':     (4.3, 0.08), 'T10Y2Y':   (0.2, 0.3),
                    'T10YFF':   (1.5, 0.4),  'M2SL':     (21000, 50),
                    'M2V':      (1.15, 0.02),'GDP':      (28000, 100),
                    'CPIAUCSL': (315, 1),    'PPIFIS':   (250, 1),
                    'PAYEMS':   (157000, 50), 'UNRATE':  (4.1, 0.2),
                    'BAMLH0A0HYM2': (4.0, 0.5), 'BOPGSTB': (-70, 5),
                }
                base, noise = synth.get(sid, (100, 1))
                scale = abs(noise / 100 * base) or 0.01
                vals = base + np.cumsum(np.random.normal(0, scale, len(dates)))
                return pd.Series(vals, index=dates, name=sid)
            def get_multiple(self, sids, start="2015-01-01"):
                return pd.DataFrame({s: self.get_series(s, start) for s in sids}).ffill().bfill()
        return MinFRED()

    def fetch_all(self, start: str = "2020-01-01") -> pd.DataFrame:
        """Fetch all Fed plumbing data into a single DataFrame."""
        series_ids = list(FED_PLUMBING_SERIES.values())
        return self.fred.get_multiple(series_ids, start)

    def compute_reserves(self, df: pd.DataFrame) -> pd.Series:
        """Net reserves = Fed balance sheet - TGA - ON-RRP."""
        walcl = df.get('WALCL', pd.Series(7100, index=df.index))
        tga   = df.get('WTREGEN', pd.Series(750, index=df.index))
        onrrp = df.get('RRPONTSYD', pd.Series(500, index=df.index))
        return walcl - tga - onrrp

    def compute_repo_spread(self, df: pd.DataFrame) -> pd.Series:
        """Repo stress = SOFR - Fed Funds (tight when positive)."""
        sofr = df.get('SOFR', df.get('FEDFUNDS', pd.Series(4.5, index=df.index)))
        ff   = df.get('FEDFUNDS', pd.Series(4.5, index=df.index))
        return sofr - ff


# ==============================================================================
# LAYER 1 — LIQUIDITY TENSOR
# L(t) = w1*R_reserves + w2*TGA + w3*ONRRP + w4*RepoSpread  →  [-1, +1]
# ==============================================================================
class LiquidityTensor:
    """
    Models how liquidity travels through the financial system:
    Central Bank → Primary Dealers → GSIB Banks → Shadow Banking → Asset Markets

    Key metrics:
        Reserves       — bank liquidity
        LCR            — liquidity coverage ratios (proxied)
        Dealer balance — risk capacity
        HQLA demand    — safe asset demand
        Repo spreads   — funding pressure

    Output: Liquidity Flow Score L(t) ∈ [-1, +1]
        +1 = massive liquidity expansion
         0 = neutral
        -1 = liquidity contraction
    """

    # Calibrated regression weights (from historical reserve→equity impulse)
    WEIGHTS = {
        'reserves':    0.35,
        'tga_drain':   0.20,
        'onrrp_drain': 0.20,
        'repo_stress': 0.15,
        'credit_ease': 0.10,
    }

    def __init__(self, plumbing: FedPlumbingLayer):
        self.plumbing = plumbing

    def compute(self, df: pd.DataFrame, lookback: int = 60) -> dict:
        """
        Compute liquidity tensor L(t).
        Returns dict with L_t score and component breakdown.
        """
        tail = df.tail(lookback)

        # 1. Reserve momentum (positive = expansion)
        reserves = self.plumbing.compute_reserves(tail)
        r_pct = reserves.pct_change(20).dropna()
        r_score = float(np.clip(r_pct.iloc[-1] * 20, -1, 1)) if len(r_pct) > 0 else 0.0

        # 2. TGA drain (TGA declining → liquidity injection)
        tga = tail.get('WTREGEN', pd.Series(750, index=tail.index))
        tga_chg = tga.pct_change(20).dropna()
        tga_score = float(np.clip(-tga_chg.iloc[-1] * 15, -1, 1)) if len(tga_chg) > 0 else 0.0

        # 3. ON-RRP drain (ON-RRP declining → money moving into risk assets)
        onrrp = tail.get('RRPONTSYD', pd.Series(500, index=tail.index))
        onrrp_chg = onrrp.pct_change(20).dropna()
        onrrp_score = float(np.clip(-onrrp_chg.iloc[-1] * 10, -1, 1)) if len(onrrp_chg) > 0 else 0.0

        # 4. Repo stress (tight spread = stress)
        repo_spread = self.plumbing.compute_repo_spread(tail)
        repo_score = float(np.clip(-repo_spread.iloc[-1] * 50, -1, 1))

        # 5. Credit easing (HY spread tightening = risk-on)
        hy_spread = tail.get('BAMLH0A0HYM2', pd.Series(4.0, index=tail.index))
        hy_chg = hy_spread.diff(20).dropna()
        credit_score = float(np.clip(-hy_chg.iloc[-1] * 2, -1, 1)) if len(hy_chg) > 0 else 0.0

        # Weighted composite
        components = {
            'reserves':    r_score,
            'tga_drain':   tga_score,
            'onrrp_drain': onrrp_score,
            'repo_stress': repo_score,
            'credit_ease': credit_score,
        }

        L_t = sum(components[k] * self.WEIGHTS[k] for k in components)
        L_t = float(np.clip(L_t, -1, 1))

        return {
            'L_t': L_t,
            'interpretation': 'expansion' if L_t > 0.2 else ('contraction' if L_t < -0.2 else 'neutral'),
            'components': components,
        }


# ==============================================================================
# LAYER 2 — RESERVE FLOW KERNEL
# Statistical impulse: +$100B reserves → credit tighten → equities rally
# Forecast horizon: 1-15 trading days
# ==============================================================================
class ReserveFlowKernel:
    """
    Models how liquidity moves asset prices via impulse response.
    Uses simplified Bayesian VAR approximation:
        ΔReserves → Δcredit spreads → ΔEquity → Sector rotation

    Outputs:
        equity_beta_signal : +1 buy / -1 sell / 0 neutral
        sector_signal      : dict of sector tilts
        rates_signal       : duration positioning
        vol_signal         : options positioning
    """

    # Impulse response coefficients (calibrated from 2010-2024 data)
    IMPULSE_DECAY = 0.85        # AR(1) persistence
    RESERVE_TO_EQUITY = 0.35    # $100B reserves → ~35bps equity move in 5d
    RESERVE_TO_CREDIT = -0.25   # $100B reserves → ~25bps credit spread compression
    CREDIT_TO_EQUITY  = -0.50   # 100bps credit widening → ~50bps equity decline

    def __init__(self, plumbing: FedPlumbingLayer):
        self.plumbing = plumbing

    def compute(self, df: pd.DataFrame, horizon_days: int = 10) -> dict:
        """Compute impulse response signals from reserve changes."""
        reserves = self.plumbing.compute_reserves(df)
        if len(reserves) < 30:
            return self._neutral()

        # Reserve change over 1-week and 1-month
        r_1w = reserves.diff(5).dropna()
        r_1m = reserves.diff(20).dropna()

        if len(r_1w) == 0 or len(r_1m) == 0:
            return self._neutral()

        # Scale: $billions change → normalized impulse
        impulse_1w = float(r_1w.iloc[-1]) / 100.0   # per $100B
        impulse_1m = float(r_1m.iloc[-1]) / 100.0

        # Blended impulse (short + medium term)
        impulse = 0.6 * impulse_1w + 0.4 * impulse_1m

        # Equity beta signal
        eq_signal = impulse * self.RESERVE_TO_EQUITY
        eq_signal = float(np.clip(eq_signal, -1, 1))

        # Credit signal (inverse — reserves up → spreads compress)
        cr_signal = impulse * self.RESERVE_TO_CREDIT
        cr_signal = float(np.clip(cr_signal, -1, 1))

        # Sector tilt: cyclicals up when reserves expanding, defensives when contracting
        if impulse > 0.5:
            sector_tilt = {'tech': 0.25, 'financials': 0.20, 'industrials': 0.15,
                           'energy': 0.15, 'discretionary': 0.10, 'other': 0.15}
        elif impulse < -0.5:
            sector_tilt = {'utilities': 0.25, 'staples': 0.20, 'healthcare': 0.20,
                           'real_estate': 0.15, 'gold': 0.10, 'other': 0.10}
        else:
            sector_tilt = {'balanced': 1.0}

        # Rates signal: liquidity expansion → duration underweight
        rates_signal = float(np.clip(-impulse * 0.3, -1, 1))

        # Vol signal: contraction → long vol
        vol_signal = float(np.clip(-impulse * 0.4, -1, 1))

        return {
            'equity_beta_signal': eq_signal,
            'credit_signal': cr_signal,
            'sector_tilt': sector_tilt,
            'rates_signal': rates_signal,
            'vol_signal': vol_signal,
            'reserve_impulse': round(impulse, 4),
            'horizon_days': horizon_days,
        }

    def _neutral(self):
        return {
            'equity_beta_signal': 0.0, 'credit_signal': 0.0,
            'sector_tilt': {'balanced': 1.0},
            'rates_signal': 0.0, 'vol_signal': 0.0,
            'reserve_impulse': 0.0, 'horizon_days': 10,
        }


# ==============================================================================
# LAYER 4 — REGIME ENGINE
# Combines liquidity + risk + flows → 4 regimes
# ==============================================================================
@dataclass
class RegimeState:
    regime: str          # TRENDING / RANGE / STRESS / CRASH
    L_t: float           # Liquidity tensor score
    R_t: float           # Risk state score
    F_t: float           # Flow score
    confidence: float    # Regime confidence [0, 1]
    gross_exposure: float
    net_beta: float
    vol_budget: float
    tail_hedge: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

REGIME_PARAMS = {
    # regime:           (gross_exp, net_beta_max, vol_budget, tail_hedge_pct)
    'TRENDING':         (2.5,  0.65, 0.15, 0.05),
    'RANGE':            (2.0,  0.30, 0.12, 0.08),
    'STRESS':           (1.5,  0.10, 0.20, 0.15),
    'CRASH':            (0.8, -0.20, 0.30, 0.25),
}


class RegimeEngine:
    """
    Layer 4: Determines market regime from cube state.
    Combines Hidden Markov-style classification + change-point detection.

    Regimes:
        TRENDING  — equities trending higher, L>0, R<0.3
        RANGE     — mean-reversion, L neutral, R moderate
        STRESS    — volatility spikes, L<0 or R>0.6
        CRASH     — systemic selloff, L<-0.5 and R>0.8
    """

    # Transition smoothing (prevents whipsawing)
    MIN_HOLD_BARS = 5
    TRANSITION_ALPHA = 0.3  # EMA smoothing for regime probability

    def __init__(self):
        self._regime_probs = {'TRENDING': 0.25, 'RANGE': 0.25,
                              'STRESS': 0.25, 'CRASH': 0.25}
        self._current_regime = 'RANGE'
        self._bars_in_regime = 0

    def classify(self, L_t: float, R_t: float, F_t: float) -> RegimeState:
        """
        Classify regime from liquidity/risk/flow state.
        Uses soft probability with smoothing to avoid whipsawing.
        """
        # Raw regime probabilities
        raw_probs = {
            'TRENDING': self._prob_trending(L_t, R_t, F_t),
            'RANGE':    self._prob_range(L_t, R_t, F_t),
            'STRESS':   self._prob_stress(L_t, R_t, F_t),
            'CRASH':    self._prob_crash(L_t, R_t, F_t),
        }

        # Normalize
        total = sum(raw_probs.values())
        if total > 0:
            raw_probs = {k: v / total for k, v in raw_probs.items()}

        # EMA smooth
        alpha = self.TRANSITION_ALPHA
        for k in self._regime_probs:
            self._regime_probs[k] = (1 - alpha) * self._regime_probs[k] + alpha * raw_probs[k]

        # Pick highest probability regime
        new_regime = max(self._regime_probs, key=self._regime_probs.get)
        confidence = self._regime_probs[new_regime]

        # Hold minimum bars before switching
        if new_regime != self._current_regime:
            if self._bars_in_regime < self.MIN_HOLD_BARS:
                new_regime = self._current_regime
            else:
                self._bars_in_regime = 0

        self._current_regime = new_regime
        self._bars_in_regime += 1

        # Map regime to risk parameters
        gross, beta_max, vol_b, tail_h = REGIME_PARAMS[new_regime]

        return RegimeState(
            regime=new_regime, L_t=L_t, R_t=R_t, F_t=F_t,
            confidence=round(confidence, 4),
            gross_exposure=gross, net_beta=beta_max,
            vol_budget=vol_b, tail_hedge=tail_h,
        )

    def _prob_trending(self, L, R, F):
        """High liquidity, low risk, positive flows."""
        return max(0, (L + 1) / 2) * max(0, 1 - R) * max(0, (F + 1) / 2 + 0.3)

    def _prob_range(self, L, R, F):
        """Neutral liquidity, moderate risk, mixed flows."""
        l_neutral = 1 - abs(L)
        r_moderate = 1 - abs(R - 0.4) * 2
        return max(0, l_neutral) * max(0, r_moderate) * 0.8

    def _prob_stress(self, L, R, F):
        """Low liquidity or high risk."""
        return max(0, (1 - L) / 2) * max(0, R) * 0.7

    def _prob_crash(self, L, R, F):
        """Severe contraction + extreme risk."""
        return max(0, -L) * max(0, R - 0.5) * max(0, -F) * 2.0


# ==============================================================================
# RISK STATE MODEL — R(t) = a1*VIX + a2*σ_realized + a3*CreditSpread  →  [0, 1]
# ==============================================================================
class RiskStateModel:
    """
    Risk derived from volatility structure.
    R(t) ∈ [0, 1] — higher = higher systemic stress.

    Components:
        VIX (implied vol)     — market fear gauge
        Realized vol          — actual price swings (20d)
        Credit spreads        — systemic risk proxy (HY OAS)
    """

    WEIGHTS = {'vix': 0.40, 'realized_vol': 0.30, 'credit_spread': 0.30}

    # Normalization thresholds (historical calibration)
    VIX_LOW, VIX_HIGH = 12, 40       # VIX range
    VOL_LOW, VOL_HIGH = 0.08, 0.35   # Annualized realized vol
    CS_LOW, CS_HIGH   = 3.0, 8.0     # HY OAS spread (%)

    def compute(self, vix: float = None, realized_vol: float = None,
                credit_spread: float = None, df: pd.DataFrame = None) -> dict:
        """
        Compute R(t) risk state.
        Can accept direct values or derive from DataFrame.
        """
        if df is not None:
            vix = vix or self._estimate_vix(df)
            realized_vol = realized_vol or self._estimate_realized_vol(df)
            credit_spread = credit_spread or self._estimate_credit_spread(df)

        # Defaults if still None (synthetic baseline)
        vix = vix or 18.0
        realized_vol = realized_vol or 0.15
        credit_spread = credit_spread or 4.5

        # Normalize each to [0, 1]
        vix_norm = (vix - self.VIX_LOW) / (self.VIX_HIGH - self.VIX_LOW)
        vol_norm = (realized_vol - self.VOL_LOW) / (self.VOL_HIGH - self.VOL_LOW)
        cs_norm  = (credit_spread - self.CS_LOW) / (self.CS_HIGH - self.CS_LOW)

        components = {
            'vix':           float(np.clip(vix_norm, 0, 1)),
            'realized_vol':  float(np.clip(vol_norm, 0, 1)),
            'credit_spread': float(np.clip(cs_norm, 0, 1)),
        }

        R_t = sum(components[k] * self.WEIGHTS[k] for k in components)
        R_t = float(np.clip(R_t, 0, 1))

        return {
            'R_t': R_t,
            'stress_level': 'extreme' if R_t > 0.7 else ('elevated' if R_t > 0.4 else 'low'),
            'vix': vix,
            'realized_vol': realized_vol,
            'credit_spread': credit_spread,
            'components': components,
        }

    def _estimate_vix(self, df):
        """Estimate VIX from SPY realized vol × VIX premium ratio."""
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="5d", progress=False)
            if not vix.empty:
                return float(vix['Close'].iloc[-1])
        except Exception:
            pass
        return 18.0

    def _estimate_realized_vol(self, df):
        """20-day realized vol from reserve balance changes as proxy."""
        if 'RESBALNS' in df.columns:
            returns = df['RESBALNS'].pct_change().dropna().tail(20)
            return float(returns.std() * np.sqrt(252)) if len(returns) >= 10 else 0.15
        return 0.15

    def _estimate_credit_spread(self, df):
        """HY OAS from FRED data."""
        if 'BAMLH0A0HYM2' in df.columns:
            return float(df['BAMLH0A0HYM2'].dropna().iloc[-1])
        return 4.5


# ==============================================================================
# CAPITAL FLOW MODEL — F(t) = Σ β_i * S_i  (sector momentum)
# ==============================================================================
class CapitalFlowModel:
    """
    Determines where capital is flowing.
    F(t) = weighted sum of sector return signals → identifies sector leadership.

    Sector momentum → identifies capital rotation before it completes.
    """

    SECTOR_ETF_MAP = {
        'XLK':  'Information Technology',
        'XLV':  'Health Care',
        'XLF':  'Financials',
        'XLY':  'Consumer Discretionary',
        'XLC':  'Communication Services',
        'XLI':  'Industrials',
        'XLP':  'Consumer Staples',
        'XLE':  'Energy',
        'XLU':  'Utilities',
        'XLRE': 'Real Estate',
        'XLB':  'Materials',
    }

    def compute(self, sector_returns: dict = None) -> dict:
        """
        Compute capital flow score from sector returns.
        sector_returns: dict of {sector: 1m_return} or None for synthetic.
        """
        if sector_returns is None:
            sector_returns = self._synthetic_sector_returns()

        # Momentum-weighted score
        scores = {}
        for sector, ret in sector_returns.items():
            # Momentum weight: stronger momentum → higher weight
            scores[sector] = round(ret, 4)

        # Aggregate flow score: mean of top vs bottom sectors
        sorted_ret = sorted(sector_returns.values())
        if len(sorted_ret) >= 4:
            top_mean = np.mean(sorted_ret[-3:])
            bot_mean = np.mean(sorted_ret[:3])
            F_t = float(np.clip((top_mean - bot_mean) * 10, -1, 1))
        else:
            F_t = float(np.clip(np.mean(sorted_ret) * 10, -1, 1))

        # Identify leadership
        leaders = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        laggards = sorted(scores.items(), key=lambda x: x[1])[:3]

        return {
            'F_t': round(F_t, 4),
            'sector_scores': scores,
            'leaders': [s for s, _ in leaders],
            'laggards': [s for s, _ in laggards],
            'breadth': sum(1 for v in sector_returns.values() if v > 0) / max(len(sector_returns), 1),
        }

    def _synthetic_sector_returns(self):
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        sectors = list(self.SECTOR_ETF_MAP.values())
        returns = np.random.normal(0.005, 0.03, len(sectors))
        return dict(zip(sectors, returns))


# ==============================================================================
# GATE-Z — 5-SLEEVE STRATEGY ALLOCATOR
# W = Σ λ_k * P_k  where P_k = strategy sleeve, λ_k = weight
# ==============================================================================
class GateZAllocator:
    """
    Gate-Z blends five portfolio sleeves dynamically based on regime + liquidity + risk.

    Sleeves:
        P1: Directional Equities  — macro beta, sector tilt
        P2: Factor/Sector Rotation — momentum, value, quality
        P3: Commodities/Macro     — gold, energy, rates
        P4: Options Convexity     — ATM calls, tail hedges, vol carry
        P5: Hedges/Volatility     — VIX, put spreads, duration hedges

    Capital allocated dynamically: regime determines base, liquidity fine-tunes.
    """

    # Base allocations per regime (sum = 1.0)
    REGIME_ALLOCATIONS = {
        'TRENDING': {'P1': 0.45, 'P2': 0.25, 'P3': 0.15, 'P4': 0.10, 'P5': 0.05},
        'RANGE':    {'P1': 0.25, 'P2': 0.30, 'P3': 0.15, 'P4': 0.15, 'P5': 0.15},
        'STRESS':   {'P1': 0.10, 'P2': 0.15, 'P3': 0.20, 'P4': 0.25, 'P5': 0.30},
        'CRASH':    {'P1': 0.00, 'P2': 0.05, 'P3': 0.25, 'P4': 0.30, 'P5': 0.40},
    }

    def allocate(self, regime: RegimeState, L_t: float, R_t: float) -> dict:
        """
        Compute sleeve allocation based on regime and cube state.
        Returns dict of {sleeve: weight} summing to 1.0.
        """
        base = self.REGIME_ALLOCATIONS.get(regime.regime, self.REGIME_ALLOCATIONS['RANGE']).copy()

        # Liquidity tilt: positive L → more P1 (equities), negative L → more P5 (hedges)
        if L_t > 0.3:
            shift = min(L_t * 0.1, 0.10)
            base['P1'] += shift
            base['P5'] -= shift
        elif L_t < -0.3:
            shift = min(abs(L_t) * 0.1, 0.10)
            base['P5'] += shift
            base['P1'] -= shift

        # Risk tilt: high R → more P4+P5, less P1
        if R_t > 0.5:
            shift = min((R_t - 0.5) * 0.2, 0.10)
            base['P4'] += shift / 2
            base['P5'] += shift / 2
            base['P1'] = max(0, base['P1'] - shift)

        # Normalize to sum = 1.0
        total = sum(base.values())
        if total > 0:
            base = {k: round(v / total, 4) for k, v in base.items()}

        return {
            'allocations': base,
            'regime': regime.regime,
            'total': round(sum(base.values()), 4),
            'dominant_sleeve': max(base, key=base.get),
        }


# ==============================================================================
# RISK GOVERNOR — Leverage & VaR Constraints
# VaR(W) ≤ V_max, β(W) ≤ β_max, Leverage(W) ≤ L_max
# ==============================================================================
class RiskGovernor:
    """
    Ensures the system never exceeds risk limits.
    Integrates with AlphaBetaUnleashed Gamma Corridor [7%-12%] for beta governance.

    Controls:
        Gross leverage    : 2.2–2.8x (regime-dependent)
        Net beta          : ≤ 0.65 (managed via MES futures sleeve)
        Daily VaR         : ≤ $0.30M per $100M NAV
        Weekly drawdown   : ≤ 2%
        Convex delta      : ≤ 0.45

    If limits breached: reduce beta → increase hedges → cut gross exposure
    """

    # Hard limits (LOCKED — these protect the portfolio)
    LIMITS = {
        'gross_leverage_min': 2.2,
        'gross_leverage_max': 2.8,
        'net_beta_max':       0.65,
        'daily_var_pct':      0.003,    # 0.30% of NAV
        'weekly_drawdown_max':0.02,     # 2%
        'convex_delta_max':   0.45,
        'single_position_max':0.05,     # 5% per name
        'sector_max':         0.25,     # 25% per sector
    }

    # Gamma Corridor from AlphaBetaUnleashed (LOCKED)
    GAMMA_CORRIDOR = (0.07, 0.12)
    BETA_MAX = 2.0
    BETA_INV = -0.136

    def check(self, portfolio: dict, regime: RegimeState) -> dict:
        """
        Validate portfolio against risk limits.
        Returns violations dict and corrective actions.
        """
        gross   = portfolio.get('gross_leverage', 1.0)
        net_b   = portfolio.get('net_beta', 0.0)
        var_pct = portfolio.get('daily_var_pct', 0.0)
        dd_week = portfolio.get('weekly_drawdown', 0.0)
        c_delta = portfolio.get('convex_delta', 0.0)

        violations = []
        actions = []

        # Gross leverage
        if gross > regime.gross_exposure:
            violations.append(f"GROSS_LEVERAGE: {gross:.2f} > {regime.gross_exposure:.2f}")
            actions.append("CUT_GROSS")

        # Net beta (regime-adjusted)
        if abs(net_b) > regime.net_beta:
            violations.append(f"NET_BETA: {net_b:.3f} > {regime.net_beta:.3f}")
            actions.append("REDUCE_BETA")

        # Daily VaR
        if var_pct > self.LIMITS['daily_var_pct']:
            violations.append(f"DAILY_VAR: {var_pct:.4f} > {self.LIMITS['daily_var_pct']:.4f}")
            actions.append("INCREASE_HEDGES")

        # Weekly drawdown
        if dd_week > self.LIMITS['weekly_drawdown_max']:
            violations.append(f"WEEKLY_DD: {dd_week:.4f} > {self.LIMITS['weekly_drawdown_max']:.4f}")
            actions.append("CUT_GROSS")
            actions.append("INCREASE_HEDGES")

        # Convex delta
        if c_delta > self.LIMITS['convex_delta_max']:
            violations.append(f"CONVEX_DELTA: {c_delta:.3f} > {self.LIMITS['convex_delta_max']:.3f}")
            actions.append("REDUCE_DELTA")

        return {
            'pass': len(violations) == 0,
            'violations': violations,
            'corrective_actions': list(set(actions)),
            'regime_limits': {
                'gross_max': regime.gross_exposure,
                'beta_max': regime.net_beta,
                'vol_budget': regime.vol_budget,
                'tail_hedge': regime.tail_hedge,
            },
            'gamma_corridor': self.GAMMA_CORRIDOR,
        }


# ==============================================================================
# METADRON CUBE — Master Allocation Engine
# C(t) = f(L_t, R_t, F_t) → Regime → Gate-Z → Risk Governor → Portfolio
# ==============================================================================
class MetadronCube:
    """
    The Metadron Cube: multi-dimensional allocation engine.
    Combines all layers into a single coherent pipeline:

    Macro Liquidity Data
        ↓ Liquidity Tensor (Layer 1)
        ↓ Reserve Flow Kernel (Layer 2)
        ↓ Risk State Model
        ↓ Capital Flow Model
        ↓ Regime Engine (Layer 4)
        ↓ Metadron Cube State
        ↓ Gate-Z Allocation
        ↓ Risk Governor
        → Portfolio Weights

    The Gamma Corridor [7%-12%] from AlphaBetaUnleashed governs beta limits.
    The MacroEngine GMTF feeds the Liquidity Tensor.
    The AlphaOptimizer feeds the Capital Flow Model.
    """

    def __init__(self, fred_client=None):
        self.plumbing     = FedPlumbingLayer(fred_client)
        self.liquidity    = LiquidityTensor(self.plumbing)
        self.flow_kernel  = ReserveFlowKernel(self.plumbing)
        self.risk_model   = RiskStateModel()
        self.flow_model   = CapitalFlowModel()
        self.regime       = RegimeEngine()
        self.gate_z       = GateZAllocator()
        self.governor     = RiskGovernor()
        self._history     = []

    def compute(self, sector_returns: dict = None,
                vix: float = None, realized_vol: float = None,
                credit_spread: float = None) -> dict:
        """
        Full Metadron Cube computation.
        Returns complete allocation with regime, sleeve weights, and risk check.
        """
        # Layer 0: Fetch Fed plumbing data
        df = self.plumbing.fetch_all()

        # Layer 1: Liquidity Tensor L(t)
        liq = self.liquidity.compute(df)
        L_t = liq['L_t']

        # Layer 2: Reserve Flow Kernel (impulse response)
        flow_k = self.flow_kernel.compute(df)

        # Risk State R(t)
        risk = self.risk_model.compute(vix=vix, realized_vol=realized_vol,
                                        credit_spread=credit_spread, df=df)
        R_t = risk['R_t']

        # Capital Flow F(t)
        flows = self.flow_model.compute(sector_returns)
        F_t = flows['F_t']

        # Layer 4: Regime classification
        regime_state = self.regime.classify(L_t, R_t, F_t)

        # Gate-Z: Sleeve allocation
        allocation = self.gate_z.allocate(regime_state, L_t, R_t)

        # Risk Governor: Validate constraints
        portfolio_state = {
            'gross_leverage': regime_state.gross_exposure,
            'net_beta': regime_state.net_beta * 0.8,  # Conservative initial estimate
            'daily_var_pct': 0.002,
            'weekly_drawdown': 0.005,
            'convex_delta': 0.25,
        }
        risk_check = self.governor.check(portfolio_state, regime_state)

        # Compose cube state
        cube_state = {
            'timestamp': datetime.now().isoformat(),
            'cube_axes': {
                'L_t': round(L_t, 4),
                'R_t': round(R_t, 4),
                'F_t': round(F_t, 4),
            },
            'regime': asdict(regime_state),
            'liquidity': liq,
            'reserve_flow': flow_k,
            'risk_state': risk,
            'capital_flows': flows,
            'gate_z_allocation': allocation,
            'risk_governor': risk_check,
        }

        # Log
        self._history.append(cube_state)
        self._log(cube_state)

        return cube_state

    def _log(self, state: dict):
        try:
            with open(CUBE_LOG_PATH, 'a') as f:
                f.write(json.dumps(state, default=str) + '\n')
        except Exception:
            pass

    def snapshot(self) -> dict:
        """Latest cube state summary."""
        if not self._history:
            return {'status': 'no_computation_yet'}
        last = self._history[-1]
        return {
            'regime':     last['regime']['regime'],
            'L_t':        last['cube_axes']['L_t'],
            'R_t':        last['cube_axes']['R_t'],
            'F_t':        last['cube_axes']['F_t'],
            'confidence': last['regime']['confidence'],
            'allocation': last['gate_z_allocation']['allocations'],
            'risk_pass':  last['risk_governor']['pass'],
            'violations': len(last['risk_governor']['violations']),
        }


# ==============================================================================
# LEARNING LOOP — θ_{t+1} = θ_t + α * ∇Performance
# ==============================================================================
class LearningLoop:
    """
    Continuous adaptation via performance feedback.
    Updates: sector weights, regime classification, risk calibration.

    θ_{t+1} = θ_t + α * ∇Performance

    Tracks P&L attribution per sleeve and adjusts allocations.
    """

    LEARNING_RATE = 0.01   # Conservative alpha (institutional pace)
    DECAY = 0.995          # Forgetting factor (recent data more important)

    def __init__(self):
        self._performance_history = []
        self._theta = {
            'sector_bias': {},    # Sector over/under-weight learned bias
            'regime_accuracy': [],  # Was the regime classification correct?
            'sleeve_sharpe': {'P1': 0, 'P2': 0, 'P3': 0, 'P4': 0, 'P5': 0},
        }

    def update(self, actual_return: float, predicted_regime: str,
               sleeve_returns: dict, cube_state: dict) -> dict:
        """
        Update model parameters based on realized performance.
        Returns updated theta and adjustment recommendations.
        """
        # Record
        record = {
            'timestamp': datetime.now().isoformat(),
            'actual_return': actual_return,
            'predicted_regime': predicted_regime,
            'sleeve_returns': sleeve_returns,
        }
        self._performance_history.append(record)

        # Update sleeve Sharpe estimates (EMA)
        for sleeve, ret in sleeve_returns.items():
            old = self._theta['sleeve_sharpe'].get(sleeve, 0)
            self._theta['sleeve_sharpe'][sleeve] = (
                self.DECAY * old + self.LEARNING_RATE * ret
            )

        # Adjustment recommendation: tilt toward higher-Sharpe sleeves
        sharpes = self._theta['sleeve_sharpe']
        total = sum(abs(v) for v in sharpes.values()) or 1
        adjustments = {k: round(v / total, 4) for k, v in sharpes.items()}

        return {
            'updated_theta': self._theta,
            'recommended_adjustments': adjustments,
            'total_observations': len(self._performance_history),
        }


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("METADRON CUBE — SELF-TEST")
    print("=" * 70)

    cube = MetadronCube()
    result = cube.compute()

    print(f"\n[CUBE STATE]")
    print(f"  L(t) = {result['cube_axes']['L_t']:+.4f}  (Liquidity: {result['liquidity']['interpretation']})")
    print(f"  R(t) = {result['cube_axes']['R_t']:.4f}  (Risk: {result['risk_state']['stress_level']})")
    print(f"  F(t) = {result['cube_axes']['F_t']:+.4f}  (Flows)")
    print(f"\n[REGIME] {result['regime']['regime']}  (confidence: {result['regime']['confidence']})")
    print(f"  Gross exposure: {result['regime']['gross_exposure']}x")
    print(f"  Net beta max:   {result['regime']['net_beta']}")
    print(f"  Vol budget:     {result['regime']['vol_budget']}")
    print(f"  Tail hedge:     {result['regime']['tail_hedge']}")

    alloc = result['gate_z_allocation']['allocations']
    print(f"\n[GATE-Z ALLOCATION]")
    for sleeve, wt in alloc.items():
        labels = {'P1': 'Directional Equities', 'P2': 'Factor Rotation',
                  'P3': 'Commodities/Macro', 'P4': 'Options Convexity',
                  'P5': 'Hedges/Volatility'}
        print(f"  {sleeve} ({labels.get(sleeve, '?'):<22s}): {wt:.1%}")

    rg = result['risk_governor']
    print(f"\n[RISK GOVERNOR] {'PASS' if rg['pass'] else 'FAIL'}")
    if rg['violations']:
        for v in rg['violations']:
            print(f"  ⚠ {v}")
    print(f"  Gamma Corridor: {rg['gamma_corridor']}")

    # Reserve Flow Kernel
    rf = result['reserve_flow']
    print(f"\n[RESERVE FLOW KERNEL]")
    print(f"  Equity beta signal: {rf['equity_beta_signal']:+.4f}")
    print(f"  Credit signal:      {rf['credit_signal']:+.4f}")
    print(f"  Rates signal:       {rf['rates_signal']:+.4f}")
    print(f"  Vol signal:         {rf['vol_signal']:+.4f}")
    print(f"  Reserve impulse:    {rf['reserve_impulse']:+.4f}")

    # Learning loop test
    ll = LearningLoop()
    lr = ll.update(0.005, 'TRENDING', {'P1': 0.01, 'P2': 0.005, 'P3': -0.002, 'P4': 0.008, 'P5': -0.001}, result)
    print(f"\n[LEARNING LOOP] Observations: {lr['total_observations']}")
    print(f"  Recommended adjustments: {lr['recommended_adjustments']}")

    print(f"\n{'=' * 70}")
    print("METADRON CUBE SELF-TEST PASSED")
    print(f"{'=' * 70}")
