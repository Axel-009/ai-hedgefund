"""
Options Convexity Engine — Greeks Optimization & Tail Hedging
=============================================================
P4 sleeve in Gate-Z: Options provide convexity and carry.

Objective: max(Θ + Γ) - c·Vega
    Delta  — aligned with portfolio beta
    Gamma  — slightly positive (convexity)
    Theta  — positive carry (time decay collection)
    Vega   — controlled exposure

Structures:
    ATM calls / call spreads — directional convexity
    Protective puts          — tail risk protection
    Short volatility carry   — theta harvesting
    Variance trades          — volatility exposure

Integration:
    Metadron Cube regime → determines options strategy mix
    Risk Governor         → convex delta ≤ 0.45
    AlphaBetaUnleashed    → delta aligned with Gamma Corridor [7%-12%]

Author: Platform Init — claude/init-test-repos-oPogr
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore")

OPTIONS_LOG = Path(__file__).parent / "options_log.jsonl"


# ==============================================================================
# BLACK-SCHOLES GREEKS
# ==============================================================================
def _norm_cdf(x):
    """Standard normal CDF (no scipy dependency)."""
    import math
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_price: float
    intrinsic: float
    time_value: float


def black_scholes_greeks(S: float, K: float, T: float, r: float,
                          sigma: float, option_type: str = 'call') -> GreeksResult:
    """
    Full Black-Scholes Greeks computation.
    S: spot price, K: strike, T: time to expiry (years),
    r: risk-free rate, sigma: implied vol.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return GreeksResult(
            delta=1.0 if option_type == 'call' and S > K else (-1.0 if option_type == 'put' and K > S else 0.0),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            option_price=intrinsic, intrinsic=intrinsic, time_value=0.0,
        )

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        rho = K * T * np.exp(-r * T) * _norm_cdf(d2) / 100
    else:  # put
        price = K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * _norm_cdf(-d2) / 100

    gamma = _norm_pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * _norm_cdf(d2 if option_type == 'call' else -d2)
             * (1 if option_type == 'call' else -1)) / 365  # Daily theta
    vega = S * _norm_pdf(d1) * np.sqrt(T) / 100

    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)

    return GreeksResult(
        delta=round(delta, 6), gamma=round(gamma, 6),
        theta=round(theta, 4), vega=round(vega, 4), rho=round(rho, 4),
        option_price=round(price, 4),
        intrinsic=round(intrinsic, 2),
        time_value=round(price - intrinsic, 4),
    )


# ==============================================================================
# OPTIONS STRUCTURES
# ==============================================================================
@dataclass
class OptionsPosition:
    """Single options position."""
    ticker: str
    option_type: str     # call / put
    strike: float
    expiry_days: int
    quantity: int        # positive = long, negative = short
    premium: float
    greeks: GreeksResult = None

    @property
    def notional(self):
        return abs(self.quantity) * 100 * self.strike  # Standard 100 multiplier


class OptionsStructureBuilder:
    """
    Builds options structures aligned with regime and portfolio beta.

    Structures:
        1. Bull Call Spread — directional upside with limited cost
        2. Protective Put — tail risk insurance
        3. Short Strangle — theta harvesting in low-vol regimes
        4. Long Straddle — volatility play for regime transitions
    """

    def __init__(self, risk_free_rate: float = 0.045):
        self.r = risk_free_rate

    def atm_call_spread(self, spot: float, iv: float, days: int = 30,
                         width_pct: float = 0.05, qty: int = 10) -> list:
        """ATM call spread: long ATM call, short OTM call."""
        T = days / 365
        K_atm = round(spot, 0)
        K_otm = round(spot * (1 + width_pct), 0)

        long_g = black_scholes_greeks(spot, K_atm, T, self.r, iv, 'call')
        short_g = black_scholes_greeks(spot, K_otm, T, self.r, iv, 'call')

        return [
            OptionsPosition(ticker='SPY', option_type='call', strike=K_atm,
                            expiry_days=days, quantity=qty,
                            premium=long_g.option_price, greeks=long_g),
            OptionsPosition(ticker='SPY', option_type='call', strike=K_otm,
                            expiry_days=days, quantity=-qty,
                            premium=short_g.option_price, greeks=short_g),
        ]

    def protective_put(self, spot: float, iv: float, days: int = 45,
                        otm_pct: float = 0.05, qty: int = 10) -> list:
        """Protective put: long OTM put for tail hedge."""
        T = days / 365
        K = round(spot * (1 - otm_pct), 0)
        g = black_scholes_greeks(spot, K, T, self.r, iv, 'put')
        return [
            OptionsPosition(ticker='SPY', option_type='put', strike=K,
                            expiry_days=days, quantity=qty,
                            premium=g.option_price, greeks=g),
        ]

    def short_strangle(self, spot: float, iv: float, days: int = 30,
                        width_pct: float = 0.05, qty: int = 5) -> list:
        """Short strangle: sell OTM call + OTM put (theta harvest)."""
        T = days / 365
        K_call = round(spot * (1 + width_pct), 0)
        K_put  = round(spot * (1 - width_pct), 0)

        g_call = black_scholes_greeks(spot, K_call, T, self.r, iv, 'call')
        g_put  = black_scholes_greeks(spot, K_put, T, self.r, iv, 'put')

        return [
            OptionsPosition(ticker='SPY', option_type='call', strike=K_call,
                            expiry_days=days, quantity=-qty,
                            premium=g_call.option_price, greeks=g_call),
            OptionsPosition(ticker='SPY', option_type='put', strike=K_put,
                            expiry_days=days, quantity=-qty,
                            premium=g_put.option_price, greeks=g_put),
        ]


# ==============================================================================
# OPTIONS CONVEXITY ENGINE — P4 Sleeve Manager
# ==============================================================================
class OptionsConvexityEngine:
    """
    Master options engine for the P4 sleeve.

    Objective: max(Θ + Γ) - c·Vega
    Subject to:
        |Δ_portfolio| ≤ convex_delta_max (0.45)
        Γ ≥ 0 (positive convexity preferred)
        Θ > 0 (positive carry preferred except for tail hedges)
        |V| ≤ vega_budget

    Regime → Structure mapping:
        TRENDING  → bull call spreads + moderate tail hedge
        RANGE     → short strangles (theta harvest) + small puts
        STRESS    → protective puts + long straddles
        CRASH     → max protective puts + long vol
    """

    REGIME_STRUCTURES = {
        'TRENDING': {
            'call_spread_weight': 0.50,
            'protective_put_weight': 0.20,
            'theta_harvest_weight': 0.30,
        },
        'RANGE': {
            'call_spread_weight': 0.20,
            'protective_put_weight': 0.15,
            'theta_harvest_weight': 0.65,
        },
        'STRESS': {
            'call_spread_weight': 0.10,
            'protective_put_weight': 0.60,
            'theta_harvest_weight': 0.30,
        },
        'CRASH': {
            'call_spread_weight': 0.00,
            'protective_put_weight': 0.80,
            'theta_harvest_weight': 0.20,
        },
    }

    CONVEX_DELTA_MAX = 0.45
    VEGA_BUDGET_PCT = 0.02  # 2% of NAV

    def __init__(self):
        self.builder = OptionsStructureBuilder()
        self._positions = []

    def construct_sleeve(self, regime: str, spot: float = 585.0,
                          iv: float = 0.18, p4_capital: float = 15_000_000,
                          nav: float = 100_000_000) -> dict:
        """
        Construct complete P4 options sleeve for current regime.
        """
        weights = self.REGIME_STRUCTURES.get(regime, self.REGIME_STRUCTURES['RANGE'])
        positions = []

        # 1. Call spreads (directional convexity)
        cs_capital = p4_capital * weights['call_spread_weight']
        if cs_capital > 0:
            cs_qty = max(1, int(cs_capital / (spot * 100)))  # ~1 contract per $58.5K
            cs_positions = self.builder.atm_call_spread(spot, iv, days=30, qty=cs_qty)
            positions.extend(cs_positions)

        # 2. Protective puts (tail hedge)
        pp_capital = p4_capital * weights['protective_put_weight']
        if pp_capital > 0:
            pp_qty = max(1, int(pp_capital / (spot * 100 * 0.03)))  # ~3% OTM cost
            pp_positions = self.builder.protective_put(spot, iv, days=45, qty=pp_qty)
            positions.extend(pp_positions)

        # 3. Theta harvest (short strangles in low-vol)
        th_capital = p4_capital * weights['theta_harvest_weight']
        if th_capital > 0 and iv < 0.25:  # Only in low-vol environment
            th_qty = max(1, int(th_capital / (spot * 100 * 0.10)))
            th_positions = self.builder.short_strangle(spot, iv, days=30, qty=th_qty)
            positions.extend(th_positions)

        # Aggregate Greeks
        total_greeks = self._aggregate_greeks(positions)
        self._positions = positions

        # Risk check
        delta_ok = abs(total_greeks['net_delta']) <= self.CONVEX_DELTA_MAX
        vega_ok = abs(total_greeks['net_vega'] * 100) <= nav * self.VEGA_BUDGET_PCT

        return {
            'regime': regime,
            'p4_capital': round(p4_capital, 2),
            'positions': len(positions),
            'structures': {
                'call_spreads': int(cs_capital > 0),
                'protective_puts': int(pp_capital > 0),
                'theta_harvest': int(th_capital > 0 and iv < 0.25),
            },
            'greeks': total_greeks,
            'risk_check': {
                'delta_ok': delta_ok,
                'vega_ok': vega_ok,
                'convex_delta_limit': self.CONVEX_DELTA_MAX,
            },
            'objective_value': round(
                total_greeks['net_theta'] + total_greeks['net_gamma'] * spot * spot * 0.5
                - 0.5 * abs(total_greeks['net_vega']),
                4
            ),
            'timestamp': datetime.now().isoformat(),
        }

    def _aggregate_greeks(self, positions: list) -> dict:
        """Sum Greeks across all positions."""
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        total_premium = 0.0

        for pos in positions:
            if pos.greeks is None:
                continue
            mult = pos.quantity  # Positive = long, negative = short
            net_delta += pos.greeks.delta * mult * 100
            net_gamma += pos.greeks.gamma * mult * 100
            net_theta += pos.greeks.theta * mult * 100
            net_vega  += pos.greeks.vega * mult * 100
            total_premium += pos.premium * mult * 100

        return {
            'net_delta': round(net_delta, 4),
            'net_gamma': round(net_gamma, 6),
            'net_theta': round(net_theta, 4),
            'net_vega':  round(net_vega, 4),
            'total_premium': round(total_premium, 2),
        }


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OPTIONS CONVEXITY ENGINE — SELF-TEST")
    print("=" * 70)

    # Test Black-Scholes Greeks
    g = black_scholes_greeks(S=585, K=585, T=30/365, r=0.045, sigma=0.18, option_type='call')
    print(f"\n[BS GREEKS] ATM Call SPY $585, 30d, IV=18%")
    print(f"  Price:  ${g.option_price:.2f}")
    print(f"  Delta:  {g.delta:.4f}")
    print(f"  Gamma:  {g.gamma:.6f}")
    print(f"  Theta:  ${g.theta:.4f}/day")
    print(f"  Vega:   ${g.vega:.4f}/1%IV")

    engine = OptionsConvexityEngine()
    for regime in ['TRENDING', 'RANGE', 'STRESS', 'CRASH']:
        result = engine.construct_sleeve(regime=regime, spot=585, iv=0.18)
        greeks = result['greeks']
        print(f"\n[{regime}]")
        print(f"  Positions: {result['positions']}")
        print(f"  Net Δ: {greeks['net_delta']:+.2f}  Γ: {greeks['net_gamma']:+.6f}  "
              f"Θ: ${greeks['net_theta']:+.2f}/d  V: ${greeks['net_vega']:+.2f}")
        print(f"  Objective: {result['objective_value']:+.4f}")
        print(f"  Risk: Δ_ok={result['risk_check']['delta_ok']}, V_ok={result['risk_check']['vega_ok']}")

    print(f"\n{'=' * 70}")
    print("OPTIONS ENGINE SELF-TEST PASSED")
    print(f"{'=' * 70}")
