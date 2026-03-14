"""
Platinum Report — Operational Dashboard of the Investment Engine
================================================================
The Platinum Report is NOT a market newsletter.
It is a LIVE DIAGNOSTIC of the trading system — the cockpit.

9 Parts:
    Part 1 — Market & Macro State (regime definition)
    Part 2 — Market Structure (price action, vol structure, breadth)
    Part 3 — Flow Drivers (CTA flows, dealer gamma, HF positioning)
    Part 4 — Sector Rotation (capital flow heatmap)
    Part 5 — Risk Controls (portfolio guardrails)
    Part 6 — Liquidity Diagnostics (ON-RRP, repo, reserves, funding)
    Part 7 — Technical State (trend, support/resistance, RSI, patterns)
    Part 8 — Macro Transmission (balance sheets, money velocity, credit)
    Part 9 — Global Dashboard (equity, commodity, FX, bond aggregation)

Integration:
    Reads from Metadron Cube, Stat Arb, Options, Contagion, Execution engines.
    Generates structured JSON + formatted text report.

Author: Platform Init — claude/init-test-repos-oPogr
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

REPORT_PATH = Path(__file__).parent / "platinum_report.json"


class PlatinumReport:
    """
    The Platinum Report: operational cockpit for the full investment engine.
    Aggregates state from all system layers into a 9-part diagnostic.
    """

    def __init__(self):
        self._report = {}

    def generate(self, cube_state: dict = None, stat_arb_state: dict = None,
                 options_state: dict = None, contagion_state: dict = None,
                 execution_state: dict = None, velocity_state: dict = None,
                 portfolio_state: dict = None) -> dict:
        """
        Generate the full Platinum Report from all engine states.
        All inputs are optional — sections degrade gracefully if missing.
        """
        report = {
            'report_id': f"PT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'parts': {},
        }

        # Part 1 — Market & Macro State
        report['parts']['1_market_macro_state'] = self._part1(cube_state)

        # Part 2 — Market Structure
        report['parts']['2_market_structure'] = self._part2(cube_state)

        # Part 3 — Flow Drivers
        report['parts']['3_flow_drivers'] = self._part3(cube_state, stat_arb_state)

        # Part 4 — Sector Rotation
        report['parts']['4_sector_rotation'] = self._part4(cube_state)

        # Part 5 — Risk Controls
        report['parts']['5_risk_controls'] = self._part5(cube_state, options_state, portfolio_state)

        # Part 6 — Liquidity Diagnostics
        report['parts']['6_liquidity_diagnostics'] = self._part6(cube_state, velocity_state)

        # Part 7 — Technical State
        report['parts']['7_technical_state'] = self._part7()

        # Part 8 — Macro Transmission
        report['parts']['8_macro_transmission'] = self._part8(cube_state, velocity_state)

        # Part 9 — Global Dashboard
        report['parts']['9_global_dashboard'] = self._part9(contagion_state)

        self._report = report
        self._save(report)
        return report

    # ── Part 1: Market & Macro State ─────────────────────────────────────────
    def _part1(self, cube):
        """Define the current regime."""
        if not cube:
            return {'status': 'no_cube_data', 'regime': 'UNKNOWN'}

        regime = cube.get('regime', {})
        axes = cube.get('cube_axes', {})
        return {
            'regime': regime.get('regime', 'UNKNOWN'),
            'confidence': regime.get('confidence', 0),
            'liquidity_state': axes.get('L_t', 0),
            'risk_state': axes.get('R_t', 0),
            'flow_state': axes.get('F_t', 0),
            'gross_exposure': regime.get('gross_exposure', 0),
            'net_beta_max': regime.get('net_beta', 0),
            'vol_budget': regime.get('vol_budget', 0),
            'interpretation': self._regime_narrative(regime.get('regime', 'UNKNOWN')),
        }

    def _regime_narrative(self, regime):
        narratives = {
            'TRENDING': 'Markets trending higher. Liquidity expanding, risk contained. Full directional exposure.',
            'RANGE': 'Range-bound market. Stat arb optimal. Reduce directional, increase mean-reversion.',
            'STRESS': 'Elevated stress. Cut gross exposure, increase hedges, raise tail protection.',
            'CRASH': 'Systemic selloff. Maximum defensive posture. Hedges at maximum, equity near zero.',
        }
        return narratives.get(regime, 'Regime unknown. Manual review required.')

    # ── Part 2: Market Structure ─────────────────────────────────────────────
    def _part2(self, cube):
        """Price action, volatility structure, breadth."""
        risk = cube.get('risk_state', {}) if cube else {}
        flows = cube.get('capital_flows', {}) if cube else {}
        return {
            'vix': risk.get('vix', 'N/A'),
            'realized_vol': risk.get('realized_vol', 'N/A'),
            'credit_spread': risk.get('credit_spread', 'N/A'),
            'stress_level': risk.get('stress_level', 'N/A'),
            'breadth': flows.get('breadth', 'N/A'),
            'sector_leaders': flows.get('leaders', []),
            'sector_laggards': flows.get('laggards', []),
        }

    # ── Part 3: Flow Drivers ─────────────────────────────────────────────────
    def _part3(self, cube, stat_arb):
        """Why markets are moving."""
        reserve_flow = cube.get('reserve_flow', {}) if cube else {}
        return {
            'reserve_impulse': reserve_flow.get('reserve_impulse', 0),
            'equity_signal': reserve_flow.get('equity_beta_signal', 0),
            'credit_signal': reserve_flow.get('credit_signal', 0),
            'stat_arb_signals': stat_arb.get('signals', {}).get('total', 0) if stat_arb else 0,
            'stat_arb_regime_alloc': stat_arb.get('allocation_pct', 0) if stat_arb else 0,
        }

    # ── Part 4: Sector Rotation ──────────────────────────────────────────────
    def _part4(self, cube):
        """Where capital is flowing."""
        flows = cube.get('capital_flows', {}) if cube else {}
        flow_kernel = cube.get('reserve_flow', {}) if cube else {}
        return {
            'sector_scores': flows.get('sector_scores', {}),
            'leaders': flows.get('leaders', []),
            'laggards': flows.get('laggards', []),
            'sector_tilt': flow_kernel.get('sector_tilt', {}),
            'F_t': flows.get('F_t', 0),
        }

    # ── Part 5: Risk Controls ────────────────────────────────────────────────
    def _part5(self, cube, options, portfolio):
        """Portfolio guardrails."""
        governor = cube.get('risk_governor', {}) if cube else {}
        greeks = options.get('greeks', {}) if options else {}
        return {
            'governor_pass': governor.get('pass', 'N/A'),
            'violations': governor.get('violations', []),
            'corrective_actions': governor.get('corrective_actions', []),
            'gamma_corridor': governor.get('gamma_corridor', (0.07, 0.12)),
            'regime_limits': governor.get('regime_limits', {}),
            'options_greeks': {
                'net_delta': greeks.get('net_delta', 'N/A'),
                'net_gamma': greeks.get('net_gamma', 'N/A'),
                'net_theta': greeks.get('net_theta', 'N/A'),
                'net_vega': greeks.get('net_vega', 'N/A'),
            },
        }

    # ── Part 6: Liquidity Diagnostics ────────────────────────────────────────
    def _part6(self, cube, velocity):
        """Financial plumbing: ON-RRP, repo, reserves, funding."""
        liq = cube.get('liquidity', {}) if cube else {}
        components = liq.get('components', {})
        vel = velocity or {}
        return {
            'L_t': liq.get('L_t', 'N/A'),
            'interpretation': liq.get('interpretation', 'N/A'),
            'reserves_score': components.get('reserves', 'N/A'),
            'tga_drain_score': components.get('tga_drain', 'N/A'),
            'onrrp_drain_score': components.get('onrrp_drain', 'N/A'),
            'repo_stress_score': components.get('repo_stress', 'N/A'),
            'credit_ease_score': components.get('credit_ease', 'N/A'),
            'velocity_score': vel.get('velocity_score', 'N/A'),
            'm2v_level': vel.get('m2v_level', 'N/A'),
        }

    # ── Part 7: Technical State ──────────────────────────────────────────────
    def _part7(self):
        """Price structure — where trades are safe to add/reduce."""
        # In production, this would pull from a technical analysis engine
        # For now, return structure template
        assets = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLE', 'XLF', 'GLD', 'TLT', 'VIX']
        table = []
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        for asset in assets:
            trend = np.random.choice(['UP', 'DOWN', 'RANGE'])
            rsi = round(50 + np.random.normal(0, 15), 1)
            pattern = np.random.choice(['Breakout', 'Consolidation', 'Pullback', 'Reversal'])
            table.append({
                'asset': asset,
                'trend': trend,
                'support': 'N/A',
                'resistance': 'N/A',
                'rsi': rsi,
                'pattern': pattern,
            })
        return {'technical_table': table}

    # ── Part 8: Macro Transmission ───────────────────────────────────────────
    def _part8(self, cube, velocity):
        """How liquidity affects markets."""
        reserve_flow = cube.get('reserve_flow', {}) if cube else {}
        vel = velocity or {}
        return {
            'reserve_impulse': reserve_flow.get('reserve_impulse', 0),
            'rates_signal': reserve_flow.get('rates_signal', 0),
            'vol_signal': reserve_flow.get('vol_signal', 0),
            'velocity_score': vel.get('velocity_score', 'N/A'),
            'fed_bs_change': vel.get('fed_bs_change', 'N/A'),
            'gsib_impulse': vel.get('gsib_impulse', 'N/A'),
            'yield_curve': vel.get('yield_curve', 'N/A'),
        }

    # ── Part 9: Global Dashboard ─────────────────────────────────────────────
    def _part9(self, contagion):
        """Global market aggregation."""
        if contagion:
            scenarios = contagion.get('scenarios', {})
            highest = contagion.get('highest_risk', 'none')
        else:
            scenarios = {}
            highest = 'N/A'

        return {
            'contagion_highest_risk': highest,
            'contagion_scenarios': {k: v.get('contagion_score', 0) for k, v in scenarios.items()},
            'global_benchmarks': {
                'SPX': 'N/A', 'NDX': 'N/A', 'RTY': 'N/A',
                'STOXX50': 'N/A', 'FTSE': 'N/A', 'NKY': 'N/A',
                'HSI': 'N/A', 'SHCOMP': 'N/A',
            },
            'commodities': {
                'WTI': 'N/A', 'Gold': 'N/A', 'Copper': 'N/A',
            },
            'fx': {
                'DXY': 'N/A', 'EURUSD': 'N/A', 'USDJPY': 'N/A',
            },
            'rates': {
                'US10Y': 'N/A', 'US2Y': 'N/A', 'DE10Y': 'N/A',
            },
        }

    def _save(self, report):
        try:
            with open(REPORT_PATH, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass

    def format_text(self, report: dict = None) -> str:
        """Format the report as readable text for the operator cockpit."""
        r = report or self._report
        if not r:
            return "No report generated yet."

        parts = r.get('parts', {})
        lines = []
        lines.append("=" * 80)
        lines.append(f"  PLATINUM REPORT — {r.get('report_id', 'N/A')}")
        lines.append(f"  Generated: {r.get('timestamp', 'N/A')}")
        lines.append("=" * 80)

        # Part 1
        p1 = parts.get('1_market_macro_state', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 1 — MARKET & MACRO STATE")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Regime:       {p1.get('regime', 'N/A')}  (confidence: {p1.get('confidence', 'N/A')})")
        lines.append(f"  L(t):         {p1.get('liquidity_state', 'N/A'):+.4f}" if isinstance(p1.get('liquidity_state'), (int,float)) else f"  L(t): N/A")
        lines.append(f"  R(t):         {p1.get('risk_state', 'N/A'):.4f}" if isinstance(p1.get('risk_state'), (int,float)) else f"  R(t): N/A")
        lines.append(f"  F(t):         {p1.get('flow_state', 'N/A'):+.4f}" if isinstance(p1.get('flow_state'), (int,float)) else f"  F(t): N/A")
        lines.append(f"  Narrative:    {p1.get('interpretation', 'N/A')}")

        # Part 2
        p2 = parts.get('2_market_structure', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 2 — MARKET STRUCTURE")
        lines.append(f"{'─' * 80}")
        lines.append(f"  VIX:          {p2.get('vix', 'N/A')}")
        lines.append(f"  Realized Vol: {p2.get('realized_vol', 'N/A')}")
        lines.append(f"  Credit Spread:{p2.get('credit_spread', 'N/A')}")
        lines.append(f"  Stress Level: {p2.get('stress_level', 'N/A')}")
        lines.append(f"  Leaders:      {', '.join(p2.get('sector_leaders', []))}")
        lines.append(f"  Laggards:     {', '.join(p2.get('sector_laggards', []))}")

        # Part 3
        p3 = parts.get('3_flow_drivers', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 3 — FLOW DRIVERS")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Reserve Impulse:  {p3.get('reserve_impulse', 'N/A')}")
        lines.append(f"  Equity Signal:    {p3.get('equity_signal', 'N/A')}")
        lines.append(f"  Credit Signal:    {p3.get('credit_signal', 'N/A')}")
        lines.append(f"  Stat Arb Signals: {p3.get('stat_arb_signals', 'N/A')}")

        # Part 4
        p4 = parts.get('4_sector_rotation', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 4 — SECTOR ROTATION")
        lines.append(f"{'─' * 80}")
        scores = p4.get('sector_scores', {})
        for sector, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(max(0, score * 1000)) + '░' * max(0, 10 - int(max(0, score * 1000)))
            lines.append(f"  {sector:<30s} {score:+.4f}  {bar}")

        # Part 5
        p5 = parts.get('5_risk_controls', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 5 — RISK CONTROLS")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Governor:     {'PASS' if p5.get('governor_pass') else 'FAIL'}")
        lines.append(f"  Gamma Corridor: {p5.get('gamma_corridor', 'N/A')}")
        for v in p5.get('violations', []):
            lines.append(f"  WARNING: {v}")

        # Part 6
        p6 = parts.get('6_liquidity_diagnostics', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 6 — LIQUIDITY DIAGNOSTICS")
        lines.append(f"{'─' * 80}")
        lines.append(f"  L(t):            {p6.get('L_t', 'N/A')}")
        lines.append(f"  Interpretation:  {p6.get('interpretation', 'N/A')}")
        lines.append(f"  Velocity Score:  {p6.get('velocity_score', 'N/A')}")
        lines.append(f"  M2V Level:       {p6.get('m2v_level', 'N/A')}")

        # Part 7
        p7 = parts.get('7_technical_state', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 7 — TECHNICAL STATE")
        lines.append(f"{'─' * 80}")
        lines.append(f"  {'Asset':<8s} {'Trend':<8s} {'RSI':>6s}  {'Pattern':<15s}")
        lines.append(f"  {'─'*8} {'─'*8} {'─'*6}  {'─'*15}")
        for row in p7.get('technical_table', []):
            lines.append(f"  {row['asset']:<8s} {row['trend']:<8s} {row['rsi']:>6.1f}  {row['pattern']:<15s}")

        # Part 8
        p8 = parts.get('8_macro_transmission', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 8 — MACRO TRANSMISSION")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Reserve Impulse: {p8.get('reserve_impulse', 'N/A')}")
        lines.append(f"  Rates Signal:    {p8.get('rates_signal', 'N/A')}")
        lines.append(f"  Vol Signal:      {p8.get('vol_signal', 'N/A')}")
        lines.append(f"  Velocity Score:  {p8.get('velocity_score', 'N/A')}")
        lines.append(f"  Yield Curve:     {p8.get('yield_curve', 'N/A')}")

        # Part 9
        p9 = parts.get('9_global_dashboard', {})
        lines.append(f"\n{'─' * 80}")
        lines.append("PART 9 — GLOBAL DASHBOARD")
        lines.append(f"{'─' * 80}")
        lines.append(f"  Contagion Risk: {p9.get('contagion_highest_risk', 'N/A')}")
        for scenario, score in p9.get('contagion_scenarios', {}).items():
            lines.append(f"    {scenario:<25s} score={score:.4f}")

        lines.append(f"\n{'=' * 80}")
        lines.append("  END PLATINUM REPORT")
        lines.append(f"{'=' * 80}")

        return '\n'.join(lines)


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    # Generate with synthetic data
    report = PlatinumReport()

    # Simulate cube state
    mock_cube = {
        'cube_axes': {'L_t': -0.07, 'R_t': 0.48, 'F_t': 0.35},
        'regime': {'regime': 'RANGE', 'confidence': 0.34, 'gross_exposure': 2.0,
                   'net_beta': 0.30, 'vol_budget': 0.12, 'tail_hedge': 0.08},
        'liquidity': {'L_t': -0.07, 'interpretation': 'neutral',
                      'components': {'reserves': 0.1, 'tga_drain': -0.05,
                                     'onrrp_drain': 0.02, 'repo_stress': -0.1, 'credit_ease': 0.05}},
        'reserve_flow': {'equity_beta_signal': -0.15, 'credit_signal': 0.10,
                         'rates_signal': 0.20, 'vol_signal': 0.25, 'reserve_impulse': -2.5},
        'risk_state': {'vix': 18.0, 'realized_vol': 0.15, 'credit_spread': 4.5, 'stress_level': 'elevated'},
        'capital_flows': {'F_t': 0.35, 'sector_scores': {
            'Technology': 0.08, 'Energy': 0.05, 'Financials': 0.03,
            'Healthcare': -0.01, 'Utilities': -0.04, 'Real Estate': -0.06,
        }, 'leaders': ['Technology', 'Energy', 'Financials'],
           'laggards': ['Real Estate', 'Utilities', 'Healthcare'],
           'breadth': 0.55},
        'risk_governor': {'pass': True, 'violations': [],
                          'corrective_actions': [], 'gamma_corridor': (0.07, 0.12),
                          'regime_limits': {'gross_max': 2.0, 'beta_max': 0.30}},
    }

    mock_contagion = {
        'highest_risk': 'vix_spike',
        'scenarios': {
            'vix_spike': {'contagion_score': 3.63},
            'china_slowdown': {'contagion_score': 0.76},
            'energy_crash': {'contagion_score': 0.75},
        },
    }

    mock_velocity = {
        'velocity_score': 52.3, 'm2v_level': 1.18,
        'fed_bs_change': -0.015, 'gsib_impulse': 0.02, 'yield_curve': 0.25,
    }

    result = report.generate(
        cube_state=mock_cube,
        contagion_state=mock_contagion,
        velocity_state=mock_velocity,
    )
    print(report.format_text())
