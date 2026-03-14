"""
Platform Orchestrator — Master Pipeline
=========================================
The complete system pipeline:

    Macro Liquidity Data
        ↓ FedPlumbingLayer (Layer 0 — Data & Market Plumbing)
        ↓ Liquidity Tensor (Layer 1)
        ↓ Reserve Flow Kernel (Layer 2)
        ↓ Contagion Engine (Layer 3)
        ↓ Regime Engine (Layer 4)
        ↓ Metadron Cube State
        ↓ Gate-Z Allocation (5 sleeves: P1-P5)
        ↓ Risk Governor (Gamma Corridor [7%-12%])
        ↓ Stat Arb Engine (Medallion-style micro alpha)
        ↓ Options Convexity Engine (Greeks optimization)
        ↓ Alpha Optimizer (walk-forward ML + fallen angel + RV)
        ↓ Alpha-Beta Unleashed (dynamic beta management)
        ↓ Execution Engine (micro-price arb)
        ↓ Platinum Report (9-part diagnostic cockpit)

Three alpha streams:
    Macro:    large directional moves
    Stat Arb: consistent micro alpha (high win rate)
    Options:  convexity / volatility

Risk governance:
    Beta managed futures sleeve within controlled Gamma Corridor [7%-12%]
    Gross leverage 2.2-2.8x, Net beta ≤ 0.65, VaR ≤ 0.30%

Repos integrated:
    ai-hedgefund       — multi-agent decision engine (THIS repo — all engines live here)
    ML-Macro-Market    — cyclical/secular regime classifier + FRB integration
    QLIB               — alpha factor research & backtesting
    quant-trading      — strategy library (TA + patterns + WonderTrader translations)
    Financial-Data     — market data infrastructure
    hedgefund-tracker  — 13F institutional flow tracker
    open-bb            — alternative data terminal
    Ruflo-agents       — Claude-flow orchestration engine
    Mav-Analysis       — MCP server for Claude tools
    Air-LLM            — lightweight LLM inference
    AI-Newton          — physics-inspired symbolic AI
    FRB (external)     — Federal Reserve data client
    wondertrader (ext) — HFT micro-price signal (translated C++ → Python)
    exchange-core (ext)— order matching engine (Java → Python adapter)
    Quant-Developers-Resources (ext) — quant research catalog

Author: Platform Init — claude/init-test-repos-oPogr
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
from datetime import datetime
from pathlib import Path

ORCHESTRATOR_LOG = Path(__file__).parent / "orchestrator_log.jsonl"


class PlatformOrchestrator:
    """
    Master orchestrator: chains all engines into the full investment pipeline.
    One call to run() executes the entire system top-to-bottom.
    """

    def __init__(self, nav: float = 100_000_000):
        self.nav = nav
        self._engines = {}
        self._init_engines()

    def _init_engines(self):
        """Initialize all engine components."""
        try:
            from agents.metadron_cube import MetadronCube
            self._engines['cube'] = MetadronCube()
        except ImportError as e:
            print(f"[ORCHESTRATOR] MetadronCube import failed: {e}")

        try:
            from agents.stat_arb_engine import StatArbEngine
            self._engines['stat_arb'] = StatArbEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] StatArbEngine import failed: {e}")

        try:
            from agents.options_engine import OptionsConvexityEngine
            self._engines['options'] = OptionsConvexityEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] OptionsEngine import failed: {e}")

        try:
            from agents.contagion_engine import ContagionEngine
            self._engines['contagion'] = ContagionEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] ContagionEngine import failed: {e}")

        try:
            from agents.macro_engine import MacroEngine
            self._engines['macro'] = MacroEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] MacroEngine import failed: {e}")

        try:
            from agents.alpha_optimizer import AlphaOptimizerEngine
            self._engines['alpha'] = AlphaOptimizerEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] AlphaOptimizer import failed: {e}")

        try:
            from agents.alpha_beta_engine import AlphaBetaUnleashed
            self._engines['beta'] = AlphaBetaUnleashed()
        except ImportError as e:
            print(f"[ORCHESTRATOR] AlphaBetaUnleashed import failed: {e}")

        try:
            from agents.platinum_report import PlatinumReport
            self._engines['report'] = PlatinumReport()
        except ImportError as e:
            print(f"[ORCHESTRATOR] PlatinumReport import failed: {e}")

        print(f"[ORCHESTRATOR] Initialized {len(self._engines)}/{8} engines")

    def run(self, spot_price: float = 585.0, iv: float = 0.18) -> dict:
        """
        Execute the full investment pipeline.
        Returns complete system state including Platinum Report.
        """
        t0 = time.time()
        results = {'timestamp': datetime.now().isoformat(), 'nav': self.nav}

        # ── STEP 1: Metadron Cube (Layers 0-4 + Gate-Z + Risk Governor) ──────
        print("\n[1/7] Computing Metadron Cube...")
        if 'cube' in self._engines:
            cube_state = self._engines['cube'].compute()
            results['cube'] = cube_state
            regime = cube_state['regime']['regime']
            print(f"       Regime: {regime}  L={cube_state['cube_axes']['L_t']:+.3f}  "
                  f"R={cube_state['cube_axes']['R_t']:.3f}  F={cube_state['cube_axes']['F_t']:+.3f}")
        else:
            cube_state = None
            regime = 'RANGE'
            print("       [SKIP] MetadronCube not available")

        # ── STEP 2: Contagion Analysis ───────────────────────────────────────
        print("[2/7] Running contagion scenarios...")
        if 'contagion' in self._engines:
            contagion_state = self._engines['contagion'].scenario_analysis()
            results['contagion'] = contagion_state
            print(f"       Highest risk: {contagion_state['highest_risk']}")
        else:
            contagion_state = None
            print("       [SKIP] ContagionEngine not available")

        # ── STEP 3: Macro Engine (GMTF + GICS + Money Velocity) ─────────────
        print("[3/7] Running Macro Engine...")
        velocity_state = None
        if 'macro' in self._engines:
            try:
                macro_result = self._engines['macro'].run()
                results['macro'] = macro_result
                velocity_state = macro_result.get('velocity', {})
                print(f"       GMTF regime: {macro_result.get('regime', 'N/A')}")
            except Exception as e:
                print(f"       [ERROR] MacroEngine: {e}")
                results['macro'] = {'error': str(e)}
        else:
            print("       [SKIP] MacroEngine not available")

        # ── STEP 4: Statistical Arbitrage (Medallion layer) ──────────────────
        print("[4/7] Running Stat Arb Engine...")
        if 'stat_arb' in self._engines:
            stat_arb_state = self._engines['stat_arb'].run(regime=regime, nav=self.nav)
            results['stat_arb'] = {
                'status': stat_arb_state['status'],
                'signals': stat_arb_state['signals'],
                'trades': len(stat_arb_state['trades']),
                'allocation_pct': stat_arb_state['allocation_pct'],
            }
            print(f"       Status: {stat_arb_state['status']}  "
                  f"Signals: {stat_arb_state['signals']['total']}  "
                  f"Trades: {len(stat_arb_state['trades'])}")
        else:
            stat_arb_state = None
            print("       [SKIP] StatArbEngine not available")

        # ── STEP 5: Options Convexity (P4 sleeve) ───────────────────────────
        print("[5/7] Constructing Options sleeve...")
        if 'options' in self._engines:
            gate_z = cube_state.get('gate_z_allocation', {}).get('allocations', {}) if cube_state else {}
            p4_pct = gate_z.get('P4', 0.15)
            p4_capital = self.nav * p4_pct
            options_state = self._engines['options'].construct_sleeve(
                regime=regime, spot=spot_price, iv=iv, p4_capital=p4_capital, nav=self.nav
            )
            results['options'] = options_state
            greeks = options_state['greeks']
            print(f"       P4 capital: ${p4_capital:,.0f}  "
                  f"Δ={greeks['net_delta']:+.0f}  Θ=${greeks['net_theta']:+.0f}/d")
        else:
            options_state = None
            print("       [SKIP] OptionsEngine not available")

        # ── STEP 6: Alpha Optimizer + Beta Engine ────────────────────────────
        print("[6/7] Running Alpha + Beta engines...")
        if 'alpha' in self._engines:
            try:
                alpha_result = self._engines['alpha'].run()
                results['alpha'] = {
                    'sleeve_beta': alpha_result.get('sleeve_beta', 0),
                    'expected_return': alpha_result.get('expected_return', 0),
                    'weights': len(alpha_result.get('weights', {})),
                }
                print(f"       Sleeve beta: {alpha_result.get('sleeve_beta', 'N/A')}")
            except Exception as e:
                print(f"       [ERROR] AlphaOptimizer: {e}")
                alpha_result = {}
        else:
            alpha_result = {}
            print("       [SKIP] AlphaOptimizer not available")

        if 'beta' in self._engines:
            try:
                beta_snapshot = self._engines['beta'].snapshot()
                results['beta'] = beta_snapshot
                print(f"       Target beta: {beta_snapshot.get('target_beta', 'N/A')}")
            except Exception as e:
                print(f"       [ERROR] BetaEngine: {e}")
        else:
            print("       [SKIP] BetaEngine not available")

        # ── STEP 7: Platinum Report ──────────────────────────────────────────
        print("[7/7] Generating Platinum Report...")
        if 'report' in self._engines:
            report = self._engines['report'].generate(
                cube_state=cube_state,
                stat_arb_state=stat_arb_state,
                options_state=options_state,
                contagion_state=contagion_state,
                velocity_state=velocity_state,
            )
            results['platinum_report_id'] = report.get('report_id', 'N/A')
            text_report = self._engines['report'].format_text()
            print(f"       Report: {report.get('report_id', 'N/A')}")
        else:
            text_report = None
            print("       [SKIP] PlatinumReport not available")

        # ── Timing ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        results['elapsed_seconds'] = round(elapsed, 2)

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print(f"PLATFORM RUN COMPLETE — {elapsed:.1f}s")
        print(f"{'=' * 70}")
        print(f"  Regime:           {regime}")
        print(f"  Engines loaded:   {len(self._engines)}/8")
        print(f"  NAV:              ${self.nav:,.0f}")
        if cube_state:
            alloc = cube_state.get('gate_z_allocation', {}).get('allocations', {})
            print(f"  Gate-Z:")
            labels = {'P1': 'Equities', 'P2': 'Factor Rotation', 'P3': 'Commodities',
                      'P4': 'Options', 'P5': 'Hedges'}
            for s, w in alloc.items():
                print(f"    {s} ({labels.get(s, '?'):<16s}): {w:.1%} = ${self.nav * w:>14,.0f}")
        print(f"  Risk Governor:    {'PASS' if cube_state and cube_state.get('risk_governor', {}).get('pass') else 'CHECK'}")
        print(f"  Gamma Corridor:   [7%, 12%]")
        print(f"{'=' * 70}\n")

        # Log
        self._log(results)

        # Print Platinum Report if available
        if text_report:
            print(text_report)

        return results

    def _log(self, results):
        try:
            log = {k: v for k, v in results.items() if k not in ['cube', 'contagion']}
            with open(ORCHESTRATOR_LOG, 'a') as f:
                f.write(json.dumps(log, default=str) + '\n')
        except Exception:
            pass


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    orchestrator = PlatformOrchestrator(nav=100_000_000)
    results = orchestrator.run(spot_price=585.0, iv=0.18)
