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
        ↓ Distressed Asset Engine (Altman+Merton+Ohlson+Zmijewski+ML ensemble)
        ↓ CVR Engine (Contingent Value Rights — 5-model fair value)
        ↓ Event-Driven Engine (12 corporate event types + Kelly sizing)
        ↓ Stat Arb Engine (Medallion-style micro alpha)
        ↓ Options Convexity Engine (Greeks optimization)
        ↓ Alpha Optimizer (walk-forward ML + fallen angel + RV)
        ↓ Alpha-Beta Unleashed (dynamic beta management)
        ↓ Execution Engine (micro-price arb)
        ↓ Platinum Report (9-part diagnostic cockpit)

Four alpha streams:
    Macro:        large directional moves (Metadron Cube)
    Special Sit:  distressed + CVR + event-driven (NEW)
    Stat Arb:     consistent micro alpha (high win rate)
    Options:      convexity / volatility

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
            from agents.distressed_asset_engine import DistressedAssetEngine
            self._engines['distress'] = DistressedAssetEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] DistressedAssetEngine import failed: {e}")

        try:
            from agents.cvr_engine import CVREngine
            self._engines['cvr'] = CVREngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] CVREngine import failed: {e}")

        try:
            from agents.event_driven_engine import EventDrivenEngine
            self._engines['event'] = EventDrivenEngine()
        except ImportError as e:
            print(f"[ORCHESTRATOR] EventDrivenEngine import failed: {e}")

        try:
            from agents.platinum_report import PlatinumReport
            self._engines['report'] = PlatinumReport()
        except ImportError as e:
            print(f"[ORCHESTRATOR] PlatinumReport import failed: {e}")

        print(f"[ORCHESTRATOR] Initialized {len(self._engines)}/{11} engines")

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

        # ── STEP 3a: Distressed Asset Engine ────────────────────────────────
        print("[3a/10] Running Distressed Asset Engine...")
        distress_state = None
        if 'distress' in self._engines:
            try:
                distress_state = self._engines['distress'].run(regime=regime, nav=self.nav)
                results['distress'] = {
                    'status': distress_state['status'],
                    'distressed_count': distress_state['distressed_count'],
                    'grey_count': distress_state['grey_count'],
                    'avg_ensemble_score': distress_state['avg_ensemble_score'],
                    'universe_stress_pct': distress_state['universe_stress_pct'],
                    'opportunity_count': distress_state['opportunity_count'],
                    'capital_deployed': distress_state['total_capital_deployed'],
                    'watchlist': distress_state['watchlist'],
                }
                print(f"       Distressed: {distress_state['distressed_count']}  "
                      f"Grey: {distress_state['grey_count']}  "
                      f"Opportunities: {distress_state['opportunity_count']}  "
                      f"AvgScore: {distress_state['avg_ensemble_score']:.3f}")
            except Exception as e:
                print(f"       [ERROR] DistressedAssetEngine: {e}")
                results['distress'] = {'error': str(e)}
        else:
            print("       [SKIP] DistressedAssetEngine not available")

        # ── STEP 3b: CVR Engine ──────────────────────────────────────────────
        print("[3b/10] Running CVR Engine...")
        cvr_state = None
        if 'cvr' in self._engines:
            try:
                cvr_state = self._engines['cvr'].run(
                    regime=regime,
                    distress_scores=distress_state,
                )
                results['cvr'] = {
                    'status': cvr_state['status'],
                    'instruments_valued': cvr_state['instruments_valued'],
                    'long_signals': cvr_state['long_signals'],
                    'short_signals': cvr_state['short_signals'],
                    'avg_mispricing_pct': cvr_state['avg_mispricing_pct'],
                    'total_alpha_score': cvr_state['total_alpha_score'],
                }
                print(f"       Instruments: {cvr_state['instruments_valued']}  "
                      f"Longs: {len(cvr_state['long_signals'])}  "
                      f"Shorts: {len(cvr_state['short_signals'])}  "
                      f"Avg misprice: {cvr_state['avg_mispricing_pct']:.2%}")
            except Exception as e:
                print(f"       [ERROR] CVREngine: {e}")
                results['cvr'] = {'error': str(e)}
        else:
            print("       [SKIP] CVREngine not available")

        # ── STEP 3c: Event-Driven Engine ─────────────────────────────────────
        print("[3c/10] Running Event-Driven Engine...")
        event_state = None
        if 'event' in self._engines:
            try:
                event_state = self._engines['event'].run(regime=regime, nav=self.nav)
                results['event_driven'] = {
                    'status': event_state['status'],
                    'events_detected': event_state['events_detected'],
                    'positions_sized': event_state['positions_sized'],
                    'long_count': event_state['long_count'],
                    'short_count': event_state['short_count'],
                    'total_exposure_pct': event_state['total_exposure_pct'],
                    'weighted_avg_alpha_bps': event_state['weighted_avg_alpha_bps'],
                    'signals_by_type': event_state['signals_by_type'],
                    'portfolio_limits_ok': event_state['portfolio_limits']['within_sleeve_limit'],
                }
                print(f"       Events: {event_state['events_detected']}  "
                      f"Positions: {event_state['positions_sized']}  "
                      f"L/S: {event_state['long_count']}/{event_state['short_count']}  "
                      f"α: {event_state['weighted_avg_alpha_bps']:.0f}bps  "
                      f"Exposure: {event_state['total_exposure_pct']:.2%}")
            except Exception as e:
                print(f"       [ERROR] EventDrivenEngine: {e}")
                results['event_driven'] = {'error': str(e)}
        else:
            print("       [SKIP] EventDrivenEngine not available")

        # ── STEP 4: Macro Engine (GMTF + GICS + Money Velocity) ─────────────
        print("[4/10] Running Macro Engine...")
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

        # ── STEP 5: Statistical Arbitrage (Medallion layer) ──────────────────
        print("[5/10] Running Stat Arb Engine...")
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

        # ── STEP 6: Options Convexity (P4 sleeve) ───────────────────────────
        print("[6/10] Constructing Options sleeve...")
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

        # ── STEP 7: Alpha Optimizer + Beta Engine ────────────────────────────
        print("[7/10] Running Alpha + Beta engines...")
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

        # ── STEP 10: Platinum Report ─────────────────────────────────────────
        print("[10/10] Generating Platinum Report...")
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
        print(f"  Engines loaded:   {len(self._engines)}/11")
        print(f"  NAV:              ${self.nav:,.0f}")
        if cube_state:
            alloc = cube_state.get('gate_z_allocation', {}).get('allocations', {})
            print(f"  Gate-Z:")
            labels = {'P1': 'Equities', 'P2': 'Factor Rotation', 'P3': 'Commodities',
                      'P4': 'Options', 'P5': 'Hedges'}
            for s, w in alloc.items():
                print(f"    {s} ({labels.get(s, '?'):<16s}): {w:.1%} = ${self.nav * w:>14,.0f}")
        if 'distress' in results and 'error' not in results.get('distress', {}):
            ds = results['distress']
            print(f"  Special Sit:")
            print(f"    Distressed:       {ds.get('distressed_count',0)} names  "
                  f"({ds.get('opportunity_count',0)} opportunities)")
            if 'cvr' in results and 'error' not in results.get('cvr', {}):
                cv = results['cvr']
                print(f"    CVR:              {cv.get('instruments_valued',0)} instruments  "
                      f"L/S={len(cv.get('long_signals',[]))}/{len(cv.get('short_signals',[]))}")
            if 'event_driven' in results and 'error' not in results.get('event_driven', {}):
                ev = results['event_driven']
                print(f"    Event-Driven:     {ev.get('positions_sized',0)} positions  "
                      f"α={ev.get('weighted_avg_alpha_bps',0):.0f}bps  "
                      f"exposure={ev.get('total_exposure_pct',0):.2%}")
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
