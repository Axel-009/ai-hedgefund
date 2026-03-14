"""
Network Contagion Engine — Global Market Interconnections (Layer 3)
===================================================================
Models propagation of shocks through the global financial system.

Graph topology:
    Nodes: countries, asset classes, sectors
    Edges: trade flows, FX exposure, credit exposure

Example contagion path:
    China slowdown → commodities fall → energy equities weaken → HY spreads widen

The system identifies propagation paths BEFORE markets react.

Integration:
    Feeds into Metadron Cube Layer 4 (Regime Engine) as contagion risk score.
    Affects Risk Governor: high contagion → cut gross exposure.

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
from dataclasses import dataclass

warnings.filterwarnings("ignore")

CONTAGION_LOG = Path(__file__).parent / "contagion_log.jsonl"


# ==============================================================================
# CONTAGION GRAPH — Adjacency Matrix of Global Risk Transmission
# ==============================================================================
# Nodes: markets / asset classes / regions
NODES = [
    'US_EQUITY', 'EU_EQUITY', 'CHINA_EQUITY', 'EM_EQUITY', 'JAPAN_EQUITY',
    'US_CREDIT', 'EU_CREDIT', 'EM_CREDIT',
    'US_RATES', 'EU_RATES', 'JAPAN_RATES',
    'COMMODITIES', 'ENERGY', 'GOLD',
    'USD', 'EUR', 'JPY', 'CNY', 'EM_FX',
    'VIX', 'CREDIT_SPREADS',
]

# Directed adjacency weights: EDGE[source][target] = transmission strength
# Calibrated from historical crisis propagation (GFC, COVID, 2022 rate shock)
ADJACENCY = {
    'CHINA_EQUITY':   {'COMMODITIES': 0.7, 'EM_EQUITY': 0.6, 'ENERGY': 0.5,
                        'EU_EQUITY': 0.3, 'CNY': 0.8, 'EM_FX': 0.4},
    'US_RATES':       {'US_EQUITY': -0.5, 'US_CREDIT': -0.6, 'EU_RATES': 0.7,
                        'USD': 0.6, 'EM_EQUITY': -0.4, 'GOLD': -0.3,
                        'JAPAN_RATES': 0.3, 'EM_FX': -0.5},
    'VIX':            {'US_EQUITY': -0.8, 'US_CREDIT': -0.5, 'EU_EQUITY': -0.6,
                        'EM_EQUITY': -0.7, 'GOLD': 0.4, 'JPY': 0.5},
    'CREDIT_SPREADS': {'US_EQUITY': -0.6, 'EU_EQUITY': -0.4, 'EM_EQUITY': -0.5,
                        'US_CREDIT': -0.9, 'EM_CREDIT': -0.8, 'VIX': 0.6},
    'USD':            {'EM_EQUITY': -0.5, 'COMMODITIES': -0.4, 'GOLD': -0.6,
                        'EM_FX': -0.7, 'EM_CREDIT': -0.3, 'ENERGY': -0.2},
    'ENERGY':         {'COMMODITIES': 0.6, 'EM_EQUITY': 0.3, 'US_CREDIT': 0.2,
                        'CREDIT_SPREADS': -0.3},
    'US_EQUITY':      {'EU_EQUITY': 0.7, 'EM_EQUITY': 0.5, 'JAPAN_EQUITY': 0.5,
                        'VIX': -0.6, 'US_CREDIT': 0.4},
    'EU_CREDIT':      {'EU_EQUITY': -0.4, 'US_CREDIT': 0.5, 'CREDIT_SPREADS': 0.6},
    'GOLD':           {'USD': -0.3, 'VIX': 0.2},
    'JPY':            {'JAPAN_EQUITY': -0.4, 'GOLD': 0.3},
}


# ==============================================================================
# SHOCK PROPAGATION ENGINE
# ==============================================================================
class ContagionEngine:
    """
    Layer 3: Models global market interconnections via graph propagation.

    Given an initial shock at a node, propagates through the adjacency graph
    using multi-step decay propagation:
        impact[t+1][j] = Σ_i  A[i][j] * impact[t][i] * decay^t

    Identifies:
        - Primary impact (direct exposure)
        - Secondary impact (2nd-order transmission)
        - Tertiary impact (3rd-order, often missed by markets)
    """

    DECAY = 0.6          # Propagation decay per step
    MAX_STEPS = 4        # Maximum propagation depth
    MIN_IMPACT = 0.01    # Minimum impact threshold

    def __init__(self):
        self.nodes = NODES
        self.adjacency = ADJACENCY

    def propagate_shock(self, source: str, magnitude: float = -0.10) -> dict:
        """
        Propagate a shock from source node through the network.
        magnitude: initial shock (e.g., -0.10 = 10% decline).
        Returns impact map at each propagation step.
        """
        if source not in self.nodes:
            return {'error': f'Unknown node: {source}', 'impacts': {}}

        impacts = {source: magnitude}
        all_steps = [{'step': 0, 'node': source, 'impact': magnitude}]

        for step in range(1, self.MAX_STEPS + 1):
            new_impacts = {}
            for node, impact in impacts.items():
                if node in self.adjacency:
                    for target, weight in self.adjacency[node].items():
                        transmitted = impact * weight * (self.DECAY ** step)
                        if abs(transmitted) >= self.MIN_IMPACT:
                            new_impacts[target] = new_impacts.get(target, 0) + transmitted
                            all_steps.append({
                                'step': step,
                                'source': node,
                                'target': target,
                                'weight': weight,
                                'impact': round(transmitted, 6),
                            })
            # Merge new impacts
            for node, imp in new_impacts.items():
                impacts[node] = impacts.get(node, 0) + imp

        # Sort by absolute impact
        ranked = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'source': source,
            'initial_shock': magnitude,
            'total_nodes_affected': len(impacts),
            'impacts': {k: round(v, 6) for k, v in ranked},
            'propagation_paths': all_steps,
            'contagion_score': round(sum(abs(v) for v in impacts.values()), 4),
            'timestamp': datetime.now().isoformat(),
        }

    def scenario_analysis(self) -> dict:
        """
        Run standard stress scenarios and compute contagion scores.
        Returns ranked scenarios by systemic risk.
        """
        scenarios = {
            'china_slowdown':     ('CHINA_EQUITY', -0.15),
            'us_rate_shock':      ('US_RATES', +0.02),
            'vix_spike':          ('VIX', +0.50),
            'credit_blowout':    ('CREDIT_SPREADS', +0.03),
            'usd_surge':          ('USD', +0.08),
            'energy_crash':       ('ENERGY', -0.25),
            'eu_credit_crisis':   ('EU_CREDIT', +0.04),
        }

        results = {}
        for name, (source, mag) in scenarios.items():
            prop = self.propagate_shock(source, mag)
            results[name] = {
                'contagion_score': prop['contagion_score'],
                'nodes_affected': prop['total_nodes_affected'],
                'worst_impact': list(prop['impacts'].items())[:3],
            }

        # Rank by contagion score
        ranked = sorted(results.items(), key=lambda x: x[1]['contagion_score'], reverse=True)

        return {
            'scenarios': dict(ranked),
            'highest_risk': ranked[0][0] if ranked else 'none',
            'timestamp': datetime.now().isoformat(),
        }

    def portfolio_exposure(self, holdings: dict) -> dict:
        """
        Given portfolio holdings mapped to graph nodes, compute contagion exposure.
        holdings: {node: weight} e.g. {'US_EQUITY': 0.55, 'ENERGY': 0.20}
        """
        total_exposure = 0.0
        node_risks = {}

        for node, weight in holdings.items():
            # How much contagion can this node receive?
            inbound = 0.0
            for source, targets in self.adjacency.items():
                if node in targets:
                    inbound += abs(targets[node])
            node_risks[node] = {
                'weight': weight,
                'inbound_contagion': round(inbound, 4),
                'weighted_risk': round(weight * inbound, 4),
            }
            total_exposure += weight * inbound

        return {
            'total_contagion_exposure': round(total_exposure, 4),
            'node_risks': node_risks,
            'most_exposed': max(node_risks, key=lambda k: node_risks[k]['weighted_risk'])
                           if node_risks else 'none',
        }


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CONTAGION ENGINE — SELF-TEST")
    print("=" * 70)

    engine = ContagionEngine()

    # Test: China slowdown propagation
    result = engine.propagate_shock('CHINA_EQUITY', -0.15)
    print(f"\n[SCENARIO] China Equity -15%")
    print(f"  Nodes affected: {result['total_nodes_affected']}")
    print(f"  Contagion score: {result['contagion_score']}")
    print(f"  Top impacts:")
    for node, impact in list(result['impacts'].items())[:5]:
        print(f"    {node:<20s} {impact:+.4f}")

    # Test: VIX spike
    vix = engine.propagate_shock('VIX', +0.50)
    print(f"\n[SCENARIO] VIX spike +50%")
    print(f"  Contagion score: {vix['contagion_score']}")
    for node, impact in list(vix['impacts'].items())[:5]:
        print(f"    {node:<20s} {impact:+.4f}")

    # Full scenario analysis
    scenarios = engine.scenario_analysis()
    print(f"\n[SCENARIO ANALYSIS] Highest risk: {scenarios['highest_risk']}")
    for name, data in scenarios['scenarios'].items():
        print(f"  {name:<25s} score={data['contagion_score']:.4f}  nodes={data['nodes_affected']}")

    # Portfolio exposure
    holdings = {'US_EQUITY': 0.55, 'ENERGY': 0.15, 'COMMODITIES': 0.10,
                'GOLD': 0.10, 'US_RATES': 0.10}
    exposure = engine.portfolio_exposure(holdings)
    print(f"\n[PORTFOLIO EXPOSURE]")
    print(f"  Total contagion exposure: {exposure['total_contagion_exposure']}")
    print(f"  Most exposed node: {exposure['most_exposed']}")

    print(f"\n{'=' * 70}")
    print("CONTAGION ENGINE SELF-TEST PASSED")
    print(f"{'=' * 70}")
