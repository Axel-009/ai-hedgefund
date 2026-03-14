"""
Portfolio Analytics and Attribution Report
Part 1: Cube Scenario Engine (A–K)
Part 2: Strategy Control Panel
Part 3: Analytics & Performance Deep Dive
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any

REPORT_DIR = "/home/user/ai-hedgefund/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part 1: Cube Scenario Engine
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    label: str                      # A through K
    name: str
    description: str
    probability: float              # 0–1
    spy_return: float               # expected 1-week SPY move
    vix_target: float
    rate_move_bps: float            # 10Y yield change
    recommended_posture: str
    key_trades: List[str]
    risk_level: str                 # LOW | MEDIUM | HIGH | EXTREME

    def to_dict(self) -> dict:
        return asdict(self)

    def format_text(self) -> str:
        pct_prob = self.probability * 100
        return (
            f"  Scenario {self.label}: {self.name}\n"
            f"    Prob: {pct_prob:.0f}% | Risk: {self.risk_level}\n"
            f"    SPY: {self.spy_return:+.1%} | VIX target: {self.vix_target:.0f} | Rates: {self.rate_move_bps:+.0f}bps\n"
            f"    Posture: {self.recommended_posture}\n"
            f"    Key Trades: {' | '.join(self.key_trades)}\n"
        )


SCENARIOS: List[Scenario] = [
    Scenario("A", "Goldilocks Melt-Up",
             "Inflation falling, rates cut, soft landing confirmed — risk assets surge",
             0.15, 0.08, 13.0, -25,
             "MAXIMUM LONG — options and leveraged equity",
             ["NVDA calls", "QQQ 3x", "SPY ITM calls", "Small cap rotation"],
             "LOW"),
    Scenario("B", "Momentum Continuation",
             "Status quo: markets grind higher on AI/tech earnings beat",
             0.25, 0.04, 17.0, -5,
             "LONG with hedges — stay the course",
             ["NVDA/MSFT calls", "Tech ETF", "Momentum longs"],
             "LOW"),
    Scenario("C", "Choppy Consolidation",
             "Market stalls after run-up, range-bound sideways chop",
             0.20, 0.01, 19.0, +10,
             "NEUTRAL — sell premium, iron condors",
             ["Covered calls", "Iron condors on SPY", "Reduce size"],
             "MEDIUM"),
    Scenario("D", "Rotation to Value",
             "Tech rolls over, value/cyclicals outperform",
             0.10, -0.01, 20.0, +15,
             "ROTATE — reduce growth, add value/energy",
             ["XLE calls", "XLF long", "QQQ puts hedge", "Short ARKK"],
             "MEDIUM"),
    Scenario("E", "Macro Shock — Fed Hawkish",
             "CPI surprise forces Fed to signal more hikes; risk-off",
             0.08, -0.05, 25.0, +30,
             "DEFENSIVE — reduce equity, buy puts",
             ["SPY puts", "TLT puts", "VIX calls", "GLD calls"],
             "HIGH"),
    Scenario("F", "Geopolitical Escalation",
             "Military conflict escalates, oil spikes, global risk-off",
             0.05, -0.07, 30.0, -20,
             "HEDGE MAXIMUM — flight to safety",
             ["VIX calls", "GLD", "USO", "XLE", "Sell equities"],
             "HIGH"),
    Scenario("G", "Credit Event / Contagion",
             "Major bank/corp default, credit spreads blow out",
             0.04, -0.10, 40.0, -50,
             "PANIC HEDGE — max short, buy vol",
             ["VIX LEAPS", "XLF puts", "HYG puts", "Raise 50% cash"],
             "EXTREME"),
    Scenario("H", "Flash Crash / Liquidity Crisis",
             "Algorithmic cascades, market drops 10%+ intraday",
             0.02, -0.12, 55.0, -80,
             "BUY THE DIP if fundamentals intact",
             ["SPY puts (pre-positioned)", "Cash ready to deploy", "TQQQ buy on dip"],
             "EXTREME"),
    Scenario("I", "Earnings Disaster",
             "NVDA/MSFT massively disappoint; AI narrative collapses",
             0.04, -0.06, 32.0, +5,
             "SHORT TECH — pivot to value",
             ["NVDA puts", "QQQ puts", "XLK short", "Rotate to XLV/XLP"],
             "HIGH"),
    Scenario("J", "China/Taiwan Escalation",
             "Taiwan Strait tensions spike; semiconductor supply chain at risk",
             0.03, -0.08, 35.0, -30,
             "REDUCE SEMIS — hedge with defense/energy",
             ["SMH puts", "AMAT short", "LMT long", "XLE calls"],
             "HIGH"),
    Scenario("K", "Slow Grind Bear Market",
             "Soft landing fails, recession confirmed, markets slowly bleed",
             0.04, -0.15, 35.0, -60,
             "BEAR MARKET MODE — inverse ETFs, put spreads",
             ["SH/SDS positions", "Put spreads SPY", "Short cyclicals", "GLD LEAPS"],
             "EXTREME"),
]


class CubeScenarioEngine:
    """
    Analyze current market conditions and assign probabilities
    to each of the 11 scenarios (A–K).
    """

    def __init__(self, scenarios: List[Scenario] = None):
        self.scenarios = scenarios or SCENARIOS

    def update_probabilities(self, market_data: dict) -> None:
        """Adjust scenario probabilities based on real-time market data."""
        vix = market_data.get("vix", 18)
        spy_ret = market_data.get("spy_1d_ret", 0)
        curve_spread = market_data.get("curve_spread", 0.1)

        # Heuristic adjustments
        for s in self.scenarios:
            if s.label == "A" and vix < 15 and spy_ret > 0.02:
                s.probability = min(0.30, s.probability * 1.5)
            elif s.label in ("G", "H") and vix > 30:
                s.probability = min(0.15, s.probability * 2)
            elif s.label == "E" and curve_spread < -0.2:
                s.probability = min(0.20, s.probability * 1.8)

        # Renormalize
        total = sum(s.probability for s in self.scenarios)
        for s in self.scenarios:
            s.probability = s.probability / total

    def get_base_case(self) -> Scenario:
        return max(self.scenarios, key=lambda s: s.probability)

    def get_top3(self) -> List[Scenario]:
        return sorted(self.scenarios, key=lambda s: s.probability, reverse=True)[:3]

    def expected_spy_return(self) -> float:
        return sum(s.probability * s.spy_return for s in self.scenarios)

    def expected_vix(self) -> float:
        return sum(s.probability * s.vix_target for s in self.scenarios)

    def risk_weighted_posture(self) -> str:
        high_risk_prob = sum(s.probability for s in self.scenarios
                             if s.risk_level in ("HIGH", "EXTREME"))
        if high_risk_prob > 0.25:
            return "DEFENSIVE"
        elif high_risk_prob > 0.10:
            return "BALANCED"
        else:
            return "OFFENSIVE"

    def format_report(self) -> str:
        lines = ["\n" + "="*70, "  CUBE SCENARIO ENGINE — 11 SCENARIOS (A through K)", "="*70]
        base = self.get_base_case()
        lines.append(f"\n  BASE CASE: Scenario {base.label} — {base.name} ({base.probability:.0%})")
        lines.append(f"  Expected SPY (1W): {self.expected_spy_return():+.2%}")
        lines.append(f"  Expected VIX: {self.expected_vix():.1f}")
        lines.append(f"  Risk-Weighted Posture: {self.risk_weighted_posture()}")
        lines.append("\n  All Scenarios:")
        for s in sorted(self.scenarios, key=lambda x: x.probability, reverse=True):
            lines.append(s.format_text())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "base_case": self.get_base_case().label,
            "expected_spy_return": round(self.expected_spy_return(), 4),
            "expected_vix": round(self.expected_vix(), 1),
            "posture": self.risk_weighted_posture(),
            "top3": [s.label for s in self.get_top3()],
        }


# ---------------------------------------------------------------------------
# Part 2: Strategy Control Panel
# ---------------------------------------------------------------------------

@dataclass
class StrategySwitch:
    name: str
    enabled: bool
    allocation_pct: float
    description: str
    performance_7d: float
    sharpe_7d: float

    def to_dict(self) -> dict:
        return asdict(self)


class StrategyControlPanel:
    """
    Control panel for all active strategies.
    Enable/disable, resize, and monitor each strategy.
    """

    DEFAULT_STRATEGIES = [
        StrategySwitch("options_momentum", True, 30.0,
                       "Buy short-dated calls on momentum names with catalysts",
                       0.042, 1.8),
        StrategySwitch("leveraged_equity", True, 25.0,
                       "Buy 2-3x leveraged ETFs aligned with regime",
                       0.028, 1.2),
        StrategySwitch("pairs_trade", False, 10.0,
                       "Long/short pairs within sectors",
                       0.012, 0.9),
        StrategySwitch("event_driven", True, 20.0,
                       "Trade around earnings, FDA, macro events",
                       0.055, 2.1),
        StrategySwitch("mean_reversion", False, 5.0,
                       "Fade extreme moves using Bollinger/RSI signals",
                       -0.008, -0.4),
        StrategySwitch("tail_hedge", True, 10.0,
                       "Protective puts and VIX calls for downside protection",
                       -0.015, 0.5),
    ]

    def __init__(self, strategies: List[StrategySwitch] = None):
        self.strategies = strategies or [StrategySwitch(**{k: v for k, v in asdict(s).items()})
                                          for s in self.DEFAULT_STRATEGIES]

    def enable(self, name: str) -> None:
        for s in self.strategies:
            if s.name == name:
                s.enabled = True
                print(f"[Control] Strategy '{name}' ENABLED")
                return
        print(f"[Control] Strategy '{name}' not found")

    def disable(self, name: str) -> None:
        for s in self.strategies:
            if s.name == name:
                s.enabled = False
                print(f"[Control] Strategy '{name}' DISABLED")
                return

    def resize(self, name: str, new_pct: float) -> None:
        for s in self.strategies:
            if s.name == name:
                s.allocation_pct = max(0, min(100, new_pct))
                print(f"[Control] Strategy '{name}' resized to {new_pct:.1f}%")
                return

    def get_active_allocation(self) -> Dict[str, float]:
        active = {s.name: s.allocation_pct for s in self.strategies if s.enabled}
        total = sum(active.values())
        if total > 100:
            active = {k: v / total * 100 for k, v in active.items()}
        return active

    def best_performing(self) -> Optional[StrategySwitch]:
        enabled = [s for s in self.strategies if s.enabled]
        return max(enabled, key=lambda s: s.sharpe_7d) if enabled else None

    def worst_performing(self) -> Optional[StrategySwitch]:
        enabled = [s for s in self.strategies if s.enabled]
        return min(enabled, key=lambda s: s.sharpe_7d) if enabled else None

    def format_report(self) -> str:
        lines = ["\n" + "="*70, "  STRATEGY CONTROL PANEL", "="*70]
        lines.append(f"\n  {'Strategy':<22} {'On':<4} {'Alloc':<8} {'7D Ret':<10} {'Sharpe'}")
        lines.append("  " + "-"*60)
        for s in self.strategies:
            status = "ON " if s.enabled else "OFF"
            lines.append(
                f"  {s.name:<22} {status:<4} {s.allocation_pct:>5.1f}%  "
                f"  {s.performance_7d:>+7.2%}   {s.sharpe_7d:>5.2f}"
            )
        best = self.best_performing()
        if best:
            lines.append(f"\n  Best Sharpe: {best.name} ({best.sharpe_7d:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategies": [s.to_dict() for s in self.strategies],
            "active_allocation": self.get_active_allocation(),
            "best_strategy": self.best_performing().name if self.best_performing() else None,
            "worst_strategy": self.worst_performing().name if self.worst_performing() else None,
        }


# ---------------------------------------------------------------------------
# Part 3: Analytics & Performance Deep Dive
# ---------------------------------------------------------------------------

class PerformanceDeepDive:
    """
    Compute and display detailed performance analytics.
    """

    def __init__(self, portfolio_snapshot: dict = None):
        self.portfolio = portfolio_snapshot or {}
        self.nav_history = self.portfolio.get("daily_nav_history", [])
        self.returns = self.portfolio.get("daily_returns", [])
        self.realized = self.portfolio.get("realized_pnls", [])
        self.metrics = self.portfolio.get("metrics", {})

    def _annualize(self, daily_ret: float) -> float:
        return (1 + daily_ret) ** 252 - 1

    def return_decomposition(self) -> dict:
        if not self.returns:
            return {}
        positive = [r for r in self.returns if r > 0]
        negative = [r for r in self.returns if r < 0]
        return {
            "total_return_pct": round(sum(self.returns) * 100, 2),
            "annualized_return_pct": round(self._annualize(sum(self.returns) / max(len(self.returns), 1)) * 100, 2),
            "positive_days": len(positive),
            "negative_days": len(negative),
            "avg_up_day_pct": round(sum(positive) / max(len(positive), 1) * 100, 2),
            "avg_down_day_pct": round(sum(negative) / max(len(negative), 1) * 100, 2),
            "best_day_pct": round(max(self.returns) * 100, 2) if self.returns else 0,
            "worst_day_pct": round(min(self.returns) * 100, 2) if self.returns else 0,
        }

    def streak_analysis(self) -> dict:
        if not self.returns:
            return {}
        max_win_streak = 0
        max_lose_streak = 0
        current_win = 0
        current_lose = 0
        for r in self.returns:
            if r > 0:
                current_win += 1
                current_lose = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_lose += 1
                current_win = 0
                max_lose_streak = max(max_lose_streak, current_lose)
        return {
            "max_winning_streak": max_win_streak,
            "max_losing_streak": max_lose_streak,
            "current_streak": current_win if self.returns[-1] > 0 else -current_lose,
        }

    def trade_analytics(self) -> dict:
        if not self.realized:
            return {}
        winners = [p for p in self.realized if p > 0]
        losers = [p for p in self.realized if p <= 0]
        avg_win = sum(winners) / max(len(winners), 1)
        avg_loss = abs(sum(losers) / max(len(losers), 1))
        return {
            "total_trades": len(self.realized),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(len(winners) / max(len(self.realized), 1) * 100, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(sum(winners) / max(abs(sum(losers)), 1e-9), 2),
            "expectancy": round(sum(self.realized) / max(len(self.realized), 1), 2),
            "total_realized_pnl": round(sum(self.realized), 2),
        }

    def risk_metrics(self) -> dict:
        if not self.returns:
            return {}
        import math
        mean_ret = sum(self.returns) / len(self.returns)
        variance = sum((r - mean_ret) ** 2 for r in self.returns) / max(len(self.returns) - 1, 1)
        std_ret = variance ** 0.5
        downside_rets = [r for r in self.returns if r < 0]
        down_std = (sum(r**2 for r in downside_rets) / max(len(downside_rets), 1)) ** 0.5
        sharpe = (mean_ret / max(std_ret, 1e-9)) * (252 ** 0.5)
        sortino = (mean_ret / max(down_std, 1e-9)) * (252 ** 0.5)

        # Max drawdown
        peak = 1.0
        nav = 1.0
        max_dd = 0.0
        for r in self.returns:
            nav *= (1 + r)
            peak = max(peak, nav)
            max_dd = max(max_dd, (peak - nav) / peak)

        calmar = ((1 + mean_ret) ** 252 - 1) / max(max_dd, 1e-9)

        return {
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar, 3),
            "daily_vol_pct": round(std_ret * 100, 3),
            "annualized_vol_pct": round(std_ret * (252 ** 0.5) * 100, 2),
            "var_95_pct": round(sorted(self.returns)[int(len(self.returns) * 0.05)] * 100, 2) if len(self.returns) >= 20 else 0,
        }

    def compound_path_analysis(self) -> dict:
        """Compare actual NAV vs $1K → $1M target path."""
        from portfolio.paper_portfolio import COMPOUND_TABLE
        nav = self.portfolio.get("nav", 1000)
        day = self.portfolio.get("day_number", 1)
        target = COMPOUND_TABLE.get(day, 1000)
        return {
            "current_day": day,
            "current_nav": round(nav, 2),
            "target_nav": round(target, 2),
            "gap": round(nav - target, 2),
            "gap_pct": round((nav - target) / max(target, 1) * 100, 2),
            "on_track": nav >= target * 0.90,
            "days_remaining": 100 - day,
            "required_daily_rate_pct": round(
                ((1_000_000 / max(nav, 1)) ** (1 / max(100 - day, 1)) - 1) * 100, 2
            ) if day < 100 else 0,
        }

    def format_report(self) -> str:
        lines = ["\n" + "="*70, "  ANALYTICS & PERFORMANCE DEEP DIVE", "="*70]

        ret_decomp = self.return_decomposition()
        if ret_decomp:
            lines.append("\n  Return Decomposition:")
            for k, v in ret_decomp.items():
                lines.append(f"    {k:<30} {v}")

        streaks = self.streak_analysis()
        if streaks:
            lines.append("\n  Streak Analysis:")
            for k, v in streaks.items():
                lines.append(f"    {k:<30} {v}")

        trades = self.trade_analytics()
        if trades:
            lines.append("\n  Trade Analytics:")
            for k, v in trades.items():
                lines.append(f"    {k:<30} {v}")

        risk = self.risk_metrics()
        if risk:
            lines.append("\n  Risk Metrics:")
            for k, v in risk.items():
                lines.append(f"    {k:<30} {v}")

        try:
            path = self.compound_path_analysis()
            lines.append("\n  Compound Path Analysis ($1K → $1M):")
            for k, v in path.items():
                lines.append(f"    {k:<30} {v}")
        except Exception:
            pass

        return "\n".join(lines)

    def to_dict(self) -> dict:
        result = {
            "return_decomposition": self.return_decomposition(),
            "streak_analysis": self.streak_analysis(),
            "trade_analytics": self.trade_analytics(),
            "risk_metrics": self.risk_metrics(),
        }
        try:
            result["compound_path"] = self.compound_path_analysis()
        except Exception:
            result["compound_path"] = {}
        return result


# ---------------------------------------------------------------------------
# Main Report Class
# ---------------------------------------------------------------------------

class PortfolioAnalyticsReport:
    """
    Generate full portfolio analytics report.
    Combines Part 1 (Cube), Part 2 (Control Panel), Part 3 (Deep Dive).
    """

    def __init__(self, portfolio_snapshot: dict = None,
                 market_data: dict = None):
        self.portfolio_snapshot = portfolio_snapshot or {}
        self.market_data = market_data or {
            "vix": 18.5, "spy_1d_ret": 0.003, "curve_spread": -0.5
        }
        self.cube_engine = CubeScenarioEngine()
        self.control_panel = StrategyControlPanel()
        self.deep_dive = PerformanceDeepDive(portfolio_snapshot)
        self.generated_at = datetime.utcnow().isoformat()

    def generate(self) -> dict:
        self.cube_engine.update_probabilities(self.market_data)
        return {
            "report_type": "portfolio_analytics",
            "generated_at": self.generated_at,
            "date": date.today().isoformat(),
            "part1_cube_scenarios": self.cube_engine.to_dict(),
            "part2_strategy_control": self.control_panel.to_dict(),
            "part3_performance_deep_dive": self.deep_dive.to_dict(),
        }

    def save_report(self) -> str:
        data = self.generate()
        filename = f"portfolio_analytics_{date.today().isoformat()}.json"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Analytics] Report saved: {filepath}")
        return filepath

    def print_full_report(self) -> None:
        self.cube_engine.update_probabilities(self.market_data)
        print(self.cube_engine.format_report())
        print(self.control_panel.format_report())
        print(self.deep_dive.format_report())

    def get_base_scenario(self) -> Scenario:
        return self.cube_engine.get_base_case()

    def get_recommended_posture(self) -> str:
        return self.cube_engine.risk_weighted_posture()


if __name__ == "__main__":
    report = PortfolioAnalyticsReport()
    report.print_full_report()
    report.save_report()
