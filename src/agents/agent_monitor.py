"""
Agent Performance Monitoring and Hierarchy System
Tracks 11 GICS sector agents, scores weekly, promotes/demotes.
"""
from __future__ import annotations

import json
import os
import sys
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GICS_SECTORS = [
    "Energy", "Materials", "Industrials", "ConsumerDiscretionary",
    "ConsumerStaples", "HealthCare", "Financials", "InformationTechnology",
    "CommunicationServices", "Utilities", "RealEstate",
]

HIERARCHY_LEVELS = {
    "ELITE": (80, 100),
    "STRONG": (65, 80),
    "DEVELOPING": (50, 65),
    "UNDERPERFORM": (0, 50),
}

PERFORMANCE_REPORT_FILE = "/home/user/ai-hedgefund/reports/agent_performance_report.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentSignal:
    agent_name: str
    symbol: str
    signal: str            # 'buy' | 'sell' | 'hold'
    confidence: float
    timestamp: str
    realized_outcome: Optional[float] = None  # actual return after signal
    was_correct: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WeeklyScore:
    week_ending: str
    agent_name: str
    signal_accuracy: float     # % correct directional calls
    alpha_generated: float     # excess return vs benchmark
    risk_adjusted_return: float  # Sharpe-like metric
    total_score: float         # 0-100
    num_signals: int
    num_correct: int
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentProfile:
    name: str
    sector: str
    current_tier: str
    score_history: List[float] = field(default_factory=list)
    weekly_scores: List[WeeklyScore] = field(default_factory=list)
    signals: List[AgentSignal] = field(default_factory=list)
    missed_opportunities: List[dict] = field(default_factory=list)
    wrong_signals: List[dict] = field(default_factory=list)
    promotion_count: int = 0
    demotion_count: int = 0
    memory_usage_mb: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def rolling_3w_score(self) -> float:
        scores = [ws.total_score for ws in self.weekly_scores[-3:]]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def current_score(self) -> float:
        return self.weekly_scores[-1].total_score if self.weekly_scores else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sector": self.sector,
            "current_tier": self.current_tier,
            "current_score": round(self.current_score, 1),
            "rolling_3w_score": round(self.rolling_3w_score, 1),
            "score_history": [round(s, 1) for s in self.score_history[-12:]],
            "weekly_scores": [ws.to_dict() for ws in self.weekly_scores[-4:]],
            "num_signals": len(self.signals),
            "missed_opportunities": self.missed_opportunities[-10:],
            "wrong_signals": self.wrong_signals[-10:],
            "promotion_count": self.promotion_count,
            "demotion_count": self.demotion_count,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Agent Scorer
# ---------------------------------------------------------------------------

class AgentScorer:
    """Score a single agent's weekly performance (0–100)."""

    # Score weights
    SIGNAL_ACCURACY_WEIGHT = 0.40
    ALPHA_WEIGHT = 0.35
    RISK_ADJUSTED_WEIGHT = 0.25

    # Accuracy score: 100 = 100% accurate, 50 = random
    ACCURACY_SCALE = 2.0   # accuracy → score multiplier

    def score_week(self, agent_name: str, sector: str,
                   signals: List[AgentSignal],
                   benchmark_return: float = 0.0,
                   week_ending: str = None) -> WeeklyScore:
        """Compute weekly score for an agent."""
        week_ending = week_ending or datetime.utcnow().date().isoformat()

        if not signals:
            return WeeklyScore(
                week_ending=week_ending, agent_name=agent_name,
                signal_accuracy=50.0, alpha_generated=0.0,
                risk_adjusted_return=0.0, total_score=50.0,
                num_signals=0, num_correct=0,
                errors=["No signals generated this week"]
            )

        # Signal accuracy
        graded = [s for s in signals if s.was_correct is not None]
        if graded:
            n_correct = sum(1 for s in graded if s.was_correct)
            accuracy = n_correct / len(graded) * 100
            num_correct = n_correct
        else:
            accuracy = 50.0  # assume random if no feedback
            num_correct = 0

        # Alpha generated
        outcomes = [s.realized_outcome for s in signals if s.realized_outcome is not None]
        if outcomes:
            avg_return = sum(outcomes) / len(outcomes)
            alpha = avg_return - benchmark_return
        else:
            alpha = 0.0
            avg_return = 0.0

        # Risk-adjusted return (simple: avg_return / std)
        if len(outcomes) >= 2:
            import statistics
            std = statistics.stdev(outcomes)
            ra_return = (avg_return / max(std, 1e-6)) * 100
            ra_return = max(-30, min(50, ra_return))  # clamp
        else:
            ra_return = 0.0

        # Composite score
        accuracy_score = min(100, accuracy * self.ACCURACY_SCALE)
        alpha_score = min(100, max(0, 50 + alpha * 1000))  # 1% alpha = 60 pts
        risk_score = min(100, max(0, 50 + ra_return))

        total = (accuracy_score * self.SIGNAL_ACCURACY_WEIGHT +
                 alpha_score * self.ALPHA_WEIGHT +
                 risk_score * self.RISK_ADJUSTED_WEIGHT)

        errors = []
        if accuracy < 40:
            errors.append(f"Signal accuracy below 40%: {accuracy:.1f}%")
        if alpha < -0.01:
            errors.append(f"Negative alpha: {alpha:.2%}")
        if num_correct == 0 and graded:
            errors.append("Zero correct signals this week")

        return WeeklyScore(
            week_ending=week_ending,
            agent_name=agent_name,
            signal_accuracy=round(accuracy, 1),
            alpha_generated=round(alpha * 100, 2),
            risk_adjusted_return=round(ra_return, 2),
            total_score=round(total, 1),
            num_signals=len(signals),
            num_correct=num_correct,
            errors=errors,
        )

    @staticmethod
    def determine_tier(score: float) -> str:
        for tier, (lo, hi) in HIERARCHY_LEVELS.items():
            if lo <= score < hi:
                return tier
        return "UNDERPERFORM"


# ---------------------------------------------------------------------------
# Hierarchy Manager
# ---------------------------------------------------------------------------

class HierarchyManager:
    """Manage promotion/demotion based on 3-week rolling performance."""

    def __init__(self, scorer: AgentScorer):
        self.scorer = scorer

    def evaluate_promotion(self, profile: AgentProfile) -> Tuple[bool, str, str]:
        """
        Returns (changed, old_tier, new_tier).
        Promotion/demotion based on 3-week rolling score.
        """
        rolling = profile.rolling_3w_score
        new_tier = self.scorer.determine_tier(rolling)
        old_tier = profile.current_tier

        if new_tier == old_tier:
            return False, old_tier, new_tier

        tiers_order = ["UNDERPERFORM", "DEVELOPING", "STRONG", "ELITE"]
        old_idx = tiers_order.index(old_tier) if old_tier in tiers_order else 0
        new_idx = tiers_order.index(new_tier) if new_tier in tiers_order else 0

        changed = True
        if new_idx > old_idx:
            profile.promotion_count += 1
            direction = "PROMOTED"
        else:
            profile.demotion_count += 1
            direction = "DEMOTED"

        profile.current_tier = new_tier
        print(f"[Hierarchy] {profile.name} {direction}: {old_tier} → {new_tier} (score: {rolling:.1f})")
        return changed, old_tier, new_tier

    def get_resource_allocation(self, profiles: List[AgentProfile]) -> Dict[str, float]:
        """
        Allocate compute/capital resources proportional to tier.
        ELITE=4x, STRONG=2x, DEVELOPING=1x, UNDERPERFORM=0.25x
        """
        multipliers = {"ELITE": 4.0, "STRONG": 2.0, "DEVELOPING": 1.0, "UNDERPERFORM": 0.25}
        weights = {p.name: multipliers.get(p.current_tier, 1.0) for p in profiles}
        total = sum(weights.values())
        return {name: round(w / total, 4) for name, w in weights.items()}


# ---------------------------------------------------------------------------
# Memory Monitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Track memory usage of agent operations."""

    def __init__(self):
        self._snapshots: Dict[str, float] = {}

    def start_tracking(self, agent_name: str) -> None:
        tracemalloc.start()
        self._snapshots[agent_name] = 0.0

    def stop_tracking(self, agent_name: str) -> float:
        """Returns peak memory usage in MB."""
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mb = peak / 1024 / 1024
            self._snapshots[agent_name] = mb
            return mb
        except Exception:
            tracemalloc.stop()
            return 0.0

    def get_usage(self, agent_name: str) -> float:
        return self._snapshots.get(agent_name, 0.0)

    def system_memory_mb(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Main tracker: stores signals, computes daily/weekly attribution."""

    def __init__(self):
        self.agent_signals: Dict[str, List[AgentSignal]] = {s: [] for s in GICS_SECTORS}
        self.benchmark_returns: List[float] = []

    def log_signal(self, sector: str, signal: AgentSignal) -> None:
        if sector in self.agent_signals:
            self.agent_signals[sector].append(signal)

    def update_outcomes(self, sector: str, symbol: str,
                         actual_return: float) -> None:
        """Mark outcomes for unresolved signals matching symbol."""
        for sig in self.agent_signals.get(sector, []):
            if sig.symbol == symbol and sig.realized_outcome is None:
                sig.realized_outcome = actual_return
                if sig.signal == "buy":
                    sig.was_correct = actual_return > 0
                elif sig.signal == "sell":
                    sig.was_correct = actual_return < 0
                else:
                    sig.was_correct = True  # hold = neutral

    def log_missed_opportunity(self, sector: str, symbol: str,
                                reason: str, missed_return: float) -> dict:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "reason": reason,
            "missed_return_pct": round(missed_return * 100, 2),
        }
        return entry

    def get_week_signals(self, sector: str, days: int = 7) -> List[AgentSignal]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        sigs = self.agent_signals.get(sector, [])
        return [s for s in sigs if datetime.fromisoformat(s.timestamp) >= cutoff]


# ---------------------------------------------------------------------------
# Main AgentMonitor
# ---------------------------------------------------------------------------

class AgentMonitor:
    """
    Central monitoring system for all 11 GICS sector agents.
    """

    def __init__(self, report_file: str = PERFORMANCE_REPORT_FILE):
        self.report_file = report_file
        self.scorer = AgentScorer()
        self.hierarchy = HierarchyManager(self.scorer)
        self.tracker = PerformanceTracker()
        self.memory = MemoryMonitor()
        self.profiles: Dict[str, AgentProfile] = {}
        self._init_profiles()
        self._load_report()

    def _init_profiles(self) -> None:
        for sector in GICS_SECTORS:
            if sector not in self.profiles:
                self.profiles[sector] = AgentProfile(
                    name=sector,
                    sector=sector,
                    current_tier="DEVELOPING",
                )

    def _load_report(self) -> None:
        if os.path.exists(self.report_file):
            try:
                with open(self.report_file, "r") as f:
                    data = json.load(f)
                for sector, prof_data in data.get("agents", {}).items():
                    if sector in self.profiles:
                        p = self.profiles[sector]
                        p.current_tier = prof_data.get("current_tier", "DEVELOPING")
                        p.score_history = prof_data.get("score_history", [])
                        p.promotion_count = prof_data.get("promotion_count", 0)
                        p.demotion_count = prof_data.get("demotion_count", 0)
                        p.missed_opportunities = prof_data.get("missed_opportunities", [])
                        p.wrong_signals = prof_data.get("wrong_signals", [])
            except Exception as e:
                print(f"[Monitor] Could not load report: {e}")

    def log_signal(self, sector: str, symbol: str, signal: str,
                   confidence: float) -> AgentSignal:
        sig = AgentSignal(
            agent_name=sector,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.tracker.log_signal(sector, sig)
        if sector in self.profiles:
            self.profiles[sector].signals.append(sig)
        return sig

    def update_outcome(self, sector: str, symbol: str, actual_return: float) -> None:
        self.tracker.update_outcomes(sector, symbol, actual_return)
        # Also update in profile signals
        for sig in self.profiles[sector].signals[-50:]:
            if sig.symbol == symbol and sig.realized_outcome is None:
                sig.realized_outcome = actual_return
                sig.was_correct = (actual_return > 0) if sig.signal == "buy" else \
                                   (actual_return < 0) if sig.signal == "sell" else True

    def record_missed_opportunity(self, sector: str, symbol: str,
                                   reason: str, missed_return: float) -> None:
        entry = self.tracker.log_missed_opportunity(sector, symbol, reason, missed_return)
        if sector in self.profiles:
            self.profiles[sector].missed_opportunities.append(entry)

    def record_wrong_signal(self, sector: str, symbol: str,
                             signal: str, actual_return: float) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "actual_return_pct": round(actual_return * 100, 2),
        }
        if sector in self.profiles:
            self.profiles[sector].wrong_signals.append(entry)

    def run_weekly_scoring(self, benchmark_return: float = 0.0) -> Dict[str, WeeklyScore]:
        """Score all agents for the current week."""
        week_ending = datetime.utcnow().date().isoformat()
        weekly_scores = {}

        for sector, profile in self.profiles.items():
            # Memory tracking
            self.memory.start_tracking(sector)
            week_signals = self.tracker.get_week_signals(sector, days=7)
            mem_usage = self.memory.stop_tracking(sector)
            profile.memory_usage_mb = mem_usage

            score = self.scorer.score_week(
                agent_name=sector,
                sector=sector,
                signals=week_signals,
                benchmark_return=benchmark_return,
                week_ending=week_ending,
            )
            profile.weekly_scores.append(score)
            profile.score_history.append(score.total_score)
            weekly_scores[sector] = score

            # Evaluate tier changes
            self.hierarchy.evaluate_promotion(profile)
            profile.last_updated = datetime.utcnow().isoformat()

        return weekly_scores

    def get_tier_summary(self) -> Dict[str, List[str]]:
        """Return agents grouped by tier."""
        summary: Dict[str, List[str]] = {tier: [] for tier in HIERARCHY_LEVELS}
        for sector, profile in self.profiles.items():
            tier = profile.current_tier
            if tier in summary:
                summary[tier].append(sector)
        return summary

    def get_resource_allocation(self) -> Dict[str, float]:
        return self.hierarchy.get_resource_allocation(list(self.profiles.values()))

    def generate_report(self) -> dict:
        """Generate full performance report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": "agent_performance",
            "agents": {sector: profile.to_dict() for sector, profile in self.profiles.items()},
            "tier_summary": self.get_tier_summary(),
            "resource_allocation": self.get_resource_allocation(),
            "system_memory_mb": round(self.memory.system_memory_mb(), 2),
            "top_performers": self._top_performers(3),
            "bottom_performers": self._bottom_performers(3),
        }
        return report

    def _top_performers(self, n: int) -> List[dict]:
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.rolling_3w_score,
            reverse=True
        )
        return [{"name": p.name, "score": round(p.rolling_3w_score, 1), "tier": p.current_tier}
                for p in sorted_profiles[:n]]

    def _bottom_performers(self, n: int) -> List[dict]:
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.rolling_3w_score
        )
        return [{"name": p.name, "score": round(p.rolling_3w_score, 1), "tier": p.current_tier}
                for p in sorted_profiles[:n]]

    def save_report(self) -> str:
        report = self.generate_report()
        os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
        with open(self.report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Monitor] Report saved: {self.report_file}")
        return self.report_file

    def print_dashboard(self) -> None:
        print("\n" + "="*70)
        print("  AGENT PERFORMANCE DASHBOARD")
        print("="*70)
        tier_summary = self.get_tier_summary()
        for tier in ["ELITE", "STRONG", "DEVELOPING", "UNDERPERFORM"]:
            agents = tier_summary.get(tier, [])
            marker = "***" if tier == "ELITE" else "  -"
            print(f"\n  [{tier}]")
            for agent in agents:
                p = self.profiles[agent]
                score = p.rolling_3w_score
                print(f"    {marker} {agent:<25} Score: {score:>5.1f}  "
                      f"Signals: {len(p.signals):>4}  "
                      f"Missed: {len(p.missed_opportunities):>3}")
        print("="*70 + "\n")


if __name__ == "__main__":
    monitor = AgentMonitor()

    # Simulate some signals
    import random
    for sector in GICS_SECTORS[:3]:
        for i in range(5):
            monitor.log_signal(sector, f"STOCK_{i}", "buy", random.uniform(0.5, 0.95))

    # Score and report
    scores = monitor.run_weekly_scoring(benchmark_return=0.001)
    monitor.print_dashboard()
    monitor.save_report()
