"""
Agent Scorecard & Hierarchy System
====================================
Tracks performance of all 18+ agents + the system engines.
Produces weekly scores and promotes/demotes agents in hierarchy.

Agent categories:
  TIER_1 (Generals)    — consistently >Sharpe 1.5, correct >60% of signals
  TIER_2 (Captains)    — Sharpe 1.0-1.5, correct 50-60%
  TIER_3 (Lieutenants) — Sharpe 0.5-1.0, correct 40-50%
  TIER_4 (Recruits)    — Sharpe <0.5 or correct <40%

Promotion: 4 consecutive weeks as top performer in tier → promoted
Demotion:  2 consecutive weeks as bottom performer in tier → demoted

Metrics tracked per agent:
  - Signal accuracy: % of signals where direction was correct
  - Conviction accuracy: when HIGH/EXTREME conviction flagged, was it right?
  - Sharpe contribution: P&L attributed to agent signals / annualized vol
  - Hit rate: wins / (wins + losses)
  - Avg return per signal
  - False positive rate
  - Sector accuracy breakdown
  - Weekly score: composite (40% signal accuracy, 30% Sharpe, 30% hit rate)

All 18 agents to track:
  Investor personas: aswath_damodaran, ben_graham, bill_ackman, cathie_wood,
    charlie_munger, michael_burry, mohnish_pabrai, peter_lynch, phil_fisher,
    rakesh_jhunjhunwala, stanley_druckenmiller, warren_buffett
  Analytical: valuation, sentiment, fundamentals, technicals, risk_manager, portfolio_manager
  System engines: metadron_cube, macro_engine, stat_arb, options_engine,
    execution_engine, pattern_recognition, contagion_engine

Saved to: logs/agent_scorecard/YYYYWW.json (weekly)
         logs/agent_scorecard/leaderboard.json (live rankings)
"""

import json
import os
import math
from datetime import datetime, date, timedelta
from typing import Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_AGENTS = [
    # Investor personas
    "aswath_damodaran", "ben_graham", "bill_ackman", "cathie_wood",
    "charlie_munger", "michael_burry", "mohnish_pabrai", "peter_lynch",
    "phil_fisher", "rakesh_jhunjhunwala", "stanley_druckenmiller", "warren_buffett",
    # Analytical
    "valuation", "sentiment", "fundamentals", "technicals", "risk_manager", "portfolio_manager",
    # System engines
    "metadron_cube", "macro_engine", "stat_arb", "options_engine",
    "execution_engine", "pattern_recognition", "contagion_engine",
]

TIER_1 = "TIER_1"
TIER_2 = "TIER_2"
TIER_3 = "TIER_3"
TIER_4 = "TIER_4"

TIER_LABELS = {
    TIER_1: "GENERALS",
    TIER_2: "CAPTAINS",
    TIER_3: "LIEUTENANTS",
    TIER_4: "RECRUITS",
}

TIER_MEDAL = {
    TIER_1: "🥇",
    TIER_2: "🥈",
    TIER_3: "🥉",
    TIER_4: "  ",
}

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "agent_scorecard",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _week_key(dt: date) -> str:
    """Return ISO-8601 year+week string, e.g. '202611'."""
    iso = dt.isocalendar()
    return f"{iso[0]}{iso[1]:02d}"


def _sharpe(returns: list[float]) -> float:
    """Annualised Sharpe from a list of per-signal returns (simple estimate)."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    # Annualise assuming ~252 trading-day signals per year
    return (mean / std) * math.sqrt(252)


def _classify_tier(sharpe: float, signal_accuracy: float) -> str:
    if sharpe >= 1.5 and signal_accuracy >= 0.60:
        return TIER_1
    if sharpe >= 1.0 and signal_accuracy >= 0.50:
        return TIER_2
    if sharpe >= 0.5 and signal_accuracy >= 0.40:
        return TIER_3
    return TIER_4


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)
    return default


def _save_json(path: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# AgentScorecard
# ---------------------------------------------------------------------------

class AgentScorecard:
    """
    Central tracking hub for agent signal performance and tier management.

    Data model (held in memory, persisted as JSON):
      signals   — list of {agent, ticker, direction, conviction, timestamp}
      outcomes  — list of {agent, ticker, actual_return, timestamp}
      tiers     — dict {agent: tier_str}
      streaks   — dict {agent: {top_streak: int, bottom_streak: int}}
      history   — dict {week_key: {agent: weekly_score_dict}}
    """

    def __init__(self):
        _ensure_dir(LOG_DIR)
        self._signals: list[dict] = []
        self._outcomes: list[dict] = []
        self._tiers: dict[str, str] = {}
        self._streaks: dict[str, dict] = {}
        self._history: dict[str, dict] = {}
        self._initialize_agents()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_agents(self) -> None:
        """Ensure every known agent has a tier record and blank streaks."""
        leaderboard_path = os.path.join(LOG_DIR, "leaderboard.json")
        saved = _load_json(leaderboard_path, {})
        for agent in ALL_AGENTS:
            self._tiers[agent] = saved.get(agent, {}).get("tier", TIER_4)
            self._streaks[agent] = saved.get(agent, {}).get(
                "streaks", {"top_streak": 0, "bottom_streak": 0}
            )

    # ------------------------------------------------------------------
    # Signal / outcome recording
    # ------------------------------------------------------------------

    def record_signal(
        self,
        agent_name: str,
        ticker: str,
        direction: str,
        conviction: str,
        timestamp: datetime | None = None,
    ) -> dict:
        """
        Log a signal emitted by an agent.

        Parameters
        ----------
        agent_name : str
        ticker     : str
        direction  : str   — "LONG" | "SHORT" | "NEUTRAL"
        conviction : str   — "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
        timestamp  : datetime (defaults to now)

        Returns
        -------
        The stored signal record.
        """
        ts = timestamp or datetime.utcnow()
        record = {
            "agent": agent_name,
            "ticker": ticker,
            "direction": direction.upper(),
            "conviction": conviction.upper(),
            "timestamp": ts.isoformat(),
        }
        self._signals.append(record)
        return record

    def record_outcome(
        self,
        agent_name: str,
        ticker: str,
        actual_return: float,
        timestamp: datetime | None = None,
    ) -> dict:
        """
        Log the actual return for a signal that was acted upon.

        Parameters
        ----------
        agent_name    : str
        ticker        : str
        actual_return : float  — e.g. 0.03 for +3%
        timestamp     : datetime (defaults to now)

        Returns
        -------
        The stored outcome record.
        """
        ts = timestamp or datetime.utcnow()
        record = {
            "agent": agent_name,
            "ticker": ticker,
            "actual_return": actual_return,
            "timestamp": ts.isoformat(),
        }
        self._outcomes.append(record)
        return record

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_weekly_score(
        self,
        agent_name: str,
        week_start: date,
        week_end: date,
    ) -> dict:
        """
        Compute the full set of metrics for *agent_name* over [week_start, week_end].

        Returns
        -------
        dict with keys:
          agent, week_start, week_end,
          signal_count, signal_accuracy, conviction_accuracy,
          hit_rate, avg_return, false_positive_rate, sharpe,
          sector_accuracy,   # always {} unless sector tags are present
          weekly_score       # composite 0-100
        """
        ws = datetime.combine(week_start, datetime.min.time())
        we = datetime.combine(week_end, datetime.max.time())

        # Signals this week for this agent
        week_signals = [
            s for s in self._signals
            if s["agent"] == agent_name
            and ws <= datetime.fromisoformat(s["timestamp"]) <= we
        ]

        # Outcomes this week for this agent (keyed by ticker for matching)
        week_outcomes_map: dict[str, list[dict]] = defaultdict(list)
        for o in self._outcomes:
            if o["agent"] == agent_name and ws <= datetime.fromisoformat(o["timestamp"]) <= we:
                week_outcomes_map[o["ticker"]].append(o)

        signal_count = len(week_signals)
        correct = 0
        high_conviction_total = 0
        high_conviction_correct = 0
        wins = 0
        losses = 0
        returns: list[float] = []
        false_positives = 0  # signal fired but return contradicted direction

        for sig in week_signals:
            ticker = sig["ticker"]
            direction = sig["direction"]
            conviction = sig["conviction"]
            matching_outcomes = week_outcomes_map.get(ticker, [])
            if not matching_outcomes:
                continue
            # Use the first matching outcome (simplification)
            outcome = matching_outcomes[0]
            ret = outcome["actual_return"]
            returns.append(ret)

            long_correct = (direction == "LONG" and ret > 0)
            short_correct = (direction == "SHORT" and ret < 0)
            neutral_correct = (direction == "NEUTRAL" and abs(ret) < 0.005)
            is_correct = long_correct or short_correct or neutral_correct

            if is_correct:
                correct += 1
                wins += 1
            else:
                losses += 1
                if conviction in ("HIGH", "EXTREME"):
                    false_positives += 1

            if conviction in ("HIGH", "EXTREME"):
                high_conviction_total += 1
                if is_correct:
                    high_conviction_correct += 1

        matched = wins + losses
        signal_accuracy = correct / matched if matched > 0 else 0.0
        conviction_accuracy = (
            high_conviction_correct / high_conviction_total
            if high_conviction_total > 0 else 0.0
        )
        hit_rate = wins / matched if matched > 0 else 0.0
        avg_return = sum(returns) / len(returns) if returns else 0.0
        false_positive_rate = false_positives / matched if matched > 0 else 0.0
        sharpe = _sharpe(returns)

        # Composite weekly score (40% accuracy, 30% Sharpe normalised, 30% hit rate)
        sharpe_norm = min(max(sharpe / 3.0, 0.0), 1.0)  # cap at Sharpe 3 → 1.0
        weekly_score = (
            0.40 * signal_accuracy * 100
            + 0.30 * sharpe_norm * 100
            + 0.30 * hit_rate * 100
        )

        return {
            "agent": agent_name,
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "signal_count": signal_count,
            "signal_accuracy": round(signal_accuracy, 4),
            "conviction_accuracy": round(conviction_accuracy, 4),
            "hit_rate": round(hit_rate, 4),
            "avg_return": round(avg_return, 6),
            "false_positive_rate": round(false_positive_rate, 4),
            "sharpe": round(sharpe, 4),
            "sector_accuracy": {},
            "weekly_score": round(weekly_score, 2),
        }

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(
        self,
        week_start: date | None = None,
        week_end: date | None = None,
    ) -> dict:
        """
        Produce the full weekly scorecard for all agents, ranked by weekly_score.

        Parameters
        ----------
        week_start : date  — defaults to last Monday
        week_end   : date  — defaults to last Sunday

        Returns
        -------
        dict with keys:
          week_start, week_end, generated_at,
          rankings: [score_dict, ...]   sorted descending by weekly_score
        """
        today = date.today()
        if week_start is None:
            # Most recent completed Monday
            week_start = today - timedelta(days=today.weekday() + 7)
        if week_end is None:
            week_end = week_start + timedelta(days=6)

        scores = []
        for agent in ALL_AGENTS:
            score = self.compute_weekly_score(agent, week_start, week_end)
            score["tier"] = self._tiers.get(agent, TIER_4)
            scores.append(score)

        scores.sort(key=lambda x: x["weekly_score"], reverse=True)

        report = {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "rankings": scores,
        }

        # Persist weekly file
        wk = _week_key(week_start)
        _save_json(os.path.join(LOG_DIR, f"{wk}.json"), report)

        # Store in history
        self._history[wk] = {s["agent"]: s for s in scores}

        return report

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def get_leaderboard(self) -> dict:
        """
        Return the current tier assignments for all agents.

        Returns
        -------
        dict: { tier_str: [agent, ...] }
        """
        board: dict[str, list] = {TIER_1: [], TIER_2: [], TIER_3: [], TIER_4: []}
        for agent, tier in self._tiers.items():
            board.setdefault(tier, []).append(agent)
        return board

    # ------------------------------------------------------------------
    # Promotions / demotions
    # ------------------------------------------------------------------

    def check_promotions_demotions(self, week_start: date | None = None) -> list[dict]:
        """
        Apply tier changes based on streak rules:
          - 4 consecutive weeks as top performer in tier → promote
          - 2 consecutive weeks as bottom performer in tier → demote

        Parameters
        ----------
        week_start : date  — week to evaluate (defaults to last Monday)

        Returns
        -------
        list of change records: {agent, old_tier, new_tier, reason}
        """
        today = date.today()
        if week_start is None:
            week_start = today - timedelta(days=today.weekday() + 7)

        wk = _week_key(week_start)
        week_scores = self._history.get(wk, {})

        # Group agents by tier and sort by score
        tier_groups: dict[str, list] = defaultdict(list)
        for agent, score_dict in week_scores.items():
            tier = self._tiers.get(agent, TIER_4)
            tier_groups[tier].append((agent, score_dict.get("weekly_score", 0.0)))

        changes = []

        for tier, agents in tier_groups.items():
            if not agents:
                continue
            agents_sorted = sorted(agents, key=lambda x: x[1], reverse=True)
            top_agent = agents_sorted[0][0]
            bottom_agent = agents_sorted[-1][0]

            # Update top streak
            if top_agent not in self._streaks:
                self._streaks[top_agent] = {"top_streak": 0, "bottom_streak": 0}
            self._streaks[top_agent]["top_streak"] += 1
            self._streaks[top_agent]["bottom_streak"] = 0

            # Update bottom streak
            if bottom_agent not in self._streaks:
                self._streaks[bottom_agent] = {"top_streak": 0, "bottom_streak": 0}
            self._streaks[bottom_agent]["bottom_streak"] += 1
            self._streaks[bottom_agent]["top_streak"] = 0

            # Promotion
            if self._streaks[top_agent]["top_streak"] >= 4:
                old_tier = tier
                new_tier = {
                    TIER_4: TIER_3,
                    TIER_3: TIER_2,
                    TIER_2: TIER_1,
                    TIER_1: TIER_1,  # already max
                }.get(tier, tier)
                if new_tier != old_tier:
                    self._tiers[top_agent] = new_tier
                    self._streaks[top_agent]["top_streak"] = 0
                    changes.append({
                        "agent": top_agent,
                        "old_tier": old_tier,
                        "new_tier": new_tier,
                        "reason": "4 consecutive weeks as top performer in tier",
                    })

            # Demotion
            if self._streaks[bottom_agent]["bottom_streak"] >= 2:
                old_tier = tier
                new_tier = {
                    TIER_1: TIER_2,
                    TIER_2: TIER_3,
                    TIER_3: TIER_4,
                    TIER_4: TIER_4,  # already min
                }.get(tier, tier)
                if new_tier != old_tier and bottom_agent != top_agent:
                    self._tiers[bottom_agent] = new_tier
                    self._streaks[bottom_agent]["bottom_streak"] = 0
                    changes.append({
                        "agent": bottom_agent,
                        "old_tier": old_tier,
                        "new_tier": new_tier,
                        "reason": "2 consecutive weeks as bottom performer in tier",
                    })

        # Recompute tiers based on latest Sharpe + accuracy where we have data
        for agent, score_dict in week_scores.items():
            sharpe = score_dict.get("sharpe", 0.0)
            acc = score_dict.get("signal_accuracy", 0.0)
            computed = _classify_tier(sharpe, acc)
            # Only override if score data is meaningful (at least 1 signal)
            if score_dict.get("signal_count", 0) > 0:
                old = self._tiers.get(agent, TIER_4)
                if computed != old:
                    # Don't log this as a change (metrics-based reclassification happens
                    # transparently; streak-based changes are the auditable events)
                    self._tiers[agent] = computed

        self._save_leaderboard()
        return changes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_leaderboard(self) -> None:
        data = {
            agent: {
                "tier": self._tiers.get(agent, TIER_4),
                "streaks": self._streaks.get(agent, {"top_streak": 0, "bottom_streak": 0}),
            }
            for agent in ALL_AGENTS
        }
        _save_json(os.path.join(LOG_DIR, "leaderboard.json"), data)

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------

    def to_formatted_report(self, week_start: date | None = None) -> str:
        """
        Return a human-readable leaderboard with ASCII box-drawing and emojis.

        Produces output such as:

        ╔══════════════════════════════════════════════════════════════╗
        ║           AGENT HIERARCHY — Week of 2026-03-09              ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ TIER 1 — GENERALS                                           ║
        ║  🥇 stanley_druckenmiller   Score: 87.3  Sharpe: 2.14      ║
        ...
        """
        today = date.today()
        if week_start is None:
            week_start = today - timedelta(days=today.weekday() + 7)

        wk = _week_key(week_start)
        week_data = self._history.get(wk, {})

        # If we have no history for this week, generate the report first
        if not week_data:
            week_end = week_start + timedelta(days=6)
            self.generate_weekly_report(week_start, week_end)
            week_data = self._history.get(wk, {})

        WIDTH = 64
        INNER = WIDTH - 2  # inside the ║ borders

        def top_border() -> str:
            return "╔" + "═" * INNER + "╗"

        def sep() -> str:
            return "╠" + "═" * INNER + "╣"

        def bottom_border() -> str:
            return "╚" + "═" * INNER + "╝"

        def row(text: str) -> str:
            padded = f" {text}"
            return "║" + padded.ljust(INNER) + "║"

        lines = [top_border()]
        title = f"AGENT HIERARCHY — Week of {week_start.strftime('%Y-%m-%d')}"
        lines.append("║" + title.center(INNER) + "║")
        lines.append(sep())

        # Build tier groups with scores
        tier_agents: dict[str, list[tuple[str, float, float]]] = {
            TIER_1: [], TIER_2: [], TIER_3: [], TIER_4: [],
        }
        for agent in ALL_AGENTS:
            tier = self._tiers.get(agent, TIER_4)
            score = week_data.get(agent, {}).get("weekly_score", 0.0)
            sharpe = week_data.get(agent, {}).get("sharpe", 0.0)
            tier_agents[tier].append((agent, score, sharpe))

        for tier in [TIER_1, TIER_2, TIER_3, TIER_4]:
            tier_num = tier.split("_")[1]
            label = TIER_LABELS[tier]
            medal = TIER_MEDAL[tier]
            lines.append(row(f"TIER {tier_num} — {label}"))

            agents_in_tier = sorted(tier_agents[tier], key=lambda x: x[1], reverse=True)
            if not agents_in_tier:
                lines.append(row("  (no agents)"))
            else:
                for agent_name, score, sharpe in agents_in_tier:
                    agent_col = f"{medal} {agent_name}"
                    score_col = f"Score: {score:5.1f}"
                    sharpe_col = f"Sharpe: {sharpe:5.2f}"
                    entry = f"  {agent_col:<32} {score_col}  {sharpe_col}"
                    lines.append(row(entry))

            if tier != TIER_4:
                lines.append(sep())

        lines.append(bottom_border())
        return "\n".join(lines)
