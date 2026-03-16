"""
Missed Opportunities Tracker
==============================
Post-close analysis: what did we miss?

At market close:
  1. Fetch all securities in universe that moved >3% on the day
  2. Compare to our actual positions and signals
  3. For each missed mover, diagnose WHY we missed it:
     - Was it in the universe? (if not -> add to watchlist)
     - Did any agent flag it? (if yes -> execution gap)
     - Was it blocked by risk? (risk governor veto)
     - Was the signal too weak? (conviction below threshold)
     - Was there a data issue? (stale/missing data)
     - Did the system predict wrong direction?
  4. Categorize: SYSTEM_MISS | RISK_VETO | EXECUTION_GAP | DATA_GAP | WRONG_DIRECTION | OUT_OF_UNIVERSE
  5. Track patterns: if same securities/sectors missed repeatedly -> flag for universe/model review

Logged to: logs/missed_opportunities/YYYYMMDD.jsonl
Weekly summary: logs/missed_opportunities/weekly_YYYYWW.json
"""

import json
import os
from datetime import date, datetime, timedelta
from collections import Counter, defaultdict
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOVE_THRESHOLD = 0.03  # 3% — minimum move to be considered a "mover"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "missed_opportunities",
)

PATTERN_FLAG_THRESHOLD = 2  # flag if missed >= this many times in the look-back window


# ---------------------------------------------------------------------------
# Miss categories
# ---------------------------------------------------------------------------

class MissCategory(str, Enum):
    """
    Taxonomy of why a large move was not captured as a position.

    OUT_OF_UNIVERSE : ticker was not in the tracked universe at all
    SYSTEM_MISS     : in universe, no agent flagged it — model blind spot
    RISK_VETO       : signal existed but risk governor blocked it
    EXECUTION_GAP   : signal existed, risk approved, but trade not executed
    DATA_GAP        : stale or missing data prevented signal generation
    WRONG_DIRECTION : signal fired but for the opposite direction
    """
    OUT_OF_UNIVERSE = "OUT_OF_UNIVERSE"
    SYSTEM_MISS = "SYSTEM_MISS"
    RISK_VETO = "RISK_VETO"
    EXECUTION_GAP = "EXECUTION_GAP"
    DATA_GAP = "DATA_GAP"
    WRONG_DIRECTION = "WRONG_DIRECTION"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _week_key(dt: date) -> str:
    iso = dt.isocalendar()
    return f"{iso[0]}{iso[1]:02d}"


def _daily_log_path(d: date) -> str:
    return os.path.join(LOG_DIR, f"{d.strftime('%Y%m%d')}.jsonl")


def _weekly_log_path(d: date) -> str:
    return os.path.join(LOG_DIR, f"weekly_{_week_key(d)}.json")


def _append_jsonl(path: str, record: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _save_json(path: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def _load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)
    return default


# ---------------------------------------------------------------------------
# MissedOpportunities
# ---------------------------------------------------------------------------

class MissedOpportunities:
    """
    Post-close analysis engine that identifies, categorises, and tracks
    securities that had large moves which the system failed to capture.

    Usage
    -----
    tracker = MissedOpportunities()

    # After market close, call:
    misses = tracker.analyze(
        date        = date.today(),
        universe_tickers  = ["AAPL", "NVDA", ...],
        signals_fired     = [{"ticker": "AAPL", "direction": "LONG", "conviction": 85,
                              "executed": True, ...}, ...],
        positions_held    = {"AAPL": 1000, ...},   # ticker -> notional
        risk_vetoes       = ["NVDA", ...],           # tickers blocked by risk governor
    )

    report = tracker.generate_report(date.today())
    recs   = tracker.track_patterns(weeks=4)
    """

    def __init__(self):
        _ensure_dir(LOG_DIR)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        date_: date,
        universe_tickers: list[str],
        signals_fired: list[dict],
        positions_held: dict[str, float],
        risk_vetoes: list[str],
        day_returns: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Identify missed opportunities for a given trading day.

        Parameters
        ----------
        date_            : date
        universe_tickers : list[str]  — all tickers in the tracked universe
        signals_fired    : list[dict] — each dict must contain:
                             ticker, direction ("LONG"/"SHORT"), conviction (0-100),
                             executed (bool), data_stale (bool, optional)
        positions_held   : dict[str, float]  — ticker -> notional held at close
        risk_vetoes      : list[str]  — tickers blocked by the risk governor today
        day_returns      : dict[str, float]  — ticker -> day_return (e.g. 0.05 for +5%)
                           If None, the method will return empty (no market data provided).

        Returns
        -------
        List of miss records, each containing:
          ticker, day_return, category, reason, date, details
        """
        if day_returns is None:
            day_returns = {}

        universe_set = set(t.upper() for t in universe_tickers)
        positions_set = set(t.upper() for t in positions_held)
        vetoes_set = set(t.upper() for t in risk_vetoes)

        # Index signals by ticker
        signals_by_ticker: dict[str, list[dict]] = defaultdict(list)
        for sig in signals_fired:
            signals_by_ticker[sig["ticker"].upper()].append(sig)

        # Movers: tickers that moved more than the threshold
        movers = {
            t.upper(): ret
            for t, ret in day_returns.items()
            if abs(ret) >= MOVE_THRESHOLD
        }

        misses: list[dict] = []

        for ticker, day_return in movers.items():
            # We captured this move — not a miss
            if ticker in positions_set and abs(positions_held.get(ticker, 0)) > 0:
                # Verify direction alignment
                pos_sign = positions_held[ticker]
                if (day_return > 0 and pos_sign > 0) or (day_return < 0 and pos_sign < 0):
                    continue  # correctly positioned

            category, reason, details = self.categorize_miss(
                ticker=ticker,
                day_return=day_return,
                signal_history=signals_by_ticker.get(ticker, []),
                risk_veto_log=vetoes_set,
                universe_set=universe_set,
            )

            record = {
                "date": date_.isoformat(),
                "ticker": ticker,
                "day_return": round(day_return, 6),
                "category": category.value,
                "reason": reason,
                "details": details,
                "logged_at": datetime.utcnow().isoformat(),
            }
            misses.append(record)
            _append_jsonl(_daily_log_path(date_), record)

        return misses

    # ------------------------------------------------------------------
    # Categorisation
    # ------------------------------------------------------------------

    def categorize_miss(
        self,
        ticker: str,
        day_return: float,
        signal_history: list[dict],
        risk_veto_log: set[str],
        universe_set: set[str] | None = None,
    ) -> tuple[MissCategory, str, dict]:
        """
        Determine the root cause of a missed move.

        Parameters
        ----------
        ticker         : str
        day_return     : float
        signal_history : list[dict] — signals fired for this ticker today
        risk_veto_log  : set[str]  — tickers blocked by risk governor
        universe_set   : set[str]  — tracked universe (optional)

        Returns
        -------
        (MissCategory, reason_str, details_dict)
        """
        ticker = ticker.upper()
        details: dict = {
            "ticker": ticker,
            "day_return": day_return,
            "signal_count": len(signal_history),
        }

        # 1. Not in universe
        if universe_set is not None and ticker not in universe_set:
            return (
                MissCategory.OUT_OF_UNIVERSE,
                f"{ticker} was not in the tracked universe — add to watchlist",
                {**details, "recommendation": "Add to universe"},
            )

        # 2. No signals at all — system/model blind spot
        if not signal_history:
            return (
                MissCategory.SYSTEM_MISS,
                f"No agent flagged {ticker} — model blind spot or data issue",
                {**details, "recommendation": "Review agent coverage for this ticker/sector"},
            )

        # Check if any signal had a data issue
        stale_signals = [s for s in signal_history if s.get("data_stale", False)]
        if stale_signals:
            return (
                MissCategory.DATA_GAP,
                f"Signal(s) for {ticker} were generated on stale/missing data",
                {**details, "stale_signals": len(stale_signals)},
            )

        # 3. Risk veto
        if ticker in risk_veto_log:
            best_signal = max(signal_history, key=lambda s: s.get("conviction", 0))
            return (
                MissCategory.RISK_VETO,
                f"Risk governor vetoed {ticker} (max conviction: {best_signal.get('conviction', 'N/A')})",
                {**details, "best_conviction": best_signal.get("conviction")},
            )

        # 4. Wrong direction
        expected_direction = "LONG" if day_return > 0 else "SHORT"
        wrong_dir_signals = [
            s for s in signal_history
            if s.get("direction", "").upper() != expected_direction
        ]
        correct_dir_signals = [
            s for s in signal_history
            if s.get("direction", "").upper() == expected_direction
        ]

        if wrong_dir_signals and not correct_dir_signals:
            return (
                MissCategory.WRONG_DIRECTION,
                f"Signals for {ticker} predicted {wrong_dir_signals[0].get('direction')} "
                f"but actual move was {expected_direction} ({day_return:+.1%})",
                {**details, "predicted_direction": wrong_dir_signals[0].get("direction"),
                 "actual_direction": expected_direction},
            )

        # 5. Execution gap — signal existed and approved, but not executed
        approved_unexecuted = [
            s for s in correct_dir_signals
            if not s.get("executed", False)
        ]
        if approved_unexecuted:
            best = max(approved_unexecuted, key=lambda s: s.get("conviction", 0))
            return (
                MissCategory.EXECUTION_GAP,
                f"Signal for {ticker} ({expected_direction}) existed but was not executed "
                f"(conviction: {best.get('conviction', 'N/A')})",
                {**details, "unexecuted_signals": len(approved_unexecuted),
                 "best_conviction": best.get("conviction")},
            )

        # 6. Weak conviction — signal existed, correct direction, but conviction too low
        # (The position wasn't taken, so it ended up as no position held)
        if correct_dir_signals:
            best = max(correct_dir_signals, key=lambda s: s.get("conviction", 0))
            return (
                MissCategory.EXECUTION_GAP,
                f"Signal for {ticker} ({expected_direction}) had conviction "
                f"{best.get('conviction', 'N/A')} — likely below threshold",
                {**details, "best_conviction": best.get("conviction"),
                 "note": "conviction below execution threshold"},
            )

        # Fallback
        return (
            MissCategory.SYSTEM_MISS,
            f"Could not determine specific cause for missing {ticker} move ({day_return:+.1%})",
            details,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, date_: date) -> str:
        """
        Return a formatted miss report for a given trading day.

        Parameters
        ----------
        date_ : date

        Returns
        -------
        Human-readable string report.
        """
        records = _read_jsonl(_daily_log_path(date_))

        lines = [
            "=" * 70,
            f"MISSED OPPORTUNITIES REPORT — {date_.strftime('%Y-%m-%d')}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 70,
        ]

        if not records:
            lines.append("  No missed opportunities recorded for this date.")
            lines.append("=" * 70)
            return "\n".join(lines)

        # Summary by category
        cat_counts: Counter = Counter(r["category"] for r in records)
        lines.append("")
        lines.append("SUMMARY BY CATEGORY:")
        for cat in MissCategory:
            count = cat_counts.get(cat.value, 0)
            bar = "█" * count
            lines.append(f"  {cat.value:<20} {count:>3}  {bar}")
        lines.append("")

        # Sort by absolute return descending
        records_sorted = sorted(records, key=lambda r: abs(r.get("day_return", 0)), reverse=True)

        lines.append("MISSED MOVERS:")
        lines.append("  " + "-" * 66)
        for r in records_sorted:
            ret = r.get("day_return", 0)
            ticker = r.get("ticker", "???")
            cat = r.get("category", "UNKNOWN")
            reason = r.get("reason", "")
            lines.append(f"  {ticker:<10} {ret:+.2%}  [{cat}]")
            lines.append(f"    -> {reason}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pattern tracking
    # ------------------------------------------------------------------

    def track_patterns(self, weeks: int = 4) -> list[dict]:
        """
        Scan the past N weeks of missed-opportunity logs to surface
        recurring misses, repeated sectors, and systematic gaps.

        Parameters
        ----------
        weeks : int — look-back window in weeks (default 4)

        Returns
        -------
        List of recommendation dicts:
          {ticker/sector, miss_count, categories, recommendation}
        """
        today = date.today()
        all_records: list[dict] = []

        for days_back in range(weeks * 7 + 1):
            d = today - timedelta(days=days_back)
            path = _daily_log_path(d)
            all_records.extend(_read_jsonl(path))

        if not all_records:
            return []

        # Count misses per ticker
        ticker_records: dict[str, list[dict]] = defaultdict(list)
        for r in all_records:
            ticker_records[r.get("ticker", "UNKNOWN")].append(r)

        recommendations: list[dict] = []

        for ticker, records in ticker_records.items():
            miss_count = len(records)
            if miss_count < PATTERN_FLAG_THRESHOLD:
                continue

            categories = Counter(r.get("category") for r in records)
            dominant_cat = categories.most_common(1)[0][0]

            rec_text = _recommendation_for_category(dominant_cat, ticker, miss_count)

            recommendations.append({
                "ticker": ticker,
                "miss_count": miss_count,
                "weeks_analyzed": weeks,
                "category_breakdown": dict(categories),
                "dominant_category": dominant_cat,
                "recommendation": rec_text,
                "avg_day_return": round(
                    sum(abs(r.get("day_return", 0)) for r in records) / miss_count, 4
                ),
            })

        # Sort by miss count descending
        recommendations.sort(key=lambda x: x["miss_count"], reverse=True)

        # Persist weekly pattern summary
        summary_path = _weekly_log_path(today)
        existing = _load_json(summary_path, {})
        existing.update({
            "generated_at": datetime.utcnow().isoformat(),
            "weeks_analyzed": weeks,
            "flagged_tickers": recommendations,
        })
        _save_json(summary_path, existing)

        return recommendations


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _recommendation_for_category(category: str, ticker: str, count: int) -> str:
    """Map dominant miss category to an actionable recommendation string."""
    mapping = {
        MissCategory.OUT_OF_UNIVERSE.value: (
            f"Add {ticker} to the tracked universe — missed {count}x in look-back window"
        ),
        MissCategory.SYSTEM_MISS.value: (
            f"Review model coverage for {ticker} — no agent flagged it {count}x; "
            "consider adding a sector-specific agent or data feed"
        ),
        MissCategory.RISK_VETO.value: (
            f"Risk governor repeatedly blocked {ticker} ({count}x); "
            "review veto rules or adjust position sizing to allow smaller entries"
        ),
        MissCategory.EXECUTION_GAP.value: (
            f"Execution gap for {ticker} ({count}x); "
            "check order routing and execution latency; lower conviction threshold"
        ),
        MissCategory.DATA_GAP.value: (
            f"Data quality issue for {ticker} ({count}x); "
            "audit data feed and add stale-data fallback"
        ),
        MissCategory.WRONG_DIRECTION.value: (
            f"Model consistently predicts wrong direction for {ticker} ({count}x); "
            "retrain or add contrarian signal"
        ),
    }
    return mapping.get(category, f"Investigate recurring miss for {ticker} ({count}x)")
