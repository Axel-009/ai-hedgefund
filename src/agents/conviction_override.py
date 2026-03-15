"""
Conviction Override — Controlled Guardrail Break
==================================================
Allows the system to exceed normal risk limits in a CONTROLLED manner
when ALL of the following conditions are met:

OVERRIDE ELIGIBILITY CRITERIA (ALL must pass):
  1. Conviction Score >= 90/100 (multi-agent consensus >= 75%)
  2. A qualified hedge is in place (or simultaneously being placed)
  3. Macro regime is NOT "CRASH" (cannot override in crash conditions)
  4. The override is for a SINGLE position (no broad leverage increase)
  5. Override positions are time-limited: auto-expire in 2 trading days
  6. Total override exposure <= 2x normal limit (hard cap)
  7. Maximum 3 simultaneous overrides active at any time
  8. Net portfolio delta remains <= BETA_MAX after override

HEDGE QUALIFICATION:
  A qualifying hedge for long equity override:
    - Long VIX calls (VXX/UVXY)
    - Long put spreads on the position or index
    - Short correlated name
    - Long TLT/SHY (duration hedge)
  For short equity override:
    - Long call spread on position
    - Long SPY calls
    - Short credit (long HYG puts)

OVERRIDE TIERS (matched to conviction level):
  CONTROLLED  (conviction 90-95%): 1.5x normal limit, 1 trading day max
  AGGRESSIVE  (conviction 95-98%): 2.0x normal limit, 2 trading day max
  MAXIMUM     (conviction >98%): 2.0x limit, requires 3+ agents in agreement

AUDIT TRAIL: Every override is logged with:
  - Timestamp, ticker, direction, conviction score, agents agreeing
  - Hedge details (what hedge was placed, size)
  - Entry price, size, override tier
  - Expiry datetime
  - Exit price, outcome (profit/loss, whether conviction was justified)

Logged to: logs/conviction_overrides/YYYYMMDD.jsonl
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERRIDE_TIER_CONTROLLED = "CONTROLLED"
OVERRIDE_TIER_AGGRESSIVE = "AGGRESSIVE"
OVERRIDE_TIER_MAXIMUM = "MAXIMUM"

# Conviction thresholds (out of 100)
CONVICTION_MIN = 90
CONVICTION_AGGRESSIVE = 95
CONVICTION_MAXIMUM = 98

# Size multipliers per tier
SIZE_MULTIPLIER = {
    OVERRIDE_TIER_CONTROLLED: 1.5,
    OVERRIDE_TIER_AGGRESSIVE: 2.0,
    OVERRIDE_TIER_MAXIMUM: 2.0,
}

# Max trading-day lifetime per tier (expressed as hours; 1 day ~ 6.5 h, 2 days ~ 13 h)
# We store expiry as an absolute datetime; for simplicity we use calendar hours
TRADING_HOURS_PER_DAY = 6.5
MAX_DAYS = {
    OVERRIDE_TIER_CONTROLLED: 1,
    OVERRIDE_TIER_AGGRESSIVE: 2,
    OVERRIDE_TIER_MAXIMUM: 2,
}

MAX_SIMULTANEOUS_OVERRIDES = 3
BETA_MAX = 1.5  # hard cap on net portfolio delta post-override

# Long-equity hedge instruments
LONG_EQUITY_HEDGES = {"VXX", "UVXY", "TLT", "SHY"}
# Short-equity hedge instruments
SHORT_EQUITY_HEDGES = {"SPY_CALLS", "HYG_PUTS"}

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "conviction_overrides",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log_file(dt: datetime) -> str:
    return os.path.join(LOG_DIR, f"{dt.strftime('%Y%m%d')}.jsonl")


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


def _trading_hours_to_timedelta(days: int) -> timedelta:
    """Convert trading days to a wall-clock timedelta approximation."""
    # Approximate: each trading day = 6.5 h of market time + overnight gap
    # For simplicity we use 24 h per calendar day so expiry lands on same weekday next day
    return timedelta(days=days)


def _validate_hedge(direction: str, hedge_details: dict) -> tuple[bool, str]:
    """
    Check that the hedge provided qualifies for the given trade direction.

    Parameters
    ----------
    direction     : "LONG" | "SHORT"
    hedge_details : dict with at least {"instrument": str, "size": float}

    Returns
    -------
    (valid: bool, reason: str)
    """
    instrument = hedge_details.get("instrument", "").upper()
    size = hedge_details.get("size", 0)

    if not instrument:
        return False, "No hedge instrument specified"
    if size <= 0:
        return False, "Hedge size must be positive"

    direction_upper = direction.upper()
    if direction_upper == "LONG":
        # Check for put-spread or index put (string matching)
        qualifies = (
            instrument in LONG_EQUITY_HEDGES
            or "PUT" in instrument
            or "SHORT" in instrument
        )
        if not qualifies:
            return False, (
                f"Hedge instrument '{instrument}' does not qualify for LONG override. "
                f"Valid: VXX, UVXY, TLT, SHY, put spreads, short correlated names."
            )
    elif direction_upper == "SHORT":
        qualifies = (
            instrument in SHORT_EQUITY_HEDGES
            or "CALL" in instrument
            or "HYG" in instrument
        )
        if not qualifies:
            return False, (
                f"Hedge instrument '{instrument}' does not qualify for SHORT override. "
                f"Valid: SPY calls, HYG puts, long call spreads on position."
            )
    else:
        return False, f"Unknown direction '{direction}'"

    return True, "Hedge qualifies"


# ---------------------------------------------------------------------------
# ConvictionOverride
# ---------------------------------------------------------------------------

class ConvictionOverride:
    """
    Manages the lifecycle of conviction-based risk-limit overrides.

    State is maintained in memory and synced to JSONL files in LOG_DIR.
    """

    def __init__(self):
        _ensure_dir(LOG_DIR)
        # Active overrides: {override_id: override_record}
        self._active: dict[str, dict] = {}
        self._load_active_overrides()

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _load_active_overrides(self) -> None:
        """Re-hydrate active overrides from today's log file (idempotent)."""
        today_path = _log_file(datetime.utcnow())
        for record in _read_jsonl(today_path):
            oid = record.get("override_id")
            if oid and record.get("status") == "ACTIVE":
                self._active[oid] = record

    def _persist(self, record: dict) -> None:
        ts = datetime.fromisoformat(record["created_at"])
        _append_jsonl(_log_file(ts), record)

    # ------------------------------------------------------------------
    # Eligibility evaluation
    # ------------------------------------------------------------------

    def evaluate_override(
        self,
        ticker: str,
        direction: str,
        conviction_score: float,
        agreeing_agents: list[str],
        hedge_details: dict,
        current_portfolio: dict,
    ) -> tuple[bool, str, float, str]:
        """
        Determine whether a conviction override is eligible.

        Parameters
        ----------
        ticker            : str   — target ticker symbol
        direction         : str   — "LONG" | "SHORT"
        conviction_score  : float — 0-100
        agreeing_agents   : list  — names of agents in agreement
        hedge_details     : dict  — {"instrument": str, "size": float, ...}
        current_portfolio : dict  — must contain:
                              "macro_regime": str
                              "net_delta": float
                              "total_agents": int  (total active agents for consensus %)

        Returns
        -------
        (eligible: bool, tier: str, max_size_multiplier: float, reason: str)
        """
        failures: list[str] = []

        # 1. Conviction score >= 90
        if conviction_score < CONVICTION_MIN:
            failures.append(
                f"Conviction {conviction_score:.1f} < minimum {CONVICTION_MIN}"
            )

        # 2. Multi-agent consensus >= 75%
        total_agents = current_portfolio.get("total_agents", len(agreeing_agents))
        consensus_pct = (len(agreeing_agents) / total_agents * 100) if total_agents > 0 else 0
        if consensus_pct < 75:
            failures.append(
                f"Agent consensus {consensus_pct:.1f}% < required 75% "
                f"({len(agreeing_agents)}/{total_agents} agents)"
            )

        # 3. Macro regime is not CRASH
        macro_regime = current_portfolio.get("macro_regime", "").upper()
        if macro_regime == "CRASH":
            failures.append("Macro regime is CRASH — override prohibited")

        # 4. Hedge qualification
        hedge_valid, hedge_reason = _validate_hedge(direction, hedge_details)
        if not hedge_valid:
            failures.append(f"Hedge invalid: {hedge_reason}")

        # 5. Maximum 3 simultaneous overrides
        active_count = len(self._active)
        if active_count >= MAX_SIMULTANEOUS_OVERRIDES:
            failures.append(
                f"Maximum simultaneous overrides reached ({active_count}/{MAX_SIMULTANEOUS_OVERRIDES})"
            )

        # 6. Net portfolio delta <= BETA_MAX after override
        net_delta = current_portfolio.get("net_delta", 0.0)
        if abs(net_delta) > BETA_MAX:
            failures.append(
                f"Net portfolio delta {net_delta:.2f} already exceeds BETA_MAX {BETA_MAX}"
            )

        if failures:
            return False, "", 0.0, "; ".join(failures)

        # Determine tier
        if conviction_score > CONVICTION_MAXIMUM:
            if len(agreeing_agents) < 3:
                return (
                    False, "", 0.0,
                    f"MAXIMUM tier requires >= 3 agents in agreement "
                    f"(got {len(agreeing_agents)})"
                )
            tier = OVERRIDE_TIER_MAXIMUM
        elif conviction_score >= CONVICTION_AGGRESSIVE:
            tier = OVERRIDE_TIER_AGGRESSIVE
        else:
            tier = OVERRIDE_TIER_CONTROLLED

        multiplier = SIZE_MULTIPLIER[tier]
        return (
            True,
            tier,
            multiplier,
            f"Override eligible: {tier} tier, {multiplier}x size, "
            f"conviction={conviction_score:.1f}, consensus={consensus_pct:.1f}%",
        )

    # ------------------------------------------------------------------
    # Applying an override
    # ------------------------------------------------------------------

    def apply_override(
        self,
        ticker: str,
        direction: str,
        size: float,
        tier: str,
        hedge_details: dict,
        conviction_score: float = 0.0,
        agreeing_agents: list[str] | None = None,
        entry_price: float = 0.0,
    ) -> str:
        """
        Record and activate a conviction override.

        Parameters
        ----------
        ticker           : str
        direction        : str    — "LONG" | "SHORT"
        size             : float  — position size (in dollars or units)
        tier             : str    — CONTROLLED | AGGRESSIVE | MAXIMUM
        hedge_details    : dict
        conviction_score : float
        agreeing_agents  : list[str]
        entry_price      : float

        Returns
        -------
        override_id : str  (UUID)
        """
        now = datetime.utcnow()
        override_id = str(uuid.uuid4())
        max_days = MAX_DAYS.get(tier, 2)
        expiry = now + _trading_hours_to_timedelta(max_days)

        record = {
            "override_id": override_id,
            "status": "ACTIVE",
            "created_at": now.isoformat(),
            "expiry_at": expiry.isoformat(),
            "ticker": ticker,
            "direction": direction.upper(),
            "size": size,
            "tier": tier,
            "max_size_multiplier": SIZE_MULTIPLIER.get(tier, 1.0),
            "conviction_score": conviction_score,
            "agreeing_agents": agreeing_agents or [],
            "hedge_details": hedge_details,
            "entry_price": entry_price,
            "exit_price": None,
            "pnl": None,
            "outcome_justified": None,
            "closed_at": None,
        }

        self._active[override_id] = record
        self._persist(record)
        return override_id

    # ------------------------------------------------------------------
    # Expiry management
    # ------------------------------------------------------------------

    def check_expiry(self) -> list[dict]:
        """
        Identify overrides that have passed their expiry time.

        Returns
        -------
        list of expired override records (status updated to EXPIRED in memory)
        """
        now = datetime.utcnow()
        expired = []
        for oid, record in list(self._active.items()):
            expiry = datetime.fromisoformat(record["expiry_at"])
            if now >= expiry:
                record["status"] = "EXPIRED"
                record["closed_at"] = now.isoformat()
                expired.append(record)
                del self._active[oid]
                self._persist(record)
        return expired

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        override_id: str,
        exit_price: float,
        pnl: float,
    ) -> dict:
        """
        Record the exit price and P&L for a closed override.

        Parameters
        ----------
        override_id : str
        exit_price  : float
        pnl         : float  — profit (positive) or loss (negative)

        Returns
        -------
        The updated override record.
        """
        now = datetime.utcnow()

        # Check active first
        record = self._active.pop(override_id, None)

        # If not active, search today's log
        if record is None:
            today_path = _log_file(now)
            for r in _read_jsonl(today_path):
                if r.get("override_id") == override_id:
                    record = r
                    break

        if record is None:
            raise ValueError(f"Override '{override_id}' not found")

        record["exit_price"] = exit_price
        record["pnl"] = pnl
        record["closed_at"] = now.isoformat()
        record["status"] = "CLOSED"
        # Conviction justified if pnl > 0 for the direction traded
        record["outcome_justified"] = pnl > 0
        self._persist(record)
        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_overrides(self) -> list[dict]:
        """Return a list of currently active override records."""
        # Refresh expiry before returning
        self.check_expiry()
        return list(self._active.values())

    # ------------------------------------------------------------------
    # Audit report
    # ------------------------------------------------------------------

    def audit_report(self, days_back: int = 30) -> str:
        """
        Return a formatted audit trail of all overrides over the past N days.

        Parameters
        ----------
        days_back : int — number of calendar days to look back (default 30)

        Returns
        -------
        Formatted string report.
        """
        now = datetime.utcnow()
        records: list[dict] = []

        for i in range(days_back + 1):
            day = now - timedelta(days=i)
            path = _log_file(day)
            records.extend(_read_jsonl(path))

        # De-duplicate by override_id, keeping the latest version of each
        seen: dict[str, dict] = {}
        for r in records:
            oid = r.get("override_id", "")
            seen[oid] = r  # later entries (CLOSED/EXPIRED) overwrite earlier ones
        records = list(seen.values())
        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        lines = [
            "=" * 72,
            f"CONVICTION OVERRIDE AUDIT REPORT — Last {days_back} days",
            f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 72,
        ]

        if not records:
            lines.append("  No override records found.")
        else:
            totals = {"ACTIVE": 0, "CLOSED": 0, "EXPIRED": 0}
            total_pnl = 0.0
            justified = 0
            unjustified = 0

            for r in records:
                status = r.get("status", "UNKNOWN")
                totals[status] = totals.get(status, 0) + 1
                pnl = r.get("pnl")
                if pnl is not None:
                    total_pnl += pnl
                if r.get("outcome_justified") is True:
                    justified += 1
                elif r.get("outcome_justified") is False:
                    unjustified += 1

                lines.append("")
                lines.append(f"  ID      : {r.get('override_id', 'N/A')}")
                lines.append(f"  Status  : {status}")
                lines.append(f"  Ticker  : {r.get('ticker')}  Dir: {r.get('direction')}  Tier: {r.get('tier')}")
                lines.append(f"  Created : {r.get('created_at')}  Expiry: {r.get('expiry_at')}")
                lines.append(f"  Conviction: {r.get('conviction_score', 'N/A')}  "
                             f"Agents: {', '.join(r.get('agreeing_agents', []))}")
                lines.append(f"  Hedge   : {r.get('hedge_details', {})}")
                lines.append(f"  Entry   : {r.get('entry_price', 'N/A')}  "
                             f"Exit: {r.get('exit_price', 'N/A')}  "
                             f"PnL: {r.get('pnl', 'N/A')}")
                lines.append(f"  Justified: {r.get('outcome_justified', 'pending')}")
                lines.append("  " + "-" * 68)

            lines.append("")
            lines.append("=" * 72)
            lines.append("SUMMARY")
            lines.append(f"  Total overrides : {len(records)}")
            for k, v in totals.items():
                lines.append(f"  {k:<10} : {v}")
            lines.append(f"  Total P&L       : {total_pnl:+.2f}")
            closed = justified + unjustified
            if closed > 0:
                lines.append(f"  Conviction justified: {justified}/{closed} ({100*justified/closed:.1f}%)")
            lines.append("=" * 72)

        return "\n".join(lines)
