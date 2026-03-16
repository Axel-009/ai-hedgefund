"""
Hourly Transaction Recap
========================
Every hour during market hours (10:30, 11:30, 12:30, 1:30, 2:30, 3:30 ET):
  - List all transactions since last recap
  - Running P&L
  - Signal quality (how many signals fired, how many executed, fill rates)
  - Position changes
  - Risk metrics update (VaR, beta, gross exposure)
  - Agent activity summary
Saved to: logs/hourly_recaps/YYYYMMDD_HH00.json
"""

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[2]
_RECAP_DIR = _BASE_DIR / "logs" / "hourly_recaps"
_RECAP_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark for beta calculation
_BENCHMARK = "SPY"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class HourlyRecap:
    """Generates an intraday hourly recap snapshot."""

    def __init__(self, recap_dir: str | Path | None = None) -> None:
        self.recap_dir = Path(recap_dir) if recap_dir else _RECAP_DIR
        self.recap_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        portfolio_state: dict,
        trades_since_last: list[dict],
        signal_log: list[dict],
    ) -> dict:
        """Generate the hourly recap.

        Parameters
        ----------
        portfolio_state : dict
            Current portfolio: cash, holdings {ticker: {qty, avg_cost}},
            starting_nav, running_realized_pnl.
        trades_since_last : list[dict]
            Trades executed since the previous recap. Each entry:
            {ticker, side, qty, fill_price, signal_type, agent, timestamp,
             avg_cost (optional), fill_qty (optional, defaults to qty)}.
        signal_log : list[dict]
            All signals fired this hour. Each entry:
            {ticker, signal_type, agent, fired_at, executed (bool),
             conviction (optional float 0-1)}.

        Returns
        -------
        dict
            Structured recap. Also saved to logs/hourly_recaps/YYYYMMDD_HH00.json.
        """
        now = datetime.utcnow()
        date_str = date.today().strftime("%Y%m%d")
        hour_str = now.strftime("%H00")

        recap: dict[str, Any] = {
            "recap_type": "HOURLY",
            "date": date.today().isoformat(),
            "hour_utc": now.strftime("%H:%M UTC"),
            "generated_at": now.isoformat() + "Z",
            "transactions": self._summarize_transactions(trades_since_last),
            "running_pnl": self._running_pnl(portfolio_state, trades_since_last),
            "signal_quality": self._signal_quality(signal_log),
            "position_changes": self._position_changes(trades_since_last),
            "risk_metrics": self._risk_metrics(portfolio_state),
            "agent_activity": self._agent_activity(trades_since_last, signal_log),
            "best_signal_of_hour": self._best_signal(signal_log, trades_since_last),
        }

        self._save(recap, date_str, hour_str)
        return recap

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _summarize_transactions(self, trades: list[dict]) -> list[dict]:
        """Return a cleaned list of transaction records."""
        out = []
        for t in trades:
            out.append(
                {
                    "ticker": str(t.get("ticker", "")),
                    "side": str(t.get("side", "")).upper(),
                    "qty": _safe_float(t.get("qty", 0)),
                    "fill_price": _safe_float(t.get("fill_price", 0)),
                    "fill_qty": _safe_float(t.get("fill_qty", t.get("qty", 0))),
                    "signal_type": str(t.get("signal_type", "unknown")),
                    "agent": str(t.get("agent", "unknown")),
                    "timestamp": str(t.get("timestamp", "")),
                    "notional": round(
                        _safe_float(t.get("fill_qty", t.get("qty", 0))) * _safe_float(t.get("fill_price", 0)), 2
                    ),
                }
            )
        return out

    def _running_pnl(self, portfolio_state: dict, trades: list[dict]) -> dict:
        """Compute running realized P&L from trades and estimate unrealized from holdings."""
        realized_hour = 0.0
        for t in trades:
            side = str(t.get("side", "")).lower()
            qty = _safe_float(t.get("qty", 0))
            fill = _safe_float(t.get("fill_price", 0))
            avg_cost = _safe_float(t.get("avg_cost", fill))
            if side == "sell":
                realized_hour += (fill - avg_cost) * qty

        realized_cumulative = _safe_float(portfolio_state.get("running_realized_pnl", 0)) + realized_hour

        # Unrealized: fetch live prices for held positions
        holdings = portfolio_state.get("holdings", {})
        tickers_held = [tk for tk, v in holdings.items() if _safe_float(v.get("qty", 0)) != 0]
        unrealized = 0.0
        if tickers_held:
            try:
                data = yf.download(tickers_held, period="1d", interval="1m", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if not close.empty:
                    last_prices = close.iloc[-1]
                    for ticker, info in holdings.items():
                        qty = _safe_float(info.get("qty", 0))
                        avg_cost = _safe_float(info.get("avg_cost", 0))
                        if ticker in last_prices.index:
                            unrealized += (_safe_float(last_prices[ticker]) - avg_cost) * qty
            except Exception as exc:
                logger.warning("_running_pnl unrealized fetch: %s", exc)

        starting_nav = _safe_float(portfolio_state.get("starting_nav", 1))
        total = realized_cumulative + unrealized
        return {
            "realized_this_hour": round(realized_hour, 2),
            "realized_cumulative": round(realized_cumulative, 2),
            "unrealized": round(unrealized, 2),
            "total_pnl": round(total, 2),
            "total_pnl_pct": round((total / starting_nav * 100) if starting_nav else 0, 4),
        }

    def _signal_quality(self, signal_log: list[dict]) -> dict:
        """Compute signal quality metrics: fired, executed, fill rates."""
        total_fired = len(signal_log)
        total_executed = sum(1 for s in signal_log if s.get("executed", False))
        fill_rate = (total_executed / total_fired * 100) if total_fired else 0.0

        by_type: dict[str, dict] = {}
        for s in signal_log:
            st = str(s.get("signal_type", "unknown"))
            if st not in by_type:
                by_type[st] = {"fired": 0, "executed": 0}
            by_type[st]["fired"] += 1
            if s.get("executed", False):
                by_type[st]["executed"] += 1

        for st, v in by_type.items():
            v["fill_rate_pct"] = round((v["executed"] / v["fired"] * 100) if v["fired"] else 0, 2)

        return {
            "signals_fired": total_fired,
            "signals_executed": total_executed,
            "fill_rate_pct": round(fill_rate, 2),
            "by_signal_type": by_type,
        }

    def _position_changes(self, trades: list[dict]) -> list[dict]:
        """Aggregate position delta by ticker."""
        changes: dict[str, dict] = {}
        for t in trades:
            ticker = str(t.get("ticker", ""))
            side = str(t.get("side", "")).lower()
            qty = _safe_float(t.get("qty", 0))
            delta = qty if side == "buy" else -qty

            if ticker not in changes:
                changes[ticker] = {"ticker": ticker, "qty_delta": 0.0, "trades": 0, "sides": set()}
            changes[ticker]["qty_delta"] += delta
            changes[ticker]["trades"] += 1
            changes[ticker]["sides"].add(side)

        result = []
        for ticker, info in changes.items():
            result.append(
                {
                    "ticker": ticker,
                    "qty_delta": round(info["qty_delta"], 4),
                    "trades": info["trades"],
                    "direction": "INCREASED" if info["qty_delta"] > 0 else ("DECREASED" if info["qty_delta"] < 0 else "FLAT"),
                    "mixed": len(info["sides"]) > 1,
                }
            )
        result.sort(key=lambda x: abs(x["qty_delta"]), reverse=True)
        return result

    def _risk_metrics(self, portfolio_state: dict) -> dict:
        """Compute VaR (historical), beta vs SPY, and gross/net exposure."""
        holdings = portfolio_state.get("holdings", {})
        tickers_held = [tk for tk, v in holdings.items() if _safe_float(v.get("qty", 0)) != 0]

        gross_exposure = sum(
            abs(_safe_float(v.get("qty", 0)) * _safe_float(v.get("avg_cost", 0)))
            for v in holdings.values()
        )
        net_exposure = sum(
            _safe_float(v.get("qty", 0)) * _safe_float(v.get("avg_cost", 0))
            for v in holdings.values()
        )

        var_95 = 0.0
        portfolio_beta = 0.0

        if tickers_held:
            try:
                fetch_tickers = tickers_held + [_BENCHMARK]
                data = yf.download(fetch_tickers, period="30d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                ret = close.pct_change().dropna()

                if not ret.empty and len(ret) >= 5:
                    # Weighted portfolio returns
                    weights: dict[str, float] = {}
                    total_val = gross_exposure if gross_exposure > 0 else 1.0
                    for ticker, info in holdings.items():
                        val = abs(_safe_float(info.get("qty", 0)) * _safe_float(info.get("avg_cost", 0)))
                        sign = 1 if _safe_float(info.get("qty", 0)) > 0 else -1
                        weights[ticker] = sign * val / total_val

                    port_ret = pd.Series(0.0, index=ret.index)
                    for ticker, w in weights.items():
                        if ticker in ret.columns:
                            port_ret += ret[ticker].fillna(0) * w

                    var_95 = float(np.percentile(port_ret, 5)) * gross_exposure

                    # Beta
                    if _BENCHMARK in ret.columns:
                        bm_ret = ret[_BENCHMARK].fillna(0)
                        cov_matrix = np.cov(port_ret, bm_ret)
                        bm_var = float(np.var(bm_ret))
                        portfolio_beta = float(cov_matrix[0, 1] / bm_var) if bm_var != 0 else 0.0

            except Exception as exc:
                logger.warning("_risk_metrics compute error: %s", exc)

        starting_nav = _safe_float(portfolio_state.get("starting_nav", 1))
        return {
            "gross_exposure": round(gross_exposure, 2),
            "net_exposure": round(net_exposure, 2),
            "gross_leverage": round(gross_exposure / starting_nav, 4) if starting_nav else 0,
            "net_leverage": round(net_exposure / starting_nav, 4) if starting_nav else 0,
            "var_95_1d": round(var_95, 2),
            "var_95_pct_nav": round((var_95 / starting_nav * 100) if starting_nav else 0, 4),
            "portfolio_beta": round(portfolio_beta, 4),
            "positions_count": len(tickers_held),
        }

    def _agent_activity(self, trades: list[dict], signal_log: list[dict]) -> dict:
        """Summarise activity by agent."""
        agents: dict[str, dict] = {}
        for t in trades:
            ag = str(t.get("agent", "unknown"))
            if ag not in agents:
                agents[ag] = {"trades": 0, "signals_fired": 0, "signals_executed": 0}
            agents[ag]["trades"] += 1

        for s in signal_log:
            ag = str(s.get("agent", "unknown"))
            if ag not in agents:
                agents[ag] = {"trades": 0, "signals_fired": 0, "signals_executed": 0}
            agents[ag]["signals_fired"] += 1
            if s.get("executed", False):
                agents[ag]["signals_executed"] += 1

        for ag, v in agents.items():
            fired = v["signals_fired"]
            v["fill_rate_pct"] = round((v["signals_executed"] / fired * 100) if fired else 0, 2)

        return agents

    def _best_signal(self, signal_log: list[dict], trades: list[dict]) -> dict | None:
        """Identify the highest-conviction executed signal of the hour."""
        executed = [s for s in signal_log if s.get("executed", False)]
        if not executed:
            return None
        best = max(executed, key=lambda s: _safe_float(s.get("conviction", 0)))
        return {
            "ticker": str(best.get("ticker", "")),
            "signal_type": str(best.get("signal_type", "")),
            "agent": str(best.get("agent", "")),
            "conviction": _safe_float(best.get("conviction", 0)),
            "fired_at": str(best.get("fired_at", "")),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, recap: dict, date_str: str, hour_str: str) -> None:
        json_path = self.recap_dir / f"{date_str}_{hour_str}.json"
        with open(json_path, "w") as f:
            json.dump(recap, f, indent=2, default=str)
        logger.info("Hourly recap saved: %s", json_path)

    def to_text(self, recap: dict) -> str:
        """Format recap as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  HOURLY RECAP  |  {recap.get('date', '')}  {recap.get('hour_utc', '')}")
        lines.append("=" * 60)

        pnl = recap.get("running_pnl", {})
        lines.append(f"  Running P&L  : ${pnl.get('total_pnl', 0):,.2f}  ({pnl.get('total_pnl_pct', 0):.2f}%)")
        lines.append(f"  Realized     : ${pnl.get('realized_cumulative', 0):,.2f}")
        lines.append(f"  Unrealized   : ${pnl.get('unrealized', 0):,.2f}")

        sq = recap.get("signal_quality", {})
        lines.append("")
        lines.append(f"  Signals Fired    : {sq.get('signals_fired', 0)}")
        lines.append(f"  Signals Executed : {sq.get('signals_executed', 0)}")
        lines.append(f"  Fill Rate        : {sq.get('fill_rate_pct', 0):.1f}%")

        rm = recap.get("risk_metrics", {})
        lines.append("")
        lines.append(f"  Gross Exposure : ${rm.get('gross_exposure', 0):,.2f}")
        lines.append(f"  Net Exposure   : ${rm.get('net_exposure', 0):,.2f}")
        lines.append(f"  VaR 95% (1d)   : ${rm.get('var_95_1d', 0):,.2f}")
        lines.append(f"  Portfolio Beta : {rm.get('portfolio_beta', 0):.3f}")

        txns = recap.get("transactions", [])
        if txns:
            lines.append("")
            lines.append(f"  Transactions ({len(txns)}):")
            for t in txns:
                lines.append(
                    f"    {t['side']:4s} {t['ticker']:6s}  qty={t['qty']:.0f}  @${t['fill_price']:.2f}  "
                    f"notional=${t['notional']:,.0f}  [{t['agent']}]"
                )

        best = recap.get("best_signal_of_hour")
        if best:
            lines.append("")
            lines.append(
                f"  Best Signal: {best['ticker']} via {best['signal_type']} "
                f"(agent={best['agent']}, conviction={best['conviction']:.2f})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    recap_engine = HourlyRecap()
    portfolio = {
        "cash": 900_000,
        "starting_nav": 2_000_000,
        "running_realized_pnl": 1_200,
        "holdings": {
            "AAPL": {"qty": 100, "avg_cost": 170.0},
            "MSFT": {"qty": 50, "avg_cost": 380.0},
        },
    }
    trades = [
        {
            "ticker": "NVDA",
            "side": "buy",
            "qty": 20,
            "fill_price": 890.0,
            "avg_cost": 880.0,
            "signal_type": "momentum",
            "agent": "alpha_agent",
            "timestamp": datetime.utcnow().isoformat(),
        }
    ]
    signals = [
        {"ticker": "NVDA", "signal_type": "momentum", "agent": "alpha_agent", "fired_at": datetime.utcnow().isoformat(), "executed": True, "conviction": 0.85},
        {"ticker": "TSLA", "signal_type": "mean_reversion", "agent": "beta_agent", "fired_at": datetime.utcnow().isoformat(), "executed": False, "conviction": 0.4},
    ]
    result = recap_engine.generate(portfolio, trades, signals)
    print(recap_engine.to_text(result))
