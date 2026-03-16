"""
Daily Report Engine
===================
Generates two reports per trading day:
  - MARKET OPEN REPORT (9:30 ET): Pre-market setup, universe scan, expected movers
  - MARKET CLOSE REPORT (4:00 ET): Full P&L, attribution, missed opportunities, anomalies

Reports are saved to: logs/daily_reports/YYYYMMDD_[open|close].json
                 and: logs/daily_reports/YYYYMMDD_[open|close].txt (human-readable)
"""

import json
import os
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Base directory is two levels up from this file: src/monitoring/ -> project root
_BASE_DIR = Path(__file__).resolve().parents[2]
_REPORT_DIR = _BASE_DIR / "logs" / "daily_reports"
_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# GICS sector ETFs used for sector rotation signals
SECTOR_ETFS = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

# Key earnings calendar is sourced from yfinance per ticker; macro events are static stubs
MACRO_KEYWORDS = ["FOMC", "CPI", "PPI", "NFP", "GDP", "PCE", "ISM", "JOLTS"]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _pct(val: Any) -> float:
    return round(_safe_float(val) * 100, 4)


class DailyReport:
    """Generates market-open and market-close daily reports for the platform."""

    def __init__(self, report_dir: str | Path | None = None) -> None:
        self.report_dir = Path(report_dir) if report_dir else _REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # OPEN REPORT
    # ------------------------------------------------------------------

    def generate_open_report(
        self,
        portfolio_state: dict,
        universe_tickers: list[str],
    ) -> dict:
        """Generate the pre-market open report.

        Parameters
        ----------
        portfolio_state : dict
            Current portfolio holdings and cash.
        universe_tickers : list[str]
            Full list of tickers in the investment universe.

        Returns
        -------
        dict
            Structured report data. Also saves JSON and TXT to report_dir.
        """
        today = date.today()
        date_str = today.strftime("%Y%m%d")
        logger.info("Generating open report for %s", date_str)

        report: dict[str, Any] = {
            "report_type": "MARKET_OPEN",
            "date": today.isoformat(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "portfolio_summary": self._portfolio_summary(portfolio_state),
            "top_expected_movers": self._top_expected_movers(universe_tickers),
            "sector_rotation_signals": self._sector_rotation_signals(),
            "options_expiry_awareness": self._options_expiry_awareness(today),
            "key_earnings_today": self._key_earnings_today(universe_tickers),
            "macro_events": self._macro_events_stub(today),
            "universe_heat_snapshot": self._universe_heat_snapshot(universe_tickers),
        }

        self._save(report, date_str, "open")
        return report

    # ------------------------------------------------------------------
    # CLOSE REPORT
    # ------------------------------------------------------------------

    def generate_close_report(
        self,
        portfolio_state: dict,
        all_trades: list[dict],
        universe_tickers: list[str],
    ) -> dict:
        """Generate the end-of-day close report.

        Parameters
        ----------
        portfolio_state : dict
            Current portfolio holdings, cash, and starting NAV.
        all_trades : list[dict]
            All trades executed today. Each entry should have keys:
            ticker, side (buy/sell), qty, fill_price, signal_type, agent.
        universe_tickers : list[str]
            Full list of tickers in the investment universe.

        Returns
        -------
        dict
            Structured report data. Also saves JSON and TXT to report_dir.
        """
        today = date.today()
        date_str = today.strftime("%Y%m%d")
        logger.info("Generating close report for %s", date_str)

        pnl = self._compute_pnl(portfolio_state, all_trades)
        attribution = self._attribution(all_trades, portfolio_state)
        missed = self._missed_opportunities(universe_tickers, portfolio_state, all_trades)
        anomalies = self._statistical_anomalies(universe_tickers)
        top_performers = self._top_performers_universe(universe_tickers)

        report: dict[str, Any] = {
            "report_type": "MARKET_CLOSE",
            "date": today.isoformat(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "portfolio_summary": self._portfolio_summary(portfolio_state),
            "pnl_breakdown": pnl,
            "attribution": attribution,
            "missed_opportunities": missed,
            "statistical_anomalies": anomalies,
            "top_performers_universe": top_performers,
            "trade_count": len(all_trades),
        }

        self._save(report, date_str, "close")
        return report

    # ------------------------------------------------------------------
    # OPEN REPORT HELPERS
    # ------------------------------------------------------------------

    def _top_expected_movers(self, tickers: list[str], top_n: int = 10) -> list[dict]:
        """Rank tickers by absolute pre-market % change (or prior close %). Returns top N."""
        results = []
        if not tickers:
            return results

        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                data = yf.download(
                    chunk,
                    period="2d",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if data.empty:
                    continue

                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()

                if len(close) < 2:
                    continue

                prev_close = close.iloc[-2]
                last_close = close.iloc[-1]
                pct_chg = ((last_close - prev_close) / prev_close.replace(0, np.nan)).fillna(0)

                for ticker in chunk:
                    if ticker in pct_chg.index:
                        results.append(
                            {
                                "ticker": ticker,
                                "pct_change_1d": round(float(pct_chg[ticker]) * 100, 2),
                                "last_close": round(_safe_float(last_close.get(ticker, 0)), 4),
                            }
                        )
            except Exception as exc:
                logger.warning("_top_expected_movers error for chunk %s: %s", chunk, exc)

        results.sort(key=lambda x: abs(x["pct_change_1d"]), reverse=True)
        return results[:top_n]

    def _sector_rotation_signals(self) -> list[dict]:
        """Compute 5-day momentum for sector ETFs to surface rotation signals."""
        signals = []
        tickers = list(SECTOR_ETFS.values())
        try:
            data = yf.download(tickers, period="10d", interval="1d", auto_adjust=True, progress=False)
            close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
            if isinstance(close, pd.Series):
                close = close.to_frame()

            if len(close) >= 5:
                ret_5d = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5].replace(0, np.nan)
                for sector, etf in SECTOR_ETFS.items():
                    if etf in ret_5d.index:
                        val = _safe_float(ret_5d[etf])
                        signals.append(
                            {
                                "sector": sector,
                                "etf": etf,
                                "momentum_5d_pct": round(val * 100, 2),
                                "signal": "OVERWEIGHT" if val > 0.01 else ("UNDERWEIGHT" if val < -0.01 else "NEUTRAL"),
                            }
                        )
        except Exception as exc:
            logger.warning("_sector_rotation_signals error: %s", exc)

        signals.sort(key=lambda x: x["momentum_5d_pct"], reverse=True)
        return signals

    def _options_expiry_awareness(self, today: date) -> dict:
        """Return metadata about weekly/monthly options expiry proximity."""
        day_of_week = today.weekday()  # 0=Mon, 4=Fri
        day_of_month = today.day

        # 3rd Friday of the month is monthly expiry
        from calendar import monthcalendar
        cal = monthcalendar(today.year, today.month)
        fridays = [week[4] for week in cal if week[4] != 0]
        monthly_expiry_day = fridays[2] if len(fridays) >= 3 else fridays[-1]

        days_to_monthly = monthly_expiry_day - day_of_month
        is_expiry_week = 0 <= days_to_monthly <= 4
        is_expiry_day = day_of_month == monthly_expiry_day and day_of_week == 4

        return {
            "today": today.isoformat(),
            "day_of_week_name": today.strftime("%A"),
            "monthly_expiry_day": monthly_expiry_day,
            "days_to_monthly_expiry": days_to_monthly,
            "is_monthly_expiry_week": is_expiry_week,
            "is_monthly_expiry_day": is_expiry_day,
            "is_weekly_expiry_day": day_of_week == 4,  # every Friday
            "note": (
                "OPEX WEEK — gamma risk elevated, pin risk possible"
                if is_expiry_week
                else "No major expiry this week"
            ),
        }

    def _key_earnings_today(self, tickers: list[str]) -> list[dict]:
        """Check yfinance calendar for earnings events today for tickers in universe."""
        today = date.today()
        earnings = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if cal is None:
                    continue
                # yfinance returns a dict with 'Earnings Date' list or similar
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date", [])
                    if not isinstance(ed, list):
                        ed = [ed]
                    for d in ed:
                        try:
                            if hasattr(d, "date"):
                                d = d.date()
                            elif isinstance(d, str):
                                d = date.fromisoformat(d[:10])
                            if d == today:
                                earnings.append(
                                    {
                                        "ticker": ticker,
                                        "earnings_date": today.isoformat(),
                                        "eps_estimate": _safe_float(cal.get("EPS Estimate")),
                                        "revenue_estimate": _safe_float(cal.get("Revenue Estimate")),
                                    }
                                )
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("_key_earnings_today error for %s: %s", ticker, exc)

        return earnings

    def _macro_events_stub(self, today: date) -> list[dict]:
        """Return a stub macro calendar. In production, integrate with an economic calendar API."""
        # Static placeholder — replace with live data feed in production
        return [
            {
                "note": "Macro calendar is a stub. Integrate with FRED, Bloomberg, or Econoday API for live events.",
                "keywords": MACRO_KEYWORDS,
                "date": today.isoformat(),
            }
        ]

    def _universe_heat_snapshot(self, tickers: list[str]) -> list[dict]:
        """Return all tickers sorted by 5-day momentum."""
        results = []
        if not tickers:
            return results

        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                data = yf.download(chunk, period="10d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if len(close) < 5:
                    continue

                ret_5d = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5].replace(0, np.nan)
                ret_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2].replace(0, np.nan) if len(close) >= 2 else pd.Series(dtype=float)

                for ticker in chunk:
                    r5 = _safe_float(ret_5d.get(ticker, 0))
                    r1 = _safe_float(ret_1d.get(ticker, 0)) if not ret_1d.empty else 0.0
                    results.append(
                        {
                            "ticker": ticker,
                            "momentum_5d_pct": round(r5 * 100, 2),
                            "return_1d_pct": round(r1 * 100, 2),
                        }
                    )
            except Exception as exc:
                logger.warning("_universe_heat_snapshot error for chunk %s: %s", chunk, exc)

        results.sort(key=lambda x: x["momentum_5d_pct"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # CLOSE REPORT HELPERS
    # ------------------------------------------------------------------

    def _compute_pnl(self, portfolio_state: dict, all_trades: list[dict]) -> dict:
        """Compute realized and unrealized P&L from portfolio state and trades."""
        starting_nav = _safe_float(portfolio_state.get("starting_nav", 0))
        cash = _safe_float(portfolio_state.get("cash", 0))
        holdings = portfolio_state.get("holdings", {})  # {ticker: {qty, avg_cost}}

        # Realized P&L from trades
        realized = 0.0
        for trade in all_trades:
            side = str(trade.get("side", "")).lower()
            qty = _safe_float(trade.get("qty", 0))
            fill = _safe_float(trade.get("fill_price", 0))
            avg_cost = _safe_float(trade.get("avg_cost", fill))
            if side == "sell":
                realized += (fill - avg_cost) * qty

        # Unrealized P&L — fetch current prices
        unrealized = 0.0
        tickers_held = [t for t in holdings if holdings[t].get("qty", 0) != 0]
        if tickers_held:
            try:
                data = yf.download(tickers_held, period="1d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                for ticker, info in holdings.items():
                    qty = _safe_float(info.get("qty", 0))
                    avg_cost = _safe_float(info.get("avg_cost", 0))
                    if ticker in close.columns and not close.empty:
                        current_price = _safe_float(close[ticker].iloc[-1])
                        unrealized += (current_price - avg_cost) * qty
            except Exception as exc:
                logger.warning("_compute_pnl unrealized fetch error: %s", exc)

        current_nav = cash + sum(
            _safe_float(holdings[t].get("qty", 0)) * _safe_float(holdings[t].get("avg_cost", 0))
            for t in holdings
        )
        total_pnl = realized + unrealized

        return {
            "starting_nav": round(starting_nav, 2),
            "current_nav_estimate": round(current_nav, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / starting_nav * 100) if starting_nav else 0, 4),
            "cash": round(cash, 2),
        }

    def _attribution(self, all_trades: list[dict], portfolio_state: dict) -> dict:
        """Break P&L down by sector, signal type, and agent."""
        by_sector: dict[str, float] = {}
        by_signal: dict[str, float] = {}
        by_agent: dict[str, float] = {}

        for trade in all_trades:
            sector = str(trade.get("sector", "Unknown"))
            signal = str(trade.get("signal_type", "Unknown"))
            agent = str(trade.get("agent", "Unknown"))
            side = str(trade.get("side", "")).lower()
            qty = _safe_float(trade.get("qty", 0))
            fill = _safe_float(trade.get("fill_price", 0))
            avg_cost = _safe_float(trade.get("avg_cost", fill))
            pnl = (fill - avg_cost) * qty if side == "sell" else 0.0

            by_sector[sector] = by_sector.get(sector, 0.0) + pnl
            by_signal[signal] = by_signal.get(signal, 0.0) + pnl
            by_agent[agent] = by_agent.get(agent, 0.0) + pnl

        return {
            "by_sector": {k: round(v, 2) for k, v in sorted(by_sector.items(), key=lambda x: -x[1])},
            "by_signal_type": {k: round(v, 2) for k, v in sorted(by_signal.items(), key=lambda x: -x[1])},
            "by_agent": {k: round(v, 2) for k, v in sorted(by_agent.items(), key=lambda x: -x[1])},
        }

    def _missed_opportunities(
        self,
        universe_tickers: list[str],
        portfolio_state: dict,
        all_trades: list[dict],
    ) -> list[dict]:
        """
        MISSED OPPORTUNITIES SECTION
        Fetch actual day's movers from yfinance, compare to what the system traded.
        Flag any security that moved >3% that we didn't hold or traded wrong way.
        """
        missed = []
        if not universe_tickers:
            return missed

        holdings = portfolio_state.get("holdings", {})
        traded_tickers = {str(t.get("ticker", "")): t for t in all_trades}

        chunk_size = 50
        all_returns: dict[str, float] = {}

        for i in range(0, len(universe_tickers), chunk_size):
            chunk = universe_tickers[i : i + chunk_size]
            try:
                data = yf.download(chunk, period="2d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if len(close) < 2:
                    continue
                prev = close.iloc[-2]
                last = close.iloc[-1]
                pct = ((last - prev) / prev.replace(0, np.nan)).fillna(0)
                for ticker in chunk:
                    if ticker in pct.index:
                        all_returns[ticker] = float(pct[ticker])
            except Exception as exc:
                logger.warning("_missed_opportunities fetch error: %s", exc)

        for ticker, ret in all_returns.items():
            ret_pct = ret * 100
            if abs(ret_pct) < 3.0:
                continue

            direction = "UP" if ret_pct > 0 else "DOWN"
            held_qty = _safe_float(holdings.get(ticker, {}).get("qty", 0))
            trade_info = traded_tickers.get(ticker)

            reason_code = "NOT_IN_UNIVERSE"
            if ticker in universe_tickers:
                if held_qty == 0 and trade_info is None:
                    reason_code = "NOT_HELD_NOT_TRADED"
                elif held_qty == 0 and trade_info is not None:
                    reason_code = "TRADED_BUT_EXITED"
                elif held_qty > 0 and ret_pct < -3:
                    reason_code = "HELD_WRONG_DIRECTION_LONG"
                elif held_qty < 0 and ret_pct > 3:
                    reason_code = "HELD_WRONG_DIRECTION_SHORT"
                else:
                    reason_code = "CAPTURED"

            if reason_code == "CAPTURED":
                continue

            missed.append(
                {
                    "ticker": ticker,
                    "day_return_pct": round(ret_pct, 2),
                    "direction": direction,
                    "held_qty": held_qty,
                    "reason_code": reason_code,
                    "magnitude": "LARGE" if abs(ret_pct) >= 5 else "MODERATE",
                }
            )

        missed.sort(key=lambda x: abs(x["day_return_pct"]), reverse=True)

        # Save to missed_opportunities log
        missed_dir = _BASE_DIR / "logs" / "missed_opportunities"
        missed_dir.mkdir(parents=True, exist_ok=True)
        mo_path = missed_dir / f"{date.today().strftime('%Y%m%d')}.json"
        try:
            with open(mo_path, "w") as f:
                json.dump(missed, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save missed opportunities: %s", exc)

        return missed

    def _statistical_anomalies(self, universe_tickers: list[str]) -> list[dict]:
        """Detect >2σ single-day moves and volume spikes in the universe."""
        anomalies = []
        if not universe_tickers:
            return anomalies

        chunk_size = 30
        for i in range(0, len(universe_tickers), chunk_size):
            chunk = universe_tickers[i : i + chunk_size]
            try:
                data = yf.download(chunk, period="65d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                volume = data["Volume"] if "Volume" in data.columns else data.get("volume", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if isinstance(volume, pd.Series):
                    volume = volume.to_frame()

                if len(close) < 5:
                    continue

                ret = close.pct_change()
                rolling_std = ret.iloc[:-1].std()
                rolling_mean = ret.iloc[:-1].mean()
                last_ret = ret.iloc[-1]

                avg_vol = volume.iloc[:-1].mean()
                last_vol = volume.iloc[-1]

                for ticker in chunk:
                    if ticker not in last_ret.index:
                        continue
                    r = _safe_float(last_ret[ticker])
                    std = _safe_float(rolling_std.get(ticker, 1))
                    mean = _safe_float(rolling_mean.get(ticker, 0))
                    if std == 0:
                        continue
                    z = (r - mean) / std
                    if abs(z) >= 2.0:
                        severity = "CRITICAL" if abs(z) >= 4 else ("HIGH" if abs(z) >= 3 else "MEDIUM")
                        anomalies.append(
                            {
                                "ticker": ticker,
                                "type": "PRICE_ANOMALY",
                                "z_score": round(z, 2),
                                "day_return_pct": round(r * 100, 2),
                                "severity": severity,
                            }
                        )

                    # Volume anomaly
                    avg_v = _safe_float(avg_vol.get(ticker, 0))
                    last_v = _safe_float(last_vol.get(ticker, 0))
                    if avg_v > 0 and last_v > 3 * avg_v:
                        anomalies.append(
                            {
                                "ticker": ticker,
                                "type": "VOLUME_ANOMALY",
                                "volume_ratio": round(last_v / avg_v, 2),
                                "severity": "HIGH" if last_v > 5 * avg_v else "MEDIUM",
                            }
                        )

            except Exception as exc:
                logger.warning("_statistical_anomalies error for chunk %s: %s", chunk, exc)

        anomalies.sort(key=lambda x: abs(x.get("z_score", x.get("volume_ratio", 0))), reverse=True)
        return anomalies

    def _top_performers_universe(self, universe_tickers: list[str], top_n: int = 5) -> dict:
        """Return top 5 best and worst performers in the universe for the day."""
        all_returns = []
        if not universe_tickers:
            return {"best": [], "worst": []}

        chunk_size = 50
        for i in range(0, len(universe_tickers), chunk_size):
            chunk = universe_tickers[i : i + chunk_size]
            try:
                data = yf.download(chunk, period="2d", interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if len(close) < 2:
                    continue
                pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2].replace(0, np.nan)).fillna(0)
                for ticker in chunk:
                    if ticker in pct.index:
                        all_returns.append({"ticker": ticker, "day_return_pct": round(float(pct[ticker]) * 100, 2)})
            except Exception as exc:
                logger.warning("_top_performers_universe error: %s", exc)

        all_returns.sort(key=lambda x: x["day_return_pct"], reverse=True)
        return {
            "best": all_returns[:top_n],
            "worst": all_returns[-top_n:][::-1] if len(all_returns) >= top_n else all_returns[::-1],
        }

    # ------------------------------------------------------------------
    # SHARED HELPERS
    # ------------------------------------------------------------------

    def _portfolio_summary(self, portfolio_state: dict) -> dict:
        holdings = portfolio_state.get("holdings", {})
        return {
            "cash": round(_safe_float(portfolio_state.get("cash", 0)), 2),
            "positions_count": sum(1 for v in holdings.values() if _safe_float(v.get("qty", 0)) != 0),
            "starting_nav": round(_safe_float(portfolio_state.get("starting_nav", 0)), 2),
            "gross_exposure": round(
                sum(abs(_safe_float(v.get("qty", 0)) * _safe_float(v.get("avg_cost", 0))) for v in holdings.values()),
                2,
            ),
        }

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def _save(self, report: dict, date_str: str, suffix: str) -> None:
        """Persist report as JSON and human-readable TXT."""
        json_path = self.report_dir / f"{date_str}_{suffix}.json"
        txt_path = self.report_dir / f"{date_str}_{suffix}.txt"

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        with open(txt_path, "w") as f:
            f.write(self._format_text(report))

        logger.info("Report saved: %s", json_path)

    def _format_text(self, report: dict) -> str:
        """Produce a human-readable text version of any report dict."""
        lines = []
        rtype = report.get("report_type", "REPORT")
        lines.append("=" * 70)
        lines.append(f"  {rtype}")
        lines.append(f"  Date: {report.get('date', '')}  |  Generated: {report.get('generated_at', '')}")
        lines.append("=" * 70)

        def _render(obj, indent=0):
            pad = "  " * indent
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{pad}{k}:")
                        _render(v, indent + 1)
                    else:
                        lines.append(f"{pad}{k}: {v}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        lines.append(f"{pad}-")
                        _render(item, indent + 1)
                    else:
                        lines.append(f"{pad}- {item}")
            else:
                lines.append(f"{pad}{obj}")

        for section, content in report.items():
            if section in ("report_type", "date", "generated_at"):
                continue
            lines.append("")
            lines.append(f"{'─' * 60}")
            lines.append(f"  {section.upper().replace('_', ' ')}")
            lines.append(f"{'─' * 60}")
            _render(content, indent=1)

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    engine = DailyReport()
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "GS", "V"]
    portfolio = {
        "cash": 1_000_000,
        "starting_nav": 2_000_000,
        "holdings": {
            "AAPL": {"qty": 100, "avg_cost": 170.0},
            "MSFT": {"qty": 50, "avg_cost": 380.0},
        },
    }
    if mode == "close":
        trades = [
            {"ticker": "AAPL", "side": "sell", "qty": 50, "fill_price": 172.0, "avg_cost": 170.0,
             "signal_type": "momentum", "agent": "alpha_agent", "sector": "Technology"},
        ]
        result = engine.generate_close_report(portfolio, trades, tickers)
    else:
        result = engine.generate_open_report(portfolio, tickers)

    print(json.dumps(result, indent=2, default=str))
