"""
Daily Market Wrap
==================
End-of-day narrative market summary covering:
  1. Market overview: SPY, QQQ, IWM, DIA performance
  2. Sector performance table (all 11 GICS sectors via XL ETFs)
  3. Best/worst performers in universe (top 5 each)
  4. Volume leaders
  5. Volatility summary (VIX change, realized vol)
  6. Fixed income: TLT, LQD, HYG performance + rate moves
  7. Commodity: GLD, USO, DBC
  8. Macro: USD index (UUP), sentiment
  9. Tomorrow's key events (earnings, macro data)

Output: structured dict + formatted text report
Saved to: logs/market_wraps/YYYYMMDD.json and .txt
"""

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[2]
_WRAP_DIR = _BASE_DIR / "logs" / "market_wraps"
_WRAP_DIR.mkdir(parents=True, exist_ok=True)

# ----- Ticker universes -----

INDICES = {
    "S&P 500": "SPY",
    "NASDAQ 100": "QQQ",
    "Russell 2000": "IWM",
    "Dow Jones": "DIA",
}

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

FIXED_INCOME = {
    "20yr Treasury (TLT)": "TLT",
    "IG Corporate (LQD)": "LQD",
    "HY Corporate (HYG)": "HYG",
    "Short Treasury (SHY)": "SHY",
}

RATES_PROXIES = {
    "2yr Yield (proxy)": "SHY",
    "10yr Yield (proxy)": "TLT",
}

COMMODITIES = {
    "Gold (GLD)": "GLD",
    "Oil (USO)": "USO",
    "Broad Commodity (DBC)": "DBC",
    "Silver (SLV)": "SLV",
}

MACRO_INSTRUMENTS = {
    "USD Index (UUP)": "UUP",
    "VIX (proxy VIXY)": "VIXY",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Fear & Greed proxy (VXX)": "VXX",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _fetch_1d_stats(tickers: list[str], period: str = "5d") -> dict[str, dict]:
    """Download price data and return per-ticker day stats."""
    result: dict[str, dict] = {}
    if not tickers:
        return result
    try:
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
        close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
        volume = data["Volume"] if "Volume" in data.columns else data.get("volume", pd.DataFrame())
        high = data["High"] if "High" in data.columns else data.get("high", pd.DataFrame())
        low = data["Low"] if "Low" in data.columns else data.get("low", pd.DataFrame())

        for df in (close, volume, high, low):
            if isinstance(df, pd.Series):
                df = df.to_frame()

        if len(close) < 2:
            return result

        for ticker in tickers:
            if ticker not in close.columns:
                continue
            prev = _safe_float(close[ticker].iloc[-2])
            last = _safe_float(close[ticker].iloc[-1])
            ret_pct = ((last - prev) / prev * 100) if prev else 0.0

            hi = _safe_float(high[ticker].iloc[-1]) if ticker in high.columns and not high.empty else 0.0
            lo = _safe_float(low[ticker].iloc[-1]) if ticker in low.columns and not low.empty else 0.0
            vol = int(_safe_float(volume[ticker].iloc[-1])) if ticker in volume.columns and not volume.empty else 0
            avg_vol = int(_safe_float(volume[ticker].mean())) if ticker in volume.columns and not volume.empty else 0

            result[ticker] = {
                "ticker": ticker,
                "last_close": round(last, 4),
                "prev_close": round(prev, 4),
                "day_return_pct": round(ret_pct, 4),
                "day_high": round(hi, 4),
                "day_low": round(lo, 4),
                "volume": vol,
                "avg_volume": avg_vol,
                "volume_ratio": round(vol / avg_vol, 2) if avg_vol > 0 else 0,
            }
    except Exception as exc:
        logger.warning("_fetch_1d_stats error: %s", exc)
    return result


def _realised_vol(ticker: str, window: int = 20) -> float:
    """Compute annualised realised vol for a ticker over the past `window` days."""
    try:
        data = yf.download(ticker, period=f"{window + 5}d", interval="1d", auto_adjust=True, progress=False)
        close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
        if isinstance(close, pd.Series):
            close = close.to_frame()
        col = ticker if ticker in close.columns else close.columns[0]
        ret = close[col].pct_change().dropna().tail(window)
        if len(ret) < 5:
            return 0.0
        return float(ret.std()) * math.sqrt(252) * 100
    except Exception:
        return 0.0


class MarketWrap:
    """Generates a comprehensive end-of-day market wrap report."""

    def __init__(self, wrap_dir: str | Path | None = None) -> None:
        self.wrap_dir = Path(wrap_dir) if wrap_dir else _WRAP_DIR
        self.wrap_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        date_: date | None = None,
        universe_tickers: list[str] | None = None,
    ) -> dict:
        """Generate the full daily market wrap.

        Parameters
        ----------
        date_ : date | None
            Trading date (default: today).
        universe_tickers : list[str] | None
            Optional list of platform universe tickers for best/worst performers.

        Returns
        -------
        dict
            Structured market wrap data. Also saves JSON and TXT.
        """
        report_date = date_ or date.today()
        date_str = report_date.strftime("%Y%m%d")
        logger.info("Generating market wrap for %s", date_str)

        wrap: dict[str, Any] = {
            "report_type": "MARKET_WRAP",
            "date": report_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "market_overview": self._market_overview(),
            "sector_performance": self._sector_performance(),
            "fixed_income": self._fixed_income(),
            "commodities": self._commodities(),
            "macro": self._macro(),
            "volatility": self._volatility_summary(),
            "volume_leaders": self._volume_leaders(universe_tickers or []),
            "universe_performers": self._universe_performers(universe_tickers or []),
            "tomorrow_events": self._tomorrow_events(report_date, universe_tickers or []),
        }

        self._save(wrap, date_str)
        return wrap

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _market_overview(self) -> dict:
        """SPY, QQQ, IWM, DIA day performance."""
        stats = _fetch_1d_stats(list(INDICES.values()))
        result = {}
        for name, ticker in INDICES.items():
            if ticker in stats:
                s = stats[ticker]
                result[name] = {
                    "ticker": ticker,
                    "last_close": s["last_close"],
                    "day_return_pct": s["day_return_pct"],
                    "day_high": s["day_high"],
                    "day_low": s["day_low"],
                    "volume": s["volume"],
                    "direction": "UP" if s["day_return_pct"] > 0 else ("DOWN" if s["day_return_pct"] < 0 else "FLAT"),
                }
        return result

    def _sector_performance(self) -> list[dict]:
        """All 11 GICS sectors via XL ETFs, sorted by day return."""
        stats = _fetch_1d_stats(list(SECTOR_ETFS.values()))
        sectors = []
        for name, ticker in SECTOR_ETFS.items():
            if ticker in stats:
                s = stats[ticker]
                sectors.append(
                    {
                        "sector": name,
                        "etf": ticker,
                        "day_return_pct": s["day_return_pct"],
                        "last_close": s["last_close"],
                        "volume": s["volume"],
                        "volume_ratio": s["volume_ratio"],
                    }
                )
        sectors.sort(key=lambda x: x["day_return_pct"], reverse=True)
        return sectors

    def _fixed_income(self) -> dict:
        """TLT, LQD, HYG, SHY performance and implied rate moves."""
        stats = _fetch_1d_stats(list(FIXED_INCOME.values()))
        result: dict[str, Any] = {"instruments": {}}
        for name, ticker in FIXED_INCOME.items():
            if ticker in stats:
                s = stats[ticker]
                result["instruments"][name] = {
                    "ticker": ticker,
                    "day_return_pct": s["day_return_pct"],
                    "last_close": s["last_close"],
                }

        # Duration / rate sensitivity approximation
        # TLT has ~18yr duration; SHY ~2yr duration
        # ΔPrice ≈ -Duration × ΔYield → ΔYield ≈ -ΔPrice / Duration
        tlt_ret = (
            result["instruments"].get("20yr Treasury (TLT)", {}).get("day_return_pct", 0) / 100
        )
        shy_ret = (
            result["instruments"].get("Short Treasury (SHY)", {}).get("day_return_pct", 0) / 100
        )
        result["implied_rate_moves"] = {
            "10yr_yield_approx_bps": round(-tlt_ret * 100 / 18 * 10000, 1),  # rough approx
            "2yr_yield_approx_bps": round(-shy_ret * 100 / 2 * 10000, 1),
            "note": "Approximate only — assumes modified duration: TLT≈18yr, SHY≈2yr",
        }
        return result

    def _commodities(self) -> dict:
        """GLD, USO, DBC, SLV day performance."""
        stats = _fetch_1d_stats(list(COMMODITIES.values()))
        result = {}
        for name, ticker in COMMODITIES.items():
            if ticker in stats:
                s = stats[ticker]
                result[name] = {
                    "ticker": ticker,
                    "day_return_pct": s["day_return_pct"],
                    "last_close": s["last_close"],
                }
        return result

    def _macro(self) -> dict:
        """USD index, VIX proxy, BTC, fear/greed proxy."""
        stats = _fetch_1d_stats(list(MACRO_INSTRUMENTS.values()))
        result = {}
        for name, ticker in MACRO_INSTRUMENTS.items():
            if ticker in stats:
                s = stats[ticker]
                result[name] = {
                    "ticker": ticker,
                    "day_return_pct": s["day_return_pct"],
                    "last_close": s["last_close"],
                }

        # Sentiment stub based on VIX proxy direction
        vixy_ret = result.get("VIX (proxy VIXY)", {}).get("day_return_pct", 0)
        spy_stats = _fetch_1d_stats(["SPY"])
        spy_ret = spy_stats.get("SPY", {}).get("day_return_pct", 0)

        if spy_ret > 0.5 and vixy_ret < -1:
            sentiment = "RISK_ON"
        elif spy_ret < -0.5 and vixy_ret > 1:
            sentiment = "RISK_OFF"
        else:
            sentiment = "NEUTRAL"

        result["market_sentiment"] = sentiment
        return result

    def _volatility_summary(self) -> dict:
        """VIX proxy change, realized vol for SPY."""
        stats = _fetch_1d_stats(["VIXY", "VXX"])
        vixy = stats.get("VIXY", {})
        vxx = stats.get("VXX", {})

        spy_rvol_20d = _realised_vol("SPY", window=20)
        spy_rvol_5d = _realised_vol("SPY", window=5)
        qqq_rvol_20d = _realised_vol("QQQ", window=20)

        return {
            "vixy_day_return_pct": vixy.get("day_return_pct", 0),
            "vixy_last_close": vixy.get("last_close", 0),
            "vxx_day_return_pct": vxx.get("day_return_pct", 0),
            "spy_realised_vol_5d_pct": round(spy_rvol_5d, 4),
            "spy_realised_vol_20d_pct": round(spy_rvol_20d, 4),
            "qqq_realised_vol_20d_pct": round(qqq_rvol_20d, 4),
            "vol_regime": (
                "HIGH" if spy_rvol_5d > spy_rvol_20d * 1.3
                else ("LOW" if spy_rvol_5d < spy_rvol_20d * 0.7 else "NORMAL")
            ),
        }

    def _volume_leaders(self, universe_tickers: list[str], top_n: int = 10) -> list[dict]:
        """Top N tickers by volume ratio (today vs average)."""
        if not universe_tickers:
            return []
        stats = _fetch_1d_stats(universe_tickers)
        leaders = [
            {
                "ticker": ticker,
                "volume": s["volume"],
                "avg_volume": s["avg_volume"],
                "volume_ratio": s["volume_ratio"],
                "day_return_pct": s["day_return_pct"],
            }
            for ticker, s in stats.items()
        ]
        leaders.sort(key=lambda x: x["volume_ratio"], reverse=True)
        return leaders[:top_n]

    def _universe_performers(self, universe_tickers: list[str], top_n: int = 5) -> dict:
        """Best and worst performers in the universe."""
        if not universe_tickers:
            return {"best": [], "worst": []}
        stats = _fetch_1d_stats(universe_tickers)
        all_ret = sorted(
            [
                {"ticker": ticker, "day_return_pct": s["day_return_pct"], "last_close": s["last_close"]}
                for ticker, s in stats.items()
            ],
            key=lambda x: x["day_return_pct"],
            reverse=True,
        )
        return {
            "best": all_ret[:top_n],
            "worst": all_ret[-top_n:][::-1] if len(all_ret) >= top_n else all_ret[::-1],
        }

    def _tomorrow_events(
        self,
        report_date: date,
        universe_tickers: list[str],
    ) -> dict:
        """Compile upcoming earnings and macro events for the next trading day."""
        tomorrow = report_date + timedelta(days=1)
        # Skip weekends
        if tomorrow.weekday() == 5:  # Saturday
            tomorrow += timedelta(days=2)
        elif tomorrow.weekday() == 6:  # Sunday
            tomorrow += timedelta(days=1)

        earnings_tomorrow = []
        for ticker in universe_tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if not isinstance(cal, dict):
                    continue
                ed_list = cal.get("Earnings Date", [])
                if not isinstance(ed_list, list):
                    ed_list = [ed_list]
                for ed in ed_list:
                    try:
                        if hasattr(ed, "date"):
                            ed = ed.date()
                        elif isinstance(ed, str):
                            ed = date.fromisoformat(ed[:10])
                        if ed == tomorrow:
                            earnings_tomorrow.append(
                                {
                                    "ticker": ticker,
                                    "date": tomorrow.isoformat(),
                                    "eps_estimate": _safe_float(cal.get("EPS Estimate")),
                                }
                            )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("_tomorrow_events error for %s: %s", ticker, exc)

        return {
            "next_trading_date": tomorrow.isoformat(),
            "earnings": earnings_tomorrow,
            "macro_note": "Integrate with FRED / Econoday for live macro calendar.",
        }

    # ------------------------------------------------------------------
    # Persistence & formatting
    # ------------------------------------------------------------------

    def _save(self, wrap: dict, date_str: str) -> None:
        json_path = self.wrap_dir / f"{date_str}.json"
        txt_path = self.wrap_dir / f"{date_str}.txt"

        with open(json_path, "w") as f:
            json.dump(wrap, f, indent=2, default=str)

        with open(txt_path, "w") as f:
            f.write(self.to_text(wrap))

        logger.info("Market wrap saved: %s", json_path)

    def to_text(self, wrap: dict) -> str:
        """Produce a human-readable market wrap report."""
        lines = []
        report_date = wrap.get("date", "")
        generated = wrap.get("generated_at", "")

        lines.append("=" * 72)
        lines.append(f"  DAILY MARKET WRAP  |  {report_date}  |  Generated: {generated}")
        lines.append("=" * 72)

        # ── Market Overview ──
        lines.append("\n  MARKET OVERVIEW")
        lines.append("  " + "─" * 60)
        header = f"  {'Index':<25} {'Ticker':<6} {'Close':>8} {'Day %':>8} {'High':>9} {'Low':>9}"
        lines.append(header)
        lines.append("  " + "─" * 60)
        for name, info in wrap.get("market_overview", {}).items():
            arrow = "▲" if info.get("day_return_pct", 0) > 0 else ("▼" if info.get("day_return_pct", 0) < 0 else "─")
            lines.append(
                f"  {name:<25} {info.get('ticker',''):<6} {info.get('last_close',0):>8.2f} "
                f"{arrow}{info.get('day_return_pct',0):>+7.2f}% "
                f"{info.get('day_high',0):>9.2f} {info.get('day_low',0):>9.2f}"
            )

        # ── Sector Performance ──
        lines.append("\n  SECTOR PERFORMANCE")
        lines.append("  " + "─" * 60)
        lines.append(f"  {'Sector':<28} {'ETF':<5} {'Day %':>8}")
        lines.append("  " + "─" * 60)
        for row in wrap.get("sector_performance", []):
            bar_len = max(0, min(20, int(abs(row["day_return_pct"]) * 3)))
            bar = ("█" * bar_len) if row["day_return_pct"] >= 0 else ("░" * bar_len)
            sign = "+" if row["day_return_pct"] >= 0 else ""
            lines.append(
                f"  {row['sector']:<28} {row['etf']:<5} {sign}{row['day_return_pct']:>+6.2f}%  {bar}"
            )

        # ── Fixed Income ──
        lines.append("\n  FIXED INCOME")
        lines.append("  " + "─" * 60)
        fi = wrap.get("fixed_income", {})
        for name, info in fi.get("instruments", {}).items():
            lines.append(f"  {name:<30} {info.get('day_return_pct',0):>+7.2f}%  @ {info.get('last_close',0):.2f}")
        rates = fi.get("implied_rate_moves", {})
        if rates:
            lines.append(f"  Implied 10yr yield move : {rates.get('10yr_yield_approx_bps', 0):>+.1f} bps (approx)")
            lines.append(f"  Implied 2yr yield move  : {rates.get('2yr_yield_approx_bps', 0):>+.1f} bps (approx)")

        # ── Commodities ──
        lines.append("\n  COMMODITIES")
        lines.append("  " + "─" * 60)
        for name, info in wrap.get("commodities", {}).items():
            lines.append(f"  {name:<30} {info.get('day_return_pct',0):>+7.2f}%  @ {info.get('last_close',0):.2f}")

        # ── Macro / FX ──
        lines.append("\n  MACRO & FX")
        lines.append("  " + "─" * 60)
        macro = wrap.get("macro", {})
        for name, info in macro.items():
            if isinstance(info, dict):
                lines.append(f"  {name:<30} {info.get('day_return_pct',0):>+7.2f}%  @ {info.get('last_close',0):.2f}")
            else:
                lines.append(f"  Market Sentiment: {info}")

        # ── Volatility ──
        lines.append("\n  VOLATILITY")
        lines.append("  " + "─" * 60)
        vol = wrap.get("volatility", {})
        lines.append(f"  VIXY Day Return       : {vol.get('vixy_day_return_pct',0):>+.2f}%")
        lines.append(f"  SPY Realised Vol (5d) : {vol.get('spy_realised_vol_5d_pct',0):.2f}% ann.")
        lines.append(f"  SPY Realised Vol (20d): {vol.get('spy_realised_vol_20d_pct',0):.2f}% ann.")
        lines.append(f"  QQQ Realised Vol (20d): {vol.get('qqq_realised_vol_20d_pct',0):.2f}% ann.")
        lines.append(f"  Vol Regime            : {vol.get('vol_regime','N/A')}")

        # ── Universe Performers ──
        perf = wrap.get("universe_performers", {})
        if perf.get("best"):
            lines.append("\n  TOP 5 UNIVERSE PERFORMERS")
            lines.append("  " + "─" * 60)
            for p in perf["best"]:
                lines.append(f"  {p['ticker']:<8} {p['day_return_pct']:>+7.2f}%  @ {p['last_close']:.2f}")

        if perf.get("worst"):
            lines.append("\n  BOTTOM 5 UNIVERSE PERFORMERS")
            lines.append("  " + "─" * 60)
            for p in perf["worst"]:
                lines.append(f"  {p['ticker']:<8} {p['day_return_pct']:>+7.2f}%  @ {p['last_close']:.2f}")

        # ── Volume Leaders ──
        vols = wrap.get("volume_leaders", [])
        if vols:
            lines.append("\n  VOLUME LEADERS")
            lines.append("  " + "─" * 60)
            lines.append(f"  {'Ticker':<8} {'Vol Ratio':>10} {'Day %':>8}  Volume")
            for v in vols[:10]:
                lines.append(
                    f"  {v['ticker']:<8} {v['volume_ratio']:>10.2f}x {v['day_return_pct']:>+7.2f}%  {v['volume']:,}"
                )

        # ── Tomorrow ──
        tmr = wrap.get("tomorrow_events", {})
        lines.append(f"\n  TOMORROW'S KEY EVENTS  ({tmr.get('next_trading_date', '')})")
        lines.append("  " + "─" * 60)
        earnings_tmr = tmr.get("earnings", [])
        if earnings_tmr:
            for e in earnings_tmr:
                lines.append(f"  EARNINGS: {e['ticker']}  EPS est: {e.get('eps_estimate',0):.2f}")
        else:
            lines.append("  No earnings from universe tickers detected.")
        lines.append(f"  {tmr.get('macro_note','')}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    universe = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "GS", "XOM", "CVX", "JNJ", "UNH", "PFE",
    ]
    wrap_engine = MarketWrap()
    result = wrap_engine.generate(universe_tickers=universe)
    print(wrap_engine.to_text(result))
