"""
Securities Universe Heatmap
============================
Generates a text-based and JSON heatmap of the full universe.
Split by GICS sector, then by security type.
Color coding:
  STRONG_UP (>2%)  | UP (0.5-2%)  | FLAT (-0.5 to 0.5%)  | DOWN (-2 to -0.5%)  | STRONG_DOWN (<-2%)

Outputs:
  - JSON heatmap: logs/heatmaps/YYYYMMDD_HHMM.json
  - ASCII heatmap for terminal: formatted table
  - Sector-level summary (breadth, avg return, vol)
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
_HEATMAP_DIR = _BASE_DIR / "logs" / "heatmaps"
_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# GICS sector ETF mapping — used for sector-level heatmap
SECTOR_ETFS: dict[str, str] = {
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

# Return thresholds for bucketing
_THRESHOLDS = {
    "STRONG_UP": 2.0,
    "UP": 0.5,
    "FLAT": -0.5,
    "DOWN": -2.0,
    # Below DOWN threshold → STRONG_DOWN
}

# ASCII display widths
_CELL_WIDTH = 12
_TICKER_WIDTH = 7


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _classify_return(ret_pct: float) -> str:
    """Map a return percentage to a heat bucket."""
    if ret_pct >= _THRESHOLDS["STRONG_UP"]:
        return "STRONG_UP"
    if ret_pct >= _THRESHOLDS["UP"]:
        return "UP"
    if ret_pct >= _THRESHOLDS["FLAT"]:
        return "FLAT"
    if ret_pct >= _THRESHOLDS["DOWN"]:
        return "DOWN"
    return "STRONG_DOWN"


# ASCII colour codes (terminal)
_ANSI = {
    "STRONG_UP": "\033[92m",    # bright green
    "UP": "\033[32m",            # green
    "FLAT": "\033[37m",          # white/grey
    "DOWN": "\033[31m",          # red
    "STRONG_DOWN": "\033[91m",   # bright red
    "RESET": "\033[0m",
}

_BUCKET_LABELS = {
    "STRONG_UP": "+2%+",
    "UP": "+0.5-2%",
    "FLAT": "FLAT",
    "DOWN": "-2--0.5%",
    "STRONG_DOWN": "-2%-",
}


class HeatmapEngine:
    """Generates universe and sector heatmaps."""

    def __init__(self, heatmap_dir: str | Path | None = None) -> None:
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir else _HEATMAP_DIR
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        self._last_universe_data: dict[str, dict] = {}  # ticker → security info

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate_universe_heatmap(
        self,
        tickers: list[str],
        period: str = "1d",
        sector_map: dict[str, str] | None = None,
    ) -> dict:
        """Fetch returns via yfinance and produce a categorised heatmap.

        Parameters
        ----------
        tickers : list[str]
            Full list of universe tickers.
        period : str
            yfinance period string (default '1d').
        sector_map : dict[str, str] | None
            Optional mapping {ticker: sector_name}. If not provided, all
            tickers are placed in an 'Unknown' sector.

        Returns
        -------
        dict
            Heatmap data with per-security and sector breakdowns.
            Also persisted to logs/heatmaps/.
        """
        now = datetime.utcnow()
        ts = now.strftime("%Y%m%d_%H%M")
        date_str = date.today().isoformat()

        security_data = self._fetch_security_data(tickers, period)
        self._last_universe_data = security_data

        # Assign sectors
        sm = sector_map or {}
        by_sector: dict[str, list[dict]] = {}
        for ticker, info in security_data.items():
            sector = sm.get(ticker, "Unknown")
            info["sector"] = sector
            by_sector.setdefault(sector, []).append(info)

        # Sort each sector by return descending
        for sector in by_sector:
            by_sector[sector].sort(key=lambda x: x["return_pct"], reverse=True)

        sector_summary = self.generate_sector_heatmap(security_data, by_sector)

        # Bucket counts
        bucket_counts: dict[str, int] = {b: 0 for b in _THRESHOLDS}
        bucket_counts["STRONG_DOWN"] = 0
        for info in security_data.values():
            bucket_counts[info["bucket"]] = bucket_counts.get(info["bucket"], 0) + 1

        heatmap = {
            "generated_at": now.isoformat() + "Z",
            "date": date_str,
            "period": period,
            "ticker_count": len(security_data),
            "bucket_counts": bucket_counts,
            "securities": security_data,
            "by_sector": by_sector,
            "sector_summary": sector_summary,
        }

        self._save(heatmap, ts)
        return heatmap

    def generate_sector_heatmap(
        self,
        security_data: dict[str, dict] | None = None,
        by_sector: dict[str, list[dict]] | None = None,
    ) -> dict[str, dict]:
        """Produce sector-level aggregation from security data.

        If security_data/by_sector are None, fetches SECTOR_ETFS directly.
        """
        if security_data is None or by_sector is None:
            # Standalone mode: fetch sector ETFs
            etf_data = self._fetch_security_data(list(SECTOR_ETFS.values()), "1d")
            result = {}
            for sector, etf in SECTOR_ETFS.items():
                if etf in etf_data:
                    info = etf_data[etf]
                    result[sector] = {
                        "etf": etf,
                        "return_pct": info["return_pct"],
                        "bucket": info["bucket"],
                        "volume": info.get("volume", 0),
                        "ticker_count": 1,
                        "breadth_positive": 1 if info["return_pct"] > 0 else 0,
                        "avg_return_pct": info["return_pct"],
                        "avg_vol_pct": info.get("vol_pct", 0),
                    }
            return result

        result = {}
        for sector, securities in by_sector.items():
            returns = [s["return_pct"] for s in securities]
            vols = [s.get("vol_pct", 0) for s in securities]
            count = len(returns)
            avg_ret = float(np.mean(returns)) if returns else 0.0
            avg_vol = float(np.mean(vols)) if vols else 0.0
            positive = sum(1 for r in returns if r > 0)
            result[sector] = {
                "ticker_count": count,
                "avg_return_pct": round(avg_ret, 4),
                "avg_vol_pct": round(avg_vol, 4),
                "breadth_positive": positive,
                "breadth_pct": round(positive / count * 100, 1) if count else 0,
                "best_ticker": securities[0]["ticker"] if securities else "",
                "best_return_pct": securities[0]["return_pct"] if securities else 0,
                "worst_ticker": securities[-1]["ticker"] if securities else "",
                "worst_return_pct": securities[-1]["return_pct"] if securities else 0,
                "bucket": _classify_return(avg_ret),
            }
        return result

    def to_ascii(self, heatmap_data: dict, use_color: bool = True) -> str:
        """Format heatmap as an ASCII table for terminal display.

        Parameters
        ----------
        heatmap_data : dict
            Output of generate_universe_heatmap().
        use_color : bool
            Whether to emit ANSI color codes (disable for file output).
        """
        lines = []
        date_str = heatmap_data.get("date", "")
        period = heatmap_data.get("period", "1d")
        total = heatmap_data.get("ticker_count", 0)
        bucket_counts = heatmap_data.get("bucket_counts", {})

        lines.append("=" * 80)
        lines.append(f"  UNIVERSE HEATMAP  |  {date_str}  |  Period: {period}  |  {total} securities")
        lines.append("=" * 80)

        # Legend
        legend_parts = []
        for bucket, label in _BUCKET_LABELS.items():
            cnt = bucket_counts.get(bucket, 0)
            col = _ANSI[bucket] if use_color else ""
            reset = _ANSI["RESET"] if use_color else ""
            legend_parts.append(f"{col}{label}: {cnt}{reset}")
        lines.append("  " + "  |  ".join(legend_parts))
        lines.append("")

        by_sector: dict[str, list[dict]] = heatmap_data.get("by_sector", {})
        sector_summary: dict[str, dict] = heatmap_data.get("sector_summary", {})

        for sector in sorted(by_sector.keys()):
            securities = by_sector[sector]
            ss = sector_summary.get(sector, {})
            avg_ret = ss.get("avg_return_pct", 0)
            breadth = ss.get("breadth_pct", 0)
            sector_bucket = ss.get("bucket", "FLAT")
            col = _ANSI[sector_bucket] if use_color else ""
            reset = _ANSI["RESET"] if use_color else ""

            lines.append(
                f"{col}  {'─' * 76}{reset}"
            )
            lines.append(
                f"{col}  {sector:<28}  avg={avg_ret:+.2f}%  breadth={breadth:.0f}%  "
                f"({ss.get('ticker_count', 0)} tickers){reset}"
            )
            lines.append(f"{'─' * 80}")

            # Render securities in rows of up to 6
            row_size = 6
            for row_start in range(0, len(securities), row_size):
                row = securities[row_start : row_start + row_size]
                cells = []
                for sec in row:
                    ticker = sec["ticker"]
                    ret = sec["return_pct"]
                    bucket = sec["bucket"]
                    col_s = _ANSI[bucket] if use_color else ""
                    reset_s = _ANSI["RESET"] if use_color else ""
                    cell = f"{col_s}{ticker:<{_TICKER_WIDTH}}{ret:+.2f}%{reset_s}"
                    cells.append(cell)
                lines.append("  " + "  ".join(cells))

            lines.append("")

        # Overall breadth summary
        lines.append("=" * 80)
        total_pos = bucket_counts.get("STRONG_UP", 0) + bucket_counts.get("UP", 0)
        total_neg = bucket_counts.get("STRONG_DOWN", 0) + bucket_counts.get("DOWN", 0)
        lines.append(
            f"  Market Breadth: {total_pos}/{total} advancing  {total_neg}/{total} declining"
        )
        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_security_data(self, tickers: list[str], period: str) -> dict[str, dict]:
        """Download price data and compute per-security metrics."""
        result: dict[str, dict] = {}
        if not tickers:
            return result

        # We need at least 2 periods to compute a return; for 1d, use 2d lookback
        dl_period = "5d" if period == "1d" else period
        chunk_size = 50

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                data = yf.download(chunk, period=dl_period, interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                volume = data["Volume"] if "Volume" in data.columns else data.get("volume", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if isinstance(volume, pd.Series):
                    volume = volume.to_frame()

                if len(close) < 2:
                    continue

                ret_series = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2].replace(0, np.nan)
                # 5-day vol as annualised daily std * sqrt(252)
                daily_ret = close.pct_change().dropna()
                rolling_vol = daily_ret.std() * math.sqrt(252) if len(daily_ret) >= 2 else pd.Series(0.0, index=close.columns)

                for ticker in chunk:
                    if ticker not in close.columns:
                        continue
                    ret_pct = round(_safe_float(ret_series.get(ticker, 0)) * 100, 4)
                    vol_pct = round(_safe_float(rolling_vol.get(ticker, 0)) * 100, 4)
                    last_close = round(_safe_float(close[ticker].iloc[-1]), 4)
                    avg_vol = int(volume[ticker].mean()) if ticker in volume.columns else 0
                    last_vol = int(_safe_float(volume[ticker].iloc[-1])) if ticker in volume.columns else 0

                    result[ticker] = {
                        "ticker": ticker,
                        "return_pct": ret_pct,
                        "bucket": _classify_return(ret_pct),
                        "last_close": last_close,
                        "vol_pct": vol_pct,
                        "volume": last_vol,
                        "avg_volume": avg_vol,
                        "volume_ratio": round(last_vol / avg_vol, 2) if avg_vol > 0 else 0,
                    }
            except Exception as exc:
                logger.warning("_fetch_security_data error for chunk %s: %s", chunk, exc)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, heatmap: dict, ts: str) -> None:
        json_path = self.heatmap_dir / f"{ts}.json"
        with open(json_path, "w") as f:
            json.dump(heatmap, f, indent=2, default=str)
        logger.info("Heatmap saved: %s", json_path)

        # Also save ASCII version
        txt_path = self.heatmap_dir / f"{ts}.txt"
        with open(txt_path, "w") as f:
            f.write(self.to_ascii(heatmap, use_color=False))
        logger.info("ASCII heatmap saved: %s", txt_path)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = HeatmapEngine()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "GS", "V",
               "XOM", "CVX", "JNJ", "UNH", "PFE", "BAC", "WFC", "NFLX", "ADBE", "CRM"]
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Communication Services",
        "TSLA": "Consumer Discretionary", "JPM": "Financials", "GS": "Financials",
        "V": "Financials", "XOM": "Energy", "CVX": "Energy", "JNJ": "Health Care",
        "UNH": "Health Care", "PFE": "Health Care", "BAC": "Financials", "WFC": "Financials",
        "NFLX": "Communication Services", "ADBE": "Technology", "CRM": "Technology",
    }
    heatmap = engine.generate_universe_heatmap(tickers, sector_map=sector_map)
    print(engine.to_ascii(heatmap))
