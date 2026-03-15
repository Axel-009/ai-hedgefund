"""
Statistical Anomaly Detector
==============================
Detects anomalies across the securities universe:
  1. Price anomalies: >2σ single-day move vs 60d rolling vol
  2. Volume anomalies: >3x average volume
  3. Correlation breaks: pair correlation moves >0.3 in a day
  4. Options anomalies: unusual put/call ratio vs 30d avg
  5. Cross-asset anomalies: equity/credit divergence
  6. Regime anomalies: sudden regime change (Metadron Cube state flip)

Anomalies are ranked by conviction:
  CRITICAL (>4σ) | HIGH (3-4σ) | MEDIUM (2-3σ) | LOW (1.5-2σ)

Logged to: logs/anomalies/YYYYMMDD.jsonl
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
_ANOMALY_DIR = _BASE_DIR / "logs" / "anomalies"
_ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

# Lookback windows
_PRICE_LOOKBACK = 60
_VOLUME_LOOKBACK = 30
_CORRELATION_LOOKBACK = 30
_CORRELATION_SHORT = 5

# Thresholds
_PRICE_SIGMA_THRESHOLD = 2.0
_VOLUME_RATIO_THRESHOLD = 3.0
_CORRELATION_BREAK_THRESHOLD = 0.30

# Cross-asset tickers
_CROSS_ASSET = {
    "equity_index": "SPY",
    "hy_credit": "HYG",
    "ig_credit": "LQD",
    "rates": "TLT",
    "vix_proxy": "VXX",
}

# Regime detection: rolling realised vol regimes
_REGIME_SHORT = 5
_REGIME_LONG = 21


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _severity(z: float) -> str:
    az = abs(z)
    if az >= 4.0:
        return "CRITICAL"
    if az >= 3.0:
        return "HIGH"
    if az >= 2.0:
        return "MEDIUM"
    return "LOW"


class AnomalyDetector:
    """Detects statistical anomalies across price, volume, correlation, and macro."""

    def __init__(self, anomaly_dir: str | Path | None = None) -> None:
        self.anomaly_dir = Path(anomaly_dir) if anomaly_dir else _ANOMALY_DIR
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def detect_price_anomalies(
        self,
        tickers: list[str],
        lookback: int = _PRICE_LOOKBACK,
    ) -> list[dict]:
        """Detect >2σ single-day price moves vs rolling historical vol.

        Parameters
        ----------
        tickers : list[str]
            Universe tickers to analyse.
        lookback : int
            Rolling lookback days for volatility (default 60).

        Returns
        -------
        list[dict]
            One entry per anomalous ticker, with z-score and severity.
        """
        anomalies = []
        chunk_size = 30

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                period_str = f"{lookback + 10}d"
                data = yf.download(chunk, period=period_str, interval="1d", auto_adjust=True, progress=False)
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if len(close) < 5:
                    continue

                ret = close.pct_change()

                # Use all rows except the last as the lookback window
                hist_ret = ret.iloc[:-1].tail(lookback)
                rolling_mean = hist_ret.mean()
                rolling_std = hist_ret.std()
                last_ret = ret.iloc[-1]

                for ticker in chunk:
                    if ticker not in last_ret.index:
                        continue
                    r = _safe_float(last_ret[ticker])
                    std = _safe_float(rolling_std.get(ticker, 0))
                    mean = _safe_float(rolling_mean.get(ticker, 0))
                    if std <= 0:
                        continue
                    z = (r - mean) / std
                    if abs(z) >= _PRICE_SIGMA_THRESHOLD:
                        last_close = _safe_float(close[ticker].iloc[-1])
                        prev_close = _safe_float(close[ticker].iloc[-2])
                        anomalies.append(
                            {
                                "type": "PRICE_ANOMALY",
                                "ticker": ticker,
                                "z_score": round(z, 4),
                                "severity": _severity(z),
                                "day_return_pct": round(r * 100, 4),
                                "rolling_vol_pct": round(std * 100, 4),
                                "last_close": round(last_close, 4),
                                "prev_close": round(prev_close, 4),
                                "lookback_days": lookback,
                                "detected_at": datetime.utcnow().isoformat() + "Z",
                            }
                        )
            except Exception as exc:
                logger.warning("detect_price_anomalies error chunk %s: %s", chunk, exc)

        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies

    def detect_volume_anomalies(
        self,
        tickers: list[str],
        lookback: int = _VOLUME_LOOKBACK,
        threshold: float = _VOLUME_RATIO_THRESHOLD,
    ) -> list[dict]:
        """Detect tickers with today's volume > threshold × average volume.

        Parameters
        ----------
        tickers : list[str]
            Universe tickers.
        lookback : int
            Days of history for computing average volume.
        threshold : float
            Volume ratio trigger (default 3.0×).

        Returns
        -------
        list[dict]
        """
        anomalies = []
        chunk_size = 30

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                period_str = f"{lookback + 5}d"
                data = yf.download(chunk, period=period_str, interval="1d", auto_adjust=True, progress=False)
                volume = data["Volume"] if "Volume" in data.columns else data.get("volume", pd.DataFrame())
                close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
                if isinstance(volume, pd.Series):
                    volume = volume.to_frame()
                if isinstance(close, pd.Series):
                    close = close.to_frame()
                if volume.empty or len(volume) < 2:
                    continue

                hist_vol = volume.iloc[:-1].tail(lookback)
                avg_vol = hist_vol.mean()
                last_vol = volume.iloc[-1]
                last_ret = (
                    (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2].replace(0, np.nan)
                    if len(close) >= 2
                    else pd.Series(dtype=float)
                )

                for ticker in chunk:
                    if ticker not in last_vol.index:
                        continue
                    lv = _safe_float(last_vol[ticker])
                    av = _safe_float(avg_vol.get(ticker, 0))
                    if av <= 0 or lv <= 0:
                        continue
                    ratio = lv / av
                    if ratio >= threshold:
                        ret_pct = _safe_float(last_ret.get(ticker, 0)) * 100 if not last_ret.empty else 0.0
                        # Compute z-score for volume
                        vol_std = _safe_float(hist_vol[ticker].std() if ticker in hist_vol.columns else 0)
                        vol_mean = av
                        z_vol = (lv - vol_mean) / vol_std if vol_std > 0 else 0.0

                        anomalies.append(
                            {
                                "type": "VOLUME_ANOMALY",
                                "ticker": ticker,
                                "volume_ratio": round(ratio, 2),
                                "z_score": round(z_vol, 4),
                                "severity": (
                                    "CRITICAL" if ratio >= 10
                                    else ("HIGH" if ratio >= 5 else "MEDIUM")
                                ),
                                "last_volume": int(lv),
                                "avg_volume": int(av),
                                "day_return_pct": round(ret_pct, 4),
                                "detected_at": datetime.utcnow().isoformat() + "Z",
                            }
                        )
            except Exception as exc:
                logger.warning("detect_volume_anomalies error chunk %s: %s", chunk, exc)

        anomalies.sort(key=lambda x: x["volume_ratio"], reverse=True)
        return anomalies

    def detect_correlation_breaks(
        self,
        rv_pairs: list[tuple[str, str]],
        lookback: int = _CORRELATION_LOOKBACK,
        short_window: int = _CORRELATION_SHORT,
        threshold: float = _CORRELATION_BREAK_THRESHOLD,
    ) -> list[dict]:
        """Detect pairs where short-window correlation diverges from long-window correlation.

        Parameters
        ----------
        rv_pairs : list[tuple[str, str]]
            List of (ticker_a, ticker_b) pairs to monitor.
        lookback : int
            Long-window lookback for baseline correlation.
        short_window : int
            Short-window for recent correlation.
        threshold : float
            Absolute difference in correlation that triggers an alert.

        Returns
        -------
        list[dict]
        """
        anomalies = []
        if not rv_pairs:
            return anomalies

        all_tickers = list({t for pair in rv_pairs for t in pair})
        try:
            period_str = f"{lookback + 10}d"
            data = yf.download(all_tickers, period=period_str, interval="1d", auto_adjust=True, progress=False)
            close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if len(close) < short_window + 2:
                return anomalies

            ret = close.pct_change().dropna()

            for ticker_a, ticker_b in rv_pairs:
                if ticker_a not in ret.columns or ticker_b not in ret.columns:
                    continue
                try:
                    long_ret = ret.tail(lookback)
                    short_ret = ret.tail(short_window)

                    corr_long = float(long_ret[ticker_a].corr(long_ret[ticker_b]))
                    corr_short = float(short_ret[ticker_a].corr(short_ret[ticker_b]))

                    if not (math.isfinite(corr_long) and math.isfinite(corr_short)):
                        continue

                    delta = corr_short - corr_long
                    if abs(delta) >= threshold:
                        z_approx = abs(delta) / 0.1  # rough z-score approximation
                        anomalies.append(
                            {
                                "type": "CORRELATION_BREAK",
                                "ticker_a": ticker_a,
                                "ticker_b": ticker_b,
                                "correlation_long": round(corr_long, 4),
                                "correlation_short": round(corr_short, 4),
                                "correlation_delta": round(delta, 4),
                                "z_score": round(z_approx, 4),
                                "severity": _severity(z_approx),
                                "direction": "DECORRELATED" if delta < 0 else "NEWLY_CORRELATED",
                                "lookback_long": lookback,
                                "lookback_short": short_window,
                                "detected_at": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                except Exception as exc:
                    logger.debug("Correlation break error %s/%s: %s", ticker_a, ticker_b, exc)

        except Exception as exc:
            logger.warning("detect_correlation_breaks fetch error: %s", exc)

        anomalies.sort(key=lambda x: abs(x["correlation_delta"]), reverse=True)
        return anomalies

    def detect_options_anomalies(
        self,
        tickers: list[str],
    ) -> list[dict]:
        """Detect unusual options activity via yfinance options chains.

        Note: yfinance does not expose historical put/call ratios.
        This method fetches the current options chain snapshot and flags
        tickers where the put/call OI ratio is extreme (>2 or <0.3).

        Returns
        -------
        list[dict]
        """
        anomalies = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                expirations = t.options
                if not expirations:
                    continue
                # Use nearest expiry
                chain = t.option_chain(expirations[0])
                calls_oi = int(chain.calls["openInterest"].sum()) if not chain.calls.empty else 0
                puts_oi = int(chain.puts["openInterest"].sum()) if not chain.puts.empty else 0

                if calls_oi == 0:
                    continue
                pc_ratio = puts_oi / calls_oi

                if pc_ratio > 2.0 or pc_ratio < 0.3:
                    z_approx = abs(math.log(pc_ratio)) * 2  # rough conviction proxy
                    anomalies.append(
                        {
                            "type": "OPTIONS_ANOMALY",
                            "ticker": ticker,
                            "put_call_ratio": round(pc_ratio, 4),
                            "calls_oi": calls_oi,
                            "puts_oi": puts_oi,
                            "z_score": round(z_approx, 4),
                            "severity": _severity(z_approx),
                            "signal": "BEARISH_SKEW" if pc_ratio > 2.0 else "BULLISH_SKEW",
                            "expiry": expirations[0],
                            "detected_at": datetime.utcnow().isoformat() + "Z",
                        }
                    )
            except Exception as exc:
                logger.debug("detect_options_anomalies error %s: %s", ticker, exc)

        anomalies.sort(key=lambda x: abs(x["put_call_ratio"] - 1), reverse=True)
        return anomalies

    def detect_cross_asset_anomalies(self) -> list[dict]:
        """Detect equity/credit and equity/rates divergence.

        Compares SPY vs HYG, LQD, TLT on 5-day rolling correlation and
        flags when equity rallies while credit/rates send conflicting signals.

        Returns
        -------
        list[dict]
        """
        anomalies = []
        tickers = list(_CROSS_ASSET.values())
        try:
            data = yf.download(tickers, period="30d", interval="1d", auto_adjust=True, progress=False)
            close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if len(close) < 5:
                return anomalies

            ret = close.pct_change().dropna()
            spy = _CROSS_ASSET["equity_index"]
            pairs = [
                ("equity/HY credit", spy, _CROSS_ASSET["hy_credit"], "divergence implies risk-off"),
                ("equity/IG credit", spy, _CROSS_ASSET["ig_credit"], "divergence implies flight to quality"),
                ("equity/rates", spy, _CROSS_ASSET["rates"], "equity/duration decoupling"),
            ]

            for label, a, b in [(p[0], p[1], p[2]) for p in pairs]:
                if a not in ret.columns or b not in ret.columns:
                    continue
                short_ret = ret.tail(5)
                long_ret = ret.tail(20)
                corr_short = float(short_ret[a].corr(short_ret[b]))
                corr_long = float(long_ret[a].corr(long_ret[b]))
                delta = corr_short - corr_long

                # Also check: both moving same direction?
                last_a = _safe_float(ret[a].iloc[-1])
                last_b = _safe_float(ret[b].iloc[-1])
                sign_diverge = (last_a > 0) != (last_b > 0) and abs(last_a) > 0.005 and abs(last_b) > 0.005

                if abs(delta) >= 0.3 or sign_diverge:
                    z = abs(delta) / 0.1 + (2.0 if sign_diverge else 0)
                    anomalies.append(
                        {
                            "type": "CROSS_ASSET_ANOMALY",
                            "label": label,
                            "ticker_a": a,
                            "ticker_b": b,
                            "correlation_delta_5_vs_20d": round(delta, 4),
                            "last_return_a_pct": round(last_a * 100, 4),
                            "last_return_b_pct": round(last_b * 100, 4),
                            "sign_divergence": sign_diverge,
                            "z_score": round(z, 4),
                            "severity": _severity(z),
                            "detected_at": datetime.utcnow().isoformat() + "Z",
                        }
                    )
        except Exception as exc:
            logger.warning("detect_cross_asset_anomalies error: %s", exc)

        return anomalies

    def detect_regime_anomalies(
        self,
        reference_ticker: str = "SPY",
    ) -> list[dict]:
        """Detect sudden regime change via rolling realised vol shift.

        Compares short-window (5d) realised vol vs long-window (21d) realised vol
        on the reference ticker. A >50% jump indicates a potential regime flip.

        Returns
        -------
        list[dict]
        """
        anomalies = []
        try:
            data = yf.download(reference_ticker, period="40d", interval="1d", auto_adjust=True, progress=False)
            close = data["Close"] if "Close" in data.columns else data.get("close", pd.DataFrame())
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if len(close) < _REGIME_LONG + 2:
                return anomalies

            col = reference_ticker if reference_ticker in close.columns else close.columns[0]
            ret = close[col].pct_change().dropna()

            short_vol = float(ret.tail(_REGIME_SHORT).std()) * math.sqrt(252)
            long_vol = float(ret.tail(_REGIME_LONG).std()) * math.sqrt(252)

            if long_vol <= 0:
                return anomalies

            vol_ratio = short_vol / long_vol
            z_approx = abs(math.log(vol_ratio)) / 0.15

            if vol_ratio >= 1.5 or vol_ratio <= 0.5:
                regime = "HIGH_VOL_REGIME" if vol_ratio >= 1.5 else "LOW_VOL_REGIME"
                anomalies.append(
                    {
                        "type": "REGIME_ANOMALY",
                        "ticker": reference_ticker,
                        "short_vol_annualised_pct": round(short_vol * 100, 4),
                        "long_vol_annualised_pct": round(long_vol * 100, 4),
                        "vol_ratio": round(vol_ratio, 4),
                        "z_score": round(z_approx, 4),
                        "severity": _severity(z_approx),
                        "regime": regime,
                        "short_window": _REGIME_SHORT,
                        "long_window": _REGIME_LONG,
                        "detected_at": datetime.utcnow().isoformat() + "Z",
                    }
                )
        except Exception as exc:
            logger.warning("detect_regime_anomalies error: %s", exc)

        return anomalies

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarize_anomalies(self, anomaly_list: list[dict]) -> dict:
        """Rank and summarise a list of anomalies.

        Parameters
        ----------
        anomaly_list : list[dict]
            Combined list from any/all detect_* methods.

        Returns
        -------
        dict
            Summary with counts by severity and type, plus top anomalies.
        """
        by_severity: dict[str, list[dict]] = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        by_type: dict[str, list[dict]] = {}

        for anomaly in anomaly_list:
            sev = anomaly.get("severity", "LOW")
            by_severity.setdefault(sev, []).append(anomaly)
            atype = anomaly.get("type", "UNKNOWN")
            by_type.setdefault(atype, []).append(anomaly)

        sorted_all = sorted(anomaly_list, key=lambda x: abs(_safe_float(x.get("z_score", 0))), reverse=True)

        return {
            "total_anomalies": len(anomaly_list),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_type": {k: len(v) for k, v in by_type.items()},
            "top_10": sorted_all[:10],
            "critical_alerts": by_severity.get("CRITICAL", []),
            "high_alerts": by_severity.get("HIGH", []),
        }

    def run_full_detection(
        self,
        tickers: list[str],
        rv_pairs: list[tuple[str, str]] | None = None,
        lookback: int = _PRICE_LOOKBACK,
    ) -> dict:
        """Run all detectors and return a combined, ranked summary.

        Parameters
        ----------
        tickers : list[str]
            Universe tickers.
        rv_pairs : list[tuple[str, str]] | None
            Relative-value pairs for correlation monitoring.
        lookback : int
            Lookback for price anomaly detection.

        Returns
        -------
        dict
            Combined detection results, saved to logs/anomalies/YYYYMMDD.jsonl.
        """
        logger.info("Running full anomaly detection on %d tickers", len(tickers))
        price = self.detect_price_anomalies(tickers, lookback=lookback)
        volume = self.detect_volume_anomalies(tickers)
        corr = self.detect_correlation_breaks(rv_pairs or [])
        cross = self.detect_cross_asset_anomalies()
        regime = self.detect_regime_anomalies()

        all_anomalies = price + volume + corr + cross + regime
        summary = self.summarize_anomalies(all_anomalies)
        summary["detected_at"] = datetime.utcnow().isoformat() + "Z"
        summary["date"] = date.today().isoformat()

        self._log(all_anomalies)
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _log(self, anomalies: list[dict]) -> None:
        """Append anomalies to today's JSONL log file."""
        log_path = self.anomaly_dir / f"{date.today().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_path, "a") as f:
                for anomaly in anomalies:
                    f.write(json.dumps(anomaly, default=str) + "\n")
            logger.info("Logged %d anomalies to %s", len(anomalies), log_path)
        except Exception as exc:
            logger.warning("_log write error: %s", exc)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    detector = AnomalyDetector()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "GS", "XOM"]
    rv_pairs = [("AAPL", "MSFT"), ("JPM", "GS"), ("SPY", "QQQ")]
    summary = detector.run_full_detection(tickers, rv_pairs=rv_pairs)
    print(json.dumps(summary, indent=2, default=str))
