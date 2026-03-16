"""
Market Pattern Recognition Engine
Detects technical, volume, macro, and earnings patterns.
Generates EXTREME CONVICTION flags when multiple signals align.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternSignal:
    pattern_name: str
    symbol: str
    detected: bool
    confidence: float        # 0.0 – 1.0
    direction: str           # 'bullish' | 'bearish' | 'neutral'
    timeframe: str           # 'intraday' | 'daily' | 'weekly'
    description: str
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "symbol": self.symbol,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "direction": self.direction,
            "timeframe": self.timeframe,
            "description": self.description,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "timestamp": self.timestamp,
        }


@dataclass
class ConvictionFlag:
    symbol: str
    level: str               # 'EXTREME' | 'HIGH' | 'MODERATE' | 'LOW'
    score: float             # 0-100
    signals: List[str]
    direction: str
    recommended_action: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Pattern Detector
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detect chart patterns from OHLCV price series."""

    def __init__(self, min_pattern_bars: int = 5):
        self.min_pattern_bars = min_pattern_bars

    # ---- Utility helpers ----

    @staticmethod
    def _local_maxima(prices: List[float], window: int = 3) -> List[int]:
        maxima = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                maxima.append(i)
        return maxima

    @staticmethod
    def _local_minima(prices: List[float], window: int = 3) -> List[int]:
        minima = []
        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i-window:i+window+1]):
                minima.append(i)
        return minima

    @staticmethod
    def _linear_trend(prices: List[float]) -> Tuple[float, float]:
        """Returns (slope, r_squared) of linear regression."""
        n = len(prices)
        if n < 2:
            return 0.0, 0.0
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(prices) / n
        ss_xy = sum((x[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
        ss_xx = sum((x[i] - x_mean) ** 2 for i in range(n))
        ss_yy = sum((prices[i] - y_mean) ** 2 for i in range(n))
        slope = ss_xy / max(ss_xx, 1e-9)
        r_sq = (ss_xy ** 2) / max(ss_xx * ss_yy, 1e-9)
        return slope, r_sq

    # ---- Technical Patterns ----

    def detect_cup_and_handle(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Cup & Handle: U-shaped recovery followed by small pullback."""
        detected = False
        confidence = 0.0
        direction = "bullish"
        desc = "No cup & handle pattern detected"

        if len(closes) >= 30:
            left = closes[:10]
            bottom = closes[10:20]
            right = closes[20:30]
            left_avg = sum(left) / len(left)
            bottom_avg = sum(bottom) / len(bottom)
            right_avg = sum(right) / len(right)
            cup_depth = (left_avg - bottom_avg) / max(left_avg, 1e-9)
            recovery = (right_avg - bottom_avg) / max(cup_depth * left_avg, 1e-9)
            if 0.1 <= cup_depth <= 0.4 and recovery >= 0.7:
                detected = True
                confidence = min(0.85, 0.5 + cup_depth + recovery * 0.2)
                desc = f"Cup depth {cup_depth:.1%}, recovery {recovery:.1%}"
                if len(closes) >= 35:
                    handle = closes[30:35]
                    handle_pullback = (right_avg - min(handle)) / max(right_avg, 1e-9)
                    if 0.03 <= handle_pullback <= 0.12:
                        confidence = min(0.95, confidence + 0.1)
                        desc += " + handle formation"

        return PatternSignal(
            pattern_name="cup_and_handle", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
            target_price=closes[-1] * 1.15 if detected else None,
            stop_price=closes[-1] * 0.95 if detected else None,
        )

    def detect_head_and_shoulders(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Head & Shoulders: bearish reversal pattern."""
        detected = False
        confidence = 0.0
        direction = "bearish"
        desc = "No head & shoulders detected"

        if len(closes) >= 20:
            maxima = self._local_maxima(closes, window=3)
            if len(maxima) >= 3:
                left_sh = closes[maxima[-3]]
                head = closes[maxima[-2]]
                right_sh = closes[maxima[-1]]
                shoulder_sym = abs(left_sh - right_sh) / max(head, 1e-9)
                head_prominence = (head - max(left_sh, right_sh)) / max(head, 1e-9)
                if head_prominence > 0.02 and shoulder_sym < 0.05:
                    detected = True
                    confidence = min(0.88, 0.6 + head_prominence * 5)
                    neckline = min(closes[maxima[-3]:maxima[-1]])
                    desc = f"H&S: head={head:.2f}, shoulders={left_sh:.2f}/{right_sh:.2f}, neckline≈{neckline:.2f}"

        return PatternSignal(
            pattern_name="head_and_shoulders", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
        )

    def detect_bull_flag(self, symbol: str, closes: List[float],
                          volumes: List[float] = None) -> PatternSignal:
        """Bull flag: sharp rally (pole) followed by consolidation."""
        detected = False
        confidence = 0.0
        direction = "bullish"
        desc = "No bull flag detected"

        if len(closes) >= 15:
            pole = closes[:8]
            flag = closes[8:15]
            pole_gain = (pole[-1] - pole[0]) / max(pole[0], 1e-9)
            flag_slope, flag_r2 = self._linear_trend(flag)
            flag_retracement = (max(flag) - min(flag)) / max(pole[-1] - pole[0], 1e-9)

            if pole_gain > 0.05 and -0.02 < flag_slope / max(pole[-1], 1) < 0:
                if flag_retracement < 0.5 and flag_r2 > 0.5:
                    detected = True
                    confidence = min(0.90, 0.65 + pole_gain)
                    vol_confirm = ""
                    if volumes and len(volumes) >= 15:
                        pole_vol = sum(volumes[:8]) / 8
                        flag_vol = sum(volumes[8:15]) / 7
                        if flag_vol < pole_vol * 0.7:
                            confidence = min(0.95, confidence + 0.05)
                            vol_confirm = " + volume contraction"
                    desc = f"Bull flag: pole gain {pole_gain:.1%}, retracement {flag_retracement:.1%}{vol_confirm}"

        return PatternSignal(
            pattern_name="bull_flag", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
            target_price=closes[-1] * 1.10 if detected else None,
            stop_price=min(closes[-7:]) * 0.99 if detected and len(closes) >= 7 else None,
        )

    def detect_bear_flag(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Bear flag: sharp drop followed by weak consolidation."""
        detected = False
        confidence = 0.0
        direction = "bearish"
        desc = "No bear flag detected"

        if len(closes) >= 15:
            pole = closes[:8]
            flag = closes[8:15]
            pole_drop = (pole[0] - pole[-1]) / max(pole[0], 1e-9)
            flag_slope, _ = self._linear_trend(flag)

            if pole_drop > 0.05 and flag_slope > 0:
                retracement = (max(flag) - min(flag)) / max(pole[0] - pole[-1], 1e-9)
                if retracement < 0.5:
                    detected = True
                    confidence = min(0.88, 0.60 + pole_drop)
                    desc = f"Bear flag: drop {pole_drop:.1%}, bounce {retracement:.1%}"

        return PatternSignal(
            pattern_name="bear_flag", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
        )

    def detect_double_top(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Double top: two peaks at similar levels = bearish reversal."""
        detected = False
        confidence = 0.0
        direction = "bearish"
        desc = "No double top detected"

        if len(closes) >= 20:
            maxima = self._local_maxima(closes, window=4)
            if len(maxima) >= 2:
                peak1 = closes[maxima[-2]]
                peak2 = closes[maxima[-1]]
                similarity = 1 - abs(peak1 - peak2) / max(peak1, 1e-9)
                valley_idx = maxima[-2] + (maxima[-1] - maxima[-2]) // 2
                valley = min(closes[maxima[-2]:maxima[-1]])
                depth = (max(peak1, peak2) - valley) / max(peak1, peak2, 1e-9)
                if similarity > 0.97 and depth > 0.03:
                    detected = True
                    confidence = min(0.87, similarity * 0.9)
                    desc = f"Double top at ${peak1:.2f}/${peak2:.2f}, valley depth {depth:.1%}"

        return PatternSignal(
            pattern_name="double_top", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
        )

    def detect_double_bottom(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Double bottom: two troughs at similar levels = bullish reversal."""
        detected = False
        confidence = 0.0
        direction = "bullish"
        desc = "No double bottom detected"

        if len(closes) >= 20:
            minima = self._local_minima(closes, window=4)
            if len(minima) >= 2:
                trough1 = closes[minima[-2]]
                trough2 = closes[minima[-1]]
                similarity = 1 - abs(trough1 - trough2) / max(trough1, 1e-9)
                peak = max(closes[minima[-2]:minima[-1]])
                depth = (peak - min(trough1, trough2)) / max(peak, 1e-9)
                if similarity > 0.97 and depth > 0.03:
                    detected = True
                    confidence = min(0.87, similarity * 0.9)
                    desc = f"Double bottom at ${trough1:.2f}/${trough2:.2f}, depth {depth:.1%}"

        return PatternSignal(
            pattern_name="double_bottom", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
            target_price=closes[-1] * 1.08 if detected else None,
        )

    def detect_wedge(self, symbol: str, closes: List[float], wedge_type: str = "rising") -> PatternSignal:
        """Rising wedge (bearish) or falling wedge (bullish)."""
        detected = False
        confidence = 0.0
        direction = "bearish" if wedge_type == "rising" else "bullish"
        desc = f"No {wedge_type} wedge detected"

        if len(closes) >= 20:
            highs = [max(closes[max(0, i-2):i+3]) for i in range(len(closes))]
            lows = [min(closes[max(0, i-2):i+3]) for i in range(len(closes))]
            high_slope, high_r2 = self._linear_trend(highs[-20:])
            low_slope, low_r2 = self._linear_trend(lows[-20:])
            converging = abs(high_slope - low_slope) < abs(high_slope) * 0.5
            if wedge_type == "rising":
                if high_slope > 0 and low_slope > 0 and converging and high_r2 > 0.6:
                    detected = True
                    confidence = min(0.82, 0.55 + high_r2 * 0.3)
                    desc = f"Rising wedge: both trendlines rising, converging"
            else:
                if high_slope < 0 and low_slope < 0 and converging and low_r2 > 0.6:
                    detected = True
                    confidence = min(0.82, 0.55 + low_r2 * 0.3)
                    desc = f"Falling wedge: both trendlines declining, converging"

        return PatternSignal(
            pattern_name=f"{wedge_type}_wedge", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
        )

    def detect_triangle(self, symbol: str, closes: List[float]) -> PatternSignal:
        """Symmetrical triangle: converging trendlines, breakout pending."""
        detected = False
        confidence = 0.0
        direction = "neutral"
        desc = "No triangle detected"

        if len(closes) >= 20:
            highs = [max(closes[max(0,i-2):i+3]) for i in range(len(closes))]
            lows = [min(closes[max(0,i-2):i+3]) for i in range(len(closes))]
            high_slope, _ = self._linear_trend(highs[-20:])
            low_slope, _ = self._linear_trend(lows[-20:])
            if high_slope < 0 and low_slope > 0:
                detected = True
                confidence = 0.72
                desc = f"Symmetrical triangle: highs declining, lows rising - breakout imminent"
                direction = "neutral"

        return PatternSignal(
            pattern_name="symmetrical_triangle", symbol=symbol, detected=detected,
            confidence=confidence, direction=direction, timeframe="daily", description=desc,
        )

    def detect_all(self, symbol: str, closes: List[float],
                   volumes: List[float] = None) -> List[PatternSignal]:
        """Run all technical pattern detectors."""
        patterns = [
            self.detect_cup_and_handle(symbol, closes),
            self.detect_head_and_shoulders(symbol, closes),
            self.detect_bull_flag(symbol, closes, volumes),
            self.detect_bear_flag(symbol, closes),
            self.detect_double_top(symbol, closes),
            self.detect_double_bottom(symbol, closes),
            self.detect_wedge(symbol, closes, "rising"),
            self.detect_wedge(symbol, closes, "falling"),
            self.detect_triangle(symbol, closes),
        ]
        return [p for p in patterns if p.detected]


# ---------------------------------------------------------------------------
# Volume Pattern Detector
# ---------------------------------------------------------------------------

class VolumeAnalyzer:
    """Detect volume-based patterns."""

    @staticmethod
    def detect_climax_volume(symbol: str, volumes: List[float],
                              closes: List[float]) -> Optional[PatternSignal]:
        if len(volumes) < 20:
            return None
        avg_vol = sum(volumes[-20:]) / 20
        std_vol = statistics.stdev(volumes[-20:]) if len(volumes) >= 2 else 1
        last_vol = volumes[-1]
        z_score = (last_vol - avg_vol) / max(std_vol, 1)
        if z_score > 2.5:
            price_dir = closes[-1] > closes[-2] if len(closes) >= 2 else True
            direction = "bullish" if price_dir else "bearish"
            return PatternSignal(
                pattern_name="climax_volume", symbol=symbol, detected=True,
                confidence=min(0.90, 0.7 + z_score * 0.05),
                direction=direction, timeframe="daily",
                description=f"Climax volume: {z_score:.1f}x std above avg ({last_vol/avg_vol:.1f}x avg)"
            )
        return None

    @staticmethod
    def detect_accumulation_distribution(symbol: str, closes: List[float],
                                          volumes: List[float]) -> Optional[PatternSignal]:
        if len(closes) < 10 or len(volumes) < 10:
            return None
        n = min(len(closes), len(volumes), 20)
        ad_values = []
        ad = 0.0
        for i in range(1, n):
            clv = ((closes[i] - closes[i-1]) / max(closes[i-1], 1e-9))
            ad += clv * volumes[i]
            ad_values.append(ad)
        if len(ad_values) >= 5:
            recent_trend = ad_values[-1] - ad_values[-5]
            price_trend = closes[-1] - closes[-5]
            if recent_trend > 0 and price_trend < 0:
                return PatternSignal(
                    pattern_name="bullish_accumulation_divergence", symbol=symbol, detected=True,
                    confidence=0.75, direction="bullish", timeframe="daily",
                    description="A/D rising while price falling = accumulation divergence (bullish)"
                )
            elif recent_trend < 0 and price_trend > 0:
                return PatternSignal(
                    pattern_name="bearish_distribution_divergence", symbol=symbol, detected=True,
                    confidence=0.75, direction="bearish", timeframe="daily",
                    description="A/D falling while price rising = distribution divergence (bearish)"
                )
        return None

    def analyze(self, symbol: str, closes: List[float],
                 volumes: List[float]) -> List[PatternSignal]:
        signals = []
        cv = self.detect_climax_volume(symbol, volumes, closes)
        if cv:
            signals.append(cv)
        ad = self.detect_accumulation_distribution(symbol, closes, volumes)
        if ad:
            signals.append(ad)
        return signals


# ---------------------------------------------------------------------------
# Macro Pattern Detector
# ---------------------------------------------------------------------------

class MacroPatternDetector:
    """Detect macro-level regime signals."""

    @staticmethod
    def yield_curve_signal(symbol: str, two_yr: float, ten_yr: float) -> PatternSignal:
        spread = ten_yr - two_yr
        if spread < 0:
            return PatternSignal(
                pattern_name="yield_curve_inversion", symbol=symbol, detected=True,
                confidence=0.85, direction="bearish", timeframe="weekly",
                description=f"Yield curve inverted: 2Y={two_yr:.2f}% 10Y={ten_yr:.2f}% spread={spread:.2f}%"
            )
        elif spread < 0.25:
            return PatternSignal(
                pattern_name="yield_curve_flat", symbol=symbol, detected=True,
                confidence=0.65, direction="bearish", timeframe="weekly",
                description=f"Yield curve flattening: spread={spread:.2f}%"
            )
        return PatternSignal(
            pattern_name="yield_curve_normal", symbol=symbol, detected=False,
            confidence=0.0, direction="neutral", timeframe="weekly",
            description=f"Normal yield curve: spread={spread:.2f}%"
        )

    @staticmethod
    def credit_spread_signal(symbol: str, hy_spread: float,
                              ig_spread: float) -> PatternSignal:
        if hy_spread > 600:
            return PatternSignal(
                pattern_name="credit_spread_blowout", symbol=symbol, detected=True,
                confidence=0.90, direction="bearish", timeframe="weekly",
                description=f"HY spread blowout: {hy_spread:.0f}bps (stress threshold 600bps)"
            )
        elif hy_spread > 400:
            return PatternSignal(
                pattern_name="credit_spread_widening", symbol=symbol, detected=True,
                confidence=0.72, direction="bearish", timeframe="weekly",
                description=f"Credit spreads widening: HY={hy_spread:.0f}bps IG={ig_spread:.0f}bps"
            )
        return PatternSignal(
            pattern_name="credit_spread_tight", symbol=symbol, detected=False,
            confidence=0.0, direction="bullish", timeframe="weekly",
            description=f"Credit spreads tight: HY={hy_spread:.0f}bps"
        )

    @staticmethod
    def vix_term_structure(symbol: str, vix_spot: float,
                            vix3m: float) -> PatternSignal:
        if vix_spot > vix3m:
            return PatternSignal(
                pattern_name="vix_backwardation", symbol=symbol, detected=True,
                confidence=0.82, direction="bearish", timeframe="intraday",
                description=f"VIX term structure inverted (backwardation): spot={vix_spot:.1f} > 3M={vix3m:.1f}"
            )
        if vix_spot > 30:
            return PatternSignal(
                pattern_name="vix_elevated", symbol=symbol, detected=True,
                confidence=0.70, direction="bearish", timeframe="daily",
                description=f"VIX elevated at {vix_spot:.1f} — fear regime"
            )
        return PatternSignal(
            pattern_name="vix_contango", symbol=symbol, detected=False,
            confidence=0.0, direction="bullish", timeframe="daily",
            description=f"VIX normal contango: spot={vix_spot:.1f} < 3M={vix3m:.1f}"
        )

    def analyze(self, symbol: str, macro_data: dict) -> List[PatternSignal]:
        signals = []
        two_yr = macro_data.get("two_yr_yield", 4.5)
        ten_yr = macro_data.get("ten_yr_yield", 4.3)
        hy_spread = macro_data.get("hy_spread", 350)
        ig_spread = macro_data.get("ig_spread", 120)
        vix = macro_data.get("vix", 18.0)
        vix3m = macro_data.get("vix3m", 20.0)

        yc = self.yield_curve_signal(symbol, two_yr, ten_yr)
        if yc.detected:
            signals.append(yc)
        cs = self.credit_spread_signal(symbol, hy_spread, ig_spread)
        if cs.detected:
            signals.append(cs)
        vx = self.vix_term_structure(symbol, vix, vix3m)
        if vx.detected:
            signals.append(vx)
        return signals


# ---------------------------------------------------------------------------
# Earnings Pattern Detector
# ---------------------------------------------------------------------------

class EarningsPatternDetector:
    """Post-Earnings Announcement Drift (PEAD) and guidance revision clusters."""

    @staticmethod
    def detect_pead(symbol: str, closes: List[float],
                     earnings_surprise_pct: float) -> PatternSignal:
        """Detect PEAD: stock drifts in direction of earnings surprise for 60 days."""
        if abs(earnings_surprise_pct) < 5:
            return PatternSignal(
                pattern_name="pead", symbol=symbol, detected=False,
                confidence=0.0, direction="neutral", timeframe="weekly",
                description="No significant earnings surprise"
            )
        direction = "bullish" if earnings_surprise_pct > 0 else "bearish"
        confidence = min(0.85, 0.60 + abs(earnings_surprise_pct) * 0.01)
        return PatternSignal(
            pattern_name="pead", symbol=symbol, detected=True,
            confidence=confidence, direction=direction, timeframe="weekly",
            description=f"PEAD: {earnings_surprise_pct:+.1f}% earnings surprise → expect drift {direction}"
        )

    @staticmethod
    def detect_guidance_revision(symbol: str, num_revisions_up: int,
                                  num_revisions_down: int) -> PatternSignal:
        net = num_revisions_up - num_revisions_down
        if abs(net) >= 3:
            direction = "bullish" if net > 0 else "bearish"
            return PatternSignal(
                pattern_name="guidance_revision_cluster", symbol=symbol, detected=True,
                confidence=min(0.80, 0.60 + abs(net) * 0.05),
                direction=direction, timeframe="weekly",
                description=f"Guidance revision cluster: {num_revisions_up} up, {num_revisions_down} down → {direction}"
            )
        return PatternSignal(
            pattern_name="guidance_revision_cluster", symbol=symbol, detected=False,
            confidence=0.0, direction="neutral", timeframe="weekly",
            description="No significant guidance revision cluster"
        )


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Z-score based statistical anomaly detection."""

    @staticmethod
    def z_score(value: float, series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        mean = sum(series) / len(series)
        std = statistics.stdev(series)
        return (value - mean) / max(std, 1e-9)

    def detect_price_anomaly(self, symbol: str, current_price: float,
                              price_history: List[float]) -> Optional[PatternSignal]:
        z = self.z_score(current_price, price_history)
        if abs(z) >= 3.0:
            direction = "bearish" if z > 0 else "bullish"  # extreme high = bearish reversal
            return PatternSignal(
                pattern_name="price_anomaly", symbol=symbol, detected=True,
                confidence=min(0.95, 0.70 + abs(z) * 0.05),
                direction=direction, timeframe="intraday",
                description=f"Statistical price anomaly: z={z:.2f} (|z|>3 threshold)"
            )
        return None

    def detect_volume_anomaly(self, symbol: str, current_volume: float,
                               volume_history: List[float]) -> Optional[PatternSignal]:
        z = self.z_score(current_volume, volume_history)
        if abs(z) >= 3.0:
            return PatternSignal(
                pattern_name="volume_anomaly", symbol=symbol, detected=True,
                confidence=min(0.90, 0.70 + abs(z) * 0.05),
                direction="neutral", timeframe="intraday",
                description=f"Volume anomaly: z={z:.2f} — unusual trading activity detected"
            )
        return None

    def detect_return_anomaly(self, symbol: str, daily_return: float,
                               return_history: List[float]) -> Optional[PatternSignal]:
        z = self.z_score(daily_return, return_history)
        if abs(z) >= 3.0:
            direction = "bullish" if daily_return > 0 else "bearish"
            return PatternSignal(
                pattern_name="return_anomaly", symbol=symbol, detected=True,
                confidence=min(0.90, 0.70 + abs(z) * 0.05),
                direction=direction, timeframe="intraday",
                description=f"Return anomaly: {daily_return:.2%} return, z={z:.2f}"
            )
        return None

    def scan(self, symbol: str, current_price: float, current_volume: float,
             daily_return: float, price_history: List[float],
             volume_history: List[float], return_history: List[float]) -> List[PatternSignal]:
        signals = []
        pa = self.detect_price_anomaly(symbol, current_price, price_history)
        if pa:
            signals.append(pa)
        va = self.detect_volume_anomaly(symbol, current_volume, volume_history)
        if va:
            signals.append(va)
        ra = self.detect_return_anomaly(symbol, daily_return, return_history)
        if ra:
            signals.append(ra)
        return signals


# ---------------------------------------------------------------------------
# Conviction Scorer
# ---------------------------------------------------------------------------

class ConvictionScorer:
    """
    Aggregate multiple signals into a conviction score.
    EXTREME CONVICTION when score >= 80 and multiple signal types align.
    """

    LEVEL_THRESHOLDS = {
        "EXTREME": 80,
        "HIGH": 60,
        "MODERATE": 40,
        "LOW": 0,
    }

    def score(self, signals: List[PatternSignal], symbol: str) -> ConvictionFlag:
        if not signals:
            return ConvictionFlag(
                symbol=symbol, level="LOW", score=0.0,
                signals=[], direction="neutral", recommended_action="HOLD"
            )

        bullish = [s for s in signals if s.direction == "bullish"]
        bearish = [s for s in signals if s.direction == "bearish"]
        total = len(signals)

        bull_score = sum(s.confidence for s in bullish) / max(total, 1) * 100
        bear_score = sum(s.confidence for s in bearish) / max(total, 1) * 100

        if bull_score >= bear_score:
            direction = "bullish"
            raw_score = bull_score
        else:
            direction = "bearish"
            raw_score = bear_score

        # Bonus for multiple signal types
        signal_types = len({s.pattern_name.split("_")[0] for s in signals})
        diversity_bonus = min(20, signal_types * 4)
        final_score = min(100, raw_score + diversity_bonus)

        # Determine level
        level = "LOW"
        for lvl, threshold in self.LEVEL_THRESHOLDS.items():
            if final_score >= threshold:
                level = lvl
                break

        action = self._recommend_action(level, direction, final_score)

        return ConvictionFlag(
            symbol=symbol,
            level=level,
            score=round(final_score, 1),
            signals=[s.pattern_name for s in signals],
            direction=direction,
            recommended_action=action,
        )

    @staticmethod
    def _recommend_action(level: str, direction: str, score: float) -> str:
        if level == "EXTREME":
            if direction == "bullish":
                return "BUY CALLS - EXTREME CONVICTION - FULL SIZE"
            else:
                return "BUY PUTS - EXTREME CONVICTION - FULL SIZE"
        elif level == "HIGH":
            if direction == "bullish":
                return "BUY EQUITY/CALLS - HIGH CONVICTION - 3/4 SIZE"
            else:
                return "SHORT/PUTS - HIGH CONVICTION - 3/4 SIZE"
        elif level == "MODERATE":
            if direction == "bullish":
                return "BUY EQUITY - MODERATE - HALF SIZE"
            else:
                return "REDUCE/SHORT - MODERATE - HALF SIZE"
        return "HOLD - LOW CONVICTION"


# ---------------------------------------------------------------------------
# Master scanner
# ---------------------------------------------------------------------------

class MarketPatternScanner:
    """Orchestrate all detectors for a universe of symbols."""

    def __init__(self):
        self.technical = PatternDetector()
        self.volume = VolumeAnalyzer()
        self.macro = MacroPatternDetector()
        self.earnings = EarningsPatternDetector()
        self.anomaly = AnomalyDetector()
        self.conviction = ConvictionScorer()

    def scan_symbol(self, symbol: str, closes: List[float],
                    volumes: List[float] = None,
                    macro_data: dict = None,
                    earnings_data: dict = None) -> dict:
        all_signals: List[PatternSignal] = []

        # Technical
        all_signals.extend(self.technical.detect_all(symbol, closes, volumes))

        # Volume
        if volumes:
            all_signals.extend(self.volume.analyze(symbol, closes, volumes))

        # Macro
        if macro_data:
            all_signals.extend(self.macro.analyze(symbol, macro_data))

        # Earnings
        if earnings_data:
            surprise = earnings_data.get("surprise_pct", 0)
            up = earnings_data.get("revisions_up", 0)
            down = earnings_data.get("revisions_down", 0)
            pead = self.earnings.detect_pead(symbol, closes, surprise)
            if pead.detected:
                all_signals.append(pead)
            gr = self.earnings.detect_guidance_revision(symbol, up, down)
            if gr.detected:
                all_signals.append(gr)

        # Anomaly (need history)
        if len(closes) > 20:
            anomalies = self.anomaly.scan(
                symbol=symbol,
                current_price=closes[-1],
                current_volume=volumes[-1] if volumes else 1e6,
                daily_return=(closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0,
                price_history=closes[:-1],
                volume_history=volumes[:-1] if volumes else [],
                return_history=[(closes[i] - closes[i-1]) / closes[i-1]
                                for i in range(1, len(closes)-1)],
            )
            all_signals.extend(anomalies)

        conviction = self.conviction.score(all_signals, symbol)

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "signals_detected": len(all_signals),
            "patterns": [s.to_dict() for s in all_signals],
            "conviction": {
                "level": conviction.level,
                "score": conviction.score,
                "direction": conviction.direction,
                "signals": conviction.signals,
                "recommended_action": conviction.recommended_action,
            },
            "extreme_conviction": conviction.level == "EXTREME",
        }

    def scan_universe(self, universe: Dict[str, dict]) -> List[dict]:
        """
        universe: {symbol: {"closes": [...], "volumes": [...], ...}}
        Returns sorted list of scan results (highest conviction first).
        """
        results = []
        for symbol, data in universe.items():
            result = self.scan_symbol(
                symbol=symbol,
                closes=data.get("closes", []),
                volumes=data.get("volumes"),
                macro_data=data.get("macro"),
                earnings_data=data.get("earnings"),
            )
            results.append(result)
        results.sort(key=lambda x: x["conviction"]["score"], reverse=True)
        return results


if __name__ == "__main__":
    import random
    scanner = MarketPatternScanner()
    closes = [100 + i * 0.5 + random.uniform(-1, 1) for i in range(50)]
    volumes = [1e6 + random.uniform(-2e5, 2e5) for _ in range(50)]
    result = scanner.scan_symbol("NVDA", closes, volumes,
                                  macro_data={"two_yr_yield": 4.8, "ten_yr_yield": 4.3,
                                              "hy_spread": 350, "ig_spread": 120,
                                              "vix": 18.5, "vix3m": 20.0})
    print(f"Symbol: {result['symbol']}")
    print(f"Conviction: {result['conviction']['level']} ({result['conviction']['score']})")
    print(f"Action: {result['conviction']['recommended_action']}")
    print(f"Patterns: {result['signals_detected']}")
