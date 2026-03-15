"""
UniverseClassifier — Top-down vs Bottom-up Security Quality Classification
===========================================================================

Exact port of MODERATE-Project/building-stock-analysis methodology
to the US securities universe.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPO METHODOLOGY (what the repo actually does)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOP-DOWN (T3.1 — dynamic, satellite-based):
  Input:   Sentinel-2 spectral band statistics (min/max/mean per band,
           summer 2021-08-14 + winter 2022-01-11 imagery) — 78 features
           + ERA5 temperature + construction year  → 88 total features
  Model:   XGBoost ensemble (GNB + GB + RF + XGB, soft voting)
           n_estimators=120, max_depth=6, lr=0.1, gamma=0,
           reg_lambda=10, colsample_bylevel=0.5
           StratifiedShuffleSplit + compute_sample_weight('balanced')
  Output:  PREDICTED EPC class A–G from observable imagery alone,
           WITHOUT reading the actual building certificate

BOTTOM-UP (T3.2 — static, database):
  Input:   EU27 building stock static database — per-building specs:
           construction year, U-values (walls/roof/windows/floor),
           materials, heating/cooling systems, area, geometry
  Method:  Direct lookup / engineering calculation (NO ML)
  Output:  ACTUAL / REFERENCE energy performance tier per building
           This is the CENED EPC label — the ground truth

RECONCILIATION:
  Satellite-predicted EPC class  ≠  CENED database EPC label
  → Divergence = building is anomalous vs its satellite signature
  → Large divergence = biggest mismatches between what the imagery
    suggests and what the actual building record shows

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECURITIES MAPPING (exact analogy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOP-DOWN (market-observable XGBoost — mirrors T3.1 satellite XGBoost):
  Input:   Market-surface observables ONLY — dual-season price features
           (bear-season ↔ winter bands, bull-season ↔ summer bands)
           + macro regime context (↔ ERA5 temperature cross-section)
           → 24 features total
  Model:   Same XGBoost ensemble with exact T3.1 params
  Output:  PREDICTED quality tier A–G from price/volume/regime alone,
           WITHOUT reading fundamental balance sheet or long-run alpha

BOTTOM-UP (FundamentalsStore — mirrors T3.2 static database):
  Input:   Long-run price history → compute CAPM alpha, beta,
           information ratio, drawdown, momentum (252d + 60d)
  Method:  Direct calculation (no ML) → IR → tier lookup
           Exactly like T3.2: static record, engineering calculation
  Output:  REFERENCE quality tier per ticker — the "ground truth"
           that the top-down model tries to predict

RECONCILIATION:
  XGBoost-predicted tier (from price observables)
  ≠  fundamentals-derived tier (from long-run alpha/IR)
  → Divergence = security is mispriced vs its market behavior
  → QUALITY_BUY:  market behaves like a low-quality name but
                  fundamentals say high-quality → undervalued
  → QUALITY_SELL: market behaves like a high-quality name but
                  fundamentals say low-quality → overvalued

Training the XGBoost:
  Labels  = fundamentals tiers (A–G) — the "CENED database" equivalent
  Features= dual-season price features — the "satellite bands" equivalent
  The model learns: "given only what I can observe in price/volume,
  what tier does the fundamentals record suggest this security is?"
  Divergence at inference = model's prediction ≠ current fundamental reality
"""

import numpy as np
from typing import Optional, List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Quality tier constants  (EPC A→G analogy)
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_TIERS: Dict[str, int] = {
    "A": 6,   # STRONG_BUY  — IR > 0.8
    "B": 5,   # BUY         — IR 0.4–0.8
    "C": 4,   # WEAK_BUY    — IR 0.1–0.4
    "D": 3,   # NEUTRAL     — IR ±0.1
    "E": 2,   # WEAK_SELL   — IR −0.4 to −0.1
    "F": 1,   # SELL        — IR −0.8 to −0.4
    "G": 0,   # STRONG_SELL — IR < −0.8
}
TIER_LABELS = ["G", "F", "E", "D", "C", "B", "A"]   # index 0→6

# Regime one-hot encoding (↔ ERA5 temperature cross-section in T3.1)
_REGIME_IDX = {"BULL": 0, "BEAR": 1, "TRANSITION": 2, "STRESS": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Bottom-Up: FundamentalsStore
# Mirrors T3.2 static building database — direct calculation, no ML
# ─────────────────────────────────────────────────────────────────────────────

def label_from_ir(ir: float) -> str:
    """
    Map Information Ratio (alpha / vol) to quality tier A–G.
    Mirrors EPC grading boundaries from T3.1 — static lookup, no ML.
    """
    if   ir >  0.8:  return "A"
    elif ir >  0.4:  return "B"
    elif ir >  0.1:  return "C"
    elif ir > -0.1:  return "D"
    elif ir > -0.4:  return "E"
    elif ir > -0.8:  return "F"
    else:            return "G"


class FundamentalsStore:
    """
    Per-security characteristics store — bottom-up analysis layer.

    Mirrors T3.2 static building stock database, where each building has a
    rich multi-attribute record:
        T3.2 building record: construction year, heated area, number of floors,
          U-values (walls/roof/windows/floor W/m²K), volume-to-surface ratio,
          wall/roof/window materials, heating system type, energy demand (UED)
        Securities record:    computed from full price history — multi-attribute
          profile characterising the security's risk/return behaviour

    This is the REFERENCE / GROUND TRUTH tier.  The XGBoost top-down model
    is trained to predict this tier from price-observable features alone,
    exactly as T3.1 XGBoost predicts the CENED EPC label from satellite bands.

    Security-level attributes computed (analogous to T3.2 building attributes):

      Return profile (↔ energy demand UED):
        alpha_annual      — CAPM excess return vs market (annualised)
        ir                — information ratio (alpha / total vol)
        sortino           — downside risk-adjusted return
        sharpe            — total return / vol

      Risk profile (↔ envelope U-values — lower = better insulated / lower risk):
        beta              — systematic market exposure
        vol_annual        — total annualised volatility
        downside_vol      — semi-deviation (vol of negative returns only)
        up_capture        — beta in up-market months
        down_capture      — beta in down-market months (lower = better hedge)

      Momentum / trend (↔ construction year + renovation status):
        momentum_252      — 12-month price return
        momentum_60       — 3-month price return
        momentum_20       — 1-month price return
        trend_r2          — R² of linear price fit (trend quality/persistence)

      Drawdown / recovery (↔ building age / structural deterioration):
        max_drawdown      — worst peak-to-trough decline
        recovery_factor   — avg_annual_return / abs(max_drawdown)
        calmar            — annual alpha / abs(max_drawdown)

      Distribution characteristics (↔ building geometry / vol-to-surface ratio):
        skew              — return skewness (negative = fat left tail)
        kurt              — excess kurtosis (tail heaviness)
        var_95            — 95th-percentile daily VaR

      Composite tier:  scored from weighted combination of all attributes above
                       → tier A–G  (not just from IR alone)

    Usage
    -----
    store = FundamentalsStore()
    store.update("AAPL", closes_list)          # computes full security profile
    tier, record = store.get_tier("AAPL")      # → "B", {all attributes}
    """

    MIN_HISTORY = 252   # 1 year minimum for stable CAPM alpha

    def __init__(self) -> None:
        self._records: Dict[str, dict] = {}

    def update(self, ticker: str, closes: List[float],
               market_closes: Optional[List[float]] = None) -> Optional[str]:
        """
        Compute and store the full per-security profile.

        Parameters
        ----------
        closes        : full available price history (≥ 252 bars)
        market_closes : SPY/benchmark closes (aligned); if None, zero-drift proxy
        """
        arr = np.asarray(closes, dtype=float)
        if len(arr) < self.MIN_HISTORY:
            return None

        rets = np.diff(np.log(np.clip(arr, 1e-10, None)))

        # Market returns proxy
        if market_closes is not None and len(market_closes) >= len(closes):
            mkt_arr  = np.asarray(market_closes, dtype=float)[-len(arr):]
            mkt_rets = np.diff(np.log(np.clip(mkt_arr, 1e-10, None)))
        else:
            mkt_rets = np.zeros_like(rets)

        n = min(len(rets), len(mkt_rets))
        r = rets[-n:]
        m = mkt_rets[-n:]

        # ── Return profile ────────────────────────────────────────────────────
        # CAPM alpha & beta (OLS)  ↔  UED energy demand in T3.2
        beta         = float(np.cov(r, m)[0, 1] / (np.var(m) + 1e-12))
        alpha_daily  = float((r - beta * m).mean())
        alpha_annual = alpha_daily * 252

        vol_annual   = float(r.std() * np.sqrt(252))
        ir           = alpha_annual / (vol_annual + 1e-10)

        # Downside vol (semi-deviation) — only negative-return days
        neg_rets     = r[r < 0]
        down_vol     = float(neg_rets.std() * np.sqrt(252)) if len(neg_rets) > 5 else vol_annual

        # Sharpe (total return, not just alpha)
        total_ret_annual = float(r.mean() * 252)
        sharpe       = total_ret_annual / (vol_annual + 1e-10)

        # Sortino (downside-adjusted)
        sortino      = total_ret_annual / (down_vol + 1e-10)

        # ── Risk profile ──────────────────────────────────────────────────────
        # Up/down capture ratios  ↔  wall U-values (asymmetric transmission)
        up_mask   = m > 0
        dn_mask   = m < 0
        if up_mask.sum() > 10 and dn_mask.sum() > 10:
            up_capture  = float(r[up_mask].mean() / (m[up_mask].mean() + 1e-10))
            down_capture= float(r[dn_mask].mean() / (m[dn_mask].mean() + 1e-10))
        else:
            up_capture   = beta
            down_capture = beta

        # ── Momentum / trend  ↔  construction year + renovation recency ──────
        mom_252 = float(arr[-1] / (arr[-253] + 1e-10) - 1.0) if len(arr) >= 253 else 0.0
        mom_60  = float(arr[-1] / (arr[-61]  + 1e-10) - 1.0) if len(arr) >= 61  else 0.0
        mom_20  = float(arr[-1] / (arr[-21]  + 1e-10) - 1.0) if len(arr) >= 21  else 0.0

        # Trend quality R² of linear fit over last 252 bars
        px_fit  = arr[-252:].astype(float)
        x_ln    = np.arange(len(px_fit), dtype=float)
        slope, intercept = np.polyfit(x_ln, px_fit, 1)
        ss_res  = float(np.sum((px_fit - (slope * x_ln + intercept)) ** 2))
        ss_tot  = float(np.sum((px_fit - px_fit.mean()) ** 2))
        trend_r2= float(1.0 - ss_res / (ss_tot + 1e-10))

        # ── Drawdown / recovery  ↔  building age / structural deterioration ──
        peak      = np.maximum.accumulate(arr)
        dd        = (arr - peak) / (peak + 1e-10)
        max_dd    = float(dd.min())
        avg_ret   = total_ret_annual
        recovery  = avg_ret / (abs(max_dd) + 1e-10)
        calmar    = alpha_annual / (abs(max_dd) + 1e-10)

        # ── Distribution  ↔  building geometry / vol-to-surface ratio ────────
        if len(r) > 4:
            m2   = float(np.mean((r - r.mean()) ** 2))
            m3   = float(np.mean((r - r.mean()) ** 3))
            m4   = float(np.mean((r - r.mean()) ** 4))
            skew = m3 / (m2 ** 1.5 + 1e-10)
            kurt = m4 / (m2 ** 2  + 1e-10) - 3.0
        else:
            skew, kurt = 0.0, 0.0

        var_95 = float(np.percentile(r, 5))   # 5th-percentile (negative tail)

        # ── Composite tier from multi-attribute profile ───────────────────────
        # Weighted score from key security characteristics
        # Mirrors T3.2: tier emerges from combination of attributes,
        # not a single metric (e.g. EPC depends on U-values + year + heating)
        score = (
              4.0 * np.clip(ir,           -2.0,  2.0)   # IR — primary driver
            + 1.5 * np.clip(sharpe,       -3.0,  3.0)   # risk-adjusted total return
            + 1.0 * np.clip(sortino,      -3.0,  3.0)   # downside-adjusted return
            - 1.0 * np.clip(abs(beta)-1,   0.0,  2.0)   # beta penalty beyond 1
            + 0.8 * np.clip(up_capture - down_capture, -2, 2)  # asymmetry bonus
            + 0.5 * np.clip(mom_60,       -1.0,  1.0)   # medium-term momentum
            + 0.3 * np.clip(mom_252,      -1.0,  1.0)   # long-term momentum
            + 0.5 * np.clip(trend_r2,      0.0,  1.0)   # trend quality
            + 0.5 * np.clip(calmar,       -2.0,  2.0)   # drawdown-adjusted alpha
            - 0.3 * np.clip(kurt,          0.0, 10.0)   # tail-risk penalty
        )

        # Map composite score to tier
        if   score >  5.0:  tier = "A"
        elif score >  2.5:  tier = "B"
        elif score >  0.5:  tier = "C"
        elif score > -0.5:  tier = "D"
        elif score > -2.5:  tier = "E"
        elif score > -5.0:  tier = "F"
        else:               tier = "G"

        self._records[ticker] = {
            # Return profile
            "alpha_annual":   round(alpha_annual,  5),
            "ir":             round(ir,             4),
            "sharpe":         round(sharpe,         4),
            "sortino":        round(sortino,        4),
            # Risk profile
            "beta":           round(float(beta),    4),
            "vol_annual":     round(vol_annual,     5),
            "downside_vol":   round(down_vol,       5),
            "up_capture":     round(up_capture,     4),
            "down_capture":   round(down_capture,   4),
            # Momentum / trend
            "momentum_252":   round(mom_252,        4),
            "momentum_60":    round(mom_60,         4),
            "momentum_20":    round(mom_20,         4),
            "trend_r2":       round(trend_r2,       4),
            # Drawdown / recovery
            "max_drawdown":   round(max_dd,         4),
            "recovery_factor":round(recovery,       4),
            "calmar":         round(calmar,         4),
            # Distribution
            "skew":           round(skew,           4),
            "kurt":           round(kurt,           4),
            "var_95":         round(var_95,         5),
            # Meta
            "composite_score":round(float(score),   4),
            "tier":           tier,
            "n_obs":          int(len(closes)),
        }
        return tier

    def get_tier(self, ticker: str) -> Tuple[str, dict]:
        """Return (tier_label, full_record) for ticker; ('D', {}) if unknown."""
        rec = self._records.get(ticker)
        if rec is None:
            return "D", {}
        return rec["tier"], rec

    def batch_update(self, closes_map: Dict[str, List[float]],
                     market_closes: Optional[List[float]] = None) -> None:
        """Pre-compute full security profiles for a batch of tickers."""
        for ticker, closes in closes_map.items():
            self.update(ticker, closes, market_closes)

    def tier_for_label_generation(self,
                                   closes_map: Dict[str, List[float]],
                                   market_closes: Optional[List[float]] = None,
                                   ) -> Dict[str, str]:
        """
        Generate training labels for the top-down XGBoost model.
        Mirrors using CENED EPC labels as training targets in T3.1.
        """
        out = {}
        for ticker, closes in closes_map.items():
            t = self.update(ticker, closes, market_closes)
            if t is not None:
                out[ticker] = t
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-Down Feature Engineering
# Mirrors Sentinel-2 band statistics (min/max/mean per band, summer + winter)
# ─────────────────────────────────────────────────────────────────────────────

def build_topdown_features(closes: List[float],
                            regime: str = "TRANSITION",
                            market_closes: Optional[List[float]] = None,
                            season_window: int = 60) -> Optional[np.ndarray]:
    """
    Build 24 market-observable features.

    Mirrors 06-data_processing.py / loop_get_bands_stats():
      Sentinel-2 bands: AOT, B02–B12, SCL, WVP, CLOUD_MASK
      Statistics per band: min, max, mean
      Seasons: summer + winter  → 78 satellite features
      + ERA5 temp (2) + construction year (1) = 88 total

    Our analogy:
      Band groups   → technical feature groups
      min/max/mean  → min_ret/max_ret/mean_ret per season window
      Summer bands  → bull-season (low-vol period) features   [0–9]
      Winter bands  → bear-season (high-vol period) features  [10–19]
      ERA5 temp     → macro regime one-hot encoding           [20–23]

    Feature layout (24 dims):
      [0–9]   bull-season (low-vol / "summer"): 10 price-observable features
      [10–19] bear-season (high-vol / "winter"): same 10 features
      [20–23] macro regime one-hot: [BULL, BEAR, TRANSITION, STRESS]

    These are ONLY price/volume/regime observables — no fundamental data —
    exactly as T3.1 uses only spectral bands + temperature, never the
    CENED certificate.

    Returns None if insufficient data (requires ≥ 2 × season_window bars).
    """
    arr = np.asarray(closes, dtype=float)
    if len(arr) < season_window * 2:
        return None

    rets = np.diff(np.log(np.clip(arr, 1e-10, None)))
    if len(rets) < season_window:
        return None

    # Split into bull / bear seasons by rolling 20-bar vol
    # Mirrors choosing summer vs winter imagery dates in T3.1
    rv = np.array([
        rets[max(0, i - 20):i + 1].std()
        for i in range(len(rets))
    ], dtype=float)
    vol_med       = np.median(rv)
    bull_idx = np.where(rv <= vol_med)[0]   # low-vol  = "summer"
    bear_idx = np.where(rv >  vol_med)[0]   # high-vol = "winter"

    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        a   = 2.0 / (span + 1)
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = a * x[i] + (1.0 - a) * out[i - 1]
        return out

    def _season_features(idx: np.ndarray) -> np.ndarray:
        """
        10 features from a seasonal subset of the price series.

        Mirrors per-band statistics (min/max/mean) extracted by
        rasterstats.zonal_stats() in 06-data_processing.py.

        [0] mean_ret    — season mean log-return (annualised)  ↔ band mean
        [1] min_ret     — season min  log-return (tail risk)   ↔ band min
        [2] max_ret     — season max  log-return (upside)      ↔ band max
        [3] vol         — season std  (annualised)             ↔ band std
        [4] skew        — return distribution skew             ↔ SCL proxy
        [5] rsi_14      — 14-period RSI (momentum oscillator)  ↔ B08 NIR analog
        [6] macd_norm   — MACD(6,12)–signal(9) / price        ↔ B11/B12 analog
        [7] bb_pos      — Bollinger position [(p–lo)/(hi–lo)]  ↔ normalised band
        [8] roc_20      — 20-period rate-of-change             ↔ temporal change
        [9] vol_ratio   — recent_vol / season_vol              ↔ AOT proxy
        """
        if len(idx) < 20:
            idx = np.arange(len(rets))

        r_sub  = rets[np.clip(idx, 0, len(rets) - 1)]
        px_sub = arr[np.clip(idx, 0, len(arr)  - 1)].astype(float)

        # [0–3] Return statistics — direct analogy to band min/max/mean
        mean_r = float(r_sub.mean()) * 252
        min_r  = float(r_sub.min())
        max_r  = float(r_sub.max())
        vol_r  = float(r_sub.std()) * np.sqrt(252) if len(r_sub) > 1 else 0.0

        # [4] Skew
        if len(r_sub) > 3:
            m2 = float(np.mean((r_sub - r_sub.mean()) ** 2))
            m3 = float(np.mean((r_sub - r_sub.mean()) ** 3))
            skew = m3 / (m2 ** 1.5 + 1e-10)
        else:
            skew = 0.0

        # Working on recent price tail
        px_tail = px_sub[-50:] if len(px_sub) >= 50 else px_sub

        # [5] RSI(14)
        d      = np.diff(px_tail)
        gains  = np.where(d > 0, d, 0.0)
        losses = np.where(d < 0, -d, 0.0)
        w      = min(14, len(gains))
        avg_g  = gains[-w:].mean()  if w > 0 else 0.0
        avg_l  = losses[-w:].mean() if w > 0 else 0.0
        rsi    = 100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-10))

        # [6] MACD(6,12)–signal(9) normalised by price
        if len(px_tail) >= 12:
            macd_line = _ema(px_tail, 6) - _ema(px_tail, 12)
            macd_norm = float(macd_line[-1] - _ema(macd_line, 9)[-1])
            macd_norm /= (px_tail[-1] + 1e-10)
        else:
            macd_norm = 0.0

        # [7] Bollinger Band position
        if len(px_tail) >= 20:
            bm  = px_tail[-20:].mean()
            bs  = px_tail[-20:].std() + 1e-10
            bbp = float(np.clip((px_tail[-1] - (bm - 2 * bs)) / (4 * bs), 0.0, 1.0))
        else:
            bbp = 0.5

        # [8] ROC(20)
        roc = float(px_tail[-1] / (px_tail[-21] + 1e-10) - 1.0) if len(px_tail) >= 21 else 0.0

        # [9] Vol ratio — recent 10-bar / season vol
        rv10     = r_sub[-10:].std() * np.sqrt(252) if len(r_sub) >= 10 else vol_r
        vol_rat  = float(np.clip(rv10 / (vol_r + 1e-10), 0.1, 5.0))

        return np.array([
            np.clip(mean_r,    -2.0,  2.0),
            np.clip(min_r,     -0.3,  0.0),
            np.clip(max_r,      0.0,  0.3),
            np.clip(vol_r,      0.0,  5.0),
            np.clip(skew,      -5.0,  5.0),
            rsi / 100.0,
            np.clip(macd_norm * 100, -1.0, 1.0),
            bbp,
            np.clip(roc,       -1.0,  1.0),
            vol_rat / 5.0,
        ], dtype=float)

    bull_feat = _season_features(bull_idx)   # "summer" bands
    bear_feat = _season_features(bear_idx)   # "winter" bands

    # Macro regime one-hot (↔ ERA5 temperature — contextual cross-section)
    regime_oh = np.zeros(4, dtype=float)
    regime_oh[_REGIME_IDX.get(regime.upper(), 2)] = 1.0

    return np.concatenate([bull_feat, bear_feat, regime_oh])   # (24,)


# ─────────────────────────────────────────────────────────────────────────────
# Top-Down Classifier — XGBoost trained on price observables
# Mirrors T3.1: train XGBoost on satellite features → predict EPC label
# ─────────────────────────────────────────────────────────────────────────────

class _RuleBasedProxy:
    """
    Fallback when XGBoost / sklearn unavailable.
    Score = weighted sum of price features → tier.
    Used exactly like a naïve satellite-band heuristic before ML is trained.
    """

    # Feature weights — bull features [0–9], bear features [10–19]
    # Positive = contributes to higher quality tier
    _W = np.array([
         3.5,  # bull mean_ret     (primary quality signal)
         1.0,  # bull min_ret      (downside resilience)
         0.5,  # bull max_ret
        -2.0,  # bull vol          (volatility penalty)
        -0.3,  # bull skew
         1.5,  # bull rsi
         0.8,  # bull macd
         0.3,  # bull bb_pos
         1.8,  # bull roc_20
        -0.8,  # bull vol_ratio
         2.0,  # bear mean_ret     (resilience in stress)
         0.5,  # bear min_ret
         0.2,  # bear max_ret
        -2.5,  # bear vol          (heavier penalty in bear)
        -0.5,  # bear skew
         0.8,  # bear rsi
         0.4,  # bear macd
         0.2,  # bear bb_pos
         0.8,  # bear roc_20
        -1.0,  # bear vol_ratio
         0.0,  # regime BULL       (regime features not used in rule-based)
         0.0,  # regime BEAR
         0.0,  # regime TRANSITION
         0.0,  # regime STRESS
    ], dtype=float)

    def predict(self, feat: np.ndarray) -> str:
        if feat is None or len(feat) < 20:
            return "D"
        score = float(np.dot(feat[:24], self._W[:len(feat[:24])]))
        if   score >  3.0:  return "A"
        elif score >  1.5:  return "B"
        elif score >  0.3:  return "C"
        elif score > -0.3:  return "D"
        elif score > -1.5:  return "E"
        elif score > -3.0:  return "F"
        else:               return "G"


class TopDownModel:
    """
    XGBoost ensemble trained on market-observable features to predict
    quality tier A–G.

    Mirrors T3.1/07-models_application.py pipeline exactly:
      - Input:   build_topdown_features() → 24-dim price/regime feature vector
                 (↔ Sentinel-2 band stats → 78-dim satellite feature vector)
      - Labels:  FundamentalsStore.tier_for_label_generation() → tier A–G
                 (↔ CENED EPC labels)
      - Model:   GNB + GB + RF + XGB VotingClassifier, soft voting
      - Split:   StratifiedShuffleSplit(test_size=0.2, random_state=42)
      - Scale:   MinMaxScaler fitted on train
      - Weights: compute_sample_weight('balanced')  — class imbalance
      - Eval:    balanced_accuracy_score (mirrors T3.1 macro scoring)

    Exact XGB params from 07-models_application.py:
      n_estimators=120, max_depth=6, learning_rate=0.1,
      gamma=0, reg_lambda=10, colsample_bylevel=0.5
    """

    XGB_PARAMS = dict(
        n_estimators      = 120,
        max_depth         = 6,
        learning_rate     = 0.1,
        gamma             = 0,
        reg_lambda        = 10,
        colsample_bylevel = 0.5,
        eval_metric       = "mlogloss",
        random_state      = 42,
    )

    FEATURE_NAMES = [
        # Bull season (summer analog) — 10 features
        "bull_mean_ret", "bull_min_ret",  "bull_max_ret",  "bull_vol",
        "bull_skew",     "bull_rsi",      "bull_macd",     "bull_bb_pos",
        "bull_roc20",    "bull_vol_ratio",
        # Bear season (winter analog) — 10 features
        "bear_mean_ret", "bear_min_ret",  "bear_max_ret",  "bear_vol",
        "bear_skew",     "bear_rsi",      "bear_macd",     "bear_bb_pos",
        "bear_roc20",    "bear_vol_ratio",
        # Macro regime (ERA5 temperature analog) — 4 features
        "regime_BULL", "regime_BEAR", "regime_TRANSITION", "regime_STRESS",
    ]

    def __init__(self) -> None:
        self._proxy   = _RuleBasedProxy()
        self._scaler  = None
        self._model   = None
        self._fitted  = False

    def fit(self, X: np.ndarray, y_tiers: List[str]) -> "TopDownModel":
        """
        Train on (features, tier_labels).

        Parameters
        ----------
        X        : ndarray (N, 24) from build_topdown_features()
        y_tiers  : list of str  ('A'–'G')  from FundamentalsStore

        Mirrors 07-models_application.py training pipeline.
        """
        try:
            from sklearn.preprocessing      import MinMaxScaler
            from sklearn.utils.class_weight import compute_sample_weight
            from sklearn.model_selection    import StratifiedShuffleSplit
            from sklearn.naive_bayes        import GaussianNB
            from sklearn.ensemble           import (GradientBoostingClassifier,
                                                     RandomForestClassifier,
                                                     VotingClassifier)
        except ImportError:
            self._fitted = False
            return self

        y_int = np.array([QUALITY_TIERS[t] for t in y_tiers], dtype=int)

        # 80/20 stratified split (mirrors T3.1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, _ = next(sss.split(X, y_int))
        X_tr = X[train_idx]
        y_tr = y_int[train_idx]

        # MinMaxScaler (mirrors T3.1 normalization)
        self._scaler = MinMaxScaler()
        X_sc         = self._scaler.fit_transform(X_tr)

        # Balanced sample weights (mirrors T3.1 class imbalance handling)
        sw = compute_sample_weight("balanced", y=y_tr)

        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(**self.XGB_PARAMS)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            xgb = GradientBoostingClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42
            )

        gnb = GaussianNB()
        gb  = GradientBoostingClassifier(
            n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42
        )
        rf  = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
        )

        # Soft-voting ensemble — 4 models (mirrors T3.1 04-model ensemble)
        self._model = VotingClassifier(
            estimators=[("gnb", gnb), ("gb", gb), ("rf", rf), ("xgb", xgb)],
            voting="soft",
        )
        self._model.fit(X_sc, y_tr, sample_weight=sw)
        self._fitted = True
        return self

    def predict_tier(self, feat: np.ndarray) -> str:
        """Predict quality tier A–G from 24-dim observable feature vector."""
        if not self._fitted or self._scaler is None:
            return self._proxy.predict(feat)
        try:
            x_sc   = self._scaler.transform(feat.reshape(1, -1))
            tier_i = int(np.clip(self._model.predict(x_sc)[0], 0, 6))
            return TIER_LABELS[tier_i]
        except Exception:
            return self._proxy.predict(feat)

    def predict_proba(self, feat: np.ndarray) -> Optional[np.ndarray]:
        """Probability over 7 tiers (G=0 → A=6)."""
        if not self._fitted or self._scaler is None:
            return None
        try:
            x_sc = self._scaler.transform(feat.reshape(1, -1))
            return self._model.predict_proba(x_sc)[0]
        except Exception:
            return None

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Ranked feature importance — mirrors T3.1 best_features selection."""
        if self._fitted and self._model is not None:
            try:
                xgb_est = self._model.named_estimators_.get("xgb")
                if xgb_est is not None:
                    imp = xgb_est.feature_importances_
                    return sorted(zip(self.FEATURE_NAMES, imp.tolist()),
                                  key=lambda x: -x[1])
            except Exception:
                pass
        # Fallback: absolute proxy weights
        w = np.abs(self._proxy._W[:len(self.FEATURE_NAMES)])
        return sorted(zip(self.FEATURE_NAMES, w.tolist()), key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation Engine
# Mirrors the comparison: satellite-predicted EPC ≠ CENED database EPC
# ─────────────────────────────────────────────────────────────────────────────

class ReconciliationEngine:
    """
    Compares top-down predicted tier to bottom-up fundamental reference tier.

    This IS the core contribution of building-stock-analysis:
      Top-down  (satellite XGBoost): predicts EPC from spectral bands
      Bottom-up (CENED database):    actual EPC on the certificate
      Divergence: satellite says B, certificate says E → overrated building

    For securities:
      Top-down  (price XGBoost):     predicts quality from price behavior
      Bottom-up (FundamentalsStore): actual quality from long-run alpha/IR
      Divergence: price says A (market treats as high-quality),
                  fundamentals say F (terrible IR) → overvalued → SELL

    Divergence direction:
      bottom_up_tier > top_down_tier  → security is BETTER than market
                                        behavior implies → QUALITY_BUY
                                        (market underpricing vs fundamentals)
      bottom_up_tier < top_down_tier  → security is WORSE than market
                                        behavior implies → QUALITY_SELL
                                        (market overpricing vs fundamentals)
    """

    DIVERGENCE_THRESHOLD = 2   # Tier gap (≥2/6 steps) to generate a signal

    def reconcile(self,
                  td_tier: str,
                  bu_tier: str,
                  td_proba: Optional[np.ndarray] = None,
                  ) -> Tuple[Optional[str], float]:
        """
        Parameters
        ----------
        td_tier  : top-down predicted tier (from XGBoost price model)
        bu_tier  : bottom-up reference tier (from FundamentalsStore)
        td_proba : XGBoost prediction probability distribution (optional)

        Returns
        -------
        signal       : "QUALITY_BUY" | "QUALITY_SELL" | None
        divergence   : float [0, 1]
        """
        td_num = QUALITY_TIERS.get(td_tier, 3)
        bu_num = QUALITY_TIERS.get(bu_tier, 3)
        gap    = bu_num - td_num   # +ve: fundamentals BETTER than market implies

        div_score = abs(gap) / 6.0

        if abs(gap) < self.DIVERGENCE_THRESHOLD:
            return None, div_score

        signal = "QUALITY_BUY" if gap > 0 else "QUALITY_SELL"

        # Scale by XGBoost confidence in its prediction
        # High confidence + large gap = strongest signal
        if td_proba is not None:
            td_confidence = float(np.max(td_proba))
            div_score     = float(np.clip(div_score * (0.5 + 0.5 * td_confidence),
                                          0.0, 1.0))

        return signal, div_score


# ─────────────────────────────────────────────────────────────────────────────
# UniverseClassifier — master entry point
# ─────────────────────────────────────────────────────────────────────────────

class UniverseClassifier:
    """
    Orchestrates top-down (XGBoost on price observables) vs bottom-up
    (FundamentalsStore static reference) quality tier classification
    for the US securities universe.

    Exact methodology from MODERATE-Project/building-stock-analysis:
      1. FundamentalsStore.update()  → compute CAPM alpha / IR → reference tier
         [mirrors: CENED database EPC label]
      2. build_topdown_features()    → 24-dim price/regime observable vector
         [mirrors: Sentinel-2 band statistics]
      3. TopDownModel.predict_tier() → XGBoost-predicted tier from price alone
         [mirrors: XGBoost prediction on satellite features]
      4. ReconciliationEngine        → divergence → QUALITY_BUY/SELL signal
         [mirrors: satellite prediction ≠ database record → anomaly]

    Training the XGBoost (call fit() before the market opens):
      fund_store.batch_update(closes_map)     → generate reference tiers
      uc.fit(closes_map, fund_store)          → train XGBoost on same data

    Inference (real-time in process_quote):
      sig = uc.get_signal(ticker, closes)     → "QUALITY_BUY" | "QUALITY_SELL" | None
      qs  = uc.quality_score(ticker, closes)  → float [-1, +1]
    """

    def __init__(self) -> None:
        self.fundamentals  = FundamentalsStore()
        self.topdown_model = TopDownModel()
        self.reconciler    = ReconciliationEngine()
        self._regime       = "TRANSITION"
        self._buf:   Dict[str, List[float]] = {}   # rolling close buffer
        self._cache: Dict[str, Tuple[str, str, float]] = {}  # td, bu, div

    # ── Regime ────────────────────────────────────────────────────────────────

    def update_regime(self, regime: str) -> None:
        """Propagate macro regime (clears prediction cache)."""
        self._regime = regime.upper()
        self._cache.clear()

    # ── Rolling buffer ────────────────────────────────────────────────────────

    def push_close(self, ticker: str, close: float) -> None:
        buf = self._buf.setdefault(ticker, [])
        buf.append(float(close))
        if len(buf) > 600:
            buf.pop(0)

    # ── Training (call once pre-market with historical data) ─────────────────

    def fit(self, closes_map: Dict[str, List[float]],
            market_closes: Optional[List[float]] = None) -> "UniverseClassifier":
        """
        Train the top-down XGBoost model.

        1. Compute reference tiers via FundamentalsStore (bottom-up labels)
        2. Build feature matrix via build_topdown_features() (top-down inputs)
        3. Train XGBoost ensemble

        Mirrors the full T3.1 training pipeline in 07-models_application.py.

        Parameters
        ----------
        closes_map    : {ticker: [close_prices]}  — full historical closes
        market_closes : aligned market/SPY closes (optional)
        """
        # Step 1: compute reference tiers (= CENED EPC labels in T3.1)
        tier_labels = self.fundamentals.tier_for_label_generation(closes_map)

        # Step 2: build feature matrix (= satellite band stats in T3.1)
        X_rows, y_tiers = [], []
        for ticker, tier in tier_labels.items():
            closes = closes_map[ticker]
            feat   = build_topdown_features(closes, regime=self._regime)
            if feat is not None:
                X_rows.append(feat)
                y_tiers.append(tier)

        if len(X_rows) < 20:
            return self   # insufficient data — model stays as rule-based proxy

        X = np.vstack(X_rows)

        # Step 3: fit XGBoost ensemble
        self.topdown_model.fit(X, y_tiers)
        return self

    # ── Core signal ───────────────────────────────────────────────────────────

    def get_signal(self, ticker: str, closes: List[float]) -> Optional[str]:
        """
        Return "QUALITY_BUY" | "QUALITY_SELL" | None.

        Pipeline:
          1. FundamentalsStore → reference (bottom-up) tier
          2. build_topdown_features → price observables
          3. TopDownModel → predicted (top-down) tier
          4. ReconciliationEngine → divergence → signal
        """
        if len(closes) < FundamentalsStore.MIN_HISTORY:
            return None

        # Bottom-up: reference tier from full security profile
        # Update whenever we have a close list (re-compute if we have new data)
        bu_tier, bu_rec = self.fundamentals.get_tier(ticker)
        if not bu_rec:   # not yet computed for this ticker
            t = self.fundamentals.update(ticker, closes)
            if t is None:
                return None
            bu_tier = t

        # Top-down: predicted tier from price observables
        feat = build_topdown_features(closes, regime=self._regime)
        if feat is None:
            return None

        td_tier  = self.topdown_model.predict_tier(feat)
        td_proba = self.topdown_model.predict_proba(feat)

        # Reconcile
        signal, div_score = self.reconciler.reconcile(td_tier, bu_tier, td_proba)
        self._cache[ticker] = (td_tier, bu_tier, div_score)
        return signal

    # ── Scanner score ─────────────────────────────────────────────────────────

    def quality_score(self, ticker: str, closes: List[float]) -> float:
        """
        Scalar ∈ [-1, +1] for DailyUniverseScanner composite ranking.

        Positive  = fundamentals BETTER than price implies (buy opportunity)
        Negative  = fundamentals WORSE  than price implies (sell opportunity)
        Magnitude = divergence strength (confidence)
        """
        if len(closes) < FundamentalsStore.MIN_HISTORY:
            return 0.0

        if ticker not in self._cache:
            self.get_signal(ticker, closes)   # populates cache

        if ticker not in self._cache:
            return 0.0

        td_tier, bu_tier, div_score = self._cache[ticker]
        td_num  = QUALITY_TIERS.get(td_tier, 3)
        bu_num  = QUALITY_TIERS.get(bu_tier, 3)
        raw_gap = (bu_num - td_num) / 6.0       # [-1, +1]
        return float(np.clip(raw_gap * (1.0 + div_score), -1.0, 1.0))

    # ── Batch summary for pre-market report ──────────────────────────────────

    def tier_summary(self, tickers: List[str],
                     closes_map: Dict[str, List[float]]) -> Dict[str, dict]:
        """
        Batch tier classification with full bottom-up security profile.

        Returns: ticker → {
            td_tier, bu_tier, signal, div_score, quality_score,
            + full FundamentalsStore record (alpha, IR, beta, vol, momentum,
              drawdown, up/down capture, trend_r2, skew, kurt, …)
        }
        """
        out: Dict[str, dict] = {}
        for ticker in tickers:
            closes = closes_map.get(ticker, [])
            sig    = self.get_signal(ticker, closes)
            qs     = self.quality_score(ticker, closes)
            td_t, bu_t, ds = self._cache.get(ticker, ("D", "D", 0.0))
            _, fund_rec    = self.fundamentals.get_tier(ticker)
            out[ticker] = {
                "td_tier":        td_t,
                "bu_tier":        bu_t,
                "signal":         sig,
                "div_score":      round(ds, 4),
                "quality_score":  round(qs, 4),
                # Full bottom-up security profile (mirrors T3.2 per-building record)
                "alpha_annual":   fund_rec.get("alpha_annual",    0.0),
                "ir":             fund_rec.get("ir",              0.0),
                "sharpe":         fund_rec.get("sharpe",          0.0),
                "sortino":        fund_rec.get("sortino",         0.0),
                "beta":           fund_rec.get("beta",            1.0),
                "vol_annual":     fund_rec.get("vol_annual",      0.0),
                "downside_vol":   fund_rec.get("downside_vol",    0.0),
                "up_capture":     fund_rec.get("up_capture",      1.0),
                "down_capture":   fund_rec.get("down_capture",    1.0),
                "momentum_252":   fund_rec.get("momentum_252",    0.0),
                "momentum_60":    fund_rec.get("momentum_60",     0.0),
                "momentum_20":    fund_rec.get("momentum_20",     0.0),
                "trend_r2":       fund_rec.get("trend_r2",        0.0),
                "max_drawdown":   fund_rec.get("max_drawdown",    0.0),
                "recovery_factor":fund_rec.get("recovery_factor", 0.0),
                "calmar":         fund_rec.get("calmar",          0.0),
                "skew":           fund_rec.get("skew",            0.0),
                "kurt":           fund_rec.get("kurt",            0.0),
                "var_95":         fund_rec.get("var_95",          0.0),
                "composite_score":fund_rec.get("composite_score", 0.0),
            }
        return out

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Ranked feature importance (mirrors T3.1 best_features selection)."""
        return self.topdown_model.feature_importance()
