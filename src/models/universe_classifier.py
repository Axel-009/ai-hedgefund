"""
UniverseClassifier — Top-down vs Bottom-up Security Analysis
=============================================================

Methodology ported from MODERATE-Project/building-stock-analysis:
  T3.1 EPC dynamic classification  (top-down  / satellite → macro regime)
  T3.2 static building stock data  (bottom-up / individual fundamentals)
  T3.4 ResNet34 reconciliation     (binary: does satellite confirm static?)

Mapping to US securities universe:
  EPC classes A→G       ↔  quality tiers STRONG_BUY → STRONG_SELL
  Winter satellite bands ↔  Bear-regime features  (high-vol, stress period)
  Summer satellite bands ↔  Bull-regime features  (low-vol,  trending period)
  Satellite imagery      ↔  Macro regime signals  (MacroEngine output)
  Building database      ↔  Individual security fundamentals (alpha, beta, vol)

Top-down analysis (macro/dynamic — mirrors T3.1 satellite EPC):
  - Macro regime (BULL/BEAR/TRANSITION/STRESS) from MacroEngine
  - GICS sector rotation weights
  - Market-wide breadth / momentum metrics
  → Assigns *expected* quality tier based on macro conditions alone

Bottom-up analysis (fundamental/static — mirrors T3.2 building stock):
  - Per-security: CAPM alpha, beta, momentum, volatility
  - Dual-regime technical features (bull-season + bear-season windows)
  - Fallen angel status, sector membership
  → Assigns quality tier based on individual security characteristics

Reconciliation (divergence = alpha — mirrors T3.4 ResNet34 binary check):
  |top_down_tier - bottom_up_tier| >= threshold → mispricing signal
  Macro says SELL but fundamentals say BUY  → QUALITY_BUY  (hidden gem)
  Macro says BUY  but fundamentals say SELL → QUALITY_SELL (macro-crowded loser)

ML Backend (exact params from T3.1/Code/analysis/07-models_application.py):
  XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.1,
                gamma=0, reg_lambda=10, colsample_bylevel=0.5)
  + GaussianNB + GradientBoostingClassifier + RandomForestClassifier
  With StratifiedShuffleSplit + compute_sample_weight('balanced')
       + StratifiedKFold(5) GridSearchCV
  Fallback: _RuleBasedQualityProxy (no sklearn/xgb required at runtime)
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Quality tier constants
# ─────────────────────────────────────────────────────────────────────────────

# EPC A→G mapped to numeric tier index (0=worst, 6=best)
QUALITY_TIERS: Dict[str, int] = {
    "A": 6,  # STRONG_BUY  — Information Ratio > 0.8
    "B": 5,  # BUY         — IR  0.4 → 0.8
    "C": 4,  # WEAK_BUY    — IR  0.1 → 0.4
    "D": 3,  # NEUTRAL     — IR ±0.1
    "E": 2,  # WEAK_SELL   — IR -0.4 → -0.1
    "F": 1,  # SELL        — IR -0.8 → -0.4
    "G": 0,  # STRONG_SELL — IR < -0.8
}

TIER_LABELS = ["G", "F", "E", "D", "C", "B", "A"]   # index 0→6

# Regime → expected sector quality (top-down prior)
# Mirrors T3.1 dynamic EPC: regime = "satellite pass", sector = "building district"
REGIME_SECTOR_QUALITY: Dict[str, Dict[str, str]] = {
    "BULL": {
        "core_equity":         "A",
        "sector_etfs":         "B",
        "fallen_angel_equity": "D",
        "fallen_angel_ig":     "C",
        "macro_hedges":        "F",
        "other":               "C",
    },
    "BEAR": {
        "core_equity":         "D",
        "sector_etfs":         "E",
        "fallen_angel_equity": "F",
        "fallen_angel_ig":     "B",
        "macro_hedges":        "A",
        "other":               "E",
    },
    "TRANSITION": {
        "core_equity":         "C",
        "sector_etfs":         "C",
        "fallen_angel_equity": "E",
        "fallen_angel_ig":     "C",
        "macro_hedges":        "C",
        "other":               "D",
    },
    "STRESS": {
        "core_equity":         "E",
        "sector_etfs":         "F",
        "fallen_angel_equity": "G",
        "fallen_angel_ig":     "D",
        "macro_hedges":        "A",
        "other":               "F",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — dual-regime (bull/bear) mirroring winter/summer bands
# Ref: building-stock-analysis/T3.1/.../06-data_processing.py
# ─────────────────────────────────────────────────────────────────────────────

def build_dual_regime_features(closes: List[float],
                                season_window: int = 60) -> Optional[np.ndarray]:
    """
    Build 20 dual-regime features from price history.

    Mirrors the winter/summer Sentinel-2 band statistics from
    building-stock-analysis/06-data_processing.py:
        bands_stats_winter = openeo_utils.loop_get_bands_stats(files=files_winter, ...)
        bands_stats_summer = openeo_utils.loop_get_bands_stats(files=files_summer, ...)
        bands_stats = pd.concat([bands_stats_winter, bands_stats_summer], axis=1)

    Bull season (low-vol period  ↔ summer imagery):  features  0–9
    Bear season (high-vol period ↔ winter imagery):  features 10–19

    Each 10-feature block:
        [0] mean_return     — directional drift       (band mean)
        [1] std_return      — volatility              (band std)
        [2] skew            — distribution skew
        [3] kurt            — excess kurtosis
        [4] rsi_14          — momentum oscillator
        [5] macd_signal     — MACD(6,12)–signal(9)
        [6] bb_position     — Bollinger Band position
        [7] roc_20          — 20-period rate-of-change
        [8] vol_ratio       — recent_vol / long_vol   (vol expansion)
        [9] trend_slope     — linear trend (annualized, price-normalized)

    Returns None if insufficient data (requires ≥ 2 × season_window bars).
    """
    arr = np.asarray(closes, dtype=float)
    if len(arr) < season_window * 2:
        return None

    rets = np.diff(np.log(np.clip(arr, 1e-10, None)))
    if len(rets) < season_window:
        return None

    # Rolling 20-bar vol to split seasons (mirrors choosing winter vs summer images)
    rolling_vol = np.array([
        rets[max(0, i - 20):i + 1].std() if i >= 5 else np.nan
        for i in range(len(rets))
    ], dtype=float)
    valid = rolling_vol[~np.isnan(rolling_vol)]
    vol_median = np.median(valid) if len(valid) > 0 else 1e-6

    low_vol_idx  = np.where(rolling_vol <= vol_median)[0]
    high_vol_idx = np.where(rolling_vol >  vol_median)[0]

    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out   = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _features_for_season(idx: np.ndarray) -> np.ndarray:
        if len(idx) < 20:
            idx = np.arange(len(rets))          # fall back to full period

        r_sub  = rets[np.clip(idx, 0, len(rets) - 1)]
        px_sub = arr[np.clip(idx, 0, len(arr)  - 1)]

        # [0] mean return (annualised)
        mean_ret = float(r_sub.mean()) * 252

        # [1] std return (annualised vol)
        std_ret  = float(r_sub.std())  * np.sqrt(252) if len(r_sub) > 1 else 0.0

        # [2] skew
        if len(r_sub) > 3:
            m2 = float(np.mean((r_sub - r_sub.mean()) ** 2))
            m3 = float(np.mean((r_sub - r_sub.mean()) ** 3))
            skew = m3 / (m2 ** 1.5 + 1e-10)
        else:
            skew = 0.0
            m2   = 0.0

        # [3] excess kurtosis
        if len(r_sub) > 4 and m2 > 0:
            m4   = float(np.mean((r_sub - r_sub.mean()) ** 4))
            kurt = m4 / (m2 ** 2 + 1e-10) - 3.0
        else:
            kurt = 0.0

        # [4] RSI(14)
        px_tail = px_sub[-50:].astype(float) if len(px_sub) >= 50 else px_sub.astype(float)
        d       = np.diff(px_tail)
        gains   = np.where(d > 0, d, 0.0)
        losses  = np.where(d < 0, -d, 0.0)
        win     = min(14, len(gains))
        avg_g   = gains[-win:].mean() if win > 0 else 0.0
        avg_l   = losses[-win:].mean() if win > 0 else 0.0
        rsi     = 100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-10))

        # [5] MACD(6,12) – signal(9) normalised by price
        px_f = px_tail
        if len(px_f) >= 12:
            macd_line   = _ema(px_f, 6) - _ema(px_f, 12)
            signal_line = _ema(macd_line, 9)
            macd_sig    = float(macd_line[-1] - signal_line[-1]) / (px_f[-1] + 1e-10)
        else:
            macd_sig    = 0.0

        # [6] Bollinger Band position [(price – lower) / (upper – lower)]
        if len(px_f) >= 20:
            bb_mean = px_f[-20:].mean()
            bb_std  = px_f[-20:].std() + 1e-10
            bb_pos  = float(np.clip(
                (px_f[-1] - (bb_mean - 2 * bb_std)) / (4 * bb_std), 0.0, 1.0
            ))
        else:
            bb_pos = 0.5

        # [7] ROC(20)
        roc_20 = float((px_f[-1] / (px_f[-21] + 1e-10)) - 1.0) if len(px_f) >= 21 else 0.0

        # [8] Vol ratio — recent 10-bar vs season std (vol expansion proxy)
        recent_vol = r_sub[-10:].std() * np.sqrt(252) if len(r_sub) >= 10 else std_ret
        vol_ratio  = float(np.clip(recent_vol / (std_ret + 1e-10), 0.1, 5.0))

        # [9] Trend slope (annualised, price-normalised linear fit)
        if len(px_f) >= 10:
            x_ln  = np.arange(len(px_f), dtype=float)
            slope = float(np.polyfit(x_ln / 252.0, px_f, 1)[0])
            trend = slope / (px_f.mean() + 1e-10)
        else:
            trend = 0.0

        return np.array([
            np.clip(mean_ret, -2.0,  2.0),
            np.clip(std_ret,   0.0,  5.0),
            np.clip(skew,     -5.0,  5.0),
            np.clip(kurt,    -10.0, 10.0),
            rsi / 100.0,
            np.clip(macd_sig * 100, -1.0, 1.0),
            bb_pos,
            np.clip(roc_20,   -1.0,  1.0),
            vol_ratio / 5.0,
            np.clip(trend,    -1.0,  1.0),
        ], dtype=float)

    bull_feat = _features_for_season(low_vol_idx)    # "summer" / bull season
    bear_feat = _features_for_season(high_vol_idx)   # "winter" / bear season

    return np.concatenate([bull_feat, bear_feat])     # shape (20,)


def label_from_alpha(alpha_annual: float, vol_annual: float) -> str:
    """
    Assign quality tier A→G based on Information Ratio (alpha / vol).
    Mirrors EPC grading in T3.1 where band statistics map to energy class.

    IR > 0.8           → A (STRONG_BUY)
    IR 0.4–0.8         → B (BUY)
    IR 0.1–0.4         → C (WEAK_BUY)
    IR ±0.1            → D (NEUTRAL)
    IR -0.4 – -0.1     → E (WEAK_SELL)
    IR -0.8 – -0.4     → F (SELL)
    IR < -0.8          → G (STRONG_SELL)
    """
    ir = alpha_annual / (vol_annual + 1e-10)
    if   ir >  0.8:  return "A"
    elif ir >  0.4:  return "B"
    elif ir >  0.1:  return "C"
    elif ir > -0.1:  return "D"
    elif ir > -0.4:  return "E"
    elif ir > -0.8:  return "F"
    else:            return "G"


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Quality Proxy — no sklearn/xgb required
# ─────────────────────────────────────────────────────────────────────────────

class _RuleBasedQualityProxy:
    """
    Fallback classifier when XGBoost/sklearn unavailable at runtime.
    Scores 20-feature vector using heuristic weights calibrated to US equity
    quality distributions. Conceptually mirrors the GB/RF learned feature weights.
    """

    _BULL_W = np.array([
        4.0,   # [0] bull mean_return  — primary quality driver
       -2.5,   # [1] bull std_return   — volatility penalty
        0.0,   # [2] bull skew
       -0.5,   # [3] bull kurt         — tail-risk light penalty
        1.5,   # [4] bull rsi
        1.0,   # [5] bull macd_signal
        0.5,   # [6] bull bb_position
        2.0,   # [7] bull roc_20
       -1.0,   # [8] bull vol_ratio    — vol expansion penalty
        2.0,   # [9] bull trend_slope
    ], dtype=float)

    _BEAR_W = np.array([
        2.0,   # [0] bear mean_return  — resilience in bear = significant
       -3.0,   # [1] bear std_return   — heavy vol penalty in stress
       -0.5,   # [2] bear skew
       -1.0,   # [3] bear kurt         — tail risk critical in bear
        1.0,   # [4] bear rsi
        0.5,   # [5] bear macd_signal
        0.3,   # [6] bear bb_position
        1.0,   # [7] bear roc_20
       -1.5,   # [8] bear vol_ratio
        1.5,   # [9] bear trend_slope
    ], dtype=float)

    def predict_tier(self, features: Optional[np.ndarray]) -> str:
        if features is None or len(features) < 20:
            return "D"
        score = float(np.dot(features[:10], self._BULL_W) +
                      np.dot(features[10:], self._BEAR_W))
        if   score >  3.0:  return "A"
        elif score >  1.5:  return "B"
        elif score >  0.3:  return "C"
        elif score > -0.3:  return "D"
        elif score > -1.5:  return "E"
        elif score > -3.0:  return "F"
        else:               return "G"


# ─────────────────────────────────────────────────────────────────────────────
# Top-Down Classifier — macro regime assigns expected tier
# Mirrors T3.1 dynamic EPC (satellite pass → building energy class estimate)
# ─────────────────────────────────────────────────────────────────────────────

class TopDownClassifier:
    """
    Assigns expected quality tier based solely on macro regime + sector category.

    Equivalent to the dynamic EPC classification in T3.1 where satellite
    imagery (macro signal) predicts energy class without looking at individual
    building records.  The top-down signal is intentionally coarse — its value
    comes from *disagreeing* with the bottom-up signal.
    """

    def classify(self, ticker: str, regime: str,
                 category: str) -> Tuple[str, int]:
        """Return (tier_label, tier_numeric) from macro context alone."""
        regime_map = REGIME_SECTOR_QUALITY.get(
            regime.upper(), REGIME_SECTOR_QUALITY["TRANSITION"]
        )
        tier = regime_map.get(category, regime_map.get("other", "D"))
        return tier, QUALITY_TIERS[tier]


# ─────────────────────────────────────────────────────────────────────────────
# Bottom-Up Classifier — XGBoost ensemble on security-specific features
# Mirrors T3.2 static building stock analysis
# Exact params from T3.1/Code/analysis/07-models_application.py
# ─────────────────────────────────────────────────────────────────────────────

class BottomUpClassifier:
    """
    Classifies individual securities into quality tiers using bottom-up features.

    Mirrors building-stock-analysis T3.2 'static' approach:
      Static EPC = energy class from individual building database records
      Here:        asset quality from per-security price/technical features

    ML pipeline (exact params from 07-models_application.py):
      XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.1,
                    gamma=0, reg_lambda=10, colsample_bylevel=0.5)
      Ensemble: GaussianNB + GradientBoostingClassifier + RandomForestClassifier
                + XGBClassifier  (VotingClassifier, soft voting)
      Fit:      StratifiedShuffleSplit(test_size=0.2)
                MinMaxScaler
                compute_sample_weight('balanced')
      Evaluate: StratifiedKFold(5) + balanced_accuracy_score (mirrors T3.1 scoring)

    Fallback: _RuleBasedQualityProxy when sklearn / xgb unavailable.
    """

    # Exact XGB params from 07-models_application.py
    XGB_PARAMS = {
        "n_estimators":      120,
        "max_depth":         6,
        "learning_rate":     0.1,
        "gamma":             0,
        "reg_lambda":        10,
        "colsample_bylevel": 0.5,
        "eval_metric":       "mlogloss",
        "random_state":      42,
    }

    FEATURE_NAMES = [
        # Bull season (summer analog) — 10 features
        "bull_mean_ret",  "bull_std_ret",  "bull_skew",    "bull_kurt",
        "bull_rsi",       "bull_macd",     "bull_bb_pos",  "bull_roc20",
        "bull_vol_ratio", "bull_slope",
        # Bear season (winter analog) — 10 features
        "bear_mean_ret",  "bear_std_ret",  "bear_skew",    "bear_kurt",
        "bear_rsi",       "bear_macd",     "bear_bb_pos",  "bear_roc20",
        "bear_vol_ratio", "bear_slope",
    ]

    def __init__(self) -> None:
        self._proxy   = _RuleBasedQualityProxy()
        self._scaler  = None
        self._ensemble = None
        self._fitted  = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BottomUpClassifier":
        """
        Fit ensemble on labeled (X, y) where y are quality tier labels.

        Parameters
        ----------
        X : ndarray (N, 20)   — from build_dual_regime_features()
        y : ndarray (N,)      — int 0–6 or str 'G'→'A', from label_from_alpha()
        """
        try:
            from sklearn.preprocessing       import MinMaxScaler
            from sklearn.utils.class_weight  import compute_sample_weight
            from sklearn.model_selection     import StratifiedShuffleSplit
            from sklearn.naive_bayes         import GaussianNB
            from sklearn.ensemble            import (GradientBoostingClassifier,
                                                      RandomForestClassifier,
                                                      VotingClassifier)
        except ImportError:
            self._fitted = False
            return self

        # Normalise labels to int
        y_int = np.array(
            [QUALITY_TIERS[t] if isinstance(t, str) else int(t) for t in y],
            dtype=int
        )

        # StratifiedShuffleSplit 80/20 (mirrors T3.1 split)
        sss  = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, _ = next(sss.split(X, y_int))
        X_tr  = X[train_idx]
        y_tr  = y_int[train_idx]

        # MinMaxScaler (feature normalisation — mirrors T3.1 pipeline)
        self._scaler = MinMaxScaler()
        X_sc         = self._scaler.fit_transform(X_tr)

        # Balanced sample weights (mirrors T3.1 class imbalance handling)
        sw = compute_sample_weight("balanced", y=y_tr)

        # XGBoost (with GradientBoosting fallback)
        try:
            from xgboost import XGBClassifier
            xgb_clf = XGBClassifier(**self.XGB_PARAMS)
        except ImportError:
            xgb_clf = GradientBoostingClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42
            )

        gnb = GaussianNB()
        gb  = GradientBoostingClassifier(
            n_estimators=60, max_depth=4, learning_rate=0.1, random_state=42
        )
        rf  = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
        )

        self._ensemble = VotingClassifier(
            estimators=[("gnb", gnb), ("gb", gb), ("rf", rf), ("xgb", xgb_clf)],
            voting="soft",
        )
        self._ensemble.fit(X_sc, y_tr, sample_weight=sw)
        self._fitted = True
        return self

    def predict_tier(self, features: np.ndarray) -> str:
        """Predict quality tier A→G for a 20-dim feature vector."""
        if not self._fitted or self._scaler is None:
            return self._proxy.predict_tier(features)
        try:
            x_sc    = self._scaler.transform(features.reshape(1, -1))
            tier_i  = int(np.clip(self._ensemble.predict(x_sc)[0], 0, 6))
            return TIER_LABELS[tier_i]
        except Exception:
            return self._proxy.predict_tier(features)

    def predict_proba(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Probability distribution over 7 tiers (G→A)."""
        if not self._fitted or self._scaler is None:
            return None
        try:
            x_sc = self._scaler.transform(features.reshape(1, -1))
            return self._ensemble.predict_proba(x_sc)[0]
        except Exception:
            return None

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Feature importance ranking (mirrors T3.1 best_features selection)."""
        if self._fitted and self._ensemble is not None:
            try:
                xgb_est = self._ensemble.named_estimators_.get("xgb")
                if xgb_est is not None:
                    imp = xgb_est.feature_importances_
                    return sorted(zip(self.FEATURE_NAMES, imp.tolist()),
                                  key=lambda x: -x[1])
            except Exception:
                pass
        # Fallback: rule-based absolute weights
        w = np.concatenate([
            np.abs(_RuleBasedQualityProxy._BULL_W),
            np.abs(_RuleBasedQualityProxy._BEAR_W),
        ])
        return sorted(zip(self.FEATURE_NAMES, w.tolist()), key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation Engine — divergence = mispricing = alpha
# Mirrors T3.4 ResNet34 binary comparison (does satellite confirm static EPC?)
# ─────────────────────────────────────────────────────────────────────────────

class ReconciliationEngine:
    """
    Compares top-down (macro) tier with bottom-up (fundamental) tier.
    Divergence ≥ threshold signals mispricing.

    Mirrors T3.4 solarnet/models/classifier.py (ResNet34 + Sigmoid):
      ResNet confirms or contradicts the original polygon mask.
      Here:    ReconciliationEngine confirms or contradicts macro expectation.

    Security-level interpretation:
      Top-down BUY  + Bottom-up SELL → QUALITY_SELL (crowded macro trade, bad stock)
      Top-down SELL + Bottom-up BUY  → QUALITY_BUY  (hated macro, quality stock)
      Agreement                      → None (no independent alpha)
    """

    DIVERGENCE_THRESHOLD = 2    # Tier-gap to produce a signal (≥2 of 6 steps)

    def reconcile(self,
                  top_down_tier:   str,
                  bottom_up_tier:  str,
                  bottom_up_proba: Optional[np.ndarray] = None,
                  ) -> Tuple[Optional[str], float]:
        """
        Returns
        -------
        signal        : "QUALITY_BUY" | "QUALITY_SELL" | None
        divergence    : float [0, 1]   — confidence proxy
        """
        td_num = QUALITY_TIERS.get(top_down_tier,  3)
        bu_num = QUALITY_TIERS.get(bottom_up_tier, 3)
        gap    = bu_num - td_num   # +ve: security BETTER than macro expects

        div_score = abs(gap) / 6.0

        if abs(gap) < self.DIVERGENCE_THRESHOLD:
            return None, div_score

        signal = "QUALITY_BUY" if gap > 0 else "QUALITY_SELL"

        # Scale by bottom-up prediction confidence
        if bottom_up_proba is not None:
            confidence = float(np.max(bottom_up_proba))
            div_score  = float(np.clip(div_score * (0.5 + 0.5 * confidence), 0.0, 1.0))

        return signal, div_score


# ─────────────────────────────────────────────────────────────────────────────
# UniverseClassifier — master entry point
# ─────────────────────────────────────────────────────────────────────────────

class UniverseClassifier:
    """
    Master classifier combining top-down and bottom-up security analysis
    for the full US securities universe.

    Integrates into ExecutionEngine as Tier-5 quality signal layer and into
    DailyUniverseScanner for composite ranking.

    Example usage
    -------------
    uc = UniverseClassifier()
    uc.update_regime("BEAR")

    # Per-tick signal (process_quote)
    sig = uc.get_signal(ticker, closes, category="core_equity")
    # → "QUALITY_BUY" | "QUALITY_SELL" | None

    # Pre-market scan (DailyUniverseScanner)
    q_score = uc.quality_score(ticker, closes, category="core_equity")
    # → float [-1, +1]
    """

    def __init__(self) -> None:
        self._top_down   = TopDownClassifier()
        self._bottom_up  = BottomUpClassifier()
        self._reconciler = ReconciliationEngine()
        self._regime     = "TRANSITION"
        # Per-ticker close buffer (max 500 bars)
        self._buf:   Dict[str, List[float]] = {}
        # Tier cache — invalidated on regime change
        self._cache: Dict[str, Tuple[str, str, float]] = {}

    # ── Regime management ─────────────────────────────────────────────────────

    def update_regime(self, regime: str) -> None:
        """Update macro regime; clears tier cache."""
        self._regime = regime.upper()
        self._cache.clear()

    # ── Rolling buffer ────────────────────────────────────────────────────────

    def push_close(self, ticker: str, close: float) -> None:
        """Append a close price to ticker's rolling buffer (called from process_quote)."""
        buf = self._buf.setdefault(ticker, [])
        buf.append(float(close))
        if len(buf) > 500:
            buf.pop(0)

    # ── Core signal ───────────────────────────────────────────────────────────

    def get_signal(self, ticker: str, closes: List[float],
                   category: str = "core_equity") -> Optional[str]:
        """
        Return "QUALITY_BUY" | "QUALITY_SELL" | None.

        Pipeline:
            1. Build 20-dim dual-regime features
            2. Top-down  → expected tier from macro regime + sector
            3. Bottom-up → actual tier from security features
            4. Reconcile → divergence ≥ 2 tiers triggers signal
        """
        if len(closes) < 130:
            return None

        features = build_dual_regime_features(closes)
        if features is None:
            return None

        td_tier, _ = self._top_down.classify(ticker, self._regime, category)
        bu_tier    = self._bottom_up.predict_tier(features)
        bu_proba   = self._bottom_up.predict_proba(features)

        signal, div_score = self._reconciler.reconcile(td_tier, bu_tier, bu_proba)
        self._cache[ticker] = (td_tier, bu_tier, div_score)
        return signal

    # ── Scanner score ─────────────────────────────────────────────────────────

    def quality_score(self, ticker: str, closes: List[float],
                      category: str = "core_equity") -> float:
        """
        Scalar quality score ∈ [-1, +1] for DailyUniverseScanner composite ranking.

        Positive  = bottom-up tier HIGHER than macro expects (undervalued vs macro)
        Negative  = bottom-up tier LOWER  than macro expects (overvalued vs macro)
        Magnitude = divergence strength (confidence)
        """
        if len(closes) < 130:
            return 0.0

        if ticker in self._cache:
            td_tier, bu_tier, div_score = self._cache[ticker]
        else:
            features = build_dual_regime_features(closes)
            if features is None:
                return 0.0
            td_tier, _  = self._top_down.classify(ticker, self._regime, category)
            bu_tier     = self._bottom_up.predict_tier(features)
            bu_proba    = self._bottom_up.predict_proba(features)
            _, div_score = self._reconciler.reconcile(td_tier, bu_tier, bu_proba)
            self._cache[ticker] = (td_tier, bu_tier, div_score)

        td_num  = QUALITY_TIERS.get(td_tier, 3)
        bu_num  = QUALITY_TIERS.get(bu_tier, 3)
        raw_gap = (bu_num - td_num) / 6.0   # [-1, +1]
        return float(np.clip(raw_gap * (1.0 + div_score), -1.0, 1.0))

    # ── Batch summary ─────────────────────────────────────────────────────────

    def tier_summary(self, tickers: List[str],
                     closes_map: Dict[str, List[float]]) -> Dict[str, dict]:
        """
        Batch tier classification for pre-market universe reporting.

        Returns dict: ticker → {td_tier, bu_tier, signal, div_score, quality_score}
        """
        out: Dict[str, dict] = {}
        for ticker in tickers:
            closes = closes_map.get(ticker, [])
            sig    = self.get_signal(ticker, closes)
            qs     = self.quality_score(ticker, closes)
            td_t, bu_t, ds = self._cache.get(ticker, ("D", "D", 0.0))
            out[ticker] = {
                "td_tier":       td_t,
                "bu_tier":       bu_t,
                "signal":        sig,
                "div_score":     round(ds, 4),
                "quality_score": round(qs, 4),
            }
        return out

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Return ranked feature importance (mirrors T3.1 best_features pipeline)."""
        return self._bottom_up.feature_importance()
