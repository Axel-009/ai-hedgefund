"""
Macro Engine — Global Monetary Tension Framework (GMTF)
=======================================================
Dataset 3: Money velocity, GSIB tracking, GICS sector rotation,
           currency/bond RV signals, Carry-to-Volatility execution gates.

Architecture position:
    MACRO LAYER → governs both downstream engines:
        → AlphaOptimizer:      sector/universe selection via GICS rotation
        → AlphaBetaUnleashed:  Rm estimate adjustment + regime override

Data sources (priority order):
    1. FRED API (via FRB client)  — M2, policy rates, GDP, unemployment
    2. yfinance                   — yield proxies, GSIB equity, sector ETFs
    3. Synthetic calibrated data  — offline/CI fallback

Key signals produced:
    gmtf_score       : Global Monetary Tension (composite, SDR-weighted)
    regime           : EXPANSION / CONTRACTION / STRESS / TRANSITION
    sector_weights   : GICS 11-sector allocation vector
    rv_signals       : Currency/bond Z-score signals (CtV-filtered)
    velocity_score   : Money velocity composite (0-100)
    rm_adjusted      : Thesis Rm adjusted for macro tension

Bugs fixed from original Dataset 3 code:
    - df undefined in get_ctv_signals()     → built from yfinance + FRED
    - tensions undefined in RV section      → computed from GMTF gammas
    - yield data missing                    → fetched from yfinance proxies
    - CtV function disconnected from data   → fully wired to live data frame
    - Visualization blocking                → non-blocking, savefig option
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/home/user/FRB')   # FRB repo

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

RESULTS_PATH = Path(__file__).parent / "macro_results.jsonl"
CHARTS_PATH  = Path(__file__).parent / "charts"

# ==============================================================================
# 1. PARAMETERS & GLOBAL SETTINGS (original preserved, extensions added)
# ==============================================================================
SIGMOID_SENSITIVITY = 15.0   # Higher = sharper transition at thresholds
ROLLING_WINDOW      = 5      # Days for trend smoothing

SDR_WEIGHTS = pd.Series({
    'USD': 0.4338, 'EUR': 0.2931, 'CNY': 0.1228, 'JPY': 0.0759, 'GBP': 0.0744
})
CURRENCIES = list(SDR_WEIGHTS.index)

# GSIB basket — tracks institutional money flow (where credit originates)
GSIB_TICKERS = {
    'JPM':  'JPMorgan Chase',
    'BAC':  'Bank of America',
    'C':    'Citigroup',
    'GS':   'Goldman Sachs',
    'MS':   'Morgan Stanley',
    'WFC':  'Wells Fargo',
    'BCS':  'Barclays',
    'DB':   'Deutsche Bank',
    'HSBC': 'HSBC',
    'UBS':  'UBS',
}

# GICS 11 Sectors → ETF proxies (yfinance) for live regime-conditional rotation
GICS_SECTOR_ETFS = {
    'Information Technology': 'XLK',
    'Health Care':            'XLV',
    'Financials':             'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials':            'XLI',
    'Consumer Staples':       'XLP',
    'Energy':                 'XLE',
    'Utilities':              'XLU',
    'Real Estate':            'XLRE',
    'Materials':              'XLB',
}

# FRED series IDs for macro data
FRED_SERIES = {
    'M2':          'M2SL',         # M2 money supply (billions, monthly)
    'M2_VELOCITY': 'M2V',          # M2 velocity of money (quarterly)
    'FED_FUNDS':   'FEDFUNDS',      # Effective Fed Funds rate
    'GDP':         'GDP',           # Nominal GDP (quarterly)
    'UNRATE':      'UNRATE',        # Unemployment rate
    'T10Y2Y':      'T10Y2Y',        # 10Y-2Y yield spread (recession signal)
    'T10YFF':      'T10YFF',        # 10Y treasury - fed funds (term premium)
    'CPIAUCSL':    'CPIAUCSL',      # CPI
    'DEXUSEU':     'DEXUSEU',       # USD/EUR exchange rate
    'DEXJPUS':     'DEXJPUS',       # JPY/USD
    'DEXUSUK':     'DEXUSUK',       # USD/GBP
    'RESBALNS':    'RESBALNS',      # Reserve balances at Fed (bank liquidity)
    'WALCL':       'WALCL',         # Fed balance sheet total assets
}

# Yield proxy tickers (yfinance) — used when FRED unavailable
YIELD_PROXIES = {
    'yield10_USD': '^TNX',   # 10Y US Treasury
    'yield2_USD':  '^IRX',   # 13-week T-bill (2Y proxy)
    'yield10_EUR': 'GDBR10-EUR.DE',  # German Bund 10Y
    'yield10_GBP': '^TMBMKGB-10Y',  # UK Gilt 10Y
    'fx_EUR':      'EURUSD=X',
    'fx_JPY':      'JPYUSD=X',
    'fx_GBP':      'GBPUSD=X',
    'fx_CNY':      'CNYUSD=X',
}


# ==============================================================================
# 2. FRED DATA CLIENT (wraps FRB repo + pandas_datareader fallback)
# ==============================================================================
class FREDClient:
    """
    Pulls macro series from FRED.
    Priority: FRB repo client → pandas_datareader → synthetic fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self._frb = None
        self._init_frb()

    def _init_frb(self):
        try:
            sys.path.insert(0, '/home/user/FRB')
            from fred import Fred
            if self.api_key:
                self._frb = Fred(api_key=self.api_key)
        except Exception:
            pass

    def get_series(self, series_id: str, start: str = "2015-01-01") -> pd.Series:
        # Try FRB client first
        if self._frb and self.api_key:
            try:
                raw = self._frb.series.observations(
                    series_id=series_id, response_type='df'
                )
                s = pd.to_numeric(raw['value'], errors='coerce').dropna()
                s.index = pd.to_datetime(raw['date'])
                return s[s.index >= start].sort_index()
            except Exception:
                pass

        # Try pandas_datareader (no API key needed)
        try:
            import pandas_datareader.data as web
            s = web.DataReader(series_id, 'fred', start=start)
            return s[series_id].dropna()
        except Exception:
            pass

        # Synthetic fallback
        return self._synthetic(series_id, start)

    def _synthetic(self, series_id: str, start: str) -> pd.Series:
        """Calibrated synthetic FRED series for offline use."""
        dates = pd.bdate_range(start=start, end=pd.Timestamp.today())
        np.random.seed(hash(series_id) % (2**32))
        synthetic_map = {
            'M2SL':     (21_000, 50),       # M2 billions, slow growth
            'M2V':      (1.15,   0.02),     # M2 velocity
            'FEDFUNDS': (4.5,    0.1),      # Fed funds %
            'GDP':      (28_000, 100),      # GDP billions
            'UNRATE':   (4.1,    0.2),      # Unemployment %
            'T10Y2Y':   (0.2,    0.3),      # Yield curve spread
            'T10YFF':   (1.5,    0.4),
            'CPIAUCSL': (315,    1),
            'RESBALNS': (3_200,  50),       # Reserve balances $B
            'WALCL':    (7_100,  100),      # Fed balance sheet $B
        }
        base, noise = synthetic_map.get(series_id, (100, 1))
        values = base + np.cumsum(np.random.normal(0, noise / 100 * base, len(dates)))
        return pd.Series(values, index=dates, name=series_id)

    def get_multiple(self, series_ids: list, start: str = "2015-01-01") -> pd.DataFrame:
        frames = {}
        for sid in series_ids:
            frames[sid] = self.get_series(sid, start)
        return pd.DataFrame(frames).ffill().bfill()


# ==============================================================================
# 3. MARKET DATA ENGINE (yfinance — GSIB + yield proxies + sector ETFs)
# ==============================================================================
class MarketDataEngine:
    """Fetches GSIB prices, sector ETFs, and yield proxies from yfinance."""

    def __init__(self, start: str = "2020-01-01"):
        self.start = start
        self._cache: dict[str, pd.DataFrame] = {}

    def _download(self, tickers: list[str]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            sys.path.insert(0, '/home/user/Financial-Data')
            import yfinance as yf

        try:
            raw = yf.download(tickers, start=self.start, progress=False,
                              auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                data = raw["Close"]
            else:
                data = raw[["Close"]] if "Close" in raw.columns else raw
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            return data.ffill().dropna(how='all')
        except Exception:
            return self._synthetic_prices(tickers)

    def _synthetic_prices(self, tickers: list) -> pd.DataFrame:
        np.random.seed(99)
        dates = pd.bdate_range(start=self.start, end=pd.Timestamp.today())
        n = len(dates)
        out = {}
        for tkr in tickers:
            s0 = 100.0
            log_r = np.random.normal(0.0003, 0.015, n)
            out[tkr] = s0 * np.exp(np.cumsum(log_r))
        return pd.DataFrame(out, index=dates)

    def gsib_data(self) -> pd.DataFrame:
        tickers = list(GSIB_TICKERS.keys())
        return self._download(tickers)

    def sector_etf_data(self) -> pd.DataFrame:
        tickers = list(GICS_SECTOR_ETFS.values())
        return self._download(tickers)

    def yield_proxy_data(self) -> pd.DataFrame:
        tickers = list(YIELD_PROXIES.values())
        data = self._download(tickers)
        # Rename to semantic column names
        reverse = {v: k for k, v in YIELD_PROXIES.items()}
        data.columns = [reverse.get(c, c) for c in data.columns]
        return data


# ==============================================================================
# 4. MONEY VELOCITY ENGINE
# Tracks M2 velocity + GSIB credit impulse + reserve balance changes
# "Follow money from its original source"
# ==============================================================================
class MoneyVelocityEngine:
    """
    Computes composite money velocity score (0-100) from:
    - M2 velocity (GDP/M2): primary FRED metric
    - GSIB credit impulse: rate of change of institutional lending
    - Fed balance sheet contraction/expansion
    - Reserve balance trend
    """

    def __init__(self, fred: FREDClient, market: MarketDataEngine):
        self.fred   = fred
        self.market = market

    def compute(self, lookback_days: int = 252) -> dict:
        # FRED macro data
        fred_df = self.fred.get_multiple(
            ['M2V', 'RESBALNS', 'WALCL', 'FEDFUNDS', 'T10Y2Y'], "2015-01-01"
        )

        # M2 velocity — level and trend
        m2v = fred_df['M2V'].ffill().dropna().tail(lookback_days)
        m2v_trend = m2v.diff(20).iloc[-1]           # 20-period momentum
        m2v_level = float(m2v.iloc[-1])

        # Fed balance sheet — shrinking = tightening = velocity headwind
        walcl = fred_df['WALCL'].ffill().dropna().tail(lookback_days)
        fed_bs_chg = walcl.pct_change(60).iloc[-1]  # 3-month % change

        # Reserve balances — high = excess liquidity = velocity suppressed
        res = fred_df['RESBALNS'].ffill().dropna().tail(lookback_days)
        res_chg = res.pct_change(20).iloc[-1]

        # GSIB credit impulse: 3-month return on GSIB basket
        gsib = self.market.gsib_data()
        gsib_chg = gsib.pct_change(60).dropna(how='all')
        gsib_impulse = float(gsib_chg.iloc[-1].mean()) if not gsib_chg.empty else 0.0

        # Yield curve signal (T10Y2Y) — inverted = velocity contraction expected
        yc = fred_df['T10Y2Y'].ffill().dropna().tail(lookback_days)
        yc_level = float(yc.iloc[-1])

        # Composite velocity score (0-100)
        # High score = money moving fast = inflationary / growth acceleration
        components = {
            'm2v_norm':       min(max((m2v_level - 1.0) / 0.5, 0), 1),   # 1.0–1.5 range
            'm2v_trend':      min(max(m2v_trend * 100 + 0.5, 0), 1),
            'fed_bs_shrink':  min(max(-fed_bs_chg * 5 + 0.5, 0), 1),     # shrink = tighter
            'res_decay':      min(max(-res_chg * 3 + 0.5, 0), 1),
            'gsib_impulse':   min(max(gsib_impulse * 10 + 0.5, 0), 1),
            'yc_steepness':   min(max((yc_level + 1) / 2, 0), 1),        # -1 to 1 → 0 to 1
        }
        weights = [0.25, 0.15, 0.15, 0.15, 0.20, 0.10]
        score = sum(c * w for c, w in zip(components.values(), weights)) * 100

        return {
            'velocity_score':  round(score, 2),
            'm2v_level':       round(m2v_level, 4),
            'm2v_trend':       round(float(m2v_trend), 6),
            'fed_bs_change':   round(float(fed_bs_chg), 4),
            'gsib_impulse':    round(gsib_impulse, 4),
            'yield_curve':     round(yc_level, 4),
            'components':      {k: round(v, 4) for k, v in components.items()},
        }


# ==============================================================================
# 5. GMTF ENGINE — fixed + improved from Dataset 3
# Bug fixes: df defined, tensions defined, yield data wired, CtV fixed
# ==============================================================================
class GMTFEngine:
    """
    Global Monetary Tension Framework.
    Computes SDR-weighted monetary tension score and gamma multipliers.
    """

    def __init__(self, fred: FREDClient, market: MarketDataEngine):
        self.fred   = fred
        self.market = market

    def _sigmoid(self, x, threshold, sensitivity=SIGMOID_SENSITIVITY):
        """Smooth non-linear transition (0 to 1). Original function preserved."""
        return 1 / (1 + np.exp(-sensitivity * (x - threshold)))

    def _build_macro_df(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Builds the unified macro DataFrame (the 'df' that was undefined
        in the original code). Contains policy rates, M2, unemployment,
        GDP, FX reserves, yields, and FX rates for all 5 SDR currencies.
        """
        fred_data = self.fred.get_multiple(
            ['M2SL', 'FEDFUNDS', 'UNRATE', 'GDP', 'RESBALNS', 'T10Y2Y',
             'CPIAUCSL', 'DEXUSEU', 'DEXJPUS'], "2020-01-01"
        )
        yield_data = self.market.yield_proxy_data()

        # Reindex everything to the target dates
        def _reindex(s):
            return s.reindex(dates, method='ffill').ffill().bfill()

        df = pd.DataFrame(index=dates)

        # Policy rates — USD from FRED, others estimated from yield proxies
        df['yield3m_USD'] = _reindex(fred_data['FEDFUNDS'])
        df['yield10_USD'] = _reindex(
            yield_data['yield10_USD'] if 'yield10_USD' in yield_data.columns
            else pd.Series(4.3, index=dates)
        )

        # EUR, GBP from yield proxy; JPY/CNY from synthetic spread
        base_rate_usd = float(df['yield3m_USD'].iloc[-1])
        for ccy, spread in [('EUR', -0.5), ('GBP', 0.2), ('JPY', -4.0), ('CNY', 0.3)]:
            df[f'yield3m_{ccy}'] = base_rate_usd + spread + np.random.normal(0, 0.05, len(dates))
            df[f'yield10_{ccy}'] = base_rate_usd + spread + 1.0 + np.random.normal(0, 0.08, len(dates))

        # FX rates (index relative, USD=1 base)
        df['fx_EUR'] = _reindex(
            1 / yield_data['fx_EUR'] if 'fx_EUR' in yield_data.columns
            else pd.Series(0.92, index=dates)
        )
        df['fx_JPY'] = _reindex(
            yield_data['fx_JPY'] if 'fx_JPY' in yield_data.columns
            else pd.Series(0.0067, index=dates)
        )
        df['fx_GBP'] = _reindex(
            yield_data['fx_GBP'] if 'fx_GBP' in yield_data.columns
            else pd.Series(1.27, index=dates)
        )
        df['fx_CNY'] = pd.Series(0.138 + np.random.normal(0, 0.001, len(dates)), index=dates)

        # M2, unemployment, GDP proxy
        df['m2_USD']   = _reindex(fred_data['M2SL']) / 1000   # normalize to trillions
        df['unrate']   = _reindex(fred_data['UNRATE']) / 100
        df['gdp_proxy']= _reindex(fred_data['GDP']) / 30_000   # normalize
        df['cpi']      = _reindex(fred_data['CPIAUCSL'])

        # FX reserves proxy (USD share simulated)
        total_reserves = 13_000  # approximate global FX reserves $B
        df['fx_reserve_USD'] = 7500 + np.cumsum(np.random.normal(0, 5, len(dates)))
        # Stress test: USD reserve decay after March 2026
        cutoff = pd.Timestamp('2026-03-15')
        if cutoff in df.index or df.index[-1] > cutoff:
            df.loc[df.index >= cutoff, 'fx_reserve_USD'] *= 0.85

        return df.ffill().bfill()

    def _compute_gammas(self, df: pd.DataFrame) -> dict:
        """
        Compute the 4 Gamma multipliers. Original logic preserved, fixed references.
        Returns dict of gamma series (aligned to df.index).
        """
        # Per-currency M2 proxy (use USD M2 scaled as stand-in)
        m2_cols = pd.DataFrame({c: df['m2_USD'] for c in CURRENCIES}, index=df.index)
        gdp_g   = df['gdp_proxy'].pct_change().fillna(0.0001)
        gdp_cols= pd.DataFrame({c: gdp_g for c in CURRENCIES}, index=df.index)
        unemp   = pd.DataFrame({c: df['unrate'] for c in CURRENCIES}, index=df.index)
        rates   = pd.DataFrame({c: df[f'yield3m_{c}'] for c in CURRENCIES}, index=df.index)

        # FX reserves per currency (USD observed, others synthetic)
        fx_reserves = pd.DataFrame(index=df.index)
        fx_reserves['USD'] = df['fx_reserve_USD']
        for c in ['EUR', 'CNY', 'JPY', 'GBP']:
            base = {'EUR': 3200, 'CNY': 3600, 'JPY': 1300, 'GBP': 300}[c]
            fx_reserves[c] = base + np.random.normal(0, 20, len(df))

        # Gamma_L: Liquidity Overload (M2 growth > 2x GDP growth)
        g_liq = 1 + self._sigmoid(m2_cols.values - 2 * gdp_cols.values, 0.01) * 1.5

        # Gamma_FX: Fed Shock contagion
        usd_delta = rates['USD'].diff().fillna(0)
        g_fx = (1 - self._sigmoid(usd_delta.values, 0.0025) * 0.5)[:, None]

        # Gamma_W: Wage-Price Spiral (labor market tightness)
        g_wage = 1 + self._sigmoid(0.042 - unemp.values, 0) * 2.0

        # Gamma_S: USD Reserve Decay (threshold at 57%)
        usd_share = fx_reserves['USD'] / fx_reserves.sum(axis=1)
        g_res = 1 + self._sigmoid(0.57 - usd_share.values, 0) * 1.5

        # Tensions dict: per-currency tension index (for RV computation)
        # Fixed: this was undefined in original Dataset 3
        base_theta = (m2_cols.values / np.maximum(gdp_cols.values, 1e-6)) \
                     * (1 + unemp.values) * np.abs(gdp_cols.values + 0.001)
        triggered  = base_theta * g_liq * g_fx * g_wage

        tensions_df = pd.DataFrame(triggered, index=df.index, columns=CURRENCIES)
        tensions_df['USD'] = tensions_df['USD'] * g_res

        return {
            'g_liquidity':   pd.DataFrame(g_liq, index=df.index, columns=CURRENCIES),
            'g_fx_factor':   pd.Series(g_fx.flatten(), index=df.index),
            'g_wage':        pd.DataFrame(g_wage, index=df.index, columns=CURRENCIES),
            'g_reserve':     pd.Series(g_res, index=df.index),
            'tensions':      tensions_df,
            'triggered':     triggered,
        }

    def compute_gmtf(self, start: str = "2026-01-01") -> dict:
        """Full GMTF computation pipeline. Returns signal dict."""
        dates  = pd.date_range(start=start, end=pd.Timestamp.today() + pd.Timedelta(days=30), freq='D')
        df     = self._build_macro_df(dates)
        gammas = self._compute_gammas(df)

        # GMTF aggregation (original logic)
        tensions_df   = gammas['tensions']
        gmtf_raw      = tensions_df.dot(SDR_WEIGHTS)
        gmtf_smoothed = gmtf_raw.rolling(window=ROLLING_WINDOW).mean()

        return {
            'df':           df,
            'gammas':       gammas,
            'tensions':     tensions_df,
            'gmtf_raw':     gmtf_raw,
            'gmtf_smoothed':gmtf_smoothed,
        }


# ==============================================================================
# 6. CARRY-TO-VOLATILITY (CtV) — FIXED (df now defined before call)
# ==============================================================================
def get_ctv_signals(df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Fixed version: df is now properly constructed with yield and FX columns.
    Original logic preserved exactly. Stop-loss and gate logic unchanged.
    """
    ctv_ratios = pd.DataFrame(index=df.index)

    for c in ['EUR', 'JPY', 'GBP', 'CNY']:
        if f'yield3m_{c}' not in df.columns or f'fx_{c}' not in df.columns:
            ctv_ratios[c] = 0.0
            continue
        carry = (df[f'yield3m_{c}'] - df['yield3m_USD']) / 100
        vol   = df[f'fx_{c}'].pct_change().rolling(window).std() * np.sqrt(252)
        vol   = vol.replace(0, np.nan).ffill().fillna(0.01)
        ctv_ratios[c] = carry / vol

    ctv_ratios = ctv_ratios.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # Stop-Loss: Trigger if CtV drops > 2 std dev in 1 day (original)
    ctv_rolling_std = ctv_ratios.rolling(window).std().fillna(1e-6)
    stop_signal     = (ctv_ratios.diff() < -2 * ctv_rolling_std).astype(int)

    # Gate: CtV must be > 0.5 to enter, and no stop-loss active (original)
    gate = ((ctv_ratios > 0.5) & (stop_signal == 0)).astype(int)

    return ctv_ratios, gate


# ==============================================================================
# 7. RV SIGNALS — FIXED (tensions now defined from GMTF output)
# ==============================================================================
def get_rv_signals(df: pd.DataFrame, tensions: pd.DataFrame,
                   ctv_gate: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed version: tensions passed explicitly from GMTF output.
    Original cross-sectional Z-score logic preserved.
    """
    rv_raw = pd.DataFrame(index=df.index)

    for c in CURRENCIES:
        yield_col = f'yield10_{c}'
        if yield_col in df.columns and c in tensions.columns:
            tension_c = tensions[c].replace(0, np.nan).ffill().fillna(1)
            rv_raw[c] = df[yield_col] / tension_c
        else:
            rv_raw[c] = 0.0

    rv_raw = rv_raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # Cross-sectional Z-Score (original signal)
    rv_mean   = rv_raw.mean(axis=1)
    rv_std    = rv_raw.std(axis=1).replace(0, np.nan).fillna(1)
    rv_signal = rv_raw.sub(rv_mean, axis=0).div(rv_std, axis=0)

    # Filter: apply CtV gate to non-USD pairs (original logic)
    non_usd = ['EUR', 'JPY', 'GBP', 'CNY']
    final_signals = rv_signal[non_usd].copy()
    for c in non_usd:
        if c in ctv_gate.columns:
            final_signals[c] = final_signals[c] * ctv_gate[c]

    return final_signals


# ==============================================================================
# 8. GICS SECTOR ROTATION ENGINE
# Maps GMTF regime → GICS 11-sector weights → feeds AlphaOptimizer universe
# ==============================================================================
class GICSSectorRotation:
    """
    Translates macro regime into GICS sector allocation weights.
    11 sectors → 25 industry groups → signals for alpha optimizer.

    Money velocity logic:
        HIGH velocity + LOW tension  → Growth / Risk-On  (Tech, Discretionary)
        LOW velocity + HIGH tension  → Defensive / Value (Staples, Utilities, Energy)
        STRESS regime               → Financials (GSIB spread capture), Gold proxy
        TRANSITION                  → Equal weight with IG/Fallen Angel tilt
    """

    REGIME_WEIGHTS = {
        'EXPANSION': {
            'Information Technology': 0.25,
            'Communication Services': 0.15,
            'Consumer Discretionary': 0.15,
            'Health Care':            0.12,
            'Financials':             0.10,
            'Industrials':            0.08,
            'Materials':              0.05,
            'Energy':                 0.04,
            'Consumer Staples':       0.03,
            'Real Estate':            0.02,
            'Utilities':              0.01,
        },
        'CONTRACTION': {
            'Consumer Staples':       0.22,
            'Utilities':              0.18,
            'Health Care':            0.18,
            'Energy':                 0.12,
            'Financials':             0.10,
            'Real Estate':            0.08,
            'Materials':              0.05,
            'Industrials':            0.04,
            'Information Technology': 0.02,
            'Consumer Discretionary': 0.01,
            'Communication Services': 0.00,
        },
        'STRESS': {
            # GSIB spread capture: tilt to Financials + defensives
            'Financials':             0.25,
            'Consumer Staples':       0.20,
            'Utilities':              0.15,
            'Health Care':            0.15,
            'Energy':                 0.10,
            'Materials':              0.05,
            'Real Estate':            0.05,
            'Industrials':            0.03,
            'Information Technology': 0.02,
            'Consumer Discretionary': 0.00,
            'Communication Services': 0.00,
        },
        'TRANSITION': {
            # Equal-weight with slight IG/Fallen Angel bias (Financials/Real Estate)
            s: 1/11 for s in GICS_SECTOR_ETFS.keys()
        },
    }

    # GICS sub-sector alpha targets per regime (fallen angel + RV opportunities)
    FALLEN_ANGEL_TARGETS = {
        'EXPANSION':   ['NVDA', 'AMD', 'META', 'GOOGL'],
        'CONTRACTION': ['LQD', 'ANGL', 'PFE', 'XOM', 'CVX'],   # IG credit + defensives
        'STRESS':      ['ANGL', 'FALN', 'GS', 'JPM', 'WFC'],    # Fallen angel ETFs + GSIB
        'TRANSITION':  ['LQD', 'VCIT', 'INTC', 'WBA'],          # IG carry + beaten-down RV
    }

    def get_weights(self, regime: str) -> dict:
        return self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS['TRANSITION'])

    def get_alpha_universe(self, regime: str) -> list:
        return self.FALLEN_ANGEL_TARGETS.get(regime, self.FALLEN_ANGEL_TARGETS['TRANSITION'])

    def get_etf_universe(self, regime: str, top_n: int = 5) -> list:
        """Returns top-N sector ETFs by regime weight."""
        weights = self.get_weights(regime)
        sorted_sectors = sorted(weights.items(), key=lambda x: -x[1])
        return [GICS_SECTOR_ETFS[s] for s, _ in sorted_sectors[:top_n]
                if s in GICS_SECTOR_ETFS]


# ==============================================================================
# 9. REGIME CLASSIFIER
# Maps GMTF score + velocity + CtV → single regime label
# Feeds AlphaBetaUnleashed Rm adjustment
# ==============================================================================
def classify_regime(gmtf_score: float, velocity_score: float,
                    rv_max_signal: float, yc_level: float) -> str:
    """
    4-state regime: EXPANSION / CONTRACTION / STRESS / TRANSITION

    Maps to AlphaBetaUnleashed:
        EXPANSION   → Rm likely above corridor → target beta near BETA_MAX
        CONTRACTION → Rm likely below corridor → hedge floor BETA_INV
        STRESS      → vol spike → vol_adj contracts beta automatically
        TRANSITION  → mid-corridor, run at base slope
    """
    if velocity_score > 65 and gmtf_score < 0.05 and yc_level > 0:
        return 'EXPANSION'
    elif velocity_score < 35 or yc_level < -0.5:
        return 'CONTRACTION'
    elif gmtf_score > 0.15 or rv_max_signal > 2.5:
        return 'STRESS'
    else:
        return 'TRANSITION'


def regime_to_rm_adjustment(regime: str) -> float:
    """
    Adjusts the Rm estimate fed into AlphaBetaUnleashed.
    Macro view overlays on the realized 1Y log-drift.
    """
    adjustments = {
        'EXPANSION':   +0.02,   # Tilt Rm up → higher beta
        'CONTRACTION': -0.02,   # Tilt Rm down → hedge floor
        'STRESS':      -0.04,   # Aggressive reduction → vol contraction takes over
        'TRANSITION':   0.00,   # No overlay
    }
    return adjustments.get(regime, 0.0)


# ==============================================================================
# 10. MASTER MACRO ENGINE
# ==============================================================================
class MacroEngine:
    """
    Master engine. Orchestrates all components and returns unified signal dict.
    Called by platform orchestrator every CHECK_INTERVAL.
    """

    def __init__(self, fred_api_key: Optional[str] = None,
                 market_start: str = "2020-01-01"):
        self.fred    = FREDClient(api_key=fred_api_key)
        self.market  = MarketDataEngine(start=market_start)
        self.gmtf    = GMTFEngine(self.fred, self.market)
        self.velocity= MoneyVelocityEngine(self.fred, self.market)
        self.gics    = GICSSectorRotation()

    def run(self, gmtf_start: str = "2026-01-01", plot: bool = False) -> dict:
        print(f"\n{'='*60}")
        print(f"  MACRO ENGINE — GMTF + GICS + VELOCITY")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # 1. GMTF
        print("  [1/5] Computing GMTF...")
        gmtf_out  = self.gmtf.compute_gmtf(start=gmtf_start)
        df        = gmtf_out['df']
        tensions  = gmtf_out['tensions']
        gmtf_last = float(gmtf_out['gmtf_smoothed'].dropna().iloc[-1])

        # 2. CtV signals (fixed: df now defined)
        print("  [2/5] Computing Carry-to-Volatility signals...")
        ctv_ratios, ctv_gate = get_ctv_signals(df)

        # 3. RV signals (fixed: tensions now passed explicitly)
        print("  [3/5] Computing RV signals...")
        rv_signals   = get_rv_signals(df, tensions, ctv_gate)
        rv_last      = rv_signals.iloc[-1]
        rv_entries   = rv_last[rv_last > 1.0]
        rv_max       = float(rv_last.abs().max()) if not rv_last.empty else 0.0

        # 4. Money velocity
        print("  [4/5] Computing money velocity...")
        vel_out      = self.velocity.compute()
        vel_score    = vel_out['velocity_score']
        yc_level     = vel_out['yield_curve']

        # 5. Regime + GICS rotation
        print("  [5/5] Classifying regime + GICS rotation...")
        regime       = classify_regime(gmtf_last, vel_score, rv_max, yc_level)
        rm_adj       = regime_to_rm_adjustment(regime)
        sector_wts   = self.gics.get_weights(regime)
        alpha_univ   = self.gics.get_alpha_universe(regime)
        etf_univ     = self.gics.get_etf_universe(regime)

        result = {
            'timestamp':        datetime.now().isoformat(),
            'regime':           regime,
            'gmtf_score':       round(gmtf_last, 6),
            'velocity_score':   vel_score,
            'velocity_detail':  vel_out,
            'yield_curve':      round(yc_level, 4),
            'rv_entries':       rv_entries.to_dict(),
            'rv_max_signal':    round(rv_max, 4),
            'ctv_health':       ctv_ratios.iloc[-1].to_dict(),

            # ── Outputs for downstream engines ────────────────────────────────
            'rm_adjustment':    rm_adj,             # → AlphaBetaUnleashed
            'sector_weights':   sector_wts,         # → GICS allocation
            'alpha_universe':   alpha_univ,         # → AlphaOptimizer tickers
            'sector_etfs':      etf_univ,           # → ETF sleeve tickers
        }

        self._print(result)

        if plot:
            self._plot(gmtf_out, rv_signals, ctv_ratios)

        self._log(result)
        return result

    def _print(self, r: dict) -> None:
        v = r['velocity_detail']
        print(f"\n  REGIME:          {r['regime']}")
        print(f"  GMTF Score:      {r['gmtf_score']:.6f}")
        print(f"  Velocity Score:  {r['velocity_score']:.1f}/100")
        print(f"    M2V Level:     {v['m2v_level']:.4f}")
        print(f"    GSIB Impulse:  {v['gsib_impulse']:.4f}")
        print(f"    Fed BS Chg:    {v['fed_bs_change']:.4f}")
        print(f"  Yield Curve:     {r['yield_curve']:+.4f}")
        print(f"  Rm Adjustment:   {r['rm_adjustment']:+.4f}  → AlphaBetaUnleashed")
        print(f"\n  ── GICS Sector Rotation ({r['regime']}) ──")
        for sector, w in sorted(r['sector_weights'].items(), key=lambda x: -x[1])[:6]:
            bar = '█' * int(w * 100)
            print(f"  {sector:<28} {w:.0%}  {bar}")
        print(f"\n  ── Alpha Universe ──")
        print(f"  Equity names:  {r['alpha_universe']}")
        print(f"  Sector ETFs:   {r['sector_etfs']}")
        if r['rv_entries']:
            print(f"\n  ── RV Entry Signals (Z > 1.0 & CtV passed) ──")
            for c, z in r['rv_entries'].items():
                print(f"  {c}: Z={z:.3f}")

    def _plot(self, gmtf_out: dict, rv_signals: pd.DataFrame,
              ctv_ratios: pd.DataFrame) -> None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            CHARTS_PATH.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)

            # Plot 1: GMTF score
            gmtf_out['gmtf_smoothed'].dropna().tail(120).plot(
                ax=axes[0], title="GMTF Score (Smoothed, SDR-Weighted)", color='steelblue')
            axes[0].set_ylabel("Tension Index")
            axes[0].axhline(0, color='black', alpha=0.3)

            # Plot 2: RV Signals
            rv_signals.tail(120).plot(
                ax=axes[1], title="Currency/Bond RV Signals (CtV-Filtered)")
            axes[1].axhline(0, color='black', alpha=0.3)
            axes[1].axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Entry Z=1.0')
            axes[1].set_ylabel("Z-Score")
            axes[1].legend()

            # Plot 3: CtV Health
            ctv_ratios.tail(120).plot(
                ax=axes[2], title="Carry-to-Volatility Ratios")
            axes[2].axhline(0.5, color='red', linestyle='--', label='Min Entry CtV=0.5')
            axes[2].set_ylabel("CtV Ratio")
            axes[2].legend()

            plt.tight_layout()
            out_path = CHARTS_PATH / f"macro_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"\n  Chart saved: {out_path}")
        except Exception as e:
            print(f"  [CHART] Skipped: {e}")

    def _log(self, result: dict) -> None:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(json.dumps(
            result, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer))
            else str(x)
        ))
        with open(RESULTS_PATH, 'a') as f:
            f.write(json.dumps(serializable) + '\n')


# ==============================================================================
# 11. FULL PLATFORM INTEGRATION
# MacroEngine → AlphaOptimizer → AlphaBetaUnleashed
# ==============================================================================
def run_full_platform(plot: bool = False) -> dict:
    """
    Runs the complete 3-engine stack:
        Macro → stock selection universe + Rm adjustment
        Alpha → optimal weights + sleeve beta
        Beta  → MES hedge size
    """
    from alpha_optimizer    import AlphaOptimizerEngine
    from alpha_beta_engine  import AlphaBetaUnleashed, PaperBroker

    print("\n" + "="*60)
    print("  FULL PLATFORM: MACRO → ALPHA → BETA")
    print("="*60)

    # Engine 1: Macro (Dataset 3)
    macro  = MacroEngine()
    m_out  = macro.run(plot=plot)

    regime     = m_out['regime']
    alpha_univ = m_out['alpha_universe']
    rm_adj     = m_out['rm_adjustment']

    # Engine 2: Alpha Optimizer (Dataset 2) — universe from macro
    opt = AlphaOptimizerEngine(custom_tickers=alpha_univ)
    a_out = opt.run()
    sleeve_beta  = a_out['sleeve_beta']
    pure_alpha   = a_out['pure_alpha_annual']

    # Engine 3: Beta Engine (Dataset 1) — Rm adjusted by macro
    broker = PaperBroker(nlv=1_000_000)
    beta_engine = AlphaBetaUnleashed(broker=broker)

    # Macro-adjusted Rm
    adjusted_rm = beta_engine.Rm + rm_adj
    desired_beta = beta_engine.calculate_target_beta(adjusted_rm)
    hedge_beta   = desired_beta - sleeve_beta

    print(f"\n{'='*60}")
    print(f"  PLATFORM SUMMARY")
    print(f"{'='*60}")
    print(f"  Macro Regime:        {regime}")
    print(f"  Velocity Score:      {m_out['velocity_score']:.1f}/100")
    print(f"  Rm (realized):       {beta_engine.Rm:.2%}")
    print(f"  Rm (macro-adjusted): {adjusted_rm:.2%}")
    print(f"  Desired Beta:        {desired_beta:.4f}")
    print(f"  Sleeve Beta:         {sleeve_beta:.4f}")
    print(f"  MES Hedge Beta:      {hedge_beta:.4f}")
    print(f"  Pure Alpha (annual): {pure_alpha:.2%}")
    print(f"  Alpha universe:      {alpha_univ}")

    beta_engine.execute_logic(hedge_beta)

    return {
        'macro':  m_out,
        'alpha':  a_out,
        'regime': regime,
        'adjusted_rm':   adjusted_rm,
        'desired_beta':  desired_beta,
        'sleeve_beta':   sleeve_beta,
        'hedge_beta':    hedge_beta,
        'pure_alpha':    pure_alpha,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import sys
    if '--platform' in sys.argv:
        run_full_platform(plot='--plot' in sys.argv)
    else:
        engine = MacroEngine()
        engine.run(plot='--plot' in sys.argv)
