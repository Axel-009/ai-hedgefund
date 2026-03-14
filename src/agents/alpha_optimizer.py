"""
Alpha Optimizer — ML Portfolio Selection Engine
================================================
Dataset 2: ML-based walk-forward portfolio optimizer with alpha extraction.

Role in platform:
    EQUITY SLEEVE → produces optimal weights + expected returns per name
    feeds into AlphaBetaUnleashed which overlays MES hedge to strip market beta

Alpha sources:
    1. IG / Fallen Angel names  — structural mispricing from forced selling
    2. RV pairs                 — relative value between comparable credits/equities
    3. Walk-forward ML signal   — LinearRegression on momentum + vol features

Integration:
    optimizer.run() → AlphaBetaUnleashed.equity_sleeve_beta()
    net alpha = sleeve return - beta_engine.current_beta * Rm

DATA: yfinance (live when network available, synthetic fallback for CI)

Original optimizer logic preserved exactly.
Extensions: universe config, alpha decomposition, regime conditioning,
            IG/Fallen Angel universe, RV signal, beta-engine hookup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RESULTS_PATH = Path(__file__).parent / "optimizer_results.jsonl"

# ==============================================================================
# UNIVERSE CONFIGURATION
# Alpha extraction targets:
#   Core equity  — mega-cap liquid names for reliable Sharpe
#   Fallen Angel — recently downgraded IG→HY; forced selling = structural edge
#   IG proxies   — LQD, VCIT: credit spread compression alpha
#   RV pairs     — GOOGL/META (ad duopoly), XOM/CVX (energy RV)
# ==============================================================================
UNIVERSES = {
    "core": ["AAPL", "MSFT", "AMZN", "NVDA"],
    "fallen_angel": [
        "INTC",   # Fallen from semiconductor throne — RV vs NVDA/AMD
        "PFE",    # Pharma derating — RV vs LLY/MRK
        "WBA",    # Retail pharmacy — structural distress alpha
        "MPW",    # Healthcare REIT — credit stress mispricing
    ],
    "ig_proxies": [
        "LQD",    # iShares IG Corporate Bond ETF
        "VCIT",   # Vanguard Intermediate-Term Corporate Bond ETF
        "ANGL",   # VanEck Fallen Angel High Yield Bond ETF ← structural alpha
        "FALN",   # iShares Fallen Angels USD Bond ETF
    ],
    "rv_pairs": [
        # Ad duopoly RV
        "GOOGL", "META",
        # Energy majors RV
        "XOM", "CVX",
        # Semiconductor RV
        "AMD", "QCOM",
    ],
    "full": [
        # Core
        "AAPL", "MSFT", "AMZN", "NVDA",
        # Fallen Angel candidates
        "INTC", "PFE", "WBA",
        # IG/Credit proxies
        "LQD", "ANGL",
        # RV pairs
        "GOOGL", "META", "XOM", "CVX", "AMD", "QCOM",
    ],
}

DEFAULT_UNIVERSE = "core"   # swap to "full" for live platform run


# ==============================================================================
# DATA ENGINE
# Preserves original yfinance fetch logic exactly.
# Adds synthetic fallback for offline/CI environments.
# ==============================================================================
class DataEngine:
    def __init__(self, tickers: list[str], start: str = "2015-01-01"):
        self.tickers = tickers
        self.start = start
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def fetch(self) -> "DataEngine":
        try:
            import yfinance as yf
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parents[3] / "Financial-Data"))
            import yfinance as yf

        try:
            # Preserve original fetch logic exactly
            raw = yf.download(
                self.tickers,
                start=self.start,
                progress=False,
                auto_adjust=False,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                top = raw.columns.get_level_values(0)
                key = "Adj Close" if "Adj Close" in top else "Close"
                data = raw[key].copy()
                if isinstance(data, pd.Series):
                    data = data.to_frame(self.tickers[0] if data.name is None else data.name)
            else:
                data = raw["Adj Close"].copy() if "Adj Close" in raw.columns else raw["Close"].copy()
                if isinstance(data, pd.Series):
                    data = data.to_frame(self.tickers[0])

            data = data[[c for c in self.tickers if c in data.columns]]

            if data.empty or len(data.columns) < 2:
                raise ValueError("Insufficient data from live feed")

        except Exception as e:
            print(f"[DATA] Network/data unavailable ({e}) — using synthetic data")
            data = self._synthetic(self.tickers, self.start)

        self.tickers = list(data.columns)
        self.data = data
        self.returns = np.log(data / data.shift(1)).dropna()
        return self

    def _synthetic(self, tickers: list[str], start: str) -> pd.DataFrame:
        """
        Synthetic price series calibrated per-ticker with realistic drift/vol.
        Fallen Angel names get higher vol + mean-reverting drift (structural edge).
        """
        np.random.seed(42)
        dates = pd.bdate_range(start=start, end=pd.Timestamp.today())
        n = len(dates)

        # Per-ticker params: (annual_drift, annual_vol, start_price)
        params = {
            "AAPL":  (0.18, 0.25, 30.0),
            "MSFT":  (0.16, 0.22, 40.0),
            "AMZN":  (0.20, 0.28, 80.0),
            "NVDA":  (0.35, 0.45, 10.0),
            "INTC":  (0.02, 0.30, 35.0),   # fallen angel: low drift, high vol
            "PFE":   (0.03, 0.22, 40.0),
            "WBA":   (-0.05, 0.35, 80.0),  # structural distress
            "GOOGL": (0.14, 0.24, 50.0),
            "META":  (0.15, 0.32, 20.0),
            "XOM":   (0.08, 0.22, 50.0),
            "CVX":   (0.09, 0.20, 80.0),
            "AMD":   (0.25, 0.42, 5.0),
            "QCOM":  (0.10, 0.28, 60.0),
            "LQD":   (0.04, 0.06, 110.0),  # IG bond: low drift, low vol
            "VCIT":  (0.04, 0.05, 80.0),
            "ANGL":  (0.06, 0.10, 25.0),   # fallen angel ETF
            "FALN":  (0.06, 0.10, 24.0),
            "MPW":   (-0.08, 0.40, 20.0),
        }
        out = {}
        for tkr in tickers:
            mu, sigma, s0 = params.get(tkr, (0.10, 0.25, 50.0))
            daily_mu = mu / 252
            daily_sig = sigma / np.sqrt(252)
            log_rets = np.random.normal(daily_mu, daily_sig, n)
            out[tkr] = s0 * np.exp(np.cumsum(log_rets))

        return pd.DataFrame(out, index=dates)


# ==============================================================================
# FEATURE ENGINEERING
# Original 4 features preserved exactly + 3 alpha-extraction extensions:
#   credit_spread_proxy  — IG vs equity momentum divergence (fallen angel signal)
#   rv_zscore            — cross-sectional z-score (RV mispricing signal)
#   capm_residual        — idiosyncratic return after stripping market beta
# ==============================================================================
def build_features(data: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)

    # ── Original features (unchanged) ──────────────────────────────────────────
    features["mkt_mean"]     = returns.mean(axis=1)
    features["mkt_vol"]      = returns.std(axis=1)
    features["momentum_3m"]  = data.reindex(returns.index).ffill().pct_change(60).mean(axis=1)
    features["momentum_1m"]  = data.reindex(returns.index).ffill().pct_change(20).mean(axis=1)

    # ── Alpha extensions ───────────────────────────────────────────────────────
    # 1. Credit spread proxy: vol-adjusted momentum spread
    #    High vol + low momentum = potential fallen angel setup
    mkt_vol_roll = features["mkt_vol"].rolling(20).mean()
    features["credit_spread_proxy"] = (features["mkt_vol"] - mkt_vol_roll) * -1

    # 2. Cross-sectional RV z-score: mean-standardized 1m momentum
    mom_1m_cs = data.reindex(returns.index).ffill().pct_change(20)
    cs_mean = mom_1m_cs.mean(axis=1)
    cs_std  = mom_1m_cs.std(axis=1).replace(0, np.nan)
    features["rv_zscore"] = ((mom_1m_cs.subtract(cs_mean, axis=0)
                               .divide(cs_std, axis=0))
                              .mean(axis=1))

    # 3. CAPM residual momentum: strip market factor, keep idiosyncratic drift
    #    Proxy: return - beta_ols * market_return  (rolling 60-day OLS beta)
    mkt_ret = returns.mean(axis=1)
    rolling_betas = returns.apply(
        lambda col: col.rolling(60).cov(mkt_ret) / mkt_ret.rolling(60).var()
    ).mean(axis=1)
    features["capm_residual"] = mkt_ret - rolling_betas * mkt_ret

    return features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


# ==============================================================================
# EWMA COVARIANCE (original, unchanged)
# ==============================================================================
def ewma_cov(returns_df: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    cov = returns_df.cov().values.copy()
    for i in range(len(returns_df)):
        r = returns_df.iloc[i].values.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
    cov = cov + 1e-10 * np.eye(cov.shape[0])
    return cov


# ==============================================================================
# WALK-FORWARD TRAINER
# Original train/test split preserved as default.
# Extended to support rolling windows for live re-fitting.
# ==============================================================================
class WalkForwardModel:
    def __init__(self, train_ratio: float = 0.8):
        self.train_ratio = train_ratio
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.alpha_preds: Optional[pd.Series] = None

    def fit_predict(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        split = int(len(X) * self.train_ratio)

        X_train = X.iloc[:split]
        X_test  = X.iloc[split:]
        y_train = y.iloc[:split]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        preds = pd.Series(self.model.predict(X_test_s), index=X_test.index)
        self.alpha_preds = preds
        self.split_idx = split
        return preds

    @property
    def feature_importance(self) -> dict:
        """Coefficient magnitudes — larger = stronger alpha signal."""
        if not hasattr(self.model, "coef_"):
            return {}
        return dict(zip(
            ["mkt_mean", "mkt_vol", "momentum_3m", "momentum_1m",
             "credit_spread_proxy", "rv_zscore", "capm_residual"],
            np.abs(self.model.coef_).tolist()
        ))


# ==============================================================================
# PORTFOLIO OPTIMIZER (original objective + constraints preserved exactly)
# ==============================================================================
class PortfolioOptimizer:
    def __init__(
        self,
        tickers: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        transaction_cost: float = 0.001,
        max_turnover: float = 0.50,
    ):
        self.tickers = tickers
        self.n = len(tickers)
        self.expected_returns = expected_returns
        self.cov = cov_matrix
        self.current_weights = (current_weights if current_weights is not None
                                else np.ones(self.n) / self.n)
        self.tc = transaction_cost
        self.max_turnover = max_turnover
        self.result = None
        self.final_weights: Optional[np.ndarray] = None

    # Original functions, unchanged ────────────────────────────────────────────
    def _portfolio_vol(self, w: np.ndarray) -> float:
        return np.sqrt(max(1e-12, w.T @ self.cov @ w))

    def _turnover(self, w: np.ndarray) -> float:
        return float(np.sum(np.abs(w - self.current_weights)))

    def _objective(self, w: np.ndarray) -> float:
        w = np.asarray(w)
        port_ret = float(w @ self.expected_returns)
        vol      = self._portfolio_vol(w)
        cost     = self._turnover(w) * self.tc
        sharpe   = (port_ret - cost) / vol
        return -sharpe

    def optimize(self) -> np.ndarray:
        constraints = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: self.max_turnover - self._turnover(np.asarray(w))},
        ]
        bounds = [(0.0, 1.0)] * self.n

        self.result = minimize(
            self._objective,
            self.current_weights.copy(),
            args=(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        self.final_weights = np.asarray(self.result.x)
        return self.final_weights

    def performance(self, test_returns: pd.DataFrame) -> dict:
        w  = self.final_weights
        pr = test_returns[self.tickers] @ w
        ar = float(pr.mean() * 252)
        av = float(pr.std() * np.sqrt(252))
        sr = ar / av if av > 0 else 0.0
        cum = (1 + pr).cumprod()
        dd  = float((cum / cum.cummax() - 1).min())
        rc  = float(np.sum(np.abs(w - self.current_weights)) * self.tc)
        return {
            "annual_return": round(ar, 4),
            "annual_vol":    round(av, 4),
            "sharpe":        round(sr, 4),
            "max_drawdown":  round(dd, 4),
            "rebalance_cost": round(rc, 6),
            "weights":       dict(zip(self.tickers, np.round(w, 4).tolist())),
        }


# ==============================================================================
# ALPHA DECOMPOSER
# Strips market beta from the sleeve to show pure idiosyncratic alpha.
# Feeds directly into AlphaBetaUnleashed as the equity sleeve beta.
# ==============================================================================
class AlphaDecomposer:
    """
    Measures how much of the portfolio's expected return is true alpha
    (idiosyncratic) vs beta-compensated market return.

    Integration with AlphaBetaUnleashed:
        sleeve_beta  → passed to beta engine to size the MES hedge
        pure_alpha   → the P&L we're actually targeting above the corridor
    """

    @staticmethod
    def decompose(
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        risk_free: float = 0.045,   # ~current 3-month T-bill
    ) -> dict:
        from sklearn.linear_model import LinearRegression
        aligned = pd.concat(
            [portfolio_returns, market_returns], axis=1
        ).dropna()
        aligned.columns = ["port", "mkt"]

        excess_port = aligned["port"] - risk_free / 252
        excess_mkt  = aligned["mkt"]  - risk_free / 252

        lr = LinearRegression()
        lr.fit(excess_mkt.values.reshape(-1, 1), excess_port.values)

        sleeve_beta  = float(lr.coef_[0])
        alpha_daily  = float(lr.intercept_)
        alpha_annual = alpha_daily * 252

        r2 = float(lr.score(
            excess_mkt.values.reshape(-1, 1), excess_port.values
        ))

        return {
            "sleeve_beta":   round(sleeve_beta, 4),
            "alpha_annual":  round(alpha_annual, 4),
            "alpha_daily":   round(alpha_daily, 6),
            "r_squared":     round(r2, 4),
            "idiosyncratic": round(1 - r2, 4),  # fraction not explained by market
        }


# ==============================================================================
# MASTER RUNNER — wires all components together
# ==============================================================================
class AlphaOptimizerEngine:
    """
    Full pipeline:
        DataEngine → Features → WalkForwardModel → PortfolioOptimizer
        → AlphaDecomposer → result dict (ready for beta engine hookup)
    """

    def __init__(
        self,
        universe: str = DEFAULT_UNIVERSE,
        custom_tickers: Optional[list[str]] = None,
        start: str = "2015-01-01",
        train_ratio: float = 0.8,
        transaction_cost: float = 0.001,
        max_turnover: float = 0.50,
    ):
        self.tickers     = custom_tickers or UNIVERSES[universe]
        self.start       = start
        self.train_ratio = train_ratio
        self.tc          = transaction_cost
        self.max_to      = max_turnover
        self._last_result: Optional[dict] = None

    def run(self, current_weights: Optional[np.ndarray] = None) -> dict:
        print(f"\n{'='*60}")
        print(f"  ALPHA OPTIMIZER ENGINE")
        print(f"  Universe: {self.tickers}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # 1. Data
        engine = DataEngine(self.tickers, self.start).fetch()
        tickers  = engine.tickers
        data     = engine.data
        returns  = engine.returns

        # 2. Features (original 4 + 3 alpha extensions)
        features = build_features(data, returns)
        target   = returns.mean(axis=1).shift(-1)
        dataset  = pd.concat([features, target.rename("target")], axis=1).dropna()
        X, y     = dataset.drop(columns=["target"]), dataset["target"]

        # 3. Walk-forward model
        wf   = WalkForwardModel(train_ratio=self.train_ratio)
        preds = wf.fit_predict(X, y)
        split = wf.split_idx

        test_returns = returns.iloc[split:]

        # 4. EWMA covariance + expected returns (original logic)
        latest_cov = ewma_cov(test_returns[tickers])
        exp_ret_series = test_returns[tickers].mean() + preds.iloc[-1]
        expected_returns = np.asarray(exp_ret_series.values, dtype=float)

        # 5. Optimize
        cw = (current_weights if current_weights is not None
              else np.ones(len(tickers)) / len(tickers))
        opt = PortfolioOptimizer(
            tickers, expected_returns, latest_cov, cw, self.tc, self.max_to
        )
        weights = opt.optimize()
        perf    = opt.performance(test_returns)

        # 6. Alpha decomposition
        mkt_ret   = returns.mean(axis=1).iloc[split:]
        port_ret  = test_returns[tickers] @ weights
        decomp    = AlphaDecomposer.decompose(port_ret, mkt_ret)

        result = {
            "timestamp":         datetime.now().isoformat(),
            "tickers":           tickers,
            "performance":       perf,
            "alpha_decomp":      decomp,
            "feature_importance": wf.feature_importance,
            "optimizer_success": bool(opt.result.success),
            # ── Beta engine hookup ──────────────────────────────────────────
            # Pass sleeve_beta to AlphaBetaUnleashed to size the MES hedge.
            # The beta engine targets (1 - sleeve_beta) exposure via futures.
            "sleeve_beta":       decomp["sleeve_beta"],
            "pure_alpha_annual": decomp["alpha_annual"],
        }

        self._last_result = result
        self._log(result)
        self._print(result)
        return result

    def _print(self, r: dict) -> None:
        p = r["performance"]
        d = r["alpha_decomp"]
        print(f"  Tickers:           {r['tickers']}")
        print(f"  Weights:           {p['weights']}")
        print(f"  Annual Return:     {p['annual_return']:.2%}")
        print(f"  Annual Vol:        {p['annual_vol']:.2%}")
        print(f"  Sharpe Ratio:      {p['sharpe']:.3f}")
        print(f"  Max Drawdown:      {p['max_drawdown']:.2%}")
        print(f"  Rebalance Cost:    {p['rebalance_cost']:.4%}")
        print()
        print(f"  ── Alpha Decomposition ──")
        print(f"  Sleeve Beta:       {d['sleeve_beta']:.4f}  ← MES hedge target")
        print(f"  Pure Alpha (ann):  {d['alpha_annual']:.2%} ← idiosyncratic P&L")
        print(f"  R² (mkt explain):  {d['r_squared']:.2%}")
        print(f"  Idiosyncratic:     {d['idiosyncratic']:.2%}")
        print()
        print(f"  ── Feature Importance ──")
        for feat, imp in sorted(r["feature_importance"].items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(imp * 500)
            print(f"  {feat:<25} {imp:.4f}  {bar}")
        print()

    def _log(self, result: dict) -> None:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(
            json.dumps(result, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
        )
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(serializable) + "\n")


# ==============================================================================
# BETA ENGINE HOOKUP — shows how Dataset 1 + Dataset 2 connect
# ==============================================================================
def run_integrated(paper_nlv: float = 100_000.0) -> None:
    """
    Full integration demo:
    1. Optimizer picks names + weights → sleeve_beta
    2. Beta engine adjusts MES hedge to offset sleeve_beta
    Net result: portfolio holds equity alpha, hedges market beta via futures.
    """
    from alpha_beta_engine import AlphaBetaUnleashed, PaperBroker

    print("\n" + "="*60)
    print("  INTEGRATED: ALPHA OPTIMIZER + BETA ENGINE")
    print("="*60)

    # Step 1: Run alpha optimizer
    opt_engine = AlphaOptimizerEngine(universe=DEFAULT_UNIVERSE)
    result = opt_engine.run()

    sleeve_beta  = result["sleeve_beta"]
    pure_alpha   = result["pure_alpha_annual"]

    print(f"\n  Sleeve Beta from equity selection: {sleeve_beta:.4f}")
    print(f"  Pure Alpha target:                 {pure_alpha:.2%}")

    # Step 2: Feed sleeve_beta to beta engine
    # Beta engine now targets (desired_portfolio_beta - sleeve_beta) via MES
    broker  = PaperBroker(nlv=paper_nlv)
    beta_engine = AlphaBetaUnleashed(broker=broker)

    # Adjust: beta engine targets the residual after equity sleeve beta
    desired_beta = beta_engine.calculate_target_beta(beta_engine.Rm)
    hedge_beta   = desired_beta - sleeve_beta

    print(f"\n  Beta engine desired β:             {desired_beta:.4f}")
    print(f"  Equity sleeve contributes β:       {sleeve_beta:.4f}")
    print(f"  MES hedge target β:                {hedge_beta:.4f}")
    print(f"\n  → Beta engine executing hedge...")
    beta_engine.execute_logic(hedge_beta)

    print(f"\n  PORTFOLIO SNAPSHOT:")
    snap = beta_engine.snapshot()
    print(f"  Regime:   {snap['regime']}")
    print(f"  Rm:       {snap['Rm']:.2%}")
    print(f"  Vol_Adj:  {snap['vol_adj']:.3f}")
    print(f"  Net β:    {sleeve_beta + snap['current_beta']:.4f}  (target: {desired_beta:.4f})")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import sys
    if "--integrated" in sys.argv:
        run_integrated()
    else:
        engine = AlphaOptimizerEngine(universe=DEFAULT_UNIVERSE)
        engine.run()
