# AI Hedge Fund — Session Context for Claude

**Read this file first in every new session.**
Run `bash setup.sh` before doing anything else to ensure all repos are present.

---

## What this system is

A fully autonomous multi-agent hedge fund platform targeting the **US securities universe only** (NYSE/NASDAQ equities, sector ETFs, IG/HY bond ETFs, US equity options). It runs three daily jobs:

```
python run_open.py     # 9:30 ET  — macro scan, universe ranking, morning signals
python run_hourly.py   # every hour — intraday updates, stop/take-profit checks
python run_close.py    # 4:00 ET  — EOD reconciliation, platinum report
```

---

## Repository layout

All repos live under `/home/user/`. The primary repo is `/home/user/ai-hedgefund/`.

### Internal repos (Axel-009, `claude/init-test-repos-oPogr` branch)
| Repo | Role |
|------|------|
| `ai-hedgefund` | **Master repo** — all agents, models, execution, backtesting |
| `Financial-Data` | Market data ingestion (yfinance fork) |
| `ML-Macro-Market` | Macro regime classifier (BULL/BEAR/TRANSITION/STRESS) |
| `Mav-Analysis` | MCP server exposing Claude tools |
| `QLIB` | Alpha factor research and backtesting |
| `Ruflo-agents` | Claude-flow multi-agent orchestration (TypeScript) |
| `AI-Newton` | Physics-inspired symbolic AI (Rust/Python) |
| `Air-LLM` | Lightweight LLM inference |
| `hedgefund-tracker` | 13F institutional holdings tracker |
| `open-bb` | Macro/alt data aggregation (OpenBB fork) |
| `quant-trading` | Strategy library (TA + patterns) |

### External repos (public GitHub, cloned as reference/source)
| Repo | Role in system |
|------|---------------|
| `Stock-Prediction-Models` | Evolution Strategy agent → `StockPredictionBridge` (Tier-1 ML vote) |
| `FinRL` | PPO/A2C/SAC DRL agents → `FinRLBridge` (Tier-2 ML vote) |
| `Deep-Trading` | 12-feature state + vol regime → `DeepTradingFeatures` (price enrichment) |
| `DeepLearningExamples` | TFT multi-horizon → `NVIDIATFTAdapter` (Tier-3 ML vote) |
| `kserve` | Production model serving → `KServeAdapter` |
| `DeepLearning` | Reference implementations |
| `wondertrader` | C++ HFT micro-price formula (translated to Python in execution engine) |
| `exchange-core` | LMAX Disruptor order book (Java, paper-mode Python mirror) |
| `gist-b16f9d8cd0a9e817fd3baa3ce3cd0194` | ARIMA+Laplacian MC notebook → `MonteCarloBridge` (Tier-4 ML vote) |
| `building-stock-analysis` | Top-down vs bottom-up methodology → `UniverseClassifier` (Tier-5) |
| `FRB` | Federal Reserve economic data |
| `Quant-Developers-Resources` | Reference quant library |

**Development branch for ALL repos:** `claude/init-test-repos-oPogr`

---

## Architecture

```
UniverseEngine (universe_engine.py)
    ↓  full investable universe: S&P 500 + S&P 400 + S&P 600 + ETFs (~1,500+ securities)
    ↓  GICS 4-tier pools (11 sectors / 25 industry groups / 74 industries / 163 sub-industries)
    ↓  data sources: yfinance (free) + OpenBB (macro)

MacroEngine (macro_engine.py)
    ↓  regime detection: BULL / BEAR / TRANSITION / STRESS
    ↓  GMTF (Global Macro Transmission Framework)
    ↓  money velocity: M1/M2, credit impulse, TGA, ON-RRP, TED spread
    ↓  sector universe → ranked by macro regime

MetadronCube (metadron_cube.py)           ← sits between MacroEngine and AlphaOptimizer
    ↓  Layer 0: FedPlumbingLayer          — H.4.1, SOMA, SOFR, HY spreads, M2V
    ↓  Layer 1: LiquidityTensor L(t)      — reserves/TGA/ON-RRP/repo/credit → [-1,+1]
    ↓  Layer 2: ReserveFlowKernel         — impulse response: ΔReserves → ΔEquity/Credit
    ↓  Risk:    RiskStateModel R(t)        — VIX + realized vol + credit spread → [0,1]
    ↓  Flow:    CapitalFlowModel F(t)      — sector momentum (leader/laggard detection)
    ↓  Layer 4: RegimeEngine              — TRENDING / RANGE / STRESS / CRASH
    ↓  Gate-Z:  GateZAllocator            — 5-sleeve capital allocation
    │             P1 Directional Equities  | P2 Factor Rotation | P3 Commodities/Macro
    │             P4 Options Convexity     | P5 Hedges/Volatility
    └  Risk Governor                       — beta / VaR / leverage / gamma corridor [7%-12%]

AlphaOptimizer (alpha_optimizer.py)
    ↓  CAPM alpha ranking + QLIB factors (150+ technical/fundamental)
    ↓  portfolio weight optimization (mean-variance, equal-weight)
    ↓  UniverseClassifier quality tiers A–G

ExecutionEngine (execution_engine.py)
    ├── MicroPriceEngine          — WonderTrader: micro_price = (bid×ask_qty + ask×bid_qty)/(ask_qty+bid_qty)
    ├── ML Vote Ensemble (5 tiers, each votes ±1)
    │     Tier-1  StockPredictionBridge  — ES agent (pure-numpy 2-layer net)
    │     Tier-2  FinRLBridge           — DRL agent (PPO/A2C/SAC, Stable-Baselines3)
    │     Tier-3  NVIDIATFTAdapter      — multi-horizon TFT (P10/P50/P90)
    │     Tier-4  MonteCarloBridge      — ARIMA(1,1,1) + Laplacian MC
    │     Tier-5  UniverseClassifier    — top-down XGBoost + bottom-up fundamentals
    ├── DeepTradingFeatures       — 12-feature state (bull/bear technicals + regime one-hot)
    ├── ConvictionOverride        — controlled guardrail break (conviction≥90 + hedge required)
    ├── KServeAdapter             — production model serving (optional)
    ├── ExchangeCoreAdapter       — order book simulation (paper mode, LMAX pattern)
    └── DailyUniverseScanner      — pre-market universe ranking (CAPM alpha + ML score)

Monitoring Layer (src/monitoring/)
    ├── daily_report.py           — open/close reports (9:30 ET / 4:00 ET)
    ├── hourly_recap.py           — hourly transaction + PnL recap
    ├── heatmap.py                — full universe heatmap by GICS sector
    ├── anomaly_detector.py       — price/volume/correlation/regime anomalies (σ-ranked)
    ├── market_wrap.py            — daily narrative: equities/rates/FX/commodities
    └── memory_monitor.py         — RAM, API usage, log sizes, token estimation

Agent Scorecard (src/agents/agent_scorecard.py)
    ├── 25 agents tracked (12 investor personas + 6 analytical + 7 engines)
    ├── Weekly scores: accuracy + Sharpe + hit rate
    └── Tier hierarchy: General → Captain → Lieutenant → Recruit

PlatformOrchestrator (platform_orchestrator.py)
    └── Ties: UniverseEngine → MacroEngine → MetadronCube → AlphaOptimizer → ExecutionEngine
```

### Signal flow
1. `MicroPriceEngine` generates `MICRO_PRICE_BUY` / `MICRO_PRICE_SELL` (primary signal)
2. Each of 5 ML tiers votes ±1 (agree/disagree with micro-price direction)
3. `vote_score` adjusts edge threshold: `effective_min_edge = 2.0 + max(0, -vote_score)` bps
4. Order placed if `edge_bps >= effective_min_edge` and position limits allow

### SignalType enum (15 values)
```python
MICRO_PRICE_BUY / MICRO_PRICE_SELL   # WonderTrader primary
RV_LONG / RV_SHORT                    # relative value pairs
FALLEN_ANGEL_BUY                      # distressed equity
ML_AGENT_BUY / ML_AGENT_SELL         # Stock-Prediction-Models ES
DRL_AGENT_BUY / DRL_AGENT_SELL       # FinRL DRL
TFT_BUY / TFT_SELL                   # NVIDIA TFT
MC_BUY / MC_SELL                      # Monte Carlo ARIMA+Laplacian
QUALITY_BUY / QUALITY_SELL           # UniverseClassifier top-down/bottom-up
HOLD
```

---

## Key source files

| File | What it does |
|------|-------------|
| `src/agents/execution_engine.py` (1137 lines) | HFT execution arm — all bridges wired here |
| `src/agents/macro_engine.py` (895 lines) | GICS macro rotation engine, regime detection |
| `src/agents/alpha_optimizer.py` (586 lines) | Portfolio optimisation, CAPM alpha ranking |
| `src/agents/platform_orchestrator.py` (415 lines) | Ties MacroEngine → AlphaOptimizer → ExecutionEngine |
| `src/models/stock_prediction_bridge.py` | ES agent from Stock-Prediction-Models (pure-numpy) |
| `src/models/finrl_bridge.py` | DRL agent from FinRL (SB3 PPO/A2C/SAC + rule-based fallback) |
| `src/models/deep_trading_features.py` | 12-feature state from Deep-Trading |
| `src/models/nvidia_tft_adapter.py` | Holt's ETS proxy for TFT (P10/P50/P90 bands) |
| `src/models/kserve_adapter.py` | KServe v2 REST inference + circuit breaker |
| `src/models/monte_carlo_bridge.py` | ARIMA(1,1,1) + Laplacian MC (Yule-Walker AR) |
| `src/models/universe_classifier.py` (994 lines) | Top-down XGBoost + bottom-up FundamentalsStore |
| `src/models/model_evaluator.py` | Balanced F1, per-class metrics, confusion matrix |

---

## UniverseClassifier — top-down vs bottom-up (most recently added)

Methodology from `building-stock-analysis` (MODERATE-Project/Horizon Europe):

**Top-down** (`TopDownModel`) — mirrors T3.1 satellite EPC XGBoost:
- Input: 24-dim price-observable features only (dual-season technicals + macro regime one-hot)
  - Bull-season features [0–9]: mean/min/max return, vol, skew, RSI, MACD, BB, ROC, vol-ratio (low-vol period = "summer imagery")
  - Bear-season features [10–19]: same 10 on high-vol periods ("winter imagery")
  - Regime one-hot [20–23]: BULL / BEAR / TRANSITION / STRESS
- Model: GNB + GB + RF + XGB soft-voting ensemble
  - XGB: `n_estimators=120, max_depth=6, lr=0.1, gamma=0, reg_lambda=10, colsample_bylevel=0.5`
  - Split: `StratifiedShuffleSplit(test_size=0.2)` + `compute_sample_weight('balanced')`
- Output: predicted quality tier A–G from price alone (no fundamental data)

**Bottom-up** (`FundamentalsStore`) — mirrors T3.2 static building database:
- Per-security 19-attribute record computed from full price history:
  - Return: `alpha_annual, ir, sharpe, sortino`
  - Risk: `beta, vol_annual, downside_vol, up_capture, down_capture`
  - Momentum: `momentum_20/60/252, trend_r2`
  - Drawdown: `max_drawdown, recovery_factor, calmar`
  - Distribution: `skew, kurt, var_95`
- Tier: composite weighted score across all 19 attributes (not just IR)

**Reconciliation**: `|XGBoost-predicted tier − fundamentals tier| >= 2` → `QUALITY_BUY` / `QUALITY_SELL`
- Fundamentals tier > predicted tier → market underpricing vs fundamentals → `QUALITY_BUY`
- Fundamentals tier < predicted tier → market overpricing vs fundamentals → `QUALITY_SELL`

---

## US Universe

Defined in `execution_engine.py :: US_UNIVERSE`:
- `core_equity`: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V, UNH, XOM, JNJ, WMT, PG, MA, HD, CVX
- `sector_etfs`: XLK, XLV, XLF, XLY, XLC, XLI, XLP, XLE, XLU, XLRE, XLB, SPY, QQQ
- `fallen_angel_ig`: ANGL, FALN, LQD, VCIT, HYG, JNK, IGLB
- `fallen_angel_equity`: INTC, PFE, WBA, MPW, VFC, PARA, DIS
- `rv_pairs`: 7 pairs (GOOGL/META, XOM/CVX, AMD/INTC, JPM/BAC, V/MA, HD/LOW, PEP/KO)
- `macro_hedges`: VXX, UVXY, GLD, TLT, SHY, IEF

---

## Environment variables needed

Copy `.env.example` → `.env` and fill in:
```
ANTHROPIC_API_KEY          # Claude models (required for LLM agents)
OPENAI_API_KEY             # GPT models (optional)
FINANCIAL_DATASETS_API_KEY # Market data (optional — yfinance is free fallback)
```

---

## Git workflow

- **Always develop on:** `claude/init-test-repos-oPogr`
- **Never push to main/master**
- Push: `git push -u origin claude/init-test-repos-oPogr`

---

## Design principles

1. **Pure-numpy fallbacks everywhere** — no bridge crashes if ML framework missing; each has a rule-based proxy
2. **Try/except on all external imports** — system runs degraded, never broken
3. **US securities only** — filter out `.HK`, `.SS`, `.SZ`, `F.`, `AMS:` tickers
4. **All ML bridges vote, never veto** — micro-price is primary; ML adjusts threshold ±1bps per vote
5. **Over-engineering is banned** — add only what is directly needed

---

## How to continue development in a new session

```bash
# 1. Setup everything
cd /home/user/ai-hedgefund
bash setup.sh

# 2. Verify it works
python -c "from src.agents.execution_engine import ExecutionEngine; e = ExecutionEngine(); print('OK')"

# 3. Read CLAUDE.md (this file) — you are here
# 4. Start working
```

When given a new task, always:
1. Read the relevant existing file before modifying it
2. Check which bridge/agent it touches
3. Follow the try/except import pattern for any new external dependency
4. Add to `src/models/__init__.py` if adding a new model
5. Wire into `ExecutionEngine.__init__` and `process_quote()` if it's a new signal
6. Run a quick smoke test: `python -c "from src.models.X import X; print('OK')"`
7. Commit to `claude/init-test-repos-oPogr` with descriptive message
