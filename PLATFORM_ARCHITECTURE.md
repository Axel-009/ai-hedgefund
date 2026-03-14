# Investment Platform Architecture
## Beyond Human Neural Assessment — Money Velocity & Cyclical/Secular Signal Intelligence

**Branch:** `claude/init-test-repos-oPogr`
**Date:** 2026-03-14
**Status:** Architectural Design Phase

---

## Vision

An investment platform that surpasses human neural assessment of markets by fusing:
- **Money Velocity Signals** — tracking M1/M2 circulation rate, credit impulse, liquidity cycles
- **Cyclical vs Secular Discrimination** — separating noise (cyclical) from structural regime changes (secular)
- **Live Allocation Continuity** — real-time portfolio rebalancing with AI agent consensus
- **Deep Portfolio Analytics** — multi-factor attribution, drawdown prediction, alpha decomposition

---

## Repository Roles & Tree of Command

```
investment-platform/
│
├── LAYER 0 — DATA INFRASTRUCTURE
│   ├── Financial-Data/          ← yfinance fork: primary market data ingestion
│   │   └── Role: OHLCV, fundamentals, options chain, real-time quotes
│   └── open-bb/                 ← OpenBB Platform: macro/alt data aggregation
│       └── Role: Fed data, FRED, macro indicators, crypto, FX, fixed income
│
├── LAYER 1 — QUANTITATIVE ENGINE
│   ├── QLIB/                    ← Microsoft Qlib: alpha factor research & backtesting
│   │   └── Role: Factor mining, ML model training, walk-forward backtests
│   ├── quant-trading/           ← Strategy library: TA + pattern recognition
│   │   └── Role: VIX calc, momentum signals, pairs trading, options strategies
│   └── ML-Macro-Market/         ← Macro ML: cyclical/secular signal detection
│       └── Role: Supervised ML on macro indicators, regime classification
│
├── LAYER 2 — AI AGENT INTELLIGENCE
│   ├── ai-hedgefund/            ← Multi-agent hedge fund: decision engine
│   │   └── Role: LangGraph agents for fundamental, technical, risk analysis
│   ├── Mav-Analysis/            ← MCP server: stock analysis as tools for Claude
│   │   └── Role: FastMCP server exposing analysis tools to Claude agents
│   ├── Air-LLM/                 ← Lightweight LLM inference: edge model execution
│   │   └── Role: Run 70B models on 4GB GPU for local signal generation
│   └── AI-Newton/               ← Rust+Python physics-inspired AI: symbolic reasoning
│       └── Role: Physics-based market models, symbolic regression on price data
│
├── LAYER 3 — PORTFOLIO MANAGEMENT
│   ├── hedgefund-tracker/       ← 13F institutional tracker: smart money positioning
│   │   └── Role: Track hedge fund filings, sector rotation, conviction changes
│   └── Ruflo-agents/            ← Claude-flow: orchestration & workflow engine
│       └── Role: Multi-agent swarm coordinator, hooks, memory, task routing
│
└── LAYER 4 — PLATFORM INTEGRATION
    └── investment-platform/     ← This workspace: unified orchestration layer
        └── Role: API gateway, portfolio analytics dashboard, live allocation engine
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│  Financial-Data ──┐                                                 │
│  (OHLCV/Options)  │                                                 │
│                   ▼                                                 │
│  open-bb ─────► Unified Market Bus (pandas DataFrames / Parquet)    │
│  (Macro/Alt)      │                                                 │
│  hedgefund-tracker►│ (13F filing events + institutional flows)      │
└───────────────────┼─────────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────────┐
│                    SIGNAL GENERATION LAYER                          │
│                                                                     │
│  ML-Macro-Market ──► Regime Classifier (Cyclical/Secular labels)    │
│  QLIB ─────────────► Alpha Factors (150+ technical + fundamental)   │
│  quant-trading ─────► Strategy Signals (momentum, mean-rev, vol)    │
│  AI-Newton ─────────► Physics-inspired signals (symbolic regression)│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         MONEY VELOCITY MODULE                               │   │
│  │  M1/M2 velocity + credit impulse + margin debt tracking     │   │
│  │  → Liquidity Score (0-100) updated hourly                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────────┐
│                    AI AGENT DECISION LAYER                          │
│                                                                     │
│  Ruflo-agents (Orchestrator)                                        │
│       │                                                             │
│       ├── ai-hedgefund agents:                                      │
│       │     ├── Fundamental Agent (P/E, FCF, earnings quality)      │
│       │     ├── Technical Agent (price action, volume)              │
│       │     ├── Sentiment Agent (options flow, put/call ratio)      │
│       │     ├── Risk Agent (VaR, correlation, tail risk)            │
│       │     ├── Macro Agent (Fed, yield curve, DXY)                 │
│       │     └── Portfolio Agent (allocation optimizer)              │
│       │                                                             │
│       ├── Mav-Analysis MCP tools (Claude API integration)           │
│       │     ├── analyze_stock(ticker)                               │
│       │     ├── screen_market(criteria)                             │
│       │     ├── backtest_strategy(params)                           │
│       │     └── generate_report(portfolio)                          │
│       │                                                             │
│       └── Air-LLM (local model for sensitive calculations)          │
│             └── On-device inference for real-time scoring           │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────────┐
│                   PORTFOLIO ANALYTICS LAYER                         │
│                                                                     │
│  Live Allocation Engine:                                            │
│    ├── Position Sizing (Kelly Criterion + conviction weighting)     │
│    ├── Risk Parity Layer (vol-adjusted allocation)                  │
│    ├── Regime-Conditional Weights (cyclical vs secular mode)        │
│    └── Execution Router (broker API abstraction)                    │
│                                                                     │
│  Analytics Dashboard:                                               │
│    ├── Attribution Analysis (factor, sector, geographic)            │
│    ├── Money Velocity Chart (real-time M2 velocity overlay)         │
│    ├── Regime Indicator Panel (cyclical/secular/transition)         │
│    ├── Agent Consensus Matrix (agreement % across all agents)       │
│    └── Drawdown Forecast (ML-predicted max drawdown horizon)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Money Velocity Module (Core Innovation)

```python
# Conceptual: money_velocity.py
class MoneyVelocityEngine:
    """
    Tracks the rate at which money changes hands in the economy.
    High velocity → inflationary pressure, growth acceleration
    Low velocity → deflationary risk, liquidity trap signal

    Key metrics:
    - M1/M2 velocity (GDP / money supply) — FRED data
    - Credit impulse (change in new credit / GDP) — BIS data
    - Margin debt velocity (change rate of NYSE margin) — FINRA
    - TED spread (credit stress proxy)
    - Real yield curve (inflation-adjusted term premium)
    """

    def compute_velocity_score(self) -> float:
        """Returns 0-100 composite velocity score"""
        ...

    def classify_regime(self) -> str:
        """Returns: 'expansion', 'peak', 'contraction', 'trough'"""
        ...

    def is_secular_shift(self) -> bool:
        """
        Detects structural (secular) vs cyclical regime change.
        Uses 5+ year rolling window comparison vs 6-month window.
        True = secular shift detected (rare, high-conviction signal)
        """
        ...
```

---

## Cyclical vs Secular Discriminator

```
CYCLICAL SIGNALS (6-18 month cycles):
  - PMI oscillations
  - Earnings revision cycles
  - Inventory cycle (Kitchin: ~40 months)
  - Credit cycle momentum

SECULAR SIGNALS (10-30+ year trends):
  - Demographic headwinds/tailwinds
  - Technology productivity S-curves
  - Debt supercycle position
  - Geopolitical regime (globalization/deglobalization)
  - Energy transition inflection

DISCRIMINATION ALGORITHM:
  1. Compute HP-filter decomposition of each signal
  2. Measure deviation from secular trend (cyclical component)
  3. Test for structural break (Chow test at rolling window)
  4. Assign conviction: cyclical (0.6-1.4σ) vs secular (>2σ persistent)
```

---

## Agent Consensus & Live Allocation Flow

```
Every 15 minutes (live market hours):

1. Data refresh: Financial-Data → fetch OHLCV + options chain
2. Macro update: open-bb → FRED/macro indicators
3. Signal compute: QLIB factors + ML-Macro regime label
4. Agent run: 6 ai-hedgefund agents analyze → vote matrix
5. Mav-Analysis: Claude API tool calls for qualitative layer
6. Consensus: Weighted average of agent outputs (by Sharpe contribution)
7. Allocation delta: compare current vs optimal weights
8. Risk gate: check VaR, correlation, regime compatibility
9. Execute: route orders if delta > threshold AND regime stable
10. Log: store full decision trace for attribution analysis
```

---

## Repository Integration Map

| Repo | Language | Framework | Integration Point |
|------|----------|-----------|------------------|
| Financial-Data | Python | yfinance | Data layer — all repos consume |
| open-bb | Python/TS | OpenBB Platform | Macro data + alt data |
| QLIB | Python | PyTorch/sklearn | Alpha engine → ai-hedgefund |
| quant-trading | Python | backtrader | Strategy signals → ai-hedgefund |
| ML-Macro-Market | Python | scikit-learn | Regime labels → allocation engine |
| ai-hedgefund | Python | LangGraph | Core decision engine |
| Mav-Analysis | Python | FastMCP | Claude tool server |
| Air-LLM | Python | HuggingFace | Local LLM inference |
| AI-Newton | Rust/Python | PyO3/maturin | Physics models |
| hedgefund-tracker | Python | - | 13F data → agent context |
| Ruflo-agents | TypeScript | claude-flow | Orchestration spine |

---

## Phase 1 Implementation Plan (Ready for Code Input)

### Phase 1A — Data Foundation
- [ ] Financial-Data: Add money velocity data fetching (FRED API)
- [ ] open-bb: Configure macro provider pipeline
- [ ] hedgefund-tracker: Wire 13F data to structured DB

### Phase 1B — Signal Engine
- [ ] ML-Macro-Market: Extend with cyclical/secular classifier
- [ ] QLIB: Configure factor library for live signals
- [ ] quant-trading: Package strategies as importable modules

### Phase 1C — Agent Intelligence
- [ ] ai-hedgefund: Add macro + velocity agents
- [ ] Mav-Analysis: Expose velocity + regime as MCP tools
- [ ] Ruflo-agents: Wire all agents into portfolio workflow

### Phase 1D — Analytics Dashboard
- [ ] Build unified portfolio analytics API
- [ ] Real-time allocation monitor
- [ ] Attribution + regime reporting

---

## Environment Setup Per Repo

```bash
# All repos are on branch: claude/init-test-repos-oPogr

# Python repos (Mav-Analysis, ai-hedgefund, QLIB, quant-trading,
#               ML-Macro-Market, Air-LLM, hedgefund-tracker, Financial-Data, open-bb)
pip install pandas numpy matplotlib scikit-learn langchain langgraph anthropic

# Rust+Python (AI-Newton)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
maturin develop

# TypeScript (Ruflo-agents)
npm install
npm run build
```

---

## Key .env Variables Required

```env
# LLM APIs
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Market Data
ALPHA_VANTAGE_API_KEY=
FINNHUB_TOKEN=

# Macro Data
FRED_API_KEY=

# Broker (optional for live execution)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

---

*Next: Awaiting 3 code sets from user with tree-of-command allocation specifics.*
