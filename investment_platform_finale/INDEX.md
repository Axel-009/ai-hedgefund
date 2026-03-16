# Investment Platform Finale — Master Index

Complete consolidated snapshot of every module built across the platform.

---

## Quick Start

```bash
pip install rich yfinance
python live_dashboard.py          # live 6-panel terminal dashboard
python run_open.py                # morning open routine
python run_hourly.py              # hourly scan
python run_close.py               # market close wrap
```

Background mode:
```bash
nohup python live_dashboard.py > dashboard.log 2>&1 & echo $! > dashboard.pid
```

---

## Architecture

```
investment_platform_finale/
│
├── live_dashboard.py             # 6-panel Rich terminal dashboard (real-time)
├── run_open.py                   # Market open routine
├── run_hourly.py                 # Hourly signal scan
├── run_close.py                  # Market close & wrap
├── init_test_all.py              # Full platform health check
├── setup.sh / setup_cron.sh      # Setup & cron automation
├── pyproject.toml                # Dependencies
│
├── src/
│   ├── agents/                   # All trading agents
│   │   ├── warren_buffett.py     # Value investing agent
│   │   ├── ben_graham.py         # Deep value agent
│   │   ├── charlie_munger.py     # Quality compounder agent
│   │   ├── michael_burry.py      # Contrarian/distressed agent
│   │   ├── peter_lynch.py        # Growth at reasonable price agent
│   │   ├── phil_fisher.py        # Growth investing agent
│   │   ├── stanley_druckenmiller.py  # Macro/momentum agent
│   │   ├── bill_ackman.py        # Activist investing agent
│   │   ├── cathie_wood.py        # Disruptive innovation agent
│   │   ├── mohnish_pabrai.py     # Focused value agent
│   │   ├── rakesh_jhunjhunwala.py # Emerging market agent
│   │   ├── aswath_damodaran.py   # Valuation specialist agent
│   │   │
│   │   ├── macro_engine.py       # Macro regime detection
│   │   ├── alpha_beta_engine.py  # Alpha/Beta analysis
│   │   ├── alpha_optimizer.py    # Signal optimizer
│   │   ├── stat_arb_engine.py    # Statistical arbitrage
│   │   ├── event_driven_engine.py # Corporate events engine
│   │   ├── options_engine.py     # Options flow analysis
│   │   ├── deep_learning_engine.py # DL-based signals
│   │   ├── cvr_engine.py         # Conviction/risk engine
│   │   ├── contagion_engine.py   # Contagion/correlation engine
│   │   ├── distressed_asset_engine.py # Distressed assets
│   │   ├── pattern_recognition.py # Technical pattern engine
│   │   ├── execution_engine.py   # Trade execution
│   │   ├── platform_orchestrator.py # Master orchestrator
│   │   ├── metadron_cube.py      # 3D signal cube
│   │   ├── gics_sector_agents.py # GICS sector coverage
│   │   ├── universe_engine.py    # Stock universe management
│   │   ├── portfolio_manager.py  # Portfolio construction
│   │   ├── risk_manager.py       # Risk management
│   │   ├── agent_monitor.py      # Agent health monitoring
│   │   ├── agent_scorecard.py    # Performance scoring
│   │   ├── conviction_override.py # High-conviction override
│   │   ├── missed_opportunities.py # Missed trade tracker
│   │   ├── platinum_report.py    # Platinum tier reporting
│   │   └── ...
│   │
│   ├── data/
│   │   ├── live_data.py          # yfinance live data feed
│   │   ├── cache.py              # Data caching layer
│   │   ├── models.py             # Data models
│   │   └── gics_universe.py      # GICS universe data
│   │
│   ├── models/
│   │   ├── deep_trading_features.py  # Deep learning features
│   │   ├── finrl_bridge.py           # FinRL RL integration
│   │   ├── kserve_adapter.py         # KServe model serving
│   │   ├── model_evaluator.py        # Model evaluation
│   │   ├── monte_carlo_bridge.py     # Monte Carlo simulation
│   │   ├── nvidia_tft_adapter.py     # NVIDIA TFT model
│   │   ├── stock_prediction_bridge.py # Prediction bridge
│   │   └── universe_classifier.py    # Universe classification
│   │
│   ├── portfolio/
│   │   └── paper_portfolio.py    # Paper trading portfolio
│   │
│   ├── backtesting/
│   │   ├── engine.py             # Backtest engine
│   │   ├── controller.py         # Backtest controller
│   │   ├── metrics.py            # Performance metrics
│   │   ├── portfolio.py          # Portfolio simulation
│   │   ├── trader.py             # Trade simulation
│   │   ├── benchmarks.py         # Benchmark comparisons
│   │   └── output.py             # Results output
│   │
│   ├── reporting/
│   │   ├── platinum_report_v2.py # Platinum tier reports
│   │   ├── portfolio_analytics.py # Analytics engine
│   │   ├── heatmap_engine.py     # Market heatmap
│   │   └── live_earnings_graph.py # Earnings visualization
│   │
│   ├── monitoring/
│   │   ├── anomaly_detector.py   # Anomaly detection
│   │   ├── daily_report.py       # Daily report generator
│   │   ├── heatmap.py            # Heatmap generator
│   │   ├── hourly_recap.py       # Hourly recap
│   │   ├── market_wrap.py        # Market wrap report
│   │   └── memory_monitor.py     # System memory monitoring
│   │
│   └── tools/
│       └── api.py                # API tools layer
│
├── app/
│   ├── backend/                  # FastAPI backend
│   │   ├── main.py               # API server entrypoint
│   │   ├── routes/               # API routes
│   │   ├── services/             # Business logic
│   │   ├── models/               # Pydantic schemas
│   │   └── database/             # DB models + migrations
│   │
│   └── frontend/                 # React/Vite frontend
│       └── src/
│           ├── nodes/            # Flow graph nodes
│           ├── components/       # UI components
│           └── services/         # API client services
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── tests/
    ├── backtesting/              # Backtesting test suite
    └── fixtures/                 # Test fixtures
```

---

## Agent Roster (12 Legend Investors + 12 Quant Engines)

| Agent | Style |
|---|---|
| Warren Buffett | Long-term value |
| Ben Graham | Deep value / margin of safety |
| Charlie Munger | Quality compounders |
| Michael Burry | Contrarian / distressed |
| Peter Lynch | GARP |
| Phil Fisher | Growth |
| Stanley Druckenmiller | Macro / momentum |
| Bill Ackman | Activist |
| Cathie Wood | Disruptive innovation |
| Mohnish Pabrai | Focused value |
| Rakesh Jhunjhunwala | Emerging market |
| Aswath Damodaran | Valuation |
| MacroEngine | Regime detection |
| StatArbEngine | Pair trading |
| EventDrivenEngine | Corporate events |
| OptionsEngine | Flow analysis |
| DeepLearningEngine | Neural signals |
| CVREngine | Conviction/risk |
| ContagionEngine | Correlation risk |
| DistressedAssetEngine | Distressed plays |
| PatternRecognition | Technical patterns |
| ExecutionEngine | Trade execution |

---

## Current Status: Learning

Agents are currently **rule-based** (static weights). To enable live learning:
- Add AgentDB + `agentdb-learning` skill for RL feedback loops
- Each agent's signal weights update based on P&L attribution
- Persistent memory across sessions via AgentDB vector store
