# Execution Arm — HFT Integration

## Components

### 1. WonderTrader (wondertrader/) — Signal Engine
- **Language:** C++ (Python bridge via MicroPriceEngine)
- **Signal:** Micro-price imbalance (volume-weighted mid vs last)
- **Translation:** `WtHftStraDemo_EN.h/.cpp` — full English translation
- **Key logic:** `micro_price = (bid*ask_qty + ask*bid_qty) / (ask_qty+bid_qty)`

### 2. exchange-core (exchange-core/) — Order Matching
- **Language:** Java (LMAX Disruptor + ART order book)
- **Latency:** 150ns match / 5M ops/sec
- **Bridge:** `ExchangeCoreAdapter` (Python paper mode → JVM gRPC in prod)
- **API surface:** `submit_order / cancel_order / get_fills`

### 3. Quant-Developers-Resources (resource hub)
- **Categories:** Technical Indicators, Reinforcement Learning, HPC, FPGA,
                  Signal Processing, Optimization, Financial Theory + more
- **Usage:** Reference library for strategy enhancement, RL training data

## Execution Flow (US Securities Only)
```
Pre-market: DailyUniverseScanner.scan()
    ├── Full US equity + ETF + credit universe
    ├── CAPM alpha rank + fallen angel bonus
    └── RV pair z-score → top 15 active names

Intraday (per tick):
    L1 quote → MicroPriceEngine.generate_signal()
        ├── edge > 2bps → Order
        ├── ExchangeCoreAdapter.submit_order()
        └── check_expiry() → cancel after 30s

Risk gates:
    - Max position: 500 shares per name
    - Long-only for US equities
    - Beta gate: net portfolio β ≤ BETA_MAX (from AlphaBetaUnleashed)
```

## US Universe (56 tickers)
- Core equity: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA...
- Sector ETFs: XLK, XLV, XLF, XLY, XLC, XLI, XLP, XLE, XLU, XLRE, XLB
- IG/Fallen Angel: ANGL, FALN, LQD, VCIT, HYG, JNK
- Fallen Angel equity: INTC, PFE, WBA, MPW, VFC, PARA, DIS
- RV pairs: GOOGL/META, XOM/CVX, AMD/INTC, JPM/BAC, V/MA, HD/LOW, PEP/KO
- Macro hedges: VXX, UVXY, GLD, TLT, SHY, IEF
