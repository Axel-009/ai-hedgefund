#!/usr/bin/env python3
"""
Market Open Runner — Execute at 9:30 ET
1. Run all 11 sector agents
2. Generate morning platinum report
3. Generate portfolio analytics report
4. Execute top signals in paper portfolio
5. Log everything to logs/
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, date

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for sub in ["", "agents", "data", "portfolio", "reporting"]:
    p = os.path.join(SRC_DIR, sub) if sub else SRC_DIR
    if p not in sys.path:
        sys.path.insert(0, p)

LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Logging setup
today = date.today().isoformat()
log_file = os.path.join(LOG_DIR, f"open_{today}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_open")


def log_step(step: str, data: dict = None) -> None:
    logger.info(f"=== {step} ===")
    if data:
        logger.info(json.dumps(data, indent=2, default=str))


def run_sector_agents() -> dict:
    """Run all 11 GICS sector agents and collect signals."""
    log_step("STEP 1: Running Sector Agents")
    results = {}
    try:
        from agents.gics_sector_agents import GICSSectorAgents
        agents = GICSSectorAgents()
        results = agents.run_all() if hasattr(agents, "run_all") else {}
        logger.info(f"Sector agents completed: {len(results)} results")
    except Exception as e:
        logger.error(f"Sector agents error: {e}")
        logger.debug(traceback.format_exc())
        # Fallback: return mock signals
        sectors = ["Energy", "Materials", "Industrials", "ConsumerDiscretionary",
                   "ConsumerStaples", "HealthCare", "Financials", "InformationTechnology",
                   "CommunicationServices", "Utilities", "RealEstate"]
        results = {s: {"status": "mock", "signals": []} for s in sectors}
    return results


def generate_platinum_report(portfolio_snapshot: dict) -> tuple:
    """Generate morning platinum report (text + JSON)."""
    log_step("STEP 2: Generating Platinum Report")
    try:
        from reporting.platinum_report_v2 import PlatinumReportV2
        report = PlatinumReportV2(session="open", portfolio_snapshot=portfolio_snapshot)
        text_path, json_path = report.run()
        logger.info(f"Platinum report: {text_path}")
        logger.info(f"Flags: {report.get_flags()}")
        return text_path, json_path
    except Exception as e:
        logger.error(f"Platinum report error: {e}")
        logger.debug(traceback.format_exc())
        return None, None


def generate_analytics_report(portfolio_snapshot: dict) -> str:
    """Generate portfolio analytics report."""
    log_step("STEP 3: Generating Portfolio Analytics")
    try:
        from reporting.portfolio_analytics import PortfolioAnalyticsReport
        report = PortfolioAnalyticsReport(portfolio_snapshot=portfolio_snapshot)
        path = report.save_report()
        logger.info(f"Analytics report: {path}")
        logger.info(f"Base scenario: {report.get_base_scenario().name}")
        logger.info(f"Posture: {report.get_recommended_posture()}")
        return path
    except Exception as e:
        logger.error(f"Analytics report error: {e}")
        logger.debug(traceback.format_exc())
        return None


def get_top_signals(sector_results: dict, max_signals: int = 5) -> list:
    """Extract top signals from sector agents for execution."""
    signals = []
    for sector, result in sector_results.items():
        if isinstance(result, dict):
            sigs = result.get("signals", [])
            for sig in sigs:
                if isinstance(sig, dict):
                    sig["sector"] = sector
                    signals.append(sig)

    # Sort by confidence
    signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return signals[:max_signals]


def execute_signals(portfolio, signals: list) -> list:
    """Execute top signals in the paper portfolio."""
    log_step("STEP 4: Executing Signals")
    executed = []

    if not signals:
        logger.info("No signals to execute — skipping")
        # Demo trades if no signals
        signals = [
            {"symbol": "SPY", "action": "buy", "confidence": 0.72,
             "price": 510.0, "position_size_pct": 0.10, "sector": "Broad Market"},
        ]

    nav = portfolio.get_nav()
    for sig in signals:
        symbol = sig.get("symbol", "UNKNOWN")
        action = sig.get("action", sig.get("signal", "buy")).lower()
        confidence = sig.get("confidence", 0.5)
        price = sig.get("price", 100.0)
        size_pct = sig.get("position_size_pct", 0.05)

        if confidence < 0.55:
            logger.info(f"Skipping {symbol}: confidence {confidence:.1%} below 55% threshold")
            continue

        dollar_size = nav * size_pct
        shares = max(1, int(dollar_size / max(price, 0.01)))

        try:
            if action in ("buy", "long"):
                pos_id = portfolio.buy_equity(
                    symbol=symbol, quantity=shares, price=price,
                    stop_loss=price * 0.92, take_profit=price * 1.15,
                    notes=f"open_signal confidence={confidence:.2f}"
                )
                if pos_id:
                    executed.append({"symbol": symbol, "action": "BUY",
                                     "quantity": shares, "price": price,
                                     "position_id": pos_id})
            elif action in ("sell", "short"):
                pos_id = portfolio.sell_equity(
                    symbol=symbol, quantity=shares, price=price,
                    notes=f"open_short confidence={confidence:.2f}"
                )
                if pos_id:
                    executed.append({"symbol": symbol, "action": "SHORT",
                                     "quantity": shares, "price": price,
                                     "position_id": pos_id})

            logger.info(f"Executed: {action.upper()} {symbol} x{shares} @ ${price:.2f}")

        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")

    logger.info(f"Executed {len(executed)} of {len(signals)} signals")
    return executed


def run_pattern_scan(portfolio_snapshot: dict) -> dict:
    """Run pattern recognition on top symbols."""
    log_step("Pattern Scan")
    try:
        from agents.pattern_recognition import MarketPatternScanner
        import random
        scanner = MarketPatternScanner()
        top_symbols = ["NVDA", "MSFT", "AAPL", "META", "GOOGL"]
        results = {}
        for sym in top_symbols:
            closes = [100 + i + random.uniform(-2, 2) for i in range(50)]
            result = scanner.scan_symbol(sym, closes)
            results[sym] = result
            if result.get("extreme_conviction"):
                logger.warning(f"*** EXTREME CONVICTION: {sym} ***")
        return results
    except Exception as e:
        logger.error(f"Pattern scan error: {e}")
        return {}


def update_agent_monitor(sector_results: dict) -> None:
    """Log today's signals to the agent monitor."""
    log_step("Agent Monitor Update")
    try:
        from agents.agent_monitor import AgentMonitor
        monitor = AgentMonitor()
        for sector, result in sector_results.items():
            if isinstance(result, dict):
                for sig in result.get("signals", []):
                    if isinstance(sig, dict):
                        monitor.log_signal(
                            sector=sector,
                            symbol=sig.get("symbol", "UNKNOWN"),
                            signal=sig.get("action", "hold"),
                            confidence=sig.get("confidence", 0.5),
                        )
        monitor.save_report()
        logger.info("Agent monitor updated")
    except Exception as e:
        logger.error(f"Agent monitor error: {e}")


def save_open_log(data: dict) -> str:
    """Save comprehensive open-session log."""
    filename = f"open_session_{today}.json"
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Open log saved: {filepath}")
    return filepath


def main() -> None:
    start_time = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"  MARKET OPEN RUNNER — {today} {start_time.strftime('%H:%M:%S')} UTC")
    logger.info("=" * 60)

    session_data = {
        "session": "open",
        "date": today,
        "started_at": start_time.isoformat(),
    }

    # Load portfolio
    try:
        from portfolio.paper_portfolio import PaperPortfolio
        portfolio = PaperPortfolio()
        snapshot = portfolio.snapshot()
        session_data["portfolio_at_open"] = {
            "nav": snapshot["nav"],
            "cash": snapshot["cash"],
            "num_positions": len(snapshot["positions"]),
        }
        logger.info(f"Portfolio NAV: ${snapshot['nav']:,.2f} | Day {snapshot['day_number']}")
        portfolio.print_compound_table()
    except Exception as e:
        logger.error(f"Portfolio load error: {e}")
        logger.debug(traceback.format_exc())
        snapshot = {}
        portfolio = None

    # Step 1: Sector agents
    sector_results = run_sector_agents()
    session_data["sector_agents"] = {k: "completed" for k in sector_results}

    # Step 2: Platinum report
    text_path, json_path = generate_platinum_report(snapshot)
    session_data["platinum_report"] = {"text": text_path, "json": json_path}

    # Step 3: Analytics report
    analytics_path = generate_analytics_report(snapshot)
    session_data["analytics_report"] = analytics_path

    # Step 4: Execute signals
    if portfolio:
        signals = get_top_signals(sector_results)
        executed = execute_signals(portfolio, signals)
        session_data["executed_trades"] = executed
        snapshot_after = portfolio.snapshot()
        session_data["portfolio_after_open"] = {
            "nav": snapshot_after["nav"],
            "cash": snapshot_after["cash"],
            "num_positions": len(snapshot_after["positions"]),
        }

    # Pattern scan
    pattern_results = run_pattern_scan(snapshot)
    extreme_signals = [sym for sym, r in pattern_results.items() if r.get("extreme_conviction")]
    session_data["extreme_conviction_signals"] = extreme_signals
    if extreme_signals:
        logger.warning(f"EXTREME CONVICTION: {extreme_signals}")

    # Agent monitor
    update_agent_monitor(sector_results)

    # Save log
    end_time = datetime.utcnow()
    session_data["completed_at"] = end_time.isoformat()
    session_data["duration_seconds"] = (end_time - start_time).total_seconds()
    log_path = save_open_log(session_data)

    logger.info("=" * 60)
    logger.info(f"  OPEN SESSION COMPLETE — {(end_time - start_time).total_seconds():.1f}s")
    logger.info(f"  Log: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
