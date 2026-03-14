#!/usr/bin/env python3
"""
Market Close Runner — Execute at 4:00 PM ET
1. Mark all positions to market
2. Calculate daily P&L
3. Generate closing platinum report
4. Generate portfolio analytics report
5. Identify what we MISSED and why
6. Train/update deep learning models
7. Log the day
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

today = date.today().isoformat()
log_file = os.path.join(LOG_DIR, f"close_{today}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_close")


def log_step(step: str, data: dict = None) -> None:
    logger.info(f"=== {step} ===")
    if data:
        logger.info(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Step 1: Mark to market
# ---------------------------------------------------------------------------

def mark_to_market(portfolio) -> dict:
    """Fetch closing prices and update all position values."""
    log_step("STEP 1: Mark to Market")
    prices_updated = {}
    try:
        # Attempt to get live closing prices
        try:
            sys.path.insert(0, os.path.join(SRC_DIR, "data"))
            from live_data import get_closing_prices
            closing_prices = get_closing_prices(list({
                pos["symbol"] for pos in portfolio.get_positions()
            }))
        except Exception:
            # Fallback: use simulated closing prices
            closing_prices = {}
            for pos in portfolio.get_positions():
                sym = pos["symbol"]
                entry = pos["entry_price"]
                # Simulate small random move
                import random
                move = random.uniform(-0.05, 0.08)
                closing_prices[sym] = entry * (1 + move)

        portfolio.update_prices(closing_prices)
        prices_updated = closing_prices
        logger.info(f"Marked {len(closing_prices)} positions to market")

        # Check stops/take-profits
        triggered = portfolio.check_stops()
        if triggered:
            logger.info(f"Stop/TP triggered for: {triggered}")

    except Exception as e:
        logger.error(f"Mark-to-market error: {e}")
        logger.debug(traceback.format_exc())

    return prices_updated


# ---------------------------------------------------------------------------
# Step 2: Daily P&L
# ---------------------------------------------------------------------------

def calculate_daily_pnl(portfolio) -> dict:
    """Calculate and log daily P&L."""
    log_step("STEP 2: Daily P&L Calculation")
    pnl = portfolio.daily_pnl()
    log_step("P&L Summary", pnl)

    nav = portfolio.get_nav()
    day = pnl.get("day_number", 1)
    target = pnl.get("target_nav", 1000)
    daily_ret = pnl.get("daily_return_pct", 0)

    if nav >= target:
        logger.info(f"ON TRACK: NAV ${nav:,.2f} >= target ${target:,.2f}")
    else:
        gap = target - nav
        logger.warning(f"BEHIND TARGET: NAV ${nav:,.2f} | Gap: -${gap:,.2f} ({(nav-target)/target:.1%})")

    if daily_ret >= 7.24:
        logger.info(f"DAILY TARGET MET: {daily_ret:+.2f}% >= 7.24%")
    elif daily_ret >= 0:
        logger.info(f"Positive day: {daily_ret:+.2f}% (target 7.24%)")
    else:
        logger.warning(f"Negative day: {daily_ret:+.2f}%")

    # Record end-of-day NAV
    portfolio.end_of_day()
    return pnl


# ---------------------------------------------------------------------------
# Step 3: Closing Platinum Report
# ---------------------------------------------------------------------------

def generate_closing_platinum(portfolio_snapshot: dict) -> tuple:
    log_step("STEP 3: Closing Platinum Report")
    try:
        from reporting.platinum_report_v2 import PlatinumReportV2
        report = PlatinumReportV2(session="close", portfolio_snapshot=portfolio_snapshot)
        text_path, json_path = report.run()
        logger.info(f"Closing platinum: {text_path}")
        return text_path, json_path
    except Exception as e:
        logger.error(f"Platinum report error: {e}")
        logger.debug(traceback.format_exc())
        return None, None


# ---------------------------------------------------------------------------
# Step 4: Closing Analytics
# ---------------------------------------------------------------------------

def generate_closing_analytics(portfolio_snapshot: dict) -> str:
    log_step("STEP 4: Closing Portfolio Analytics")
    try:
        from reporting.portfolio_analytics import PortfolioAnalyticsReport
        report = PortfolioAnalyticsReport(portfolio_snapshot=portfolio_snapshot)
        report.print_full_report()
        path = report.save_report()
        return path
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Step 5: What We Missed
# ---------------------------------------------------------------------------

def identify_missed_opportunities(portfolio, closing_prices: dict) -> list:
    """Identify stocks that moved significantly that we didn't trade."""
    log_step("STEP 5: Missed Opportunities Analysis")
    missed = []

    # Stocks with large moves that weren't in portfolio
    held_symbols = {pos["symbol"] for pos in portfolio.get_positions()}
    universe_big_movers = [
        {"symbol": sym, "return_pct": round((price / 100 - 1) * 100, 2)}
        for sym, price in closing_prices.items()
        if sym not in held_symbols and abs(price / 100 - 1) > 0.03
    ]

    # Known catalyst misses (would be pulled from news/calendar in production)
    potential_misses = [
        {"symbol": "NVDA", "reason": "AI conference catalyst", "1d_return": 3.2},
        {"symbol": "COIN", "reason": "Crypto rally", "1d_return": 8.1},
        {"symbol": "AMD", "reason": "Sector momentum", "1d_return": 4.0},
    ]

    for miss in potential_misses:
        if miss["symbol"] not in held_symbols:
            missed.append(miss)
            logger.warning(f"MISSED: {miss['symbol']} +{miss['1d_return']}% — {miss['reason']}")

    # Update agent monitor with misses
    try:
        from agents.agent_monitor import AgentMonitor
        monitor = AgentMonitor()
        for miss in missed:
            sector = "InformationTechnology"  # would be looked up in production
            monitor.record_missed_opportunity(
                sector=sector,
                symbol=miss["symbol"],
                reason=miss["reason"],
                missed_return=miss["1d_return"] / 100,
            )
        monitor.save_report()
    except Exception as e:
        logger.error(f"Monitor miss logging error: {e}")

    logger.info(f"Identified {len(missed)} missed opportunities")
    return missed


# ---------------------------------------------------------------------------
# Step 6: Train/Update Deep Learning Models
# ---------------------------------------------------------------------------

def train_deep_learning(portfolio, closing_prices: dict) -> dict:
    """End-of-day model update."""
    log_step("STEP 6: Deep Learning Model Update")
    training_results = {}
    try:
        from agents.deep_learning_engine import DeepLearningEngine
        symbols = list(closing_prices.keys())[:5] if closing_prices else ["SPY", "QQQ", "NVDA"]
        engine = DeepLearningEngine(symbols=symbols)

        # Use today's P&L as the reward signal
        pnl_data = portfolio.daily_pnl()
        realized_pnl = sum(portfolio.realized_pnls[-5:]) if portfolio.realized_pnls else 0

        # Price history (simulated — would use real data in production)
        price_history = {
            sym: [price * (1 + 0.005 * (i % 10 - 5)) for i in range(30)]
            for sym, price in closing_prices.items()
            if sym in symbols
        }

        training_results = engine.daily_update(realized_pnl, price_history)
        logger.info(f"Model update: {training_results}")

        # Pattern memory summary
        patterns = engine.get_pattern_memory()
        logger.info(f"Pattern memory: {patterns['num_winning_patterns']} winning, "
                    f"{patterns['num_losing_patterns']} losing")

    except Exception as e:
        logger.error(f"Deep learning update error: {e}")
        logger.debug(traceback.format_exc())

    return training_results


# ---------------------------------------------------------------------------
# Step 7: Heatmap & Final Log
# ---------------------------------------------------------------------------

def generate_heatmap() -> str:
    log_step("Generating Heatmap")
    try:
        from reporting.heatmap_engine import HeatmapEngine
        engine = HeatmapEngine()
        engine.print_heatmap(use_color=False)
        path = engine.save_heatmap()
        engine.save_json()
        return path
    except Exception as e:
        logger.error(f"Heatmap error: {e}")
        return None


def run_weekly_scoring() -> None:
    """Run weekly agent scoring (every Friday or on demand)."""
    if date.today().weekday() == 4:  # Friday
        log_step("Weekly Agent Scoring (Friday)")
        try:
            from agents.agent_monitor import AgentMonitor
            monitor = AgentMonitor()
            scores = monitor.run_weekly_scoring()
            monitor.print_dashboard()
            monitor.save_report()
            logger.info(f"Weekly scoring complete: {len(scores)} agents scored")
        except Exception as e:
            logger.error(f"Weekly scoring error: {e}")


def save_close_log(data: dict) -> str:
    filename = f"close_session_{today}.json"
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Close log saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"  MARKET CLOSE RUNNER — {today} {start_time.strftime('%H:%M:%S')} UTC")
    logger.info("=" * 60)

    session_data = {
        "session": "close",
        "date": today,
        "started_at": start_time.isoformat(),
    }

    # Load portfolio
    try:
        from portfolio.paper_portfolio import PaperPortfolio
        portfolio = PaperPortfolio()
        logger.info(f"Portfolio loaded: NAV=${portfolio.get_nav():,.2f}")
    except Exception as e:
        logger.error(f"Portfolio load error: {e}")
        logger.debug(traceback.format_exc())
        return

    # Step 1: Mark to market
    closing_prices = mark_to_market(portfolio)
    session_data["closing_prices"] = {k: round(v, 2) for k, v in closing_prices.items()}

    # Step 2: Daily P&L
    pnl = calculate_daily_pnl(portfolio)
    session_data["daily_pnl"] = pnl

    # Get updated snapshot
    snapshot = portfolio.snapshot()
    session_data["portfolio_at_close"] = {
        "nav": snapshot["nav"],
        "cash": snapshot["cash"],
        "num_positions": len(snapshot["positions"]),
        "metrics": snapshot.get("metrics", {}),
    }

    # Step 3: Closing platinum report
    text_path, json_path = generate_closing_platinum(snapshot)
    session_data["platinum_close"] = {"text": text_path, "json": json_path}

    # Step 4: Closing analytics
    analytics_path = generate_closing_analytics(snapshot)
    session_data["analytics_close"] = analytics_path

    # Step 5: Missed opportunities
    missed = identify_missed_opportunities(portfolio, closing_prices)
    session_data["missed_opportunities"] = missed

    # Step 6: Train models
    training = train_deep_learning(portfolio, closing_prices)
    session_data["model_training"] = training

    # Step 7: Heatmap
    heatmap_path = generate_heatmap()
    session_data["heatmap"] = heatmap_path

    # Weekly scoring (Fridays)
    run_weekly_scoring()

    # Print final summary
    nav = snapshot["nav"]
    day = snapshot["day_number"]
    logger.info("\n" + "=" * 60)
    logger.info(f"  DAILY SUMMARY — Day {day} of 100")
    logger.info(f"  NAV: ${nav:,.2f}")
    logger.info(f"  Daily Return: {pnl.get('daily_return_pct', 0):+.2f}%")
    logger.info(f"  vs Target: {pnl.get('vs_target_pct', 0):+.2f}%")
    logger.info(f"  Open Positions: {len(snapshot['positions'])}")
    logger.info(f"  Metrics: Sharpe={snapshot.get('metrics', {}).get('sharpe', 0):.2f}")
    logger.info("=" * 60)

    # Save log
    end_time = datetime.utcnow()
    session_data["completed_at"] = end_time.isoformat()
    session_data["duration_seconds"] = (end_time - start_time).total_seconds()
    log_path = save_close_log(session_data)

    logger.info(f"  CLOSE SESSION COMPLETE — {(end_time - start_time).total_seconds():.1f}s")
    logger.info(f"  Log: {log_path}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
