#!/usr/bin/env python3
"""
Hourly Update Runner
- Update position values
- Check for stop losses / take profits
- Log any triggered alerts
- Update heatmap data
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
now_str = datetime.utcnow().strftime("%H%M")
log_file = os.path.join(LOG_DIR, f"hourly_{today}_{now_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_hourly")


# ---------------------------------------------------------------------------
# Price updater
# ---------------------------------------------------------------------------

def fetch_current_prices(symbols: list) -> dict:
    """Fetch current intraday prices for all held symbols."""
    prices = {}
    try:
        sys.path.insert(0, os.path.join(SRC_DIR, "data"))
        from live_data import get_intraday_prices
        prices = get_intraday_prices(symbols)
        logger.info(f"Live prices fetched for {len(prices)} symbols")
    except Exception:
        # Fallback: simulate small intraday moves
        import random
        for sym in symbols:
            base = 100.0  # would use last known price
            prices[sym] = base * (1 + random.uniform(-0.015, 0.020))
        logger.info(f"Using simulated prices for {len(prices)} symbols")
    return prices


# ---------------------------------------------------------------------------
# Stop/TP checker
# ---------------------------------------------------------------------------

def check_risk_levels(portfolio, prices: dict) -> dict:
    """
    Update prices and check stop losses / take profits.
    Returns dict with triggered alerts.
    """
    portfolio.update_prices(prices)
    triggered_positions = portfolio.check_stops()

    alerts = {
        "stop_losses_triggered": [],
        "take_profits_triggered": [],
        "near_stop": [],
        "near_take_profit": [],
    }

    for pos in portfolio.get_positions():
        sym = pos["symbol"]
        current = pos["current_price"]
        entry = pos["entry_price"]
        direction = pos["direction"]
        stop = pos.get("stop_loss")
        tp = pos.get("take_profit")

        # Near-stop warning (within 2%)
        if stop and direction == "long":
            pct_from_stop = (current - stop) / max(stop, 1e-9)
            if 0 < pct_from_stop < 0.02:
                alerts["near_stop"].append({
                    "symbol": sym, "current": round(current, 2),
                    "stop": round(stop, 2), "pct_away": round(pct_from_stop * 100, 2)
                })
                logger.warning(f"NEAR STOP: {sym} @ ${current:.2f} stop @ ${stop:.2f} "
                               f"({pct_from_stop:.1%} away)")

        # Near-TP warning (within 2%)
        if tp and direction == "long":
            pct_from_tp = (tp - current) / max(tp, 1e-9)
            if 0 < pct_from_tp < 0.02:
                alerts["near_take_profit"].append({
                    "symbol": sym, "current": round(current, 2),
                    "tp": round(tp, 2), "pct_away": round(pct_from_tp * 100, 2)
                })
                logger.info(f"NEAR TP: {sym} @ ${current:.2f} TP @ ${tp:.2f} "
                            f"({pct_from_tp:.1%} away)")

    return alerts


# ---------------------------------------------------------------------------
# NAV & P&L snapshot
# ---------------------------------------------------------------------------

def log_intraday_snapshot(portfolio) -> dict:
    """Log current intraday portfolio state."""
    nav = portfolio.get_nav()
    pnl = portfolio.daily_pnl()
    positions = portfolio.get_positions()

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "nav": round(nav, 2),
        "daily_pnl": pnl,
        "num_positions": len(positions),
        "unrealized_pnl": round(sum(p.get("unrealized_pnl", 0) for p in positions), 2),
        "positions_summary": [
            {
                "symbol": p["symbol"],
                "direction": p["direction"],
                "unrealized_pnl": round(p.get("unrealized_pnl", 0), 2),
                "unrealized_pnl_pct": round(p.get("unrealized_pnl_pct", 0), 2),
            }
            for p in sorted(positions, key=lambda x: x.get("unrealized_pnl", 0), reverse=True)
        ],
    }

    logger.info(f"Intraday NAV: ${nav:,.2f} | "
                f"Unrealized: ${snapshot['unrealized_pnl']:+,.2f} | "
                f"Positions: {len(positions)}")

    # Warn if significantly behind target
    vs_target = pnl.get("vs_target_pct", 0)
    if vs_target < -15:
        logger.warning(f"SIGNIFICANTLY BEHIND TARGET: {vs_target:.1f}%")

    return snapshot


# ---------------------------------------------------------------------------
# Heatmap update
# ---------------------------------------------------------------------------

def update_heatmap(prices: dict) -> str:
    """Update heatmap with current intraday prices."""
    try:
        from reporting.heatmap_engine import HeatmapEngine
        engine = HeatmapEngine()
        engine.update_returns(prices)
        path = engine.save_heatmap()
        engine.save_json()
        logger.info(f"Heatmap updated: {path}")
        return path
    except Exception as e:
        logger.error(f"Heatmap update error: {e}")
        return None


# ---------------------------------------------------------------------------
# Alert logger
# ---------------------------------------------------------------------------

def log_alerts(alerts: dict, snapshot: dict) -> str:
    """Save all alerts and snapshot to log file."""
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "alerts": alerts,
        "snapshot": snapshot,
    }
    filename = f"hourly_alerts_{today}_{now_str}.json"
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Alert log saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Intraday signal scan (lightweight)
# ---------------------------------------------------------------------------

def quick_signal_scan(portfolio, prices: dict) -> list:
    """
    Quick intraday scan for new entry opportunities.
    Only runs if we have significant cash available.
    """
    nav = portfolio.get_nav()
    cash = portfolio.cash
    if cash / max(nav, 1) < 0.15:
        logger.info(f"Cash too low for new entries ({cash/nav:.0%}), skipping signal scan")
        return []

    signals = []
    # Simple momentum signals (would be replaced by full agent logic)
    for sym, price in list(prices.items())[:5]:
        intraday_ret = (price / 100) - 1  # simplified
        if intraday_ret > 0.02:  # up >2% intraday
            signals.append({
                "symbol": sym, "signal": "buy",
                "confidence": min(0.75, 0.5 + intraday_ret * 10),
                "intraday_return": round(intraday_ret * 100, 2),
                "reason": f"Intraday momentum {intraday_ret:+.1%}",
            })
        elif intraday_ret < -0.025:  # down >2.5% intraday
            signals.append({
                "symbol": sym, "signal": "short",
                "confidence": min(0.70, 0.5 + abs(intraday_ret) * 8),
                "intraday_return": round(intraday_ret * 100, 2),
                "reason": f"Breakdown {intraday_ret:+.1%}",
            })

    if signals:
        logger.info(f"Intraday signals: {len(signals)} found")
        for s in signals:
            logger.info(f"  {s['symbol']}: {s['signal'].upper()} ({s['reason']})")

    return signals


# ---------------------------------------------------------------------------
# Options expiry check
# ---------------------------------------------------------------------------

def check_options_expiry(portfolio) -> list:
    """Flag options positions expiring within 2 days."""
    from datetime import timedelta
    expiring = []
    today_dt = date.today()
    for pos in portfolio.get_positions():
        if pos["asset_type"] in ("call", "put") and pos.get("expiry"):
            try:
                exp_date = date.fromisoformat(pos["expiry"])
                days_left = (exp_date - today_dt).days
                if days_left <= 2:
                    expiring.append({
                        "position_id": pos["position_id"],
                        "symbol": pos["symbol"],
                        "type": pos["asset_type"],
                        "strike": pos.get("strike"),
                        "expiry": pos["expiry"],
                        "days_left": days_left,
                        "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    })
                    logger.warning(f"OPTIONS EXPIRY WARNING: {pos['symbol']} {pos['asset_type'].upper()} "
                                   f"${pos.get('strike')} expires in {days_left} day(s)")
            except Exception:
                pass
    return expiring


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_time = datetime.utcnow()
    logger.info(f"{'='*50}")
    logger.info(f"  HOURLY UPDATE — {run_time.strftime('%Y-%m-%d %H:%M')} UTC")
    logger.info(f"{'='*50}")

    hourly_data = {
        "timestamp": run_time.isoformat(),
        "date": today,
        "hour": run_time.strftime("%H%M"),
    }

    # Load portfolio
    try:
        from portfolio.paper_portfolio import PaperPortfolio
        portfolio = PaperPortfolio()
        logger.info(f"Portfolio: NAV=${portfolio.get_nav():,.2f} | {len(portfolio.positions)} positions")
    except Exception as e:
        logger.error(f"Portfolio load error: {e}")
        logger.debug(traceback.format_exc())
        return

    # Get current symbols
    held_symbols = list({pos["symbol"] for pos in portfolio.get_positions()})
    # Add universe symbols for heatmap
    universe_symbols = held_symbols + ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "META"]
    universe_symbols = list(set(universe_symbols))

    # Fetch prices
    prices = fetch_current_prices(universe_symbols)

    # Check risk levels (stops/TPs)
    alerts = check_risk_levels(portfolio, prices)
    hourly_data["alerts"] = alerts

    total_alerts = sum(len(v) for v in alerts.values())
    if total_alerts > 0:
        logger.warning(f"ALERTS: {total_alerts} triggered")
    else:
        logger.info("No alerts triggered")

    # Options expiry check
    expiring = check_options_expiry(portfolio)
    if expiring:
        hourly_data["expiring_options"] = expiring
        logger.warning(f"{len(expiring)} options positions near expiry!")

    # Portfolio snapshot
    snapshot = log_intraday_snapshot(portfolio)
    hourly_data["snapshot"] = snapshot

    # Intraday signal scan
    signals = quick_signal_scan(portfolio, prices)
    hourly_data["intraday_signals"] = signals

    # Heatmap update
    heatmap_path = update_heatmap(prices)
    hourly_data["heatmap_updated"] = heatmap_path

    # Save portfolio state
    portfolio.save_state()

    # Log alerts
    log_path = log_alerts(alerts, snapshot)
    hourly_data["log_path"] = log_path

    # Live earnings graph — regenerated every hour
    try:
        from reporting.live_earnings_graph import generate as gen_graph
        graph_path = gen_graph()
        hourly_data["earnings_graph"] = graph_path
        logger.info(f"Earnings graph updated: {graph_path}")
    except Exception as _eg:
        logger.debug(f"Earnings graph skipped: {_eg}")

    logger.info(f"{'='*50}")
    logger.info(f"  HOURLY UPDATE COMPLETE — NAV: ${snapshot['nav']:,.2f}")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()
