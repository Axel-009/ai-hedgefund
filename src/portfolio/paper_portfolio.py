"""
Paper Trading Portfolio Engine
Target: $1,000 → $1,000,000 in 100 days (7.24%/day compounded)
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Compound path table: day -> target NAV
# ---------------------------------------------------------------------------
DAILY_RATE = 0.0724  # 7.24% per day
STARTING_CAPITAL = 1_000.0
COMPOUND_TABLE: Dict[int, float] = {
    day: STARTING_CAPITAL * (1 + DAILY_RATE) ** day for day in range(0, 101)
}
# Key milestones
MILESTONES = {1: 1_072, 5: 1_420, 10: 2_014, 15: 2_790, 30: 8_013, 50: 31_691, 75: 197_735, 100: 1_000_000}

STATE_FILE = "/home/user/ai-hedgefund/portfolio_state.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    position_id: str
    symbol: str
    asset_type: str          # 'equity' | 'call' | 'put'
    direction: str           # 'long' | 'short'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: str
    expiry: Optional[str] = None          # options only
    strike: Optional[float] = None        # options only
    underlying: Optional[str] = None      # options only
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: str = ""

    @property
    def market_value(self) -> float:
        if self.direction == "long":
            return self.quantity * self.current_price
        else:  # short: we received premium, value goes against us
            return -self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.direction == "long":
            return self.quantity * (self.current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.quantity * self.entry_price) * 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["market_value"] = self.market_value
        d["unrealized_pnl"] = self.unrealized_pnl
        d["unrealized_pnl_pct"] = self.unrealized_pnl_pct
        return d


@dataclass
class Trade:
    trade_id: str
    position_id: str
    symbol: str
    asset_type: str
    direction: str
    action: str              # 'open' | 'close' | 'partial_close'
    quantity: float
    price: float
    timestamp: str
    realized_pnl: float = 0.0
    commission: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Portfolio Metrics
# ---------------------------------------------------------------------------

class PortfolioMetrics:
    """Compute Sharpe, Sortino, max drawdown, win rate."""

    def __init__(self, daily_returns: List[float]):
        self.returns = daily_returns

    def sharpe(self, risk_free: float = 0.0) -> float:
        if len(self.returns) < 2:
            return 0.0
        import statistics
        excess = [r - risk_free for r in self.returns]
        mean = statistics.mean(excess)
        std = statistics.stdev(excess) if len(excess) > 1 else 1e-9
        return (mean / std) * math.sqrt(252) if std > 0 else 0.0

    def sortino(self, risk_free: float = 0.0) -> float:
        if len(self.returns) < 2:
            return 0.0
        import statistics
        downside = [r for r in self.returns if r < risk_free]
        if not downside:
            return float("inf")
        downside_std = statistics.stdev(downside) if len(downside) > 1 else 1e-9
        mean_excess = statistics.mean(self.returns) - risk_free
        return (mean_excess / downside_std) * math.sqrt(252)

    def max_drawdown(self) -> float:
        if not self.returns:
            return 0.0
        peak = 1.0
        nav = 1.0
        max_dd = 0.0
        for r in self.returns:
            nav *= (1 + r)
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def win_rate(self, realized_pnls: List[float]) -> float:
        if not realized_pnls:
            return 0.0
        winners = sum(1 for p in realized_pnls if p > 0)
        return winners / len(realized_pnls)

    def calmar(self) -> float:
        dd = self.max_drawdown()
        if dd == 0:
            return float("inf")
        ann_return = (1 + sum(self.returns) / max(len(self.returns), 1)) ** 252 - 1
        return ann_return / dd

    def summary(self, realized_pnls: List[float] = None) -> dict:
        return {
            "sharpe": round(self.sharpe(), 3),
            "sortino": round(self.sortino(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "calmar": round(self.calmar(), 3),
            "win_rate_pct": round(self.win_rate(realized_pnls or []) * 100, 1),
            "total_return_pct": round((sum(self.returns)) * 100, 2),
            "num_days": len(self.returns),
        }


# ---------------------------------------------------------------------------
# Main Portfolio Class
# ---------------------------------------------------------------------------

class PaperPortfolio:
    """
    Paper trading engine targeting $1,000 → $1,000,000 in 100 days.

    Strategy mix:
    - Primary (60%): Short-dated options on high-momentum names with catalysts
    - Secondary (30%): Leveraged equity on regime-aligned names
    - Hedge (10%): Protective puts / inverse ETFs
    """

    COMMISSION_PER_SHARE = 0.005
    OPTION_COMMISSION = 0.65  # per contract
    CONTRACTS_PER_LOT = 100

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash: float = STARTING_CAPITAL
        self.starting_capital: float = STARTING_CAPITAL
        self.daily_nav_history: List[dict] = []
        self.daily_returns: List[float] = []
        self.realized_pnls: List[float] = []
        self.guardrails_active: bool = True
        self.day_number: int = 0
        self.created_at: str = datetime.utcnow().isoformat()

        # Risk limits
        self.max_position_pct: float = 0.25       # max 25% in one position
        self.max_drawdown_limit: float = 0.15     # stop trading if DD > 15%
        self.min_cash_pct: float = 0.05           # keep 5% cash
        self.max_options_pct: float = 0.70        # max 70% in options

        if os.path.exists(state_file):
            self.load_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _commission(self, asset_type: str, quantity: float) -> float:
        if asset_type == "equity":
            return max(1.0, quantity * self.COMMISSION_PER_SHARE)
        else:
            contracts = quantity
            return contracts * self.OPTION_COMMISSION

    def _nav(self) -> float:
        pos_value = sum(p.quantity * p.current_price for p in self.positions.values()
                        if p.direction == "long")
        short_value = sum(p.quantity * p.entry_price for p in self.positions.values()
                          if p.direction == "short")
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + sum(p.quantity * p.current_price
                                for p in self.positions.values()
                                if p.direction == "long") + \
               sum(p.quantity * (p.entry_price - p.current_price)  # short gains
                   for p in self.positions.values()
                   if p.direction == "short")

    def get_nav(self) -> float:
        nav = self.cash
        for pos in self.positions.values():
            nav += pos.unrealized_pnl + pos.quantity * pos.entry_price if pos.direction == "long" \
                else pos.quantity * (pos.entry_price - pos.current_price)
        # simpler calculation
        nav = self.cash
        for pos in self.positions.values():
            if pos.direction == "long":
                nav += pos.quantity * pos.current_price
            else:
                # short: cash was credited at entry, mark current exposure
                nav += pos.quantity * (pos.entry_price - pos.current_price)
        return nav

    def risk_check(self, symbol: str, value: float, asset_type: str = "equity") -> Tuple[bool, str]:
        """Return (allowed, reason). Checks position limits."""
        if not self.guardrails_active:
            return True, "Guardrails disabled"

        nav = self.get_nav()

        # Drawdown check
        if self.daily_returns:
            peak = self.starting_capital
            for r in self.daily_returns:
                peak = max(peak, peak * (1 + r))
            current_dd = (peak - nav) / peak if peak > 0 else 0
            if current_dd > self.max_drawdown_limit:
                return False, f"Max drawdown {current_dd:.1%} exceeded limit {self.max_drawdown_limit:.1%}"

        # Cash floor
        if (self.cash - value) / nav < self.min_cash_pct:
            return False, f"Would breach min cash floor ({self.min_cash_pct:.0%} of NAV)"

        # Position concentration
        if value / nav > self.max_position_pct:
            return False, f"Position size {value/nav:.1%} exceeds max {self.max_position_pct:.1%}"

        # Options concentration
        if asset_type in ("call", "put"):
            current_options = sum(p.quantity * p.current_price
                                  for p in self.positions.values()
                                  if p.asset_type in ("call", "put") and p.direction == "long")
            if (current_options + value) / nav > self.max_options_pct:
                return False, f"Options allocation would exceed {self.max_options_pct:.0%}"

        return True, "OK"

    def override_guardrails(self, disable: bool = True, reason: str = "") -> None:
        """Enable or disable risk guardrails."""
        self.guardrails_active = not disable
        status = "DISABLED" if disable else "ENABLED"
        print(f"[PaperPortfolio] Guardrails {status}. Reason: {reason}")

    # ------------------------------------------------------------------
    # Trading methods
    # ------------------------------------------------------------------

    def buy_equity(self, symbol: str, quantity: float, price: float,
                   stop_loss: float = None, take_profit: float = None,
                   notes: str = "") -> Optional[str]:
        value = quantity * price
        comm = self._commission("equity", quantity)
        allowed, reason = self.risk_check(symbol, value, "equity")
        if not allowed:
            print(f"[RISK BLOCK] {symbol} buy blocked: {reason}")
            return None
        if self.cash < value + comm:
            print(f"[CASH] Insufficient cash for {symbol}: need ${value+comm:.2f}, have ${self.cash:.2f}")
            return None

        pos_id = self._new_id()
        self.cash -= (value + comm)
        pos = Position(
            position_id=pos_id, symbol=symbol, asset_type="equity",
            direction="long", quantity=quantity, entry_price=price,
            current_price=price, entry_time=datetime.utcnow().isoformat(),
            stop_loss=stop_loss, take_profit=take_profit, notes=notes
        )
        self.positions[pos_id] = pos
        trade = Trade(trade_id=self._new_id(), position_id=pos_id, symbol=symbol,
                      asset_type="equity", direction="long", action="open",
                      quantity=quantity, price=price,
                      timestamp=datetime.utcnow().isoformat(),
                      commission=comm, notes=notes)
        self.trades.append(trade)
        print(f"[BUY] {symbol} x{quantity} @ ${price:.2f} | Cost: ${value+comm:.2f} | NAV: ${self.get_nav():.2f}")
        self.save_state()
        return pos_id

    def sell_equity(self, symbol: str, quantity: float, price: float, notes: str = "") -> Optional[str]:
        """Open a short equity position."""
        proceeds = quantity * price
        comm = self._commission("equity", quantity)
        pos_id = self._new_id()
        self.cash += (proceeds - comm)  # credit short proceeds minus commission
        pos = Position(
            position_id=pos_id, symbol=symbol, asset_type="equity",
            direction="short", quantity=quantity, entry_price=price,
            current_price=price, entry_time=datetime.utcnow().isoformat(), notes=notes
        )
        self.positions[pos_id] = pos
        trade = Trade(trade_id=self._new_id(), position_id=pos_id, symbol=symbol,
                      asset_type="equity", direction="short", action="open",
                      quantity=quantity, price=price,
                      timestamp=datetime.utcnow().isoformat(),
                      commission=comm, notes=notes)
        self.trades.append(trade)
        print(f"[SHORT] {symbol} x{quantity} @ ${price:.2f} | Proceeds: ${proceeds-comm:.2f}")
        self.save_state()
        return pos_id

    def buy_option(self, symbol: str, option_type: str, strike: float,
                   expiry: str, contracts: int, premium: float,
                   underlying: str = None, stop_loss: float = None,
                   take_profit: float = None, notes: str = "") -> Optional[str]:
        """Buy calls or puts. premium is per share; 1 contract = 100 shares."""
        option_type = option_type.lower()
        assert option_type in ("call", "put"), "option_type must be 'call' or 'put'"
        value = contracts * self.CONTRACTS_PER_LOT * premium
        comm = self._commission(option_type, contracts)
        allowed, reason = self.risk_check(symbol, value, option_type)
        if not allowed:
            print(f"[RISK BLOCK] {symbol} {option_type} blocked: {reason}")
            return None
        if self.cash < value + comm:
            print(f"[CASH] Insufficient for {symbol} {option_type}: need ${value+comm:.2f}")
            return None

        pos_id = self._new_id()
        self.cash -= (value + comm)
        pos = Position(
            position_id=pos_id, symbol=symbol, asset_type=option_type,
            direction="long", quantity=float(contracts),
            entry_price=premium, current_price=premium,
            entry_time=datetime.utcnow().isoformat(),
            expiry=expiry, strike=strike,
            underlying=underlying or symbol,
            stop_loss=stop_loss, take_profit=take_profit, notes=notes
        )
        self.positions[pos_id] = pos
        trade = Trade(trade_id=self._new_id(), position_id=pos_id, symbol=symbol,
                      asset_type=option_type, direction="long", action="open",
                      quantity=float(contracts), price=premium,
                      timestamp=datetime.utcnow().isoformat(),
                      commission=comm, notes=notes)
        self.trades.append(trade)
        print(f"[OPTION] {symbol} {option_type.upper()} ${strike} exp {expiry} x{contracts} @ ${premium:.2f} | Cost: ${value+comm:.2f}")
        self.save_state()
        return pos_id

    def sell_option(self, symbol: str, option_type: str, strike: float,
                    expiry: str, contracts: int, premium: float,
                    underlying: str = None, notes: str = "") -> Optional[str]:
        """Sell (write) calls or puts."""
        option_type = option_type.lower()
        proceeds = contracts * self.CONTRACTS_PER_LOT * premium
        comm = self._commission(option_type, contracts)
        pos_id = self._new_id()
        self.cash += (proceeds - comm)
        pos = Position(
            position_id=pos_id, symbol=symbol, asset_type=option_type,
            direction="short", quantity=float(contracts),
            entry_price=premium, current_price=premium,
            entry_time=datetime.utcnow().isoformat(),
            expiry=expiry, strike=strike,
            underlying=underlying or symbol, notes=notes
        )
        self.positions[pos_id] = pos
        trade = Trade(trade_id=self._new_id(), position_id=pos_id, symbol=symbol,
                      asset_type=option_type, direction="short", action="open",
                      quantity=float(contracts), price=premium,
                      timestamp=datetime.utcnow().isoformat(),
                      commission=comm, notes=notes)
        self.trades.append(trade)
        print(f"[WRITE] {symbol} {option_type.upper()} ${strike} exp {expiry} x{contracts} @ ${premium:.2f}")
        self.save_state()
        return pos_id

    def close_position(self, position_id: str, close_price: float,
                       quantity: float = None, notes: str = "") -> float:
        """Close all or part of a position. Returns realized P&L."""
        if position_id not in self.positions:
            print(f"[ERROR] Position {position_id} not found")
            return 0.0

        pos = self.positions[position_id]
        pos.current_price = close_price
        close_qty = quantity if quantity is not None else pos.quantity

        # Calculate realized P&L
        if pos.asset_type == "equity":
            if pos.direction == "long":
                gross = close_qty * close_price
                comm = self._commission("equity", close_qty)
                realized = close_qty * (close_price - pos.entry_price) - comm
                self.cash += gross - comm
            else:  # short
                # buy back to cover
                gross = close_qty * close_price
                comm = self._commission("equity", close_qty)
                realized = close_qty * (pos.entry_price - close_price) - comm
                self.cash -= (gross + comm)
        else:  # option
            multiplier = self.CONTRACTS_PER_LOT
            comm = self._commission(pos.asset_type, close_qty)
            if pos.direction == "long":
                gross = close_qty * multiplier * close_price
                realized = close_qty * multiplier * (close_price - pos.entry_price) - comm
                self.cash += gross - comm
            else:  # short option
                gross = close_qty * multiplier * close_price
                realized = close_qty * multiplier * (pos.entry_price - close_price) - comm
                self.cash -= (gross + comm)

        self.realized_pnls.append(realized)

        trade = Trade(trade_id=self._new_id(), position_id=position_id,
                      symbol=pos.symbol, asset_type=pos.asset_type,
                      direction=pos.direction,
                      action="close" if close_qty >= pos.quantity else "partial_close",
                      quantity=close_qty, price=close_price,
                      timestamp=datetime.utcnow().isoformat(),
                      realized_pnl=realized, notes=notes)
        self.trades.append(trade)

        if close_qty >= pos.quantity:
            del self.positions[position_id]
        else:
            pos.quantity -= close_qty

        print(f"[CLOSE] {pos.symbol} {close_qty} @ ${close_price:.2f} | Realized P&L: ${realized:.2f}")
        self.save_state()
        return realized

    def update_prices(self, price_map: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for pos in self.positions.values():
            key = pos.symbol
            if key in price_map:
                pos.current_price = price_map[key]
            elif pos.underlying and pos.underlying in price_map:
                pass  # option pricing would need BS model

    def check_stops(self) -> List[str]:
        """Check stop losses and take profits. Returns list of closed position IDs."""
        closed = []
        for pid, pos in list(self.positions.items()):
            if pos.stop_loss and pos.direction == "long" and pos.current_price <= pos.stop_loss:
                print(f"[STOP LOSS] {pos.symbol} hit ${pos.stop_loss:.2f}")
                self.close_position(pid, pos.current_price, notes="stop_loss_triggered")
                closed.append(pid)
            elif pos.take_profit and pos.direction == "long" and pos.current_price >= pos.take_profit:
                print(f"[TAKE PROFIT] {pos.symbol} hit ${pos.take_profit:.2f}")
                self.close_position(pid, pos.current_price, notes="take_profit_triggered")
                closed.append(pid)
        return closed

    def kelly_size(self, win_prob: float, avg_win: float, avg_loss: float, nav: float) -> float:
        """Kelly criterion position sizing. Returns dollar amount to risk."""
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss  # win/loss ratio
        f = (b * win_prob - (1 - win_prob)) / b  # Kelly fraction
        f = max(0.0, min(f, 0.25))  # cap at 25% (half-Kelly conservative)
        return nav * f

    def get_positions(self) -> List[dict]:
        return [p.to_dict() for p in self.positions.values()]

    def daily_pnl(self) -> dict:
        nav = self.get_nav()
        if self.daily_nav_history:
            prev_nav = self.daily_nav_history[-1]["nav"]
            day_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        else:
            day_return = (nav - self.starting_capital) / self.starting_capital

        day_num = len(self.daily_nav_history) + 1
        target_nav = COMPOUND_TABLE.get(day_num, STARTING_CAPITAL)
        milestone = MILESTONES.get(day_num)

        return {
            "date": date.today().isoformat(),
            "day_number": day_num,
            "nav": round(nav, 2),
            "cash": round(self.cash, 2),
            "daily_return_pct": round(day_return * 100, 2),
            "target_nav": round(target_nav, 2),
            "vs_target_pct": round((nav - target_nav) / target_nav * 100, 2) if target_nav else 0,
            "milestone": milestone,
            "num_positions": len(self.positions),
            "total_realized_pnl": round(sum(self.realized_pnls), 2),
            "unrealized_pnl": round(sum(p.unrealized_pnl for p in self.positions.values()), 2),
        }

    def snapshot(self) -> dict:
        """Full portfolio snapshot."""
        nav = self.get_nav()
        metrics = PortfolioMetrics(self.daily_returns)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "nav": round(nav, 2),
            "cash": round(self.cash, 2),
            "starting_capital": self.starting_capital,
            "day_number": self.day_number,
            "positions": self.get_positions(),
            "recent_trades": [t.to_dict() for t in self.trades[-20:]],
            "metrics": metrics.summary(self.realized_pnls),
            "compound_targets": {str(d): round(v, 2) for d, v in COMPOUND_TABLE.items() if d % 5 == 0},
            "daily_pnl": self.daily_pnl(),
            "guardrails_active": self.guardrails_active,
        }

    def get_summary(self) -> dict:
        """Alias for snapshot() — returns full portfolio state dict."""
        snap = self.snapshot()
        # Normalise key names used by run scripts
        snap.setdefault("total_equity", snap.get("nav", 0))
        snap.setdefault("unrealized_pnl", snap.get("daily_pnl", {}).get("unrealized_pnl", 0))
        snap.setdefault("n_positions", len(snap.get("positions", [])))
        return snap

    def end_of_day(self) -> None:
        """Call at end of each trading day to record NAV and returns."""
        nav = self.get_nav()
        if self.daily_nav_history:
            prev = self.daily_nav_history[-1]["nav"]
            ret = (nav - prev) / prev if prev > 0 else 0.0
        else:
            ret = (nav - self.starting_capital) / self.starting_capital
        self.daily_returns.append(ret)
        self.day_number += 1
        self.daily_nav_history.append({
            "date": date.today().isoformat(),
            "day": self.day_number,
            "nav": round(nav, 2),
            "return_pct": round(ret * 100, 2),
        })
        self.save_state()

    def save_state(self) -> None:
        state = {
            "cash": self.cash,
            "starting_capital": self.starting_capital,
            "day_number": self.day_number,
            "created_at": self.created_at,
            "guardrails_active": self.guardrails_active,
            "positions": {pid: p.to_dict() for pid, p in self.positions.items()},
            "trades": [t.to_dict() for t in self.trades],
            "daily_returns": self.daily_returns,
            "realized_pnls": self.realized_pnls,
            "daily_nav_history": self.daily_nav_history,
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> None:
        with open(self.state_file, "r") as f:
            state = json.load(f)
        self.cash = state.get("cash", STARTING_CAPITAL)
        self.starting_capital = state.get("starting_capital", STARTING_CAPITAL)
        self.day_number = state.get("day_number", 0)
        self.created_at = state.get("created_at", datetime.utcnow().isoformat())
        self.guardrails_active = state.get("guardrails_active", True)
        self.daily_returns = state.get("daily_returns", [])
        self.realized_pnls = state.get("realized_pnls", [])
        self.daily_nav_history = state.get("daily_nav_history", [])
        self.positions = {}
        for pid, pd in state.get("positions", {}).items():
            pos = Position(
                position_id=pd["position_id"], symbol=pd["symbol"],
                asset_type=pd["asset_type"], direction=pd["direction"],
                quantity=pd["quantity"], entry_price=pd["entry_price"],
                current_price=pd["current_price"], entry_time=pd["entry_time"],
                expiry=pd.get("expiry"), strike=pd.get("strike"),
                underlying=pd.get("underlying"),
                stop_loss=pd.get("stop_loss"), take_profit=pd.get("take_profit"),
                notes=pd.get("notes", "")
            )
            self.positions[pid] = pos
        self.trades = []
        for td in state.get("trades", []):
            trade = Trade(
                trade_id=td["trade_id"], position_id=td["position_id"],
                symbol=td["symbol"], asset_type=td["asset_type"],
                direction=td["direction"], action=td["action"],
                quantity=td["quantity"], price=td["price"],
                timestamp=td["timestamp"],
                realized_pnl=td.get("realized_pnl", 0.0),
                commission=td.get("commission", 0.0),
                notes=td.get("notes", "")
            )
            self.trades.append(trade)
        print(f"[Portfolio] Loaded state: NAV=${self.get_nav():.2f}, Day {self.day_number}")

    def print_compound_table(self, days: List[int] = None) -> None:
        """Print the $1K → $1M compound path."""
        if days is None:
            days = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]
        print("\n" + "="*50)
        print("  $1,000 → $1,000,000 Compound Path (7.24%/day)")
        print("="*50)
        print(f"  {'Day':>4}  {'Target NAV':>14}  {'Milestone':>12}")
        print("-"*50)
        for d in days:
            nav = COMPOUND_TABLE.get(d, 0)
            ms = MILESTONES.get(d, "")
            ms_str = f"  *** ${ms:,.0f}" if ms else ""
            print(f"  {d:>4}  ${nav:>13,.0f}{ms_str}")
        print("="*50 + "\n")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_portfolio(state_file: str = STATE_FILE) -> PaperPortfolio:
    """Get or create the singleton paper portfolio."""
    return PaperPortfolio(state_file=state_file)


if __name__ == "__main__":
    port = PaperPortfolio()
    port.print_compound_table()
    print("NAV:", port.get_nav())
    print("Snapshot:", json.dumps(port.snapshot(), indent=2))
