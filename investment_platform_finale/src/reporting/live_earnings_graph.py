#!/usr/bin/env python3
"""
Live Intraday Earnings Graph
============================
Reads today's hourly JSON alert logs and open/close session logs,
plots NAV and cumulative P&L throughout the trading day, and saves:
  - reports/earnings_graph_{date}.png  (static image)
  - reports/earnings_graph_{date}.html (interactive matplotlib JS embed)

Run standalone:
    python src/reporting/live_earnings_graph.py

Or import and call:
    from src.reporting.live_earnings_graph import generate
    generate()                   # today
    generate(date_str='2026-03-14')
"""
from __future__ import annotations

import json
import os
import sys
import glob
from datetime import date, datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load_snapshots(date_str: str) -> list[dict]:
    """
    Collect all NAV/P&L snapshots for a given date from:
      - logs/hourly_alerts_{date}_*.json  (intraday hourly snapshots)
      - logs/open_{date}.json             (open session summary)
      - logs/close_{date}.json            (close session summary)
    Returns list of dicts sorted by timestamp:
      {timestamp: datetime, nav: float, daily_pnl_usd: float, label: str}
    """
    snapshots: list[dict] = []

    # --- Hourly alert logs ---
    pattern = os.path.join(LOG_DIR, f"hourly_alerts_{date_str}_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                data = json.load(f)
            snap = data.get("snapshot", {})
            ts_raw = snap.get("timestamp") or data.get("timestamp")
            if not ts_raw:
                continue
            ts = datetime.fromisoformat(ts_raw)
            nav = float(snap.get("nav", 0) or 0)
            pnl = snap.get("daily_pnl", {})
            pnl_usd = float(pnl.get("daily_pnl_usd", pnl.get("pnl_usd", 0)) or 0)
            unrealized = float(snap.get("unrealized_pnl", 0) or 0)
            snapshots.append({
                "timestamp": ts,
                "nav": nav,
                "daily_pnl_usd": pnl_usd,
                "unrealized_pnl": unrealized,
                "num_positions": int(snap.get("num_positions", 0) or 0),
                "label": f"Hourly {ts.strftime('%H:%M')}",
            })
        except Exception:
            pass

    # --- Open/Close session JSON logs ---
    for session in ("open", "close"):
        path = os.path.join(LOG_DIR, f"{session}_{date_str}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            ts_raw = data.get("timestamp") or data.get("start_time") or data.get("end_time")
            if not ts_raw:
                continue
            ts = datetime.fromisoformat(ts_raw)
            nav = float(data.get("nav", data.get("portfolio", {}).get("nav", 0)) or 0)
            pnl_usd = float(
                data.get("daily_pnl_usd", data.get("daily_pnl", {}).get("daily_pnl_usd", 0)) or 0
            )
            snapshots.append({
                "timestamp": ts,
                "nav": nav,
                "daily_pnl_usd": pnl_usd,
                "unrealized_pnl": 0.0,
                "num_positions": 0,
                "label": session.upper(),
            })
        except Exception:
            pass

    snapshots.sort(key=lambda x: x["timestamp"])
    return snapshots


def _synthetic_day(date_str: str, start_nav: float = 1_000_000.0) -> list[dict]:
    """
    Generate plausible synthetic intraday data when no real logs exist.
    Used so the chart renders beautifully even on first run.
    """
    rng = np.random.default_rng(seed=int(date_str.replace("-", "")) % (2**32))
    base_dt = datetime.fromisoformat(f"{date_str}T13:30:00")  # 9:30 ET = 13:30 UTC
    n = 7  # 7 hourly ticks (9:30–15:30)
    returns = rng.normal(0.0008, 0.004, n).cumsum()
    navs = start_nav * (1 + returns)
    pnls = navs - start_nav
    result = []
    for i in range(n):
        ts = base_dt.replace(hour=base_dt.hour + i) if base_dt.hour + i < 24 else base_dt
        result.append({
            "timestamp": ts,
            "nav": round(float(navs[i]), 2),
            "daily_pnl_usd": round(float(pnls[i]), 2),
            "unrealized_pnl": round(float(pnls[i]) * 0.8, 2),
            "num_positions": rng.integers(3, 12),
            "label": f"Hourly {ts.strftime('%H:%M')}",
            "synthetic": True,
        })
    return result


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------

HEDGE_BLUE   = "#0047AB"
PROFIT_GREEN = "#00A878"
LOSS_RED     = "#E63946"
NEUTRAL_GRAY = "#6B7280"
BACKGROUND   = "#0D1117"
PANEL_BG     = "#161B22"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#E6EDF3"
ACCENT_GOLD  = "#D4A017"


def generate(date_str: Optional[str] = None, start_nav: float = 1_000_000.0) -> str:
    """
    Generate the earnings graph. Returns path to the saved PNG.

    Args:
        date_str:  ISO date string (default: today)
        start_nav: Starting NAV for synthetic data fallback
    """
    if date_str is None:
        date_str = date.today().isoformat()

    snapshots = _load_snapshots(date_str)
    synthetic = False
    if not snapshots:
        snapshots = _synthetic_day(date_str, start_nav)
        synthetic = True

    # Extract series
    times     = [s["timestamp"] for s in snapshots]
    navs      = [s["nav"] for s in snapshots]
    pnls      = [s["daily_pnl_usd"] for s in snapshots]
    positions = [s["num_positions"] for s in snapshots]

    final_nav   = navs[-1]  if navs else start_nav
    final_pnl   = pnls[-1]  if pnls else 0.0
    pnl_pct     = (final_pnl / start_nav) * 100 if start_nav else 0.0
    pnl_color   = PROFIT_GREEN if final_pnl >= 0 else LOSS_RED
    max_nav     = max(navs) if navs else start_nav
    min_nav     = min(navs) if navs else start_nav

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.5, 1], hspace=0.08)

    ax_nav  = fig.add_subplot(gs[0])
    ax_pnl  = fig.add_subplot(gs[1], sharex=ax_nav)
    ax_pos  = fig.add_subplot(gs[2], sharex=ax_nav)

    for ax in (ax_nav, ax_pnl, ax_pos):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    # ── NAV Panel ───────────────────────────────────────────────────────────
    ax_nav.plot(times, navs, color=HEDGE_BLUE, linewidth=2.5, zorder=3)
    ax_nav.fill_between(times, navs, min_nav * 0.9995,
                        color=HEDGE_BLUE, alpha=0.12, zorder=2)

    # Annotate peak
    peak_idx = navs.index(max_nav)
    ax_nav.annotate(
        f"Peak ${max_nav:,.0f}",
        xy=(times[peak_idx], max_nav),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8, color=ACCENT_GOLD,
        arrowprops=dict(arrowstyle="->", color=ACCENT_GOLD, lw=0.8),
    )

    # Start NAV reference line
    ax_nav.axhline(start_nav, color=NEUTRAL_GRAY, linewidth=0.8,
                   linestyle="--", alpha=0.5, zorder=1, label=f"Start ${start_nav:,.0f}")

    ax_nav.set_ylabel("Portfolio NAV ($)", color=TEXT_COLOR, fontsize=10)
    ax_nav.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.3f}M" if x >= 1e6 else f"${x:,.0f}"))
    ax_nav.tick_params(axis="x", labelbottom=False)
    ax_nav.set_title(
        f"AI HEDGE FUND — LIVE EARNINGS  ·  {date_str}"
        + ("  [SYNTHETIC DATA]" if synthetic else ""),
        color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12,
    )

    # Current NAV big badge (top right)
    ax_nav.text(0.98, 0.95,
                f"NAV  ${final_nav:,.2f}\n{pnl_pct:+.2f}% today",
                transform=ax_nav.transAxes, fontsize=11, fontweight="bold",
                color=pnl_color, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                          edgecolor=pnl_color, linewidth=1.5))

    # ── P&L Panel ───────────────────────────────────────────────────────────
    colors_pnl = [PROFIT_GREEN if p >= 0 else LOSS_RED for p in pnls]
    ax_pnl.bar(times, pnls, color=colors_pnl, width=0.02, alpha=0.85, zorder=3)
    ax_pnl.axhline(0, color=NEUTRAL_GRAY, linewidth=0.8, linestyle="-", alpha=0.5)
    ax_pnl.set_ylabel("Daily P&L ($)", color=TEXT_COLOR, fontsize=9)
    ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:+,.0f}"))
    ax_pnl.tick_params(axis="x", labelbottom=False)

    # P&L running line overlay
    ax_pnl.plot(times, pnls, color=ACCENT_GOLD, linewidth=1.2,
                linestyle="--", alpha=0.6, zorder=4)

    # ── Positions Panel ─────────────────────────────────────────────────────
    ax_pos.step(times, positions, where="post", color=NEUTRAL_GRAY,
                linewidth=1.5, zorder=3)
    ax_pos.fill_between(times, positions, step="post",
                        color=NEUTRAL_GRAY, alpha=0.15)
    ax_pos.set_ylabel("Positions", color=TEXT_COLOR, fontsize=9)
    ax_pos.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_pos.set_xlabel("Time (UTC)", color=TEXT_COLOR, fontsize=9)

    # x-axis format
    ax_pos.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_pos.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax_pos.xaxis.get_majorticklabels(), rotation=30, ha="right",
             color=TEXT_COLOR, fontsize=8)

    # Legend bottom
    legend_elements = [
        mpatches.Patch(facecolor=HEDGE_BLUE,   label=f"NAV  ${final_nav:,.0f}"),
        mpatches.Patch(facecolor=pnl_color,    label=f"P&L  ${final_pnl:+,.0f}  ({pnl_pct:+.2f}%)"),
        mpatches.Patch(facecolor=NEUTRAL_GRAY, label=f"Positions: {positions[-1] if positions else 0}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = os.path.join(REPORT_DIR, f"earnings_graph_{date_str}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND, edgecolor="none")
    plt.close(fig)

    # ── Minimal HTML wrapper (self-contained, embeds PNG as base64) ─────────
    import base64
    with open(out_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Hedge Fund — Earnings {date_str}</title>
<style>
  body {{background:#0D1117;color:#E6EDF3;font-family:monospace;margin:0;padding:20px;}}
  h1  {{color:#D4A017;font-size:1.2em;border-bottom:1px solid #21262D;padding-bottom:8px;}}
  .badge {{display:inline-block;padding:6px 14px;border-radius:6px;margin:4px;font-size:0.9em;}}
  .green {{background:#00A878;color:#000;}}
  .red   {{background:#E63946;color:#fff;}}
  .blue  {{background:#0047AB;color:#fff;}}
  img   {{max-width:100%;border-radius:8px;margin-top:16px;}}
  .meta {{color:#6B7280;font-size:0.8em;margin-top:8px;}}
  .auto-refresh {{color:#6B7280;font-size:0.75em;}}
</style>
<script>
  // Auto-refresh every 5 minutes during market hours
  function autoRefresh() {{
    var now = new Date();
    var h = now.getUTCHours(), m = now.getUTCMinutes();
    if ((h > 13 || (h === 13 && m >= 30)) && h < 21) {{
      setTimeout(function() {{ location.reload(); }}, 5 * 60 * 1000);
    }}
  }}
  window.onload = autoRefresh;
</script>
</head>
<body>
<h1>AI HEDGE FUND — LIVE EARNINGS DASHBOARD</h1>
<span class="badge blue">Date: {date_str}</span>
<span class="badge {'green' if final_pnl >= 0 else 'red'}">P&L: ${final_pnl:+,.2f} ({pnl_pct:+.2f}%)</span>
<span class="badge blue">NAV: ${final_nav:,.2f}</span>
{'<span class="badge" style="background:#6B7280">⚠ SYNTHETIC DATA</span>' if synthetic else ''}
<p class="auto-refresh">Auto-refreshes every 5 min during market hours (13:30–21:00 UTC)</p>
<img src="data:image/png;base64,{img_b64}" alt="Earnings Graph"/>
<p class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
 · Source: logs/hourly_alerts_{date_str}_*.json</p>
</body>
</html>"""

    html_path = os.path.join(REPORT_DIR, f"earnings_graph_{date_str}.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"[earnings_graph] Saved: {out_path}")
    print(f"[earnings_graph] Saved: {html_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate live earnings graph")
    p.add_argument("--date", default=None, help="ISO date (default: today)")
    p.add_argument("--nav",  type=float, default=1_000_000.0,
                   help="Starting NAV for synthetic fallback (default: 1000000)")
    args = p.parse_args()
    path = generate(date_str=args.date, start_nav=args.nav)
    print(f"Chart saved: {path}")
