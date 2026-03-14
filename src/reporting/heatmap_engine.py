"""
Securities Universe Heatmap Generator
Color-coded ASCII heatmap by sector and sub-industry.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

REPORT_DIR = "/home/user/ai-hedgefund/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
COLORS = {
    "dark_green":  "\033[38;5;22m",
    "green":       "\033[32m",
    "light_green": "\033[92m",
    "yellow":      "\033[33m",
    "orange":      "\033[38;5;208m",
    "red":         "\033[31m",
    "dark_red":    "\033[38;5;52m",
    "white":       "\033[37m",
    "bold":        "\033[1m",
    "reset":       "\033[0m",
    "bg_dark_green": "\033[42m",
    "bg_red":      "\033[41m",
}

# Thresholds: return → color
COLOR_THRESHOLDS = [
    (0.03,  "dark_green"),   # > +3%
    (0.01,  "green"),        # +1% to +3%
    (0.00,  "light_green"),  # 0% to +1%
    (-0.01, "yellow"),       # -1% to 0%
    (-0.03, "orange"),       # -3% to -1%
    (float("-inf"), "red"),  # < -3%
]


def get_color(ret: float) -> str:
    for threshold, color in COLOR_THRESHOLDS:
        if ret >= threshold:
            return COLORS.get(color, "")
    return COLORS["dark_red"]


def color_text(text: str, ret: float) -> str:
    color = get_color(ret)
    return f"{color}{text}{COLORS['reset']}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SecurityCell:
    symbol: str
    name: str
    return_1d: float
    market_cap_bn: float
    volume_rel: float  # relative to 20D avg
    sector: str
    sub_industry: str

    def cell_text(self, width: int = 10) -> str:
        ret_str = f"{self.return_1d:+.1%}"
        sym_str = self.symbol[:width-6].ljust(width-6)
        return f"{sym_str}{ret_str}"

    def colored_cell(self, width: int = 10) -> str:
        return color_text(self.cell_text(width), self.return_1d)

    def ascii_bar(self) -> str:
        """ASCII bar representing return magnitude."""
        pct = self.return_1d * 100
        n = min(10, int(abs(pct) * 3))
        if pct >= 0:
            return "+" + "█" * n
        else:
            return "-" + "█" * n


@dataclass
class SectorHeatmap:
    sector: str
    etf: str
    sector_return: float
    securities: List[SecurityCell] = field(default_factory=list)

    @property
    def avg_return(self) -> float:
        if not self.securities:
            return self.sector_return
        return sum(s.return_1d for s in self.securities) / len(self.securities)

    @property
    def num_advancing(self) -> int:
        return sum(1 for s in self.securities if s.return_1d > 0)

    @property
    def num_declining(self) -> int:
        return sum(1 for s in self.securities if s.return_1d < 0)


# ---------------------------------------------------------------------------
# Sample universe data (would be replaced by live_data.py in production)
# ---------------------------------------------------------------------------

SAMPLE_UNIVERSE: Dict[str, Dict] = {
    "InformationTechnology": {
        "etf": "XLK", "return_1d": 0.008,
        "securities": [
            {"symbol": "AAPL", "name": "Apple Inc", "return_1d": 0.006, "market_cap_bn": 2800, "volume_rel": 1.2, "sub_industry": "Technology Hardware"},
            {"symbol": "MSFT", "name": "Microsoft", "return_1d": 0.012, "market_cap_bn": 3100, "volume_rel": 1.4, "sub_industry": "Systems Software"},
            {"symbol": "NVDA", "name": "NVIDIA", "return_1d": 0.025, "market_cap_bn": 2200, "volume_rel": 2.1, "sub_industry": "Semiconductors"},
            {"symbol": "AVGO", "name": "Broadcom", "return_1d": 0.009, "market_cap_bn": 750, "volume_rel": 1.1, "sub_industry": "Semiconductors"},
            {"symbol": "AMD",  "name": "AMD", "return_1d": 0.018, "market_cap_bn": 300, "volume_rel": 1.8, "sub_industry": "Semiconductors"},
            {"symbol": "ORCL", "name": "Oracle", "return_1d": 0.003, "market_cap_bn": 400, "volume_rel": 0.9, "sub_industry": "Systems Software"},
            {"symbol": "CRM",  "name": "Salesforce", "return_1d": -0.004, "market_cap_bn": 280, "volume_rel": 1.0, "sub_industry": "Application Software"},
            {"symbol": "AMAT", "name": "Applied Materials", "return_1d": 0.014, "market_cap_bn": 180, "volume_rel": 1.3, "sub_industry": "Semiconductor Equipment"},
        ]
    },
    "HealthCare": {
        "etf": "XLV", "return_1d": -0.002,
        "securities": [
            {"symbol": "LLY",  "name": "Eli Lilly", "return_1d": 0.005, "market_cap_bn": 750, "volume_rel": 1.1, "sub_industry": "Pharmaceuticals"},
            {"symbol": "UNH",  "name": "UnitedHealth", "return_1d": -0.008, "market_cap_bn": 480, "volume_rel": 0.9, "sub_industry": "Managed Health Care"},
            {"symbol": "JNJ",  "name": "J&J", "return_1d": -0.003, "market_cap_bn": 390, "volume_rel": 0.8, "sub_industry": "Pharmaceuticals"},
            {"symbol": "ABBV", "name": "AbbVie", "return_1d": 0.004, "market_cap_bn": 310, "volume_rel": 1.0, "sub_industry": "Pharmaceuticals"},
            {"symbol": "MRK",  "name": "Merck", "return_1d": -0.005, "market_cap_bn": 290, "volume_rel": 0.9, "sub_industry": "Pharmaceuticals"},
        ]
    },
    "Financials": {
        "etf": "XLF", "return_1d": 0.004,
        "securities": [
            {"symbol": "BRK.B","name": "Berkshire B", "return_1d": 0.002, "market_cap_bn": 860, "volume_rel": 0.8, "sub_industry": "Multi-line Insurance"},
            {"symbol": "JPM",  "name": "JPMorgan", "return_1d": 0.007, "market_cap_bn": 590, "volume_rel": 1.2, "sub_industry": "Diversified Banks"},
            {"symbol": "V",    "name": "Visa", "return_1d": 0.004, "market_cap_bn": 530, "volume_rel": 1.0, "sub_industry": "Transaction & Payment"},
            {"symbol": "MA",   "name": "Mastercard", "return_1d": 0.003, "market_cap_bn": 450, "volume_rel": 1.0, "sub_industry": "Transaction & Payment"},
            {"symbol": "GS",   "name": "Goldman Sachs", "return_1d": 0.009, "market_cap_bn": 150, "volume_rel": 1.3, "sub_industry": "Investment Banking"},
        ]
    },
    "Energy": {
        "etf": "XLE", "return_1d": -0.005,
        "securities": [
            {"symbol": "XOM",  "name": "ExxonMobil", "return_1d": -0.004, "market_cap_bn": 490, "volume_rel": 0.9, "sub_industry": "Integrated Oil"},
            {"symbol": "CVX",  "name": "Chevron", "return_1d": -0.006, "market_cap_bn": 290, "volume_rel": 0.8, "sub_industry": "Integrated Oil"},
            {"symbol": "COP",  "name": "ConocoPhillips", "return_1d": -0.007, "market_cap_bn": 140, "volume_rel": 0.9, "sub_industry": "E&P"},
            {"symbol": "SLB",  "name": "Schlumberger", "return_1d": -0.008, "market_cap_bn": 80, "volume_rel": 1.1, "sub_industry": "Oil Field Services"},
        ]
    },
    "ConsumerDiscretionary": {
        "etf": "XLY", "return_1d": 0.007,
        "securities": [
            {"symbol": "AMZN", "name": "Amazon", "return_1d": 0.009, "market_cap_bn": 1900, "volume_rel": 1.3, "sub_industry": "Internet Retail"},
            {"symbol": "TSLA", "name": "Tesla", "return_1d": 0.015, "market_cap_bn": 650, "volume_rel": 2.5, "sub_industry": "Automobile Manufacturers"},
            {"symbol": "HD",   "name": "Home Depot", "return_1d": 0.003, "market_cap_bn": 360, "volume_rel": 0.9, "sub_industry": "Home Improvement Retail"},
            {"symbol": "MCD",  "name": "McDonald's", "return_1d": -0.002, "market_cap_bn": 210, "volume_rel": 0.8, "sub_industry": "Restaurants"},
        ]
    },
    "CommunicationServices": {
        "etf": "XLC", "return_1d": 0.006,
        "securities": [
            {"symbol": "META", "name": "Meta Platforms", "return_1d": 0.011, "market_cap_bn": 1300, "volume_rel": 1.6, "sub_industry": "Interactive Media"},
            {"symbol": "GOOGL","name": "Alphabet", "return_1d": 0.007, "market_cap_bn": 2000, "volume_rel": 1.2, "sub_industry": "Interactive Media"},
            {"symbol": "NFLX", "name": "Netflix", "return_1d": 0.004, "market_cap_bn": 280, "volume_rel": 1.1, "sub_industry": "Movies & Entertainment"},
            {"symbol": "DIS",  "name": "Disney", "return_1d": -0.005, "market_cap_bn": 195, "volume_rel": 0.9, "sub_industry": "Movies & Entertainment"},
        ]
    },
    "Industrials": {
        "etf": "XLI", "return_1d": 0.003,
        "securities": [
            {"symbol": "GE",   "name": "GE Aerospace", "return_1d": 0.006, "market_cap_bn": 180, "volume_rel": 1.1, "sub_industry": "Aerospace & Defense"},
            {"symbol": "CAT",  "name": "Caterpillar", "return_1d": 0.004, "market_cap_bn": 190, "volume_rel": 0.9, "sub_industry": "Construction Machinery"},
            {"symbol": "RTX",  "name": "RTX Corp", "return_1d": 0.002, "market_cap_bn": 145, "volume_rel": 0.8, "sub_industry": "Aerospace & Defense"},
            {"symbol": "UNP",  "name": "Union Pacific", "return_1d": -0.001, "market_cap_bn": 140, "volume_rel": 0.7, "sub_industry": "Railroads"},
        ]
    },
    "ConsumerStaples": {
        "etf": "XLP", "return_1d": -0.001,
        "securities": [
            {"symbol": "PG",   "name": "Procter & Gamble", "return_1d": -0.002, "market_cap_bn": 380, "volume_rel": 0.8, "sub_industry": "Household Products"},
            {"symbol": "KO",   "name": "Coca-Cola", "return_1d": 0.001, "market_cap_bn": 270, "volume_rel": 0.7, "sub_industry": "Soft Drinks"},
            {"symbol": "WMT",  "name": "Walmart", "return_1d": 0.003, "market_cap_bn": 500, "volume_rel": 1.0, "sub_industry": "Hypermarkets"},
            {"symbol": "COST", "name": "Costco", "return_1d": 0.002, "market_cap_bn": 360, "volume_rel": 0.9, "sub_industry": "Hypermarkets"},
        ]
    },
    "Materials": {
        "etf": "XLB", "return_1d": 0.002,
        "securities": [
            {"symbol": "LIN",  "name": "Linde", "return_1d": 0.003, "market_cap_bn": 220, "volume_rel": 0.8, "sub_industry": "Industrial Gases"},
            {"symbol": "APD",  "name": "Air Products", "return_1d": 0.001, "market_cap_bn": 55, "volume_rel": 0.7, "sub_industry": "Industrial Gases"},
            {"symbol": "FCX",  "name": "Freeport-McMoRan", "return_1d": 0.005, "market_cap_bn": 65, "volume_rel": 1.2, "sub_industry": "Copper"},
        ]
    },
    "Utilities": {
        "etf": "XLU", "return_1d": -0.003,
        "securities": [
            {"symbol": "NEE",  "name": "NextEra Energy", "return_1d": -0.004, "market_cap_bn": 150, "volume_rel": 0.9, "sub_industry": "Electric Utilities"},
            {"symbol": "SO",   "name": "Southern Company", "return_1d": -0.002, "market_cap_bn": 80, "volume_rel": 0.7, "sub_industry": "Electric Utilities"},
            {"symbol": "DUK",  "name": "Duke Energy", "return_1d": -0.003, "market_cap_bn": 75, "volume_rel": 0.8, "sub_industry": "Electric Utilities"},
        ]
    },
    "RealEstate": {
        "etf": "XLRE", "return_1d": -0.004,
        "securities": [
            {"symbol": "PLD",  "name": "Prologis", "return_1d": -0.003, "market_cap_bn": 115, "volume_rel": 0.9, "sub_industry": "Industrial REITs"},
            {"symbol": "EQIX", "name": "Equinix", "return_1d": -0.006, "market_cap_bn": 80, "volume_rel": 0.8, "sub_industry": "Data Center REITs"},
            {"symbol": "AMT",  "name": "American Tower", "return_1d": -0.005, "market_cap_bn": 90, "volume_rel": 0.9, "sub_industry": "Specialized REITs"},
        ]
    },
}


# ---------------------------------------------------------------------------
# Heatmap Engine
# ---------------------------------------------------------------------------

class HeatmapEngine:
    """Generate ASCII/text heatmaps for the securities universe."""

    def __init__(self, universe_data: Dict[str, Dict] = None):
        self.universe = universe_data or SAMPLE_UNIVERSE
        self.heatmaps: Dict[str, SectorHeatmap] = {}
        self._build_heatmaps()

    def _build_heatmaps(self) -> None:
        for sector, data in self.universe.items():
            cells = []
            for sec in data.get("securities", []):
                cells.append(SecurityCell(
                    symbol=sec["symbol"],
                    name=sec["name"],
                    return_1d=sec["return_1d"],
                    market_cap_bn=sec["market_cap_bn"],
                    volume_rel=sec["volume_rel"],
                    sector=sector,
                    sub_industry=sec.get("sub_industry", ""),
                ))
            self.heatmaps[sector] = SectorHeatmap(
                sector=sector,
                etf=data.get("etf", ""),
                sector_return=data.get("return_1d", 0),
                securities=sorted(cells, key=lambda x: x.return_1d, reverse=True),
            )

    def update_returns(self, price_updates: Dict[str, float]) -> None:
        """Update return data from live prices."""
        for sector, hm in self.heatmaps.items():
            for sec in hm.securities:
                if sec.symbol in price_updates:
                    sec.return_1d = price_updates[sec.symbol]
            # Update sector return
            if hm.securities:
                hm.sector_return = hm.avg_return

    def _color_legend(self) -> str:
        return (
            f"\n  LEGEND: "
            f"{COLORS['dark_green']}██{COLORS['reset']} >+3%  "
            f"{COLORS['green']}██{COLORS['reset']} +1-3%  "
            f"{COLORS['light_green']}██{COLORS['reset']} 0-1%  "
            f"{COLORS['yellow']}██{COLORS['reset']} -1-0%  "
            f"{COLORS['orange']}██{COLORS['reset']} -1-3%  "
            f"{COLORS['red']}██{COLORS['reset']} <-3%\n"
        )

    def generate_sector_row(self, hm: SectorHeatmap, cell_width: int = 12) -> str:
        """Generate one row of colored cells for a sector."""
        colored_cells = []
        for sec in hm.securities[:8]:  # max 8 per row
            cell = f"{sec.symbol[:5]:<5}{sec.return_1d:+.1%}"
            cell = cell.ljust(cell_width)
            colored_cells.append(color_text(cell, sec.return_1d))
        return " ".join(colored_cells)

    def generate_heatmap(self, use_color: bool = True) -> str:
        """Generate full ASCII heatmap sorted by sector performance."""
        sorted_sectors = sorted(
            self.heatmaps.values(),
            key=lambda h: h.sector_return,
            reverse=True
        )
        lines = [
            "",
            "=" * 80,
            f"  MARKET HEATMAP — {date.today().isoformat()} | {datetime.utcnow().strftime('%H:%M')} UTC",
            "=" * 80,
        ]
        if use_color:
            lines.append(self._color_legend())

        for hm in sorted_sectors:
            ret = hm.sector_return
            adv = hm.num_advancing
            dec = hm.num_declining
            total = len(hm.securities)

            sector_header = (
                f"  {hm.sector:<25} {hm.etf:<6} "
                f"{ret:+.2%}  ({adv}↑ {dec}↓ of {total})"
            )
            if use_color:
                sector_header = color_text(sector_header, ret)
            lines.append(sector_header)

            # Securities row
            if use_color:
                sec_row = "    " + self.generate_sector_row(hm)
            else:
                sec_cells = [f"{s.symbol:<5}{s.return_1d:+.1%}".ljust(12)
                             for s in hm.securities[:8]]
                sec_row = "    " + " ".join(sec_cells)
            lines.append(sec_row)
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def generate_sub_industry_heatmap(self, sector: str) -> str:
        """Detailed heatmap for a single sector with sub-industry breakdown."""
        if sector not in self.heatmaps:
            return f"Sector '{sector}' not found"

        hm = self.heatmaps[sector]
        lines = [
            "",
            "=" * 70,
            f"  SECTOR DEEP DIVE: {sector} ({hm.etf}) {hm.sector_return:+.2%}",
            "=" * 70,
        ]

        # Group by sub-industry
        sub_industries: Dict[str, List[SecurityCell]] = {}
        for sec in hm.securities:
            si = sec.sub_industry or "Other"
            sub_industries.setdefault(si, []).append(sec)

        for si, secs in sub_industries.items():
            avg_ret = sum(s.return_1d for s in secs) / len(secs)
            si_header = color_text(f"\n  [{si}]  avg: {avg_ret:+.2%}", avg_ret)
            lines.append(si_header)
            for sec in sorted(secs, key=lambda x: x.return_1d, reverse=True):
                vol_flag = " **HIGH VOL**" if sec.volume_rel > 1.5 else ""
                row = f"    {sec.symbol:<7} {sec.return_1d:+.2%}  cap:${sec.market_cap_bn:.0f}B  vol:{sec.volume_rel:.1f}x{vol_flag}"
                lines.append(color_text(row, sec.return_1d))

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def generate_ascii_grid(self, cols: int = 4) -> str:
        """Plain ASCII grid (no color) for file output."""
        sorted_secs = sorted(
            [sec for hm in self.heatmaps.values() for sec in hm.securities],
            key=lambda x: x.return_1d,
            reverse=True
        )
        lines = [
            f"MARKET HEATMAP (PLAIN) — {date.today().isoformat()}",
            "=" * 80,
            f"{'Symbol':<8} {'Return':>8} {'Sector':<25} {'Sub-Industry':<25} {'Cap':>8}",
            "-" * 80,
        ]
        for sec in sorted_secs:
            ret_bar = sec.ascii_bar()
            lines.append(
                f"{sec.symbol:<8} {sec.return_1d:>+7.2%}  "
                f"{sec.sector:<25} {sec.sub_industry:<25} "
                f"${sec.market_cap_bn:>6.0f}B  {ret_bar}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def get_movers(self, top_n: int = 5) -> dict:
        """Return top gainers and losers."""
        all_secs = [sec for hm in self.heatmaps.values() for sec in hm.securities]
        sorted_secs = sorted(all_secs, key=lambda x: x.return_1d, reverse=True)
        return {
            "top_gainers": [
                {"symbol": s.symbol, "return_pct": round(s.return_1d * 100, 2),
                 "sector": s.sector, "volume_rel": s.volume_rel}
                for s in sorted_secs[:top_n]
            ],
            "top_losers": [
                {"symbol": s.symbol, "return_pct": round(s.return_1d * 100, 2),
                 "sector": s.sector, "volume_rel": s.volume_rel}
                for s in sorted_secs[-top_n:]
            ],
            "high_volume_movers": [
                {"symbol": s.symbol, "return_pct": round(s.return_1d * 100, 2),
                 "volume_rel": s.volume_rel, "sector": s.sector}
                for s in sorted(all_secs, key=lambda x: x.volume_rel, reverse=True)[:top_n]
            ],
        }

    def save_heatmap(self, use_color: bool = False) -> str:
        """Save plain ASCII heatmap to reports directory."""
        content = self.generate_ascii_grid()
        filename = f"heatmap_{date.today().isoformat()}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"[Heatmap] Saved: {filepath}")
        return filepath

    def save_json(self) -> str:
        """Save heatmap data as JSON."""
        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "date": date.today().isoformat(),
            "sectors": {
                sector: {
                    "etf": hm.etf,
                    "sector_return_pct": round(hm.sector_return * 100, 2),
                    "advancing": hm.num_advancing,
                    "declining": hm.num_declining,
                    "securities": [
                        {
                            "symbol": s.symbol,
                            "return_pct": round(s.return_1d * 100, 2),
                            "market_cap_bn": s.market_cap_bn,
                            "volume_rel": round(s.volume_rel, 2),
                            "sub_industry": s.sub_industry,
                        }
                        for s in hm.securities
                    ],
                }
                for sector, hm in self.heatmaps.items()
            },
            "movers": self.get_movers(),
        }
        filename = f"heatmap_{date.today().isoformat()}.json"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Heatmap] JSON saved: {filepath}")
        return filepath

    def print_heatmap(self, use_color: bool = True) -> None:
        print(self.generate_heatmap(use_color=use_color))


if __name__ == "__main__":
    engine = HeatmapEngine()
    engine.print_heatmap(use_color=True)
    engine.save_heatmap()
    engine.save_json()
    movers = engine.get_movers()
    print("\nTop Gainers:")
    for m in movers["top_gainers"]:
        print(f"  {m['symbol']}: {m['return_pct']:+.2f}%")
