"""
Securities Universe Heatmap Generator
======================================
Full GICS universe heatmap — ASCII color-coded by sector and sub-industry.

ASCII block character legend:
  ████  dark_green  > +3%
  ███   green       +1% to +3%
  ██    light_green  0% to +1%
  ▒     yellow      -1% to  0%
  ░     orange      -3% to -1%
  ▓     red         < -3%

Class HeatmapEngine:
  generate_heatmap(prices: dict = None) -> dict
  format_ascii(data: dict) -> str
  save(session: str = 'CLOSE') -> str  (returns file path)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_SRC, os.path.join(_SRC, 'data'), os.path.join(_SRC, 'portfolio')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = "/home/user/ai-hedgefund/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ==============================================================================
# ASCII block legend
# ==============================================================================

def _ascii_bar(ret: float) -> str:
    """
    Return ASCII block chars representing the return level.
      ████  > +3%
      ███   +1–3%
      ██    0–1%
      ▒     -1–0%
      ░     -3–-1%
      ▓     < -3%
    """
    if   ret >=  0.03: return '████'
    elif ret >=  0.01: return '███ '
    elif ret >=  0.00: return '██  '
    elif ret >= -0.01: return '▒   '
    elif ret >= -0.03: return '░   '
    else:              return '▓   '


def _return_str(ret: float) -> str:
    return f"{ret:+.1%}"


# ==============================================================================
# GICS universe — 11 sectors, key tickers per sub-industry
# ==============================================================================

FULL_UNIVERSE: Dict[str, Dict] = {
    'InformationTechnology': {
        'etf': 'XLK', 'name': 'Information Technology', 'return_1d': 0.0184,
        'sub_industries': {
            'Semiconductors':      [
                ('NVDA','NVIDIA',             0.032), ('AMD','AMD',               0.021),
                ('INTC','Intel',             -0.003), ('QCOM','Qualcomm',          0.011),
                ('AVGO','Broadcom',           0.009), ('MU',  'Micron',            0.014),
                ('AMAT','Applied Materials',  0.013), ('KLAC','KLA Corp',          0.008),
            ],
            'Systems Software':    [
                ('MSFT','Microsoft',          0.014), ('ORCL','Oracle',           -0.001),
                ('PANW','Palo Alto Nets',     0.019), ('FTNT','Fortinet',          0.007),
            ],
            'Application Software':[
                ('CRM', 'Salesforce',         0.028), ('NOW', 'ServiceNow',        0.017),
                ('ADBE','Adobe',              0.011), ('WDAY','Workday',            0.006),
                ('INTU','Intuit',             0.009),
            ],
            'Technology Hardware':  [
                ('AAPL','Apple',              0.006), ('HPQ', 'HP Inc',           -0.012),
                ('DELL','Dell Technologies', -0.004),
            ],
        }
    },
    'HealthCare': {
        'etf': 'XLV', 'name': 'Health Care', 'return_1d': -0.0022,
        'sub_industries': {
            'Pharmaceuticals':     [
                ('LLY', 'Eli Lilly',          0.005), ('JNJ', 'J&J',              -0.003),
                ('ABBV','AbbVie',             0.004), ('MRK', 'Merck',             -0.005),
                ('PFE', 'Pfizer',            -0.008),
            ],
            'Managed Health Care':  [
                ('UNH', 'UnitedHealth',      -0.008), ('CVS', 'CVS Health',        -0.002),
                ('CI',  'Cigna',              0.001),
            ],
            'Biotechnology':        [
                ('AMGN','Amgen',              0.003), ('GILD','Gilead',            -0.002),
                ('REGN','Regeneron',          0.007), ('MRNA','Moderna',            0.015),
            ],
            'Medical Devices':      [
                ('MDT', 'Medtronic',         -0.001), ('ABT', 'Abbott Labs',        0.002),
                ('SYK', 'Stryker',            0.004),
            ],
        }
    },
    'Financials': {
        'etf': 'XLF', 'name': 'Financials', 'return_1d': 0.0041,
        'sub_industries': {
            'Diversified Banks':    [
                ('JPM', 'JPMorgan',           0.007), ('BAC', 'Bank of America',   0.005),
                ('WFC', 'Wells Fargo',        0.004), ('C',   'Citigroup',          0.006),
            ],
            'Investment Banking':   [
                ('GS',  'Goldman Sachs',      0.009), ('MS',  'Morgan Stanley',     0.008),
                ('BX',  'Blackstone',         0.012),
            ],
            'Transaction & Payment':[
                ('V',   'Visa',               0.004), ('MA',  'Mastercard',         0.003),
                ('PYPL','PayPal',            -0.006), ('AXP', 'Amex',              0.005),
            ],
            'Insurance':            [
                ('BRK.B','Berkshire B',       0.002), ('MET', 'MetLife',            0.001),
                ('PGR', 'Progressive',        0.007),
            ],
        }
    },
    'ConsumerDiscretionary': {
        'etf': 'XLY', 'name': 'Consumer Discretionary', 'return_1d': 0.0068,
        'sub_industries': {
            'Internet & Direct Retail': [
                ('AMZN','Amazon',             0.009), ('EBAY','eBay',               0.003),
                ('ETSY','Etsy',              -0.005),
            ],
            'Automobile Manufacturers': [
                ('TSLA','Tesla',              0.015), ('GM',  'General Motors',      0.003),
                ('F',   'Ford',               0.001), ('RIVN','Rivian',             -0.008),
            ],
            'Restaurants':          [
                ('MCD', "McDonald's",        -0.002), ('SBUX','Starbucks',          -0.004),
                ('CMG', 'Chipotle',           0.005),
            ],
            'Specialty Retail':     [
                ('HD',  'Home Depot',         0.003), ('LOW', "Lowe's",              0.002),
                ('BKNG','Booking Holdings',   0.007),
            ],
        }
    },
    'CommunicationServices': {
        'etf': 'XLC', 'name': 'Communication Services', 'return_1d': 0.0058,
        'sub_industries': {
            'Interactive Media':    [
                ('META','Meta Platforms',     0.011), ('GOOGL','Alphabet A',         0.007),
                ('SNAP','Snap Inc',           0.015),
            ],
            'Entertainment':        [
                ('NFLX','Netflix',            0.004), ('DIS', 'Disney',             -0.005),
                ('WBD', 'Warner Bros',        -0.012),
            ],
            'Wireless Telecom':     [
                ('VZ',  'Verizon',           -0.003), ('T',   'AT&T',              -0.001),
                ('TMUS','T-Mobile',           0.004),
            ],
        }
    },
    'Industrials': {
        'etf': 'XLI', 'name': 'Industrials', 'return_1d': 0.0031,
        'sub_industries': {
            'Aerospace & Defense':  [
                ('GE',  'GE Aerospace',       0.006), ('RTX', 'RTX Corp',           0.002),
                ('LMT', 'Lockheed Martin',    0.004), ('NOC', 'Northrop Grumman',   0.003),
            ],
            'Construction Machinery':[
                ('CAT', 'Caterpillar',        0.004), ('DE',  'Deere & Co',         0.002),
            ],
            'Air Freight':          [
                ('FDX', 'FedEx',              0.003), ('UPS', 'UPS',               -0.001),
            ],
            'Railroads':            [
                ('UNP', 'Union Pacific',     -0.001), ('CSX', 'CSX Corp',           0.001),
            ],
        }
    },
    'Energy': {
        'etf': 'XLE', 'name': 'Energy', 'return_1d': -0.0052,
        'sub_industries': {
            'Integrated Oil & Gas': [
                ('XOM', 'ExxonMobil',        -0.004), ('CVX', 'Chevron',           -0.006),
                ('BP',  'BP',                -0.007), ('SHEL','Shell',              -0.005),
            ],
            'E&P':                  [
                ('COP', 'ConocoPhillips',    -0.007), ('DVN', 'Devon Energy',      -0.009),
                ('EOG', 'EOG Resources',     -0.006), ('FANG','Diamondback Energy', -0.008),
            ],
            'Oil Field Services':   [
                ('SLB', 'Schlumberger',      -0.008), ('HAL', 'Halliburton',       -0.010),
                ('BKR', 'Baker Hughes',      -0.007),
            ],
        }
    },
    'ConsumerStaples': {
        'etf': 'XLP', 'name': 'Consumer Staples', 'return_1d': -0.0012,
        'sub_industries': {
            'Household Products':   [
                ('PG',  'Procter & Gamble',  -0.002), ('CL',  'Colgate',            0.001),
                ('KMB', 'Kimberly-Clark',    -0.001),
            ],
            'Beverages':            [
                ('KO',  'Coca-Cola',          0.001), ('PEP', 'PepsiCo',             0.000),
                ('MNST','Monster Beverage',   0.005),
            ],
            'Hypermarkets':         [
                ('WMT', 'Walmart',            0.003), ('COST','Costco',              0.002),
                ('TGT', 'Target',            -0.004),
            ],
            'Food Products':        [
                ('MKC', 'McCormick',         -0.001), ('CPB', "Campbell's",         -0.003),
            ],
        }
    },
    'Materials': {
        'etf': 'XLB', 'name': 'Materials', 'return_1d': 0.0021,
        'sub_industries': {
            'Industrial Gases':     [
                ('LIN', 'Linde',              0.003), ('APD', 'Air Products',        0.001),
            ],
            'Specialty Chemicals':  [
                ('ECL', 'Ecolab',             0.002), ('SHW', 'Sherwin-Williams',    0.004),
                ('ALB', 'Albemarle',         -0.008),
            ],
            'Metals & Mining':      [
                ('FCX', 'Freeport-McMoRan',   0.005), ('NEM', 'Newmont',            -0.002),
                ('AA',  'Alcoa',              0.001),
            ],
        }
    },
    'Utilities': {
        'etf': 'XLU', 'name': 'Utilities', 'return_1d': -0.0031,
        'sub_industries': {
            'Electric Utilities':   [
                ('NEE', 'NextEra Energy',    -0.004), ('SO',  'Southern Company',   -0.002),
                ('DUK', 'Duke Energy',       -0.003), ('AEE', 'Ameren Corp',        -0.001),
            ],
            'Water Utilities':      [
                ('AWK', 'American Water',    -0.002),
            ],
        }
    },
    'RealEstate': {
        'etf': 'XLRE', 'name': 'Real Estate', 'return_1d': -0.0043,
        'sub_industries': {
            'Industrial REITs':     [
                ('PLD', 'Prologis',          -0.003), ('EXR', 'Extra Space Storage',-0.005),
            ],
            'Data Center REITs':    [
                ('EQIX','Equinix',           -0.006), ('DLR', 'Digital Realty',    -0.008),
            ],
            'Specialized REITs':    [
                ('AMT', 'American Tower',    -0.005), ('CCI', 'Crown Castle',       -0.009),
            ],
            'Residential REITs':    [
                ('AVB', 'AvalonBay',         -0.002), ('EQR', 'Equity Residential', -0.003),
            ],
        }
    },
}

# Sector ordering (best performers first, relative to typical GICS display)
SECTOR_ORDER = [
    'InformationTechnology', 'CommunicationServices', 'ConsumerDiscretionary',
    'Financials', 'Industrials', 'Materials', 'ConsumerStaples',
    'HealthCare', 'Utilities', 'RealEstate', 'Energy',
]


# ==============================================================================
# Data classes
# ==============================================================================

@dataclass
class SecurityData:
    symbol:      str
    name:        str
    return_1d:   float
    sub_industry: str
    sector:      str
    market_cap_bn: float = 0.0

    def ascii_cell(self) -> str:
        """Return fixed-width ASCII cell: bar+symbol+return."""
        bar = _ascii_bar(self.return_1d)
        ret = _return_str(self.return_1d)
        return f"{bar}{self.symbol:<5}{ret}"


@dataclass
class SubIndustryData:
    name:       str
    sector:     str
    securities: List[SecurityData] = field(default_factory=list)

    @property
    def avg_return(self) -> float:
        if not self.securities:
            return 0.0
        return sum(s.return_1d for s in self.securities) / len(self.securities)


@dataclass
class SectorData:
    key:        str
    name:       str
    etf:        str
    return_1d:  float
    sub_industries: List[SubIndustryData] = field(default_factory=list)

    @property
    def num_advancing(self) -> int:
        return sum(1 for si in self.sub_industries for s in si.securities if s.return_1d > 0)

    @property
    def num_declining(self) -> int:
        return sum(1 for si in self.sub_industries for s in si.securities if s.return_1d < 0)

    @property
    def all_securities(self) -> List[SecurityData]:
        return [s for si in self.sub_industries for s in si.securities]


# ==============================================================================
# Heatmap Engine
# ==============================================================================

class HeatmapEngine:
    """
    Full GICS universe heatmap generator.

    Methods
    -------
    generate_heatmap(prices: dict = None) -> dict
        Build heatmap data structure. Returns dict with 'sectors_covered', etc.
    format_ascii(data: dict) -> str
        Render the heatmap dict as an ASCII-art string.
    save(session: str = 'CLOSE') -> str
        Generate + save ASCII heatmap, return file path.
    """

    def __init__(self, universe: Dict[str, Dict] = None):
        self._universe = universe or FULL_UNIVERSE
        self._sectors: Dict[str, SectorData] = {}
        self._last_data: dict = {}

    # ── public API ────────────────────────────────────────────────────────────

    def generate_heatmap(self, prices: dict = None) -> dict:
        """
        Build heatmap data.

        Parameters
        ----------
        prices : optional dict of {symbol: return_1d_pct}
                 If provided, overrides synthetic universe returns.
                 Values expected as fractions (0.03 = +3%), or
                 percentage floats > 1 will be auto-divided by 100.

        Returns
        -------
        dict with keys:
            sectors_covered, generated_at, date, sectors, movers,
            market_summary, legend
        """
        # Normalise optional price overrides
        price_map: Dict[str, float] = {}
        if prices:
            for sym, val in prices.items():
                # Accept both fraction (0.03) and percent (3.0)
                price_map[sym] = val / 100.0 if abs(val) > 1.5 else val

        # Try to pull live sector data if available
        live_sector_rets = self._try_live_sectors()

        # Build sector objects
        self._sectors = {}
        for key in SECTOR_ORDER:
            raw = self._universe.get(key, {})
            if not raw:
                continue
            # Override sector ETF return if live available
            sec_ret = live_sector_rets.get(raw.get('etf', ''), raw.get('return_1d', 0.0))

            sub_list = []
            for si_name, tickers in raw.get('sub_industries', {}).items():
                secs = []
                for t_data in tickers:
                    sym, name, ret = t_data
                    # Apply override if provided
                    if sym in price_map:
                        ret = price_map[sym]
                    secs.append(SecurityData(
                        symbol=sym, name=name, return_1d=ret,
                        sub_industry=si_name, sector=raw.get('name', key)
                    ))
                sub_list.append(SubIndustryData(name=si_name, sector=key, securities=secs))

            self._sectors[key] = SectorData(
                key=key, name=raw.get('name', key), etf=raw.get('etf', ''),
                return_1d=sec_ret, sub_industries=sub_list
            )

        # Build output dict
        sectors_out = {}
        for key, sec in self._sectors.items():
            sub_out = {}
            for si in sec.sub_industries:
                sub_out[si.name] = {
                    'avg_return': round(si.avg_return, 4),
                    'securities': [
                        {'symbol': s.symbol, 'name': s.name,
                         'return_1d': round(s.return_1d, 4),
                         'return_1d_pct': round(s.return_1d * 100, 2),
                         'bar': _ascii_bar(s.return_1d)}
                        for s in sorted(si.securities, key=lambda x: x.return_1d, reverse=True)
                    ]
                }
            sectors_out[key] = {
                'name':          sec.name,
                'etf':           sec.etf,
                'return_1d':     round(sec.return_1d, 4),
                'return_1d_pct': round(sec.return_1d * 100, 2),
                'bar':           _ascii_bar(sec.return_1d),
                'num_advancing': sec.num_advancing,
                'num_declining': sec.num_declining,
                'sub_industries': sub_out,
            }

        movers = self._get_movers()
        adv_sectors = sum(1 for s in self._sectors.values() if s.return_1d > 0)
        all_secs = [s for sec in self._sectors.values() for s in sec.all_securities]
        adv_stocks = sum(1 for s in all_secs if s.return_1d > 0)

        self._last_data = {
            'sectors_covered':  len(self._sectors),
            'securities_total': len(all_secs),
            'generated_at':     datetime.utcnow().isoformat(),
            'date':             date.today().isoformat(),
            'sectors':          sectors_out,
            'movers':           movers,
            'market_summary': {
                'sectors_advancing':  adv_sectors,
                'sectors_declining':  len(self._sectors) - adv_sectors,
                'stocks_advancing':   adv_stocks,
                'stocks_declining':   len(all_secs) - adv_stocks,
                'breadth_pct':        round(adv_stocks / max(len(all_secs), 1) * 100, 1),
            },
            'legend': {
                '████': '>+3%  (strong bullish)',
                '███ ': '+1-3% (bullish)',
                '██  ': '0-1%  (mild green)',
                '▒   ': '-1-0% (mild red)',
                '░   ': '-3-1% (bearish)',
                '▓   ': '<-3%  (strong bearish)',
            }
        }
        return self._last_data

    def format_ascii(self, data: dict) -> str:
        """
        Render heatmap dict as an ASCII art string.

        Parameters
        ----------
        data : dict returned by generate_heatmap()

        Returns
        -------
        str — terminal-ready heatmap
        """
        today   = data.get('date', date.today().isoformat())
        gen_at  = data.get('generated_at', '')
        summary = data.get('market_summary', {})
        legend  = data.get('legend', {})

        lines = [
            '',
            '═' * 74,
            f'  SECURITIES UNIVERSE HEATMAP — {today}  16:00 ET',
            f'  Generated: {gen_at}',
            '═' * 74,
            '',
            '  LEGEND:',
        ]
        for bar, desc in legend.items():
            lines.append(f"    {bar}  {desc}")
        lines.append('')
        lines.append(
            f"  Market Breadth: {summary.get('sectors_advancing',0)} of "
            f"{summary.get('sectors_advancing',0)+summary.get('sectors_declining',0)} "
            f"sectors advancing  |  "
            f"{summary.get('stocks_advancing',0)} of "
            f"{summary.get('securities_total', data.get('securities_total',0))} stocks advancing  "
            f"({summary.get('breadth_pct',0):.1f}%)"
        )
        lines += ['', '─' * 74]

        # Sectors sorted by return (best first)
        sectors = data.get('sectors', {})
        ordered_keys = sorted(sectors, key=lambda k: sectors[k].get('return_1d', 0), reverse=True)

        for key in ordered_keys:
            sec = sectors[key]
            etf = sec['etf']
            ret = sec['return_1d_pct']
            bar = sec['bar']
            adv = sec['num_advancing']
            dec = sec['num_declining']
            lines.append('')
            lines.append(
                f"  {bar} {sec['name'].upper()} ({etf}: {ret:+.2f}%)  "
                f"[{adv}↑ {dec}↓]"
            )

            for si_name, si_data in sec.get('sub_industries', {}).items():
                si_ret = si_data['avg_return'] * 100
                si_bar = _ascii_bar(si_data['avg_return'])
                secs_row_parts = []
                for s in si_data.get('securities', []):
                    s_bar = s['bar']
                    s_ret = s['return_1d_pct']
                    secs_row_parts.append(f"{s['symbol']:<5} {s_bar}{s_ret:>+5.1f}%")
                secs_str = '  '.join(secs_row_parts)
                lines.append(f"    {si_bar} {si_name:<28}  {secs_str}")

        # Movers section
        movers = data.get('movers', {})
        lines += ['', '─' * 74, '', '  TOP MOVERS:']
        lines.append(f"  {'GAINERS':<35}  {'LOSERS'}")
        lines.append('  ' + '─'*68)
        gainers = movers.get('top_gainers', [])
        losers  = movers.get('top_losers', [])
        for i in range(max(len(gainers), len(losers))):
            g = gainers[i] if i < len(gainers) else {}
            l = losers[i]  if i < len(losers)  else {}
            g_str = f"{g.get('symbol',''):<7} {_ascii_bar(g.get('return_1d',0))} {g.get('return_1d_pct',0):>+6.2f}%  {g.get('sector','')[:18]}" if g else ''
            l_str = f"{l.get('symbol',''):<7} {_ascii_bar(l.get('return_1d',0))} {l.get('return_1d_pct',0):>+6.2f}%  {l.get('sector','')[:18]}" if l else ''
            lines.append(f"  {g_str:<35}  {l_str}")

        lines += ['', '═'*74, '  END OF HEATMAP', '═'*74, '']
        return '\n'.join(lines)

    def save(self, session: str = 'CLOSE') -> str:
        """
        Generate heatmap (if not already done) and save to file.

        Returns
        -------
        str — file path
        """
        if not self._last_data:
            self.generate_heatmap()
        text     = self.format_ascii(self._last_data)
        today    = date.today().isoformat()
        filename = f"heatmap_{today}.txt"
        filepath = os.path.join(REPORT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[HeatmapEngine] Saved: {filepath}")
        return filepath

    # ── internal helpers ──────────────────────────────────────────────────────

    def _try_live_sectors(self) -> Dict[str, float]:
        """Attempt to get live sector returns from live_data.py."""
        try:
            from live_data import get_sector_performance
            data = get_sector_performance()
            return {etf: v.get('ret_1d', 0) / 100.0 for etf, v in data.items()}
        except Exception:
            return {}

    def _get_movers(self, n: int = 5) -> dict:
        all_secs = []
        for sec in self._sectors.values():
            for si in sec.sub_industries:
                for s in si.securities:
                    all_secs.append({
                        'symbol':       s.symbol,
                        'name':         s.name,
                        'return_1d':    s.return_1d,
                        'return_1d_pct': round(s.return_1d * 100, 2),
                        'sector':       s.sector,
                        'sub_industry': s.sub_industry,
                        'bar':          _ascii_bar(s.return_1d),
                    })
        sorted_all = sorted(all_secs, key=lambda x: x['return_1d'], reverse=True)
        return {
            'top_gainers': sorted_all[:n],
            'top_losers':  sorted_all[-n:][::-1],
        }

    # ── legacy compat ─────────────────────────────────────────────────────────

    def generate_ascii_grid(self) -> str:
        """Legacy: plain ASCII grid sorted by return."""
        if not self._last_data:
            self.generate_heatmap()
        return self.format_ascii(self._last_data)

    def save_heatmap(self) -> str:
        """Legacy: alias for save()."""
        return self.save()

    def get_movers(self, top_n: int = 5) -> dict:
        if not self._last_data:
            self.generate_heatmap()
        return self._last_data.get('movers', {})

    def print_heatmap(self) -> None:
        if not self._last_data:
            self.generate_heatmap()
        print(self.format_ascii(self._last_data))


# ==============================================================================
# Standalone runner
# ==============================================================================

if __name__ == '__main__':
    engine = HeatmapEngine()
    data   = engine.generate_heatmap()
    print(f"Sectors covered: {data['sectors_covered']}")
    print(f"Securities total: {data['securities_total']}")
    text = engine.format_ascii(data)
    print(text[:4000])
    path = engine.save()
    print(f"Saved: {path}")
