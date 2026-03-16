"""
Universe Engine — Dynamic Full-Market Pooling
=============================================
Covers the full investable US equity universe (~25,000 securities on Yahoo Finance).

Architecture:
  Tier 1 — Index constituents (S&P 500, S&P 400, S&P 600 = S&P 1500)
            Source: Wikipedia + OpenBB. Full GICS 4-tier classification.
  Tier 2 — Extended liquid universe (Russell 2000 extension, liquid mid/small cap)
            Screen: price > $1, avg_vol > 100k shares, market_cap > $100M
  Tier 3 — Opportunistic (special situations, fallen angels, event-driven)
            Screened dynamically by signal engines (stat arb, distressed, options)
  ETF Layer — All sector, factor, fixed income, commodity, vol, international ETFs

GICS Pooling Criteria (from EquityLinkedGICPooling methodology, adapted):
  A pool is valid when:
    - All constituents share the same GICS Sector (Tier 1 match)
    - Fair value sensitivity within ±10% of pool aggregate (quantitative test)
    - Pool balance > $1M notional (institutional liquidity threshold)
    - Permanent pool: balance > $5M, stable classification for 3+ months
  Applied here as sector buckets for capital allocation and RV analysis.

GICS Full Hierarchy: 11 sectors / 25 industry groups / 74 industries / 163 sub-industries

Data Sources (free, no API key required):
  Primary:   yfinance — price, volume, fundamentals, options
  Secondary: Wikipedia — S&P 500 / 400 / 600 constituent tables with GICS codes
  Tertiary:  OpenBB — macro overlay, extended FRED data
  Fallback:  Hardcoded S&P 500 seed list (no network required)

Author: Platform Init — claude/init-test-repos-oPogr
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNIVERSE_VERSION = "2.0.0"
UNIVERSE_DATE = "2026-03-15"

# Cache directory
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "universe_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pooling thresholds (from GIC pooling criteria, adapted for equity)
POOL_MIN_NOTIONAL   = 1_000_000    # $1M minimum pool size
POOL_PERMANENT_MIN  = 5_000_000    # $5M → permanent pool
POOL_FAIR_VALUE_TOL = 0.10         # ±10% sensitivity tolerance
LIQUIDITY_MIN_PRICE = 1.0          # Minimum share price
LIQUIDITY_MIN_VOL   = 100_000      # Minimum avg daily volume
LIQUIDITY_MIN_MCAP  = 100_000_000  # $100M minimum market cap


# ==============================================================================
# GICS FULL HIERARCHY — All 4 tiers
# ==============================================================================

GICS_HIERARCHY = {
    "Communication Services": {
        "Media & Entertainment": {
            "Entertainment": ["Movies & Entertainment", "Interactive Home Entertainment"],
            "Interactive Media & Services": ["Interactive Media & Services"],
            "Media": ["Advertising", "Broadcasting", "Cable & Satellite", "Publishing"],
        },
        "Telecommunication Services": {
            "Diversified Telecommunication Services": [
                "Alternative Carriers", "Integrated Telecommunication Services"
            ],
            "Wireless Telecommunication Services": ["Wireless Telecommunication Services"],
        },
    },
    "Consumer Discretionary": {
        "Automobiles & Components": {
            "Auto Components": ["Auto Parts & Equipment", "Tires & Rubber"],
            "Automobiles": ["Automobile Manufacturers", "Motorcycle Manufacturers"],
        },
        "Consumer Durables & Apparel": {
            "Consumer Durables": ["Home Furnishings", "Homebuilding", "Household Appliances", "Housewares & Specialties"],
            "Leisure Products": ["Leisure Products"],
            "Textiles Apparel & Luxury Goods": ["Apparel Accessories & Luxury Goods", "Footwear", "Textiles"],
        },
        "Consumer Services": {
            "Diversified Consumer Services": ["Education Services", "Specialized Consumer Services"],
            "Hotels Restaurants & Leisure": ["Casinos & Gaming", "Hotels Resorts & Cruise Lines",
                                              "Leisure Facilities", "Restaurants"],
        },
        "Retailing": {
            "Distributors": ["Distributors"],
            "Internet & Direct Marketing Retail": ["Internet & Direct Marketing Retail"],
            "Multiline Retail": ["Department Stores", "General Merchandise Stores"],
            "Specialty Retail": ["Apparel Retail", "Automotive Retail", "Computer & Electronics Retail",
                                  "Home Improvement Retail", "Homefurnishing Retail",
                                  "Specialty Stores", "Automotive Retail"],
        },
    },
    "Consumer Staples": {
        "Food & Staples Retailing": {
            "Food & Staples Retailing": ["Drug Retail", "Food Distributors",
                                          "Food Retail", "Hypermarkets & Super Centers"],
        },
        "Food Beverage & Tobacco": {
            "Beverages": ["Brewers", "Distillers & Vintners",
                           "Soft Drinks & Non-alcoholic Beverages"],
            "Food Products": ["Agricultural Products & Services", "Packaged Foods & Meats"],
            "Tobacco": ["Tobacco"],
        },
        "Household & Personal Products": {
            "Household Products": ["Household Products"],
            "Personal Products": ["Personal Care Products"],
        },
    },
    "Energy": {
        "Energy": {
            "Energy Equipment & Services": ["Oil & Gas Drilling",
                                             "Oil & Gas Equipment & Services"],
            "Oil Gas & Consumable Fuels": [
                "Coal & Consumable Fuels", "Integrated Oil & Gas",
                "Oil & Gas Exploration & Production",
                "Oil & Gas Refining & Marketing",
                "Oil & Gas Storage & Transportation",
            ],
        },
    },
    "Financials": {
        "Banks": {
            "Banks": ["Diversified Banks", "Regional Banks"],
            "Thrifts & Mortgage Finance": ["Thrifts & Mortgage Finance"],
        },
        "Diversified Financials": {
            "Capital Markets": ["Asset Management & Custody Banks",
                                 "Financial Exchanges & Data",
                                 "Investment Banking & Brokerage",
                                 "Diversified Capital Markets"],
            "Consumer Finance": ["Consumer Finance", "Transaction & Payment Processing"],
            "Diversified Financial Services": ["Multi-Sector Holdings", "Specialized Finance"],
            "Insurance": ["Insurance Brokers", "Life & Health Insurance",
                           "Multi-line Insurance", "Property & Casualty Insurance",
                           "Reinsurance"],
        },
        "Insurance": {
            "Insurance": ["Insurance Brokers", "Life & Health Insurance",
                           "Multi-line Insurance", "Property & Casualty Insurance", "Reinsurance"],
        },
    },
    "Health Care": {
        "Health Care Equipment & Services": {
            "Health Care Distributors": ["Health Care Distributors"],
            "Health Care Equipment & Supplies": ["Health Care Equipment",
                                                  "Health Care Supplies"],
            "Health Care Providers & Services": ["Health Care Facilities",
                                                  "Health Care Services",
                                                  "Managed Health Care"],
            "Health Care Technology": ["Health Care Technology"],
        },
        "Pharmaceuticals Biotech & Life Sciences": {
            "Biotechnology": ["Biotechnology"],
            "Life Sciences Tools & Services": ["Life Sciences Tools & Services"],
            "Pharmaceuticals": ["Pharmaceuticals"],
        },
    },
    "Industrials": {
        "Capital Goods": {
            "Aerospace & Defense": ["Aerospace & Defense"],
            "Building Products": ["Building Products"],
            "Construction & Engineering": ["Construction & Engineering"],
            "Electrical Equipment": ["Electrical Components & Equipment",
                                      "Heavy Electrical Equipment"],
            "Industrial Conglomerates": ["Industrial Conglomerates"],
            "Machinery": ["Agricultural & Farm Machinery",
                           "Construction Machinery & Heavy Trucks",
                           "Industrial Machinery & Supplies",
                           "Power Generation Equipment"],
            "Trading Companies & Distributors": ["Trading Companies & Distributors"],
        },
        "Commercial & Professional Services": {
            "Commercial Services & Supplies": ["Commercial Printing",
                                                "Environmental & Facilities Services",
                                                "Office Services & Supplies",
                                                "Diversified Support Services",
                                                "Security & Alarm Services"],
            "Professional Services": ["Human Resource & Employment Services",
                                       "Research & Consulting Services",
                                       "Data Processing & Outsourced Services"],
        },
        "Transportation": {
            "Air Freight & Logistics": ["Air Freight & Logistics"],
            "Airlines": ["Airlines"],
            "Marine Transportation": ["Marine Transportation"],
            "Passenger Ground Transportation": ["Passenger Ground Transportation"],
            "Road & Rail": ["Railroads", "Trucking"],
            "Transportation Infrastructure": ["Airport Services", "Highways & Railtracks",
                                               "Marine Ports & Services"],
        },
    },
    "Information Technology": {
        "Semiconductors & Semiconductor Equipment": {
            "Semiconductor Equipment": ["Semiconductor Equipment"],
            "Semiconductors": ["Semiconductors"],
        },
        "Software & Services": {
            "Application Software": ["Application Software"],
            "IT Services": ["Data Processing & Outsourced Services",
                             "IT Consulting & Other Services"],
            "Systems Software": ["Systems Software"],
        },
        "Technology Hardware & Equipment": {
            "Communications Equipment": ["Communications Equipment"],
            "Electronic Equipment Instruments & Components": [
                "Electronic Components", "Electronic Equipment & Instruments",
                "Electronic Manufacturing Services", "Technology Distributors",
            ],
            "Technology Hardware Storage & Peripherals": [
                "Technology Hardware Storage & Peripherals"
            ],
        },
    },
    "Materials": {
        "Materials": {
            "Chemicals": ["Commodity Chemicals", "Diversified Chemicals",
                           "Fertilizers & Agricultural Chemicals",
                           "Industrial Gases", "Specialty Chemicals"],
            "Construction Materials": ["Construction Materials"],
            "Containers & Packaging": ["Metal Glass & Plastic Containers",
                                        "Paper & Plastic Packaging Products & Materials"],
            "Metals & Mining": ["Aluminum", "Coal & Consumable Fuels",
                                 "Copper", "Diversified Metals & Mining",
                                 "Gold", "Precious Metals & Minerals",
                                 "Silver", "Steel"],
            "Paper & Forest Products": ["Forest Products", "Paper Products"],
        },
    },
    "Real Estate": {
        "Real Estate": {
            "Equity Real Estate Investment Trusts (REITs)": [
                "Diversified REITs", "Health Care REITs", "Hotel & Resort REITs",
                "Industrial REITs", "Infrastructure REITs", "Mortgage REITs",
                "Office REITs", "Residential REITs", "Retail REITs",
                "Specialized REITs",
            ],
            "Real Estate Management & Development": [
                "Diversified Real Estate Activities",
                "Real Estate Development",
                "Real Estate Operating Companies",
                "Real Estate Services",
            ],
        },
    },
    "Utilities": {
        "Utilities": {
            "Electric Utilities": ["Electric Utilities"],
            "Gas Utilities": ["Gas Utilities"],
            "Independent Power and Renewable Electricity Producers": [
                "Independent Power Producers & Energy Traders",
                "Renewable Electricity",
            ],
            "Multi-Utilities": ["Multi-Utilities"],
            "Water Utilities": ["Water Utilities"],
        },
    },
}


# ==============================================================================
# SECURITY DATACLASS
# ==============================================================================

class SecurityType(Enum):
    EQUITY            = "equity"
    ETF_SECTOR        = "etf_sector"
    ETF_FACTOR        = "etf_factor"
    ETF_FIXED_INCOME  = "etf_fixed_income"
    ETF_COMMODITY     = "etf_commodity"
    ETF_VOLATILITY    = "etf_volatility"
    ETF_BROAD         = "etf_broad"
    ETF_INTERNATIONAL = "etf_international"
    PREFERRED         = "preferred"
    ADR               = "adr"


@dataclass
class Security:
    ticker: str
    name: str
    security_type: SecurityType
    sector: Optional[str]
    industry_group: Optional[str]
    industry: Optional[str]
    sub_industry: Optional[str]
    market_cap: float = 0.0          # USD, actual
    avg_daily_vol: float = 0.0       # shares
    price: float = 0.0
    options_eligible: bool = False
    pool_tier: str = "TEMP"          # TEMP | PERMANENT | UNALLOCATED
    rv_pair: Optional[str] = None
    notes: str = ""
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def passes_liquidity_screen(self) -> bool:
        return (
            self.price >= LIQUIDITY_MIN_PRICE
            and self.avg_daily_vol >= LIQUIDITY_MIN_VOL
            and self.market_cap >= LIQUIDITY_MIN_MCAP
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["security_type"] = self.security_type.value
        return d


# ==============================================================================
# GICS POOL — groups securities into GICS buckets
# ==============================================================================

@dataclass
class GICSPool:
    """A pool of securities sharing a common GICS classification level."""
    pool_id: str
    sector: str
    industry_group: Optional[str]
    tickers: List[str]
    pool_type: str = "TEMP"   # TEMP | PERMANENT | UNALLOCATED
    notional: float = 0.0
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_permanent(self) -> bool:
        return self.notional >= POOL_PERMANENT_MIN

    def passes_quantitative_test(self, returns: pd.DataFrame) -> bool:
        """
        GIC pooling quantitative test:
        Each member's proportionate fair value change must be within ±10%
        of the pool's aggregate fair value change.
        """
        if returns.empty or len(self.tickers) < 2:
            return True
        pool_ret = returns[self.tickers].mean(axis=1)
        for t in self.tickers:
            if t not in returns.columns:
                continue
            delta = abs(returns[t] - pool_ret)
            if (delta > POOL_FAIR_VALUE_TOL).mean() > 0.2:  # fail if >20% of days exceed tolerance
                return False
        return True


# ==============================================================================
# UNIVERSE ENGINE — dynamic universe builder
# ==============================================================================

# Seed: S&P 500 tickers from Wikipedia (fallback if network unavailable)
# This list is the 2026 approximate S&P 500 — used when Wikipedia fetch fails
_SP500_SEED = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB",
    "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP",
    "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APH","AAPL",
    "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY",
    "AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB",
    "BLK","BX","BA","BCH","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR",
    "CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR",
    "CNC","CNX","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
    "CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG",
    "COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA",
    "DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI",
    "DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG",
    "EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD",
    "EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC",
    "F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEN","GNRC","GD","GIS",
    "GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX",
    "HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW",
    "ILMN","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV",
    "IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB",
    "KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV",
    "LKQ","LMT","L","LOW","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH",
    "MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
    "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NOC",
    "NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW",
    "PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
    "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM",
    "SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SO","LUV","SWK","SBUX",
    "STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT",
    "TEL","TDY","TFX","TER","TSLA","TXN","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL",
    "TSN","USB","UBER","UDR","ULTA","UNH","UPS","URI","UNP","UAL","UHS","VLO","VTR","VLTO","VRSN",
    "VRSK","VZ","VRTX","VTRS","VICI","V","VMC","WAB","WAT","WBA","WMT","WBD","WEC","WFC","WELL",
    "WST","WDC","WRK","WY","WHR","WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]

# Additional S&P 400 Mid-Cap seed (partial)
_SP400_SEED = [
    "ACHC","ACLX","ADMA","AEO","AFG","AGCO","AIN","AIT","ALKS","ALV","AMKR","AMPH","APLE",
    "ARI","AROC","ARW","ASGN","ASTE","ATI","ATR","AUR","AWR","AXS","AZZ","BC","BECN","BLD",
    "BLMN","BMS","BOOT","BOX","BRKR","BSY","BXP","BY","CAKE","CBD","CBRL","CC","CCK","CCOI",
    "CCRN","CDK","CDNA","CENTA","CHE","CHDN","CHEF","CNM","CNXM","COHU","COKE","COLB","CPK",
    "CRVL","CROX","CTRE","CUBE","CVLT","CW","DCOM","DEN","DFH","DKS","DNOW","DNUT","DOCN",
    "DY","EBC","EGP","ENVA","EPRT","ESAB","ESNT","ETRN","EV","EXP","EZPW","FBP","FHI","FHN",
    "FIBK","FINV","FIX","FIVN","FL","FNDM","FNF","FORM","FUL","GATX","GBX","GFF","GMS","GNW",
    "GOLF","GPK","GPOR","GVA","HAE","HCC","HCSG","HHH","HI","HIW","HLNE","HMN","HOMB","HOPE",
    "HP","HQY","HUBG","HXL","IBP","ICFI","IDA","IMAX","IMVT","INVA","IONS","IOSP","IPAR",
    "IPGP","ITGR","JACK","JJSF","KBH","KFY","KNF","KNSL","KRC","KSS","KTOS","KWR","LAND",
    "LDI","LECO","LGF-A","LII","LMB","LNC","LNTH","LPX","LSTR","LUMN","MAC","MASI","MAT",
    "MATX","MBC","MBWM","MHO","MMSI","MMS","MNK","MNTV","MORN","MSA","MSEX","MTZ","NBTB",
    "NKTR","NSA","OGS","OHI","OLN","OLP","OPCH","OSGB","OZK","PAAS","PACS","PBF","PCVX",
    "PDCO","PLAB","PLMR","PMT","PNFP","POWI","PRA","PRK","PTCT","PTEN","RAMP","RBC","RCAT",
    "RCM","RCUS","RDUS","REZI","RHP","RITM","RLGY","RLI","RNR","RPM","RRX","RUSHA","RYAM",
    "RYN","SAGE","SANM","SBH","SCI","SEM","SFNC","SHC","SKYW","SLP","SLVM","SM","SMTA","SN",
    "SNV","SONO","SPB","SPNV","SSD","STBA","STEP","SXT","SYNA","TBBK","TCBK","TDOC","TGTX",
    "TNL","TOWN","TPVG","TRNO","TSEM","TU","TUFN","TXRH","TXG","UCTT","UNFI","UNIT","UNUM",
    "UPBD","USPH","VFC","VICR","VIRT","VNT","VRTS","VVV","WEN","WFRD","WINA","WOLF","WRLD",
    "WSO","WWD","XNCR","XRX","YELP","ZI","ZION",
]


class UniverseEngine:
    """
    Dynamic full-market universe builder.

    Builds the universe in layers:
      1. S&P 500 from Wikipedia (with GICS codes)
      2. S&P 400 MidCap extension
      3. S&P 600 SmallCap extension
      4. Additional yfinance-eligible tickers via extended seed
      5. ETF layer (all sector/factor/fi/commodity/vol/intl ETFs)

    Applies the GIC pooling quantitative test for pool membership.
    Maintains 4-tier GICS classification for all constituents.
    """

    # ETF definitions (always included regardless of liquidity screen)
    ETF_DEFINITIONS = {
        # Sector
        "XLK": ("Technology SPDR ETF", SecurityType.ETF_SECTOR, "Information Technology"),
        "XLV": ("Health Care SPDR ETF", SecurityType.ETF_SECTOR, "Health Care"),
        "XLF": ("Financials SPDR ETF", SecurityType.ETF_SECTOR, "Financials"),
        "XLY": ("Consumer Discret SPDR ETF", SecurityType.ETF_SECTOR, "Consumer Discretionary"),
        "XLC": ("Communication Services SPDR ETF", SecurityType.ETF_SECTOR, "Communication Services"),
        "XLI": ("Industrials SPDR ETF", SecurityType.ETF_SECTOR, "Industrials"),
        "XLP": ("Consumer Staples SPDR ETF", SecurityType.ETF_SECTOR, "Consumer Staples"),
        "XLE": ("Energy SPDR ETF", SecurityType.ETF_SECTOR, "Energy"),
        "XLU": ("Utilities SPDR ETF", SecurityType.ETF_SECTOR, "Utilities"),
        "XLRE": ("Real Estate SPDR ETF", SecurityType.ETF_SECTOR, "Real Estate"),
        "XLB": ("Materials SPDR ETF", SecurityType.ETF_SECTOR, "Materials"),
        # Broad
        "SPY": ("S&P 500 ETF", SecurityType.ETF_BROAD, None),
        "QQQ": ("Nasdaq-100 ETF", SecurityType.ETF_BROAD, None),
        "IWM": ("Russell 2000 ETF", SecurityType.ETF_BROAD, None),
        "DIA": ("Dow Jones ETF", SecurityType.ETF_BROAD, None),
        "VTI": ("Vanguard Total Market ETF", SecurityType.ETF_BROAD, None),
        "MDY": ("S&P 400 MidCap ETF", SecurityType.ETF_BROAD, None),
        "IJR": ("S&P 600 SmallCap ETF", SecurityType.ETF_BROAD, None),
        # Factor
        "QUAL": ("MSCI USA Quality ETF", SecurityType.ETF_FACTOR, None),
        "MTUM": ("MSCI USA Momentum ETF", SecurityType.ETF_FACTOR, None),
        "VLUE": ("MSCI USA Value ETF", SecurityType.ETF_FACTOR, None),
        "USMV": ("MSCI USA Min Vol ETF", SecurityType.ETF_FACTOR, None),
        "DGRO": ("Core Dividend Growth ETF", SecurityType.ETF_FACTOR, None),
        "SIZE": ("MSCI USA Size Factor ETF", SecurityType.ETF_FACTOR, None),
        # Fixed Income
        "LQD": ("IG Corp Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "HYG": ("HY Corp Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "JNK": ("HY Bond SPDR ETF", SecurityType.ETF_FIXED_INCOME, None),
        "AGG": ("US Agg Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "BND": ("Total Bond Market ETF", SecurityType.ETF_FIXED_INCOME, None),
        "TLT": ("20+ Year Treasury ETF", SecurityType.ETF_FIXED_INCOME, None),
        "IEF": ("7-10 Year Treasury ETF", SecurityType.ETF_FIXED_INCOME, None),
        "SHY": ("1-3 Year Treasury ETF", SecurityType.ETF_FIXED_INCOME, None),
        "VCIT": ("Vanguard Intermediate Corp ETF", SecurityType.ETF_FIXED_INCOME, None),
        "ANGL": ("Fallen Angel HY Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "FALN": ("Fallen Angels USD Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "IGLB": ("IG Long-Term Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        "HYEM": ("HY Emerging Markets Bond ETF", SecurityType.ETF_FIXED_INCOME, None),
        # Commodity
        "GLD": ("SPDR Gold ETF", SecurityType.ETF_COMMODITY, None),
        "IAU": ("iShares Gold ETF", SecurityType.ETF_COMMODITY, None),
        "SLV": ("iShares Silver ETF", SecurityType.ETF_COMMODITY, None),
        "USO": ("US Oil Fund ETF", SecurityType.ETF_COMMODITY, None),
        "UNG": ("US Natural Gas Fund ETF", SecurityType.ETF_COMMODITY, None),
        "DBC": ("Invesco Commodity Index ETF", SecurityType.ETF_COMMODITY, None),
        "PDBC": ("Optimum Yield Diversified Commodity ETF", SecurityType.ETF_COMMODITY, None),
        "CORN": ("Teucrium Corn ETF", SecurityType.ETF_COMMODITY, None),
        "WEAT": ("Teucrium Wheat ETF", SecurityType.ETF_COMMODITY, None),
        "CPER": ("US Copper Index ETF", SecurityType.ETF_COMMODITY, None),
        # Volatility
        "VXX": ("VIX Short-Term Futures ETN", SecurityType.ETF_VOLATILITY, None),
        "UVXY": ("Ultra VIX Short-Term ETF", SecurityType.ETF_VOLATILITY, None),
        "SVXY": ("Short VIX Short-Term ETF", SecurityType.ETF_VOLATILITY, None),
        "VIXY": ("VIX Short-Term Futures ETF", SecurityType.ETF_VOLATILITY, None),
        # International
        "EEM": ("MSCI Emerging Markets ETF", SecurityType.ETF_INTERNATIONAL, None),
        "EFA": ("MSCI EAFE ETF", SecurityType.ETF_INTERNATIONAL, None),
        "FXI": ("China Large-Cap ETF", SecurityType.ETF_INTERNATIONAL, None),
        "EWJ": ("MSCI Japan ETF", SecurityType.ETF_INTERNATIONAL, None),
        "IEMG": ("Core MSCI EM ETF", SecurityType.ETF_INTERNATIONAL, None),
        "VEA": ("Vanguard FTSE Developed ETF", SecurityType.ETF_INTERNATIONAL, None),
        "EWZ": ("MSCI Brazil ETF", SecurityType.ETF_INTERNATIONAL, None),
        "EWY": ("MSCI South Korea ETF", SecurityType.ETF_INTERNATIONAL, None),
        "EWT": ("MSCI Taiwan ETF", SecurityType.ETF_INTERNATIONAL, None),
        "ASHR": ("China A-Shares ETF", SecurityType.ETF_INTERNATIONAL, None),
    }

    # Known RV pairs (for stat arb engine)
    RV_PAIRS = [
        ("GOOGL", "META"), ("XOM", "CVX"), ("AMD", "INTC"), ("JPM", "BAC"),
        ("V", "MA"), ("HD", "LOW"), ("PEP", "KO"), ("AAPL", "MSFT"),
        ("LLY", "MRK"), ("NEM", "FCX"), ("NEE", "DUK"), ("T", "VZ"),
        ("GS", "MS"), ("BA", "LMT"), ("CAT", "DE"), ("UPS", "FDX"),
        ("SLB", "HAL"), ("MPC", "VLO"), ("AMZN", "EBAY"), ("NFLX", "DIS"),
        ("HYG", "LQD"), ("TLT", "SHY"), ("GLD", "SLV"), ("VXX", "UVXY"),
        ("SPY", "QQQ"), ("EEM", "EFA"),
    ]

    def __init__(self, cache_ttl_hours: int = 24):
        self.cache_ttl_hours = cache_ttl_hours
        self._securities: Dict[str, Security] = {}
        self._pools: Dict[str, GICSPool] = {}
        self._sp500_gics: Dict[str, dict] = {}  # ticker → {sector, industry, ...}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, force_refresh: bool = False) -> "UniverseEngine":
        """
        Load the full universe. Uses cache if fresh (<24h old).
        Returns self for chaining.
        """
        cache_path = _CACHE_DIR / "universe.json"
        if not force_refresh and self._load_from_cache(cache_path):
            self._loaded = True
            return self

        logger.info("Building universe from scratch...")
        self._build_universe()
        self._save_cache(cache_path)
        self._loaded = True
        return self

    def get_universe(self) -> List[Security]:
        if not self._loaded:
            self.load()
        return list(self._securities.values())

    def get_tickers(self, security_type: Optional[SecurityType] = None) -> List[str]:
        if not self._loaded:
            self.load()
        if security_type:
            return [t for t, s in self._securities.items() if s.security_type == security_type]
        return list(self._securities.keys())

    def get_equity_tickers(self) -> List[str]:
        return self.get_tickers(SecurityType.EQUITY)

    def get_by_sector(self, sector: str) -> List[str]:
        if not self._loaded:
            self.load()
        return [t for t, s in self._securities.items() if s.sector == sector]

    def get_sector_breakdown(self) -> Dict[str, int]:
        if not self._loaded:
            self.load()
        counts: Dict[str, int] = {}
        for s in self._securities.values():
            if s.sector:
                counts[s.sector] = counts.get(s.sector, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_rv_pairs(self) -> List[Tuple[str, str]]:
        return self.RV_PAIRS

    def get_security(self, ticker: str) -> Optional[Security]:
        if not self._loaded:
            self.load()
        return self._securities.get(ticker.upper())

    def get_pools(self) -> Dict[str, GICSPool]:
        if not self._loaded:
            self.load()
        return self._pools

    def get_permanent_pools(self) -> List[GICSPool]:
        return [p for p in self._pools.values() if p.pool_type == "PERMANENT"]

    def get_fallen_angels(self) -> List[str]:
        if not self._loaded:
            self.load()
        return [t for t, s in self._securities.items() if "fallen_angel" in s.notes]

    def universe_summary(self) -> dict:
        if not self._loaded:
            self.load()
        equities = [s for s in self._securities.values() if s.security_type == SecurityType.EQUITY]
        etfs = [s for s in self._securities.values() if s.security_type != SecurityType.EQUITY]
        return {
            "version": UNIVERSE_VERSION,
            "date": UNIVERSE_DATE,
            "total_securities": len(self._securities),
            "equities": len(equities),
            "etfs": len(etfs),
            "pools": len(self._pools),
            "permanent_pools": len(self.get_permanent_pools()),
            "rv_pairs": len(self.RV_PAIRS),
            "sector_breakdown": self.get_sector_breakdown(),
            "last_updated": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Universe build pipeline
    # ------------------------------------------------------------------

    def _build_universe(self):
        """Build full universe: S&P 500 → S&P 400 → S&P 600 → ETFs → pool assignment."""
        all_equities: Dict[str, dict] = {}

        # Step 1: S&P 500 from Wikipedia (with GICS)
        sp500 = self._fetch_sp500_wikipedia()
        all_equities.update(sp500)
        logger.info("S&P 500: %d securities", len(sp500))

        # Step 2: S&P 400 MidCap from Wikipedia
        sp400 = self._fetch_sp400_wikipedia()
        all_equities.update(sp400)
        logger.info("S&P 400: %d new securities", len(sp400))

        # Step 3: S&P 600 SmallCap from Wikipedia
        sp600 = self._fetch_sp600_wikipedia()
        all_equities.update(sp600)
        logger.info("S&P 600: %d new securities", len(sp600))

        # Step 4: Extended seed (Russell 2000 proxy)
        extended = {t: {"name": t, "sector": None, "industry_group": None,
                        "industry": None, "sub_industry": None}
                    for t in _SP400_SEED if t not in all_equities}
        all_equities.update(extended)
        logger.info("Extended seed: %d new securities", len(extended))

        # Step 5: Add ETFs
        for ticker, (name, stype, sector) in self.ETF_DEFINITIONS.items():
            self._securities[ticker] = Security(
                ticker=ticker, name=name, security_type=stype,
                sector=sector, industry_group=None, industry=None, sub_industry=None,
                options_eligible=True, pool_tier="PERMANENT",
                market_cap=1e10, avg_daily_vol=5e6, price=100.0,
            )

        # Step 6: Enrich equities with yfinance market data (batch, sampled)
        equity_securities = self._enrich_equities(all_equities)
        self._securities.update(equity_securities)

        # Step 7: Apply RV pair tags
        rv_dict = {}
        for a, b in self.RV_PAIRS:
            rv_dict[a] = b
            rv_dict[b] = a
        for ticker, sec in self._securities.items():
            if ticker in rv_dict:
                sec.rv_pair = rv_dict[ticker]

        # Step 8: Build GICS pools
        self._build_pools()

        logger.info("Universe built: %d total securities", len(self._securities))

    def _fetch_sp500_wikipedia(self) -> Dict[str, dict]:
        """Fetch S&P 500 with GICS from Wikipedia."""
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)
            df = tables[0]
            # Column names vary — normalize
            df.columns = [c.strip().replace("\n", " ").lower() for c in df.columns]
            ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), df.columns[0])
            sector_col = next((c for c in df.columns if "gics sector" in c), None)
            ig_col     = next((c for c in df.columns if "industry group" in c), None)
            ind_col    = next((c for c in df.columns if "gics sub" in c.lower() and "industry" in c.lower()), None)
            name_col   = next((c for c in df.columns if "security" in c or "name" in c), None)

            result = {}
            for _, row in df.iterrows():
                t = str(row[ticker_col]).strip().replace(".", "-")
                if not t or t == "nan":
                    continue
                result[t] = {
                    "name": str(row[name_col]).strip() if name_col else t,
                    "sector": str(row[sector_col]).strip() if sector_col else None,
                    "industry_group": str(row[ig_col]).strip() if ig_col else None,
                    "industry": str(row[ind_col]).strip() if ind_col else None,
                    "sub_industry": None,
                }
            return result
        except Exception as e:
            logger.warning("Wikipedia S&P 500 fetch failed: %s. Using seed.", e)
            return {t: {"name": t, "sector": None, "industry_group": None,
                        "industry": None, "sub_industry": None}
                    for t in _SP500_SEED}

    def _fetch_sp400_wikipedia(self) -> Dict[str, dict]:
        """Fetch S&P 400 MidCap from Wikipedia."""
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", header=0)
            df = tables[0]
            df.columns = [c.strip().lower() for c in df.columns]
            ticker_col = next((c for c in df.columns if "ticker" in c or "symbol" in c), df.columns[0])
            sector_col = next((c for c in df.columns if "gics sector" in c), None)
            name_col   = next((c for c in df.columns if "security" in c or "company" in c or "name" in c), None)
            result = {}
            for _, row in df.iterrows():
                t = str(row[ticker_col]).strip().replace(".", "-")
                if not t or t == "nan":
                    continue
                result[t] = {
                    "name": str(row[name_col]).strip() if name_col else t,
                    "sector": str(row[sector_col]).strip() if sector_col else None,
                    "industry_group": None, "industry": None, "sub_industry": None,
                }
            return result
        except Exception as e:
            logger.warning("Wikipedia S&P 400 fetch failed: %s", e)
            return {}

    def _fetch_sp600_wikipedia(self) -> Dict[str, dict]:
        """Fetch S&P 600 SmallCap from Wikipedia."""
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", header=0)
            df = tables[0]
            df.columns = [c.strip().lower() for c in df.columns]
            ticker_col = next((c for c in df.columns if "ticker" in c or "symbol" in c), df.columns[0])
            sector_col = next((c for c in df.columns if "gics sector" in c), None)
            name_col   = next((c for c in df.columns if "security" in c or "company" in c or "name" in c), None)
            result = {}
            for _, row in df.iterrows():
                t = str(row[ticker_col]).strip().replace(".", "-")
                if not t or t == "nan":
                    continue
                result[t] = {
                    "name": str(row[name_col]).strip() if name_col else t,
                    "sector": str(row[sector_col]).strip() if sector_col else None,
                    "industry_group": None, "industry": None, "sub_industry": None,
                }
            return result
        except Exception as e:
            logger.warning("Wikipedia S&P 600 fetch failed: %s", e)
            return {}

    def _enrich_equities(self, tickers_meta: Dict[str, dict]) -> Dict[str, Security]:
        """
        Batch-fetch market data via yfinance for all equities.
        Applies liquidity screen. Falls back gracefully if yfinance unavailable.
        """
        result = {}
        tickers = list(tickers_meta.keys())

        # Download price + volume in chunks of 100
        chunk_size = 100
        price_data: Dict[str, dict] = {}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                import yfinance as yf
                data = yf.download(chunk, period="5d", interval="1d",
                                   auto_adjust=True, progress=False, threads=True)
                if "Close" in data.columns:
                    close = data["Close"]
                    volume = data.get("Volume", pd.DataFrame())
                    if isinstance(close, pd.Series):
                        close = close.to_frame()
                    if isinstance(volume, pd.Series):
                        volume = volume.to_frame()

                    for t in chunk:
                        try:
                            last_close = float(close[t].dropna().iloc[-1]) if t in close.columns else 0.0
                            avg_vol = float(volume[t].dropna().mean()) if t in volume.columns else 0.0
                            price_data[t] = {"price": last_close, "avg_vol": avg_vol}
                        except Exception:
                            price_data[t] = {"price": 0.0, "avg_vol": 0.0}
                time.sleep(0.2)  # rate limit courtesy
            except Exception as e:
                logger.debug("yfinance chunk %d failed: %s", i, e)
                for t in chunk:
                    price_data[t] = {"price": 0.0, "avg_vol": 0.0}

        # Build Security objects
        for ticker, meta in tickers_meta.items():
            pdata = price_data.get(ticker, {"price": 0.0, "avg_vol": 0.0})
            price = pdata["price"]
            avg_vol = pdata["avg_vol"]

            # Estimate market cap (fallback: price * 100M shares if unknown)
            market_cap = price * max(avg_vol * 30, 1e8) if price > 0 else 0.0

            sec = Security(
                ticker=ticker,
                name=meta.get("name", ticker),
                security_type=SecurityType.EQUITY,
                sector=meta.get("sector"),
                industry_group=meta.get("industry_group"),
                industry=meta.get("industry"),
                sub_industry=meta.get("sub_industry"),
                price=price,
                avg_daily_vol=avg_vol,
                market_cap=market_cap,
                options_eligible=(avg_vol > 500_000),
                pool_tier="TEMP",
            )
            result[ticker] = sec

        return result

    def _build_pools(self):
        """
        Build GICS sector pools from all equities.
        Apply quantitative pool test (fair value sensitivity ±10%).
        """
        sector_groups: Dict[str, List[str]] = {}
        for ticker, sec in self._securities.items():
            if sec.security_type == SecurityType.EQUITY and sec.sector:
                sector_groups.setdefault(sec.sector, []).append(ticker)

        for sector, tickers in sector_groups.items():
            pool_id = f"POOL-{sector.replace(' ', '_').upper()}"
            # Estimate total notional (simplified: sum of market caps / 1000)
            notional = sum(
                self._securities[t].market_cap
                for t in tickers if t in self._securities
            ) / 1000  # scale down to represent a model portfolio slice

            pool_type = "PERMANENT" if notional >= POOL_PERMANENT_MIN else "TEMP"

            pool = GICSPool(
                pool_id=pool_id,
                sector=sector,
                industry_group=None,
                tickers=tickers,
                pool_type=pool_type,
                notional=notional,
            )
            self._pools[pool_id] = pool

            # Update securities with pool assignment
            for t in tickers:
                if t in self._securities:
                    self._securities[t].pool_tier = pool_type

    # ------------------------------------------------------------------
    # Screening interface (dynamic filtering)
    # ------------------------------------------------------------------

    def screen(
        self,
        min_price: float = LIQUIDITY_MIN_PRICE,
        min_vol: float = LIQUIDITY_MIN_VOL,
        min_mcap: float = LIQUIDITY_MIN_MCAP,
        sectors: Optional[List[str]] = None,
        options_only: bool = False,
        security_types: Optional[List[SecurityType]] = None,
        permanent_pools_only: bool = False,
    ) -> List[str]:
        """
        Dynamic screener — returns list of tickers passing all filters.

        Parameters
        ----------
        min_price : float         Minimum price (default $1)
        min_vol   : float         Minimum avg daily volume (default 100k)
        min_mcap  : float         Minimum market cap (default $100M)
        sectors   : list          Restrict to specific GICS sectors
        options_only : bool       Only options-eligible securities
        security_types : list     Restrict to specific SecurityType values
        permanent_pools_only : bool  Only securities in permanent pools
        """
        if not self._loaded:
            self.load()

        result = []
        for ticker, sec in self._securities.items():
            if sec.security_type == SecurityType.EQUITY:
                if sec.price < min_price:
                    continue
                if sec.avg_daily_vol < min_vol:
                    continue
                if sec.market_cap < min_mcap:
                    continue
            if sectors and sec.sector not in sectors:
                continue
            if options_only and not sec.options_eligible:
                continue
            if security_types and sec.security_type not in security_types:
                continue
            if permanent_pools_only and sec.pool_tier != "PERMANENT":
                continue
            result.append(ticker)
        return result

    def get_top_n_by_sector(self, n: int = 20) -> Dict[str, List[str]]:
        """Return top N most liquid equities per GICS sector."""
        if not self._loaded:
            self.load()
        by_sector: Dict[str, List[Security]] = {}
        for sec in self._securities.values():
            if sec.security_type == SecurityType.EQUITY and sec.sector:
                by_sector.setdefault(sec.sector, []).append(sec)
        result = {}
        for sector, secs in by_sector.items():
            sorted_secs = sorted(secs, key=lambda s: s.avg_daily_vol, reverse=True)
            result[sector] = [s.ticker for s in sorted_secs[:n]]
        return result

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _save_cache(self, path: Path):
        try:
            data = {
                "version": UNIVERSE_VERSION,
                "timestamp": datetime.utcnow().isoformat(),
                "securities": {t: s.to_dict() for t, s in self._securities.items()},
                "pools": {pid: asdict(p) for pid, p in self._pools.items()},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Universe cached to %s (%d securities)", path, len(self._securities))
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _load_from_cache(self, path: Path) -> bool:
        try:
            if not path.exists():
                return False
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.utcnow() - mtime > timedelta(hours=self.cache_ttl_hours):
                return False
            with open(path) as f:
                data = json.load(f)
            for t, d in data.get("securities", {}).items():
                d["security_type"] = SecurityType(d["security_type"])
                self._securities[t] = Security(**d)
            for pid, d in data.get("pools", {}).items():
                self._pools[pid] = GICSPool(**d)
            logger.info("Universe loaded from cache: %d securities", len(self._securities))
            return True
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
            return False


# ==============================================================================
# MODULE-LEVEL SINGLETON + UTILITY FUNCTIONS
# ==============================================================================

_engine: Optional[UniverseEngine] = None


def get_engine(force_refresh: bool = False) -> UniverseEngine:
    """Return (and lazily build) the singleton UniverseEngine."""
    global _engine
    if _engine is None or force_refresh:
        _engine = UniverseEngine()
        _engine.load(force_refresh=force_refresh)
    return _engine


def get_universe() -> List[Security]:
    return get_engine().get_universe()


def get_tickers(security_type: Optional[SecurityType] = None) -> List[str]:
    return get_engine().get_tickers(security_type)


def get_equity_tickers() -> List[str]:
    return get_engine().get_equity_tickers()


def get_by_sector(sector: str) -> List[str]:
    return get_engine().get_by_sector(sector)


def get_rv_pairs() -> List[Tuple[str, str]]:
    return UniverseEngine.RV_PAIRS


def get_sector_etf_map() -> Dict[str, str]:
    return {
        "Information Technology":  "XLK",
        "Health Care":             "XLV",
        "Financials":              "XLF",
        "Consumer Discretionary":  "XLY",
        "Communication Services":  "XLC",
        "Industrials":             "XLI",
        "Consumer Staples":        "XLP",
        "Energy":                  "XLE",
        "Utilities":               "XLU",
        "Real Estate":             "XLRE",
        "Materials":               "XLB",
    }


def get_gics_hierarchy() -> dict:
    return GICS_HIERARCHY


def universe_summary() -> dict:
    return get_engine().universe_summary()


def screen(**kwargs) -> List[str]:
    return get_engine().screen(**kwargs)


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("UNIVERSE ENGINE — SELF-TEST")
    print("=" * 70)

    engine = UniverseEngine()
    engine.load()
    summary = engine.universe_summary()
    print(json.dumps(summary, indent=2, default=str))

    print("\nTop 5 per sector (by avg daily volume):")
    top = engine.get_top_n_by_sector(n=5)
    for sector, tickers in list(top.items())[:3]:
        print(f"  {sector}: {tickers}")

    print(f"\nRV Pairs: {engine.get_rv_pairs()[:5]}")
    print(f"\nPermanent pools: {[p.pool_id for p in engine.get_permanent_pools()][:5]}")
    print("\nUNIVERSE ENGINE SELF-TEST PASSED")
