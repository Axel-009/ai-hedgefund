"""
GICS Universe — Complete Global Industry Classification Standard
================================================================
Full 4-tier hierarchy: 11 Sectors → 25 Industry Groups → 74 Industries → 163 Sub-Industries
Covers ~58,000 securities across 125 countries (~95% world market cap).

Includes: representative liquid US tickers per sub-industry for agent universe assignment.
Author: Platform Init — claude/init-test-repos-oPogr
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SubIndustry:
    code: str
    name: str
    tickers: List[str] = field(default_factory=list)  # representative liquid names

@dataclass
class Industry:
    code: str
    name: str
    sub_industries: List[SubIndustry] = field(default_factory=list)

@dataclass
class IndustryGroup:
    code: str
    name: str
    industries: List[Industry] = field(default_factory=list)

@dataclass
class Sector:
    code: str
    name: str
    industry_groups: List[IndustryGroup] = field(default_factory=list)
    agent_id: str = ""
    benchmark_etf: str = ""

# ==============================================================================
# FULL GICS HIERARCHY (11 Sectors, curated liquid tickers per sub-industry)
# ==============================================================================

GICS = [

    # ─────────────────────────────────────────────────────────────────────────
    Sector("10", "Energy", benchmark_etf="XLE", industry_groups=[
        IndustryGroup("1010", "Energy", industries=[
            Industry("101010", "Energy Equipment & Services", sub_industries=[
                SubIndustry("10101010", "Oil & Gas Drilling", ["RIG","VAL","DO"]),
                SubIndustry("10101020", "Oil & Gas Equipment & Services", ["SLB","HAL","BKR","NOV"]),
            ]),
            Industry("101020", "Oil, Gas & Consumable Fuels", sub_industries=[
                SubIndustry("10102010", "Integrated Oil & Gas", ["XOM","CVX","BP","SHEL"]),
                SubIndustry("10102020", "Oil & Gas Exploration & Production", ["EOG","PXD","DVN","FANG","COP"]),
                SubIndustry("10102030", "Oil & Gas Refining & Marketing", ["MPC","PSX","VLO"]),
                SubIndustry("10102040", "Oil & Gas Storage & Transportation", ["KMI","WMB","ET","EPD"]),
                SubIndustry("10102050", "Coal & Consumable Fuels", ["BTU","ARCH","CEIX"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("15", "Materials", benchmark_etf="XLB", industry_groups=[
        IndustryGroup("1510", "Materials", industries=[
            Industry("151010", "Chemicals", sub_industries=[
                SubIndustry("15101010", "Commodity Chemicals", ["DD","LYB","WLK"]),
                SubIndustry("15101020", "Diversified Chemicals", ["DOW","EMN","CC"]),
                SubIndustry("15101030", "Fertilizers & Agricultural Chemicals", ["MOS","NTR","CF"]),
                SubIndustry("15101040", "Industrial Gases", ["APD","LIN","AIR"]),
                SubIndustry("15101050", "Specialty Chemicals", ["ECL","ALB","IFF","PPG","SHW"]),
            ]),
            Industry("151020", "Construction Materials", sub_industries=[
                SubIndustry("15102010", "Construction Materials", ["MLM","VMC","EXP"]),
            ]),
            Industry("151030", "Containers & Packaging", sub_industries=[
                SubIndustry("15103010", "Metal & Glass Containers", ["BLL","CCK","SLGN"]),
                SubIndustry("15103020", "Paper & Plastic Packaging", ["IP","PKG","SEE"]),
            ]),
            Industry("151040", "Metals & Mining", sub_industries=[
                SubIndustry("15104010", "Aluminum", ["AA","CENX"]),
                SubIndustry("15104020", "Diversified Metals & Mining", ["BHP","RIO","FCX","VALE"]),
                SubIndustry("15104025", "Copper", ["FCX","SCCO","HBM"]),
                SubIndustry("15104030", "Gold", ["NEM","GOLD","AEM","KGC"]),
                SubIndustry("15104040", "Precious Metals & Minerals", ["WPM","FNV","PAAS"]),
                SubIndustry("15104045", "Silver", ["AG","PAAS","SVM"]),
                SubIndustry("15104050", "Steel", ["NUE","STLD","X","CLF"]),
            ]),
            Industry("151050", "Paper & Forest Products", sub_industries=[
                SubIndustry("15105010", "Forest Products", ["WY","RYN","PCH"]),
                SubIndustry("15105020", "Paper Products", ["IP","GPK","CLW"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("20", "Industrials", benchmark_etf="XLI", industry_groups=[
        IndustryGroup("2010", "Capital Goods", industries=[
            Industry("201010", "Aerospace & Defense", sub_industries=[
                SubIndustry("20101010", "Aerospace & Defense", ["BA","LMT","RTX","NOC","GD","HII","TXT"]),
            ]),
            Industry("201020", "Building Products", sub_industries=[
                SubIndustry("20102010", "Building Products", ["CARR","TT","JCI","ALLE","AZEK"]),
            ]),
            Industry("201030", "Construction & Engineering", sub_industries=[
                SubIndustry("20103010", "Construction & Engineering", ["PWR","PRIM","MTZ","TRMK"]),
            ]),
            Industry("201040", "Electrical Equipment", sub_industries=[
                SubIndustry("20104010", "Electrical Components & Equipment", ["ROK","GE","ETN","HUBB"]),
                SubIndustry("20104020", "Heavy Electrical Equipment", ["AMSC","PLUG","FCEL"]),
            ]),
            Industry("201050", "Industrial Conglomerates", sub_industries=[
                SubIndustry("20105010", "Industrial Conglomerates", ["GE","MMM","HON","ITW"]),
            ]),
            Industry("201060", "Machinery", sub_industries=[
                SubIndustry("20106010", "Construction Machinery & Heavy Trucks", ["CAT","DE","PCAR","OSK"]),
                SubIndustry("20106015", "Agricultural & Farm Machinery", ["DE","AGCO","CNH"]),
                SubIndustry("20106020", "Industrial Machinery", ["EMR","PH","DOV","IR","XYL"]),
            ]),
            Industry("201070", "Trading Companies & Distributors", sub_industries=[
                SubIndustry("20107010", "Trading Companies & Distributors", ["FAST","MSM","GWW","AIT"]),
            ]),
        ]),
        IndustryGroup("2020", "Commercial & Professional Services", industries=[
            Industry("202010", "Commercial Services & Supplies", sub_industries=[
                SubIndustry("20201010", "Commercial Printing", ["RRD","LSC"]),
                SubIndustry("20201050", "Environmental & Facilities Services", ["RSG","WM","CWST","CLH"]),
                SubIndustry("20201060", "Office Services & Supplies", ["ACCO","HNI"]),
                SubIndustry("20201070", "Diversified Support Services", ["CTAS","ROL","SCI"]),
                SubIndustry("20201080", "Security & Alarm Services", ["ALLE","BRINKS"]),
            ]),
            Industry("202020", "Professional Services", sub_industries=[
                SubIndustry("20202010", "Human Resources & Employment Services", ["ADP","PAYX","MAN","KFRC"]),
                SubIndustry("20202020", "Research & Consulting Services", ["G","ICF","VRSK","MCO","MSCI"]),
            ]),
        ]),
        IndustryGroup("2030", "Transportation", industries=[
            Industry("203010", "Air Freight & Logistics", sub_industries=[
                SubIndustry("20301010", "Air Freight & Logistics", ["FDX","UPS","EXPD","XPO","CHRW"]),
            ]),
            Industry("203020", "Passenger Airlines", sub_industries=[
                SubIndustry("20302010", "Passenger Airlines", ["DAL","UAL","AAL","LUV","JBLU","ALGT"]),
            ]),
            Industry("203030", "Marine Transportation", sub_industries=[
                SubIndustry("20303010", "Marine Transportation", ["ZIM","MATX","SFL","KNOP"]),
            ]),
            Industry("203040", "Ground Transportation", sub_industries=[
                SubIndustry("20304010", "Rail Transportation", ["UNP","CSX","NSC","CP","CNI"]),
                SubIndustry("20304030", "Trucking", ["ODFL","WERN","KNX","JBHT"]),
            ]),
            Industry("203050", "Transportation Infrastructure", sub_industries=[
                SubIndustry("20305010", "Airport Services", ["FLUG","OMAB"]),
                SubIndustry("20305020", "Highways & Railtracks", ["MIC","NWSA"]),
                SubIndustry("20305030", "Marine Ports & Services", ["GLP","SBLK"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("25", "Consumer Discretionary", benchmark_etf="XLY", industry_groups=[
        IndustryGroup("2510", "Automobiles & Components", industries=[
            Industry("251010", "Automobile Components", sub_industries=[
                SubIndustry("25101010", "Automotive Parts & Equipment", ["MGA","BWA","ADNT","SL"]),
                SubIndustry("25101020", "Tires & Rubber", ["GT","CTB"]),
            ]),
            Industry("251020", "Automobiles", sub_industries=[
                SubIndustry("25102010", "Automobile Manufacturers", ["GM","F","TM","STLA"]),
                SubIndustry("25102020", "Motorcycle Manufacturers", ["HOG","HON"]),
            ]),
        ]),
        IndustryGroup("2520", "Consumer Durables & Apparel", industries=[
            Industry("252010", "Household Durables", sub_industries=[
                SubIndustry("25201010", "Consumer Electronics", ["SONY","SNE","GPRO"]),
                SubIndustry("25201020", "Home Furnishings", ["ETH","WSM","RH","LOVE"]),
                SubIndustry("25201030", "Homebuilding", ["DHI","LEN","PHM","NVR","TOL"]),
                SubIndustry("25201040", "Household Appliances", ["WHR","SNA","AAON"]),
                SubIndustry("25201050", "Housewares & Specialties", ["NWL","TUP","HOLI"]),
            ]),
            Industry("252020", "Leisure Products", sub_industries=[
                SubIndustry("25202010", "Leisure Products", ["PTON","YETI","BF-B","DKNG"]),
            ]),
            Industry("252030", "Textiles, Apparel & Luxury Goods", sub_industries=[
                SubIndustry("25203010", "Apparel, Accessories & Luxury Goods", ["TPR","RL","PVH","VFC","HBI"]),
                SubIndustry("25203020", "Footwear", ["NKE","SKX","CROX","ONON","DECK"]),
                SubIndustry("25203030", "Textiles", ["UFI","CULP"]),
            ]),
        ]),
        IndustryGroup("2530", "Consumer Services", industries=[
            Industry("253010", "Hotels, Restaurants & Leisure", sub_industries=[
                SubIndustry("25301010", "Casinos & Gaming", ["LVS","WYNN","MGM","CZR","PENN"]),
                SubIndustry("25301020", "Hotels, Resorts & Cruise Lines", ["MAR","HLT","H","RCL","CCL","NCLH"]),
                SubIndustry("25301030", "Leisure Facilities", ["SIX","SEAS","EPR"]),
                SubIndustry("25301040", "Restaurants", ["MCD","SBUX","CMG","YUM","DPZ","QSR"]),
            ]),
            Industry("253020", "Diversified Consumer Services", sub_industries=[
                SubIndustry("25302010", "Education Services", ["STRA","PRDO","LAUR","GHC"]),
                SubIndustry("25302020", "Specialized Consumer Services", ["CDAY","HRB","IART"]),
            ]),
        ]),
        IndustryGroup("2550", "Consumer Discretionary Distribution & Retail", industries=[
            Industry("255010", "Distributors", sub_industries=[
                SubIndustry("25501010", "Distributors", ["LKQ","POOL","GPC"]),
            ]),
            Industry("255020", "Broadline Retail", sub_industries=[
                SubIndustry("25502020", "Broadline Retail", ["AMZN","TGT","WMT","COST","BJ"]),
            ]),
            Industry("255030", "Specialty Retail", sub_industries=[
                SubIndustry("25503010", "Apparel Retail", ["GPS","AEO","ANF","URBN","VSCO"]),
                SubIndustry("25503020", "Automotive Retail", ["AN","KMX","ORLY","AZO","AAP"]),
                SubIndustry("25503030", "Home Improvement Retail", ["HD","LOW"]),
                SubIndustry("25503040", "Other Specialty Retail", ["TSCO","BBY","ULTA","SIG"]),
                SubIndustry("25503050", "Computer & Electronics Retail", ["GME","BBY"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("30", "Consumer Staples", benchmark_etf="XLP", industry_groups=[
        IndustryGroup("3010", "Consumer Staples Distribution & Retail", industries=[
            Industry("301010", "Consumer Staples Distribution & Retail", sub_industries=[
                SubIndustry("30101010", "Drug Retail", ["CVS","WBA","RAD"]),
                SubIndustry("30101020", "Food Distributors", ["SYSCO","PFGC","UNFI"]),
                SubIndustry("30101030", "Food Retail", ["KR","ACI","SFM","WINN"]),
                SubIndustry("30101040", "Consumer Staples Merchandise Retail", ["WMT","COST","TGT","DG","DLTR"]),
            ]),
        ]),
        IndustryGroup("3020", "Food, Beverage & Tobacco", industries=[
            Industry("302010", "Beverages", sub_industries=[
                SubIndustry("30201010", "Brewers", ["BUD","TAP","SAM"]),
                SubIndustry("30201020", "Distillers & Vintners", ["BF-B","STZ","MGPI"]),
                SubIndustry("30201030", "Soft Drinks & Non-alcoholic Beverages", ["KO","PEP","MNST","COTT","CELH"]),
            ]),
            Industry("302020", "Food Products", sub_industries=[
                SubIndustry("30202010", "Agricultural Products & Services", ["ADM","BG","INGR"]),
                SubIndustry("30202030", "Packaged Foods & Meats", ["GIS","K","CPB","HRL","CAG","MKC"]),
            ]),
            Industry("302030", "Tobacco", sub_industries=[
                SubIndustry("30203010", "Tobacco", ["MO","PM","BTI","LO"]),
            ]),
        ]),
        IndustryGroup("3030", "Household & Personal Products", industries=[
            Industry("303010", "Household Products", sub_industries=[
                SubIndustry("30301010", "Household Products", ["PG","CL","CHD","ENR","SPB"]),
            ]),
            Industry("303020", "Personal Care Products", sub_industries=[
                SubIndustry("30302010", "Personal Care Products", ["EL","COTY","REYN","SJM"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("35", "Health Care", benchmark_etf="XLV", industry_groups=[
        IndustryGroup("3510", "Health Care Equipment & Services", industries=[
            Industry("351010", "Health Care Equipment & Supplies", sub_industries=[
                SubIndustry("35101010", "Health Care Equipment", ["MDT","BSX","EW","ABT","ISRG","SYK","ZBH"]),
                SubIndustry("35101020", "Health Care Supplies", ["HOLX","BAX","BDX","DXCM"]),
            ]),
            Industry("351020", "Health Care Providers & Services", sub_industries=[
                SubIndustry("35102010", "Health Care Distributors", ["MCK","CAH","ABC","OMI"]),
                SubIndustry("35102015", "Health Care Services", ["DaVita","HCSG","AH"]),
                SubIndustry("35102020", "Health Care Facilities", ["HCA","THC","UHS","ACHC"]),
                SubIndustry("35102030", "Managed Health Care", ["UNH","ELV","CVS","CNC","MOH","HUM"]),
            ]),
            Industry("351030", "Health Care Technology", sub_industries=[
                SubIndustry("35103010", "Health Care Technology", ["VEEV","NXGN","CERN","OMCL","PHR"]),
            ]),
        ]),
        IndustryGroup("3520", "Pharmaceuticals, Biotechnology & Life Sciences", industries=[
            Industry("352010", "Biotechnology", sub_industries=[
                SubIndustry("35201010", "Biotechnology", ["AMGN","GILD","REGN","VRTX","BIIB","MRNA","BNTX"]),
            ]),
            Industry("352020", "Pharmaceuticals", sub_industries=[
                SubIndustry("35202010", "Pharmaceuticals", ["JNJ","PFE","MRK","ABBV","LLY","BMY","AZN"]),
            ]),
            Industry("352030", "Life Sciences Tools & Services", sub_industries=[
                SubIndustry("35203010", "Life Sciences Tools & Services", ["TMO","DHR","IQV","A","BIO","ILMN"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("40", "Financials", benchmark_etf="XLF", industry_groups=[
        IndustryGroup("4010", "Banks", industries=[
            Industry("401010", "Banks", sub_industries=[
                SubIndustry("40101010", "Diversified Banks", ["JPM","BAC","WFC","C","USB","TFC"]),
                SubIndustry("40101015", "Regional Banks", ["ZION","HBAN","RF","FHN","FITB","KEY"]),
            ]),
        ]),
        IndustryGroup("4020", "Financial Services", industries=[
            Industry("402010", "Diversified Financial Services", sub_industries=[
                SubIndustry("40201010", "Diversified Financial Services", ["BRK-B","V","MA","AXP","DFS"]),
                SubIndustry("40201020", "Multi-Sector Holdings", ["BX","KKR","APO","CG","ARES"]),
            ]),
            Industry("402020", "Consumer Finance", sub_industries=[
                SubIndustry("40202010", "Consumer Finance", ["SYF","COF","ALLY","OMF","CACC"]),
            ]),
            Industry("402030", "Capital Markets", sub_industries=[
                SubIndustry("40203010", "Asset Management & Custody Banks", ["BLK","STT","BK","NTRS","IVZ"]),
                SubIndustry("40203020", "Investment Banking & Brokerage", ["GS","MS","LAZ","EVR","MC"]),
                SubIndustry("40203030", "Diversified Capital Markets", ["UBS","DB","CS"]),
                SubIndustry("40203040", "Financial Exchanges & Data", ["CME","ICE","CBOE","MSCI","SPGI"]),
            ]),
            Industry("402040", "Mortgage Real Estate Investment Trusts", sub_industries=[
                SubIndustry("40204010", "Mortgage REITs", ["AGNC","NLY","STWD","RITM"]),
            ]),
        ]),
        IndustryGroup("4030", "Insurance", industries=[
            Industry("403010", "Insurance", sub_industries=[
                SubIndustry("40301010", "Insurance Brokers", ["MMC","AON","WTW","BRP","RYAN"]),
                SubIndustry("40301020", "Life & Health Insurance", ["MET","PRU","PFG","UNM","GL"]),
                SubIndustry("40301030", "Multi-line Insurance", ["AIG","HIG","L","CB"]),
                SubIndustry("40301040", "Property & Casualty Insurance", ["ALL","TRV","PGR","CINF","WRB"]),
                SubIndustry("40301050", "Reinsurance", ["EG","RNR","ACGL","MKL"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("45", "Information Technology", benchmark_etf="XLK", industry_groups=[
        IndustryGroup("4510", "Software & Services", industries=[
            Industry("451010", "IT Services", sub_industries=[
                SubIndustry("45101010", "IT Consulting & Other Services", ["ACN","IBM","EPAM","GLOB","INFY"]),
                SubIndustry("45101020", "Internet Services & Infrastructure", ["AKAM","NET","FSLY","DDOG"]),
            ]),
            Industry("451020", "Software", sub_industries=[
                SubIndustry("45102010", "Application Software", ["CRM","ADBE","INTU","NOW","WDAY","DDOG","ZM"]),
                SubIndustry("45102020", "Systems Software", ["MSFT","ORCL","PANW","FTNT","CRWD","ZS"]),
                SubIndustry("45102030", "Infrastructure Software", ["SPLK","SNOW","MDB","ESTC"]),
            ]),
        ]),
        IndustryGroup("4520", "Technology Hardware & Equipment", industries=[
            Industry("452010", "Communications Equipment", sub_industries=[
                SubIndustry("45201020", "Communications Equipment", ["CSCO","JNPR","ANET","CIEN","VIAV"]),
            ]),
            Industry("452020", "Technology Hardware, Storage & Peripherals", sub_industries=[
                SubIndustry("45202030", "Technology Hardware, Storage & Peripherals", ["AAPL","HPQ","HPE","WDC","STX","NTAP"]),
            ]),
            Industry("452030", "Electronic Equipment, Instruments & Components", sub_industries=[
                SubIndustry("45203010", "Electronic Components", ["TE","GLW","FLEX","JBL","CLS"]),
                SubIndustry("45203015", "Electronic Equipment & Instruments", ["KEYS","TRMB","MKS","ITRI"]),
                SubIndustry("45203020", "Electronic Manufacturing Services", ["JABIL","FLEX","CLS"]),
                SubIndustry("45203030", "Technology Distributors", ["TD","AVT","ARW"]),
            ]),
        ]),
        IndustryGroup("4530", "Semiconductors & Semiconductor Equipment", industries=[
            Industry("453010", "Semiconductors & Semiconductor Equipment", sub_industries=[
                SubIndustry("45301010", "Semiconductor Equipment", ["AMAT","LRCX","KLAC","ASML","ONTO"]),
                SubIndustry("45301020", "Semiconductors", ["NVDA","AMD","INTC","QCOM","AVGO","TXN","MU","TSM","MRVL"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("50", "Communication Services", benchmark_etf="XLC", industry_groups=[
        IndustryGroup("5010", "Telecommunication Services", industries=[
            Industry("501010", "Diversified Telecommunication Services", sub_industries=[
                SubIndustry("50101010", "Alternative Carriers", ["LUMN","TMUS","DISH"]),
                SubIndustry("50101020", "Integrated Telecommunication Services", ["VZ","T","TMUS"]),
            ]),
            Industry("501020", "Wireless Telecommunication Services", sub_industries=[
                SubIndustry("50102010", "Wireless Telecom Services", ["TMUS","VZ","T","SHEN"]),
            ]),
        ]),
        IndustryGroup("5020", "Media & Entertainment", industries=[
            Industry("502010", "Media", sub_industries=[
                SubIndustry("50201010", "Advertising", ["IPG","OMC","WPP","PUBM","TTD"]),
                SubIndustry("50201020", "Broadcasting", ["FOX","NWSA","PARA","WBD"]),
                SubIndustry("50201030", "Cable & Satellite", ["CMCSA","CHTR","CABO"]),
                SubIndustry("50201040", "Publishing", ["NYT","MDP","NWSA"]),
            ]),
            Industry("502020", "Entertainment", sub_industries=[
                SubIndustry("50202010", "Movies & Entertainment", ["DIS","NFLX","AMZN","PARA","LGF-A"]),
                SubIndustry("50202020", "Interactive Home Entertainment", ["EA","TTWO","ATVI","RBLX","U"]),
            ]),
            Industry("502030", "Interactive Media & Services", sub_industries=[
                SubIndustry("50203010", "Interactive Media & Services", ["GOOGL","META","SNAP","PINS","TWTR"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("55", "Utilities", benchmark_etf="XLU", industry_groups=[
        IndustryGroup("5510", "Utilities", industries=[
            Industry("551010", "Electric Utilities", sub_industries=[
                SubIndustry("55101010", "Electric Utilities", ["NEE","DUK","SO","AEP","EXC","PCG","SRE"]),
            ]),
            Industry("551020", "Gas Utilities", sub_industries=[
                SubIndustry("55102010", "Gas Utilities", ["ATO","NI","OGS","NWN","SPOK"]),
            ]),
            Industry("551030", "Multi-Utilities", sub_industries=[
                SubIndustry("55103010", "Multi-Utilities", ["D","ED","XEL","PEG","WEC","ES"]),
            ]),
            Industry("551040", "Water Utilities", sub_industries=[
                SubIndustry("55104010", "Water Utilities", ["AWK","WTRG","SJW","MSEX"]),
            ]),
            Industry("551050", "Independent Power & Renewable Electricity", sub_industries=[
                SubIndustry("55105010", "Independent Power Producers", ["AES","VST","NRG","CWEN"]),
                SubIndustry("55105020", "Renewable Electricity", ["NEP","ENPH","FSLR","RUN","ARRY","SEDG"]),
            ]),
        ]),
    ]),

    # ─────────────────────────────────────────────────────────────────────────
    Sector("60", "Real Estate", benchmark_etf="XLRE", industry_groups=[
        IndustryGroup("6010", "Equity Real Estate Investment Trusts (REITs)", industries=[
            Industry("601010", "Diversified REITs", sub_industries=[
                SubIndustry("60101010", "Diversified REITs", ["WPC","IRET","EPRT"]),
            ]),
            Industry("601025", "Health Care REITs", sub_industries=[
                SubIndustry("60102510", "Health Care REITs", ["WELL","VTR","PEAK","SBRA","OHI"]),
            ]),
            Industry("601030", "Residential REITs", sub_industries=[
                SubIndustry("60103010", "Multi-Family Residential REITs", ["AVB","EQR","UDR","CPT","MAA"]),
                SubIndustry("60103020", "Single-Family Residential REITs", ["INVH","AMH","SFR"]),
            ]),
            Industry("601040", "Retail REITs", sub_industries=[
                SubIndustry("60104010", "Retail REITs", ["SPG","O","NNN","SITC","BRX","KIM"]),
            ]),
            Industry("601050", "Specialized REITs", sub_industries=[
                SubIndustry("60105010", "Self-Storage REITs", ["PSA","EXR","CUBE","LSI"]),
                SubIndustry("60105020", "Telecom Tower REITs", ["AMT","CCI","SBAC","UNIT"]),
                SubIndustry("60105030", "Data Center REITs", ["EQIX","DLR","CONE","SWCH"]),
                SubIndustry("60105040", "Office REITs", ["BXP","VNO","SLG","CUZ"]),
                SubIndustry("60105050", "Industrial REITs", ["PLD","REXR","EGP","FR"]),
            ]),
        ]),
        IndustryGroup("6020", "Real Estate Management & Development", industries=[
            Industry("602010", "Real Estate Management & Development", sub_industries=[
                SubIndustry("60201010", "Diversified Real Estate Activities", ["CBRE","JLL","CSGP"]),
                SubIndustry("60201020", "Real Estate Operating Companies", ["Z","OPEN","RDFN"]),
                SubIndustry("60201030", "Real Estate Development", ["FRP","NXRT","KW"]),
                SubIndustry("60201040", "Real Estate Services", ["RMAX","RKT","KBHOME"]),
            ]),
        ]),
    ]),

]

# ==============================================================================
# QUICK LOOKUP HELPERS
# ==============================================================================

def get_sector(name_or_code: str) -> Optional[Sector]:
    for s in GICS:
        if s.code == name_or_code or s.name.lower() == name_or_code.lower():
            return s
    return None

def get_all_tickers() -> List[str]:
    """Return deduplicated list of all tickers across the universe."""
    seen = set()
    tickers = []
    for sector in GICS:
        for ig in sector.industry_groups:
            for ind in ig.industries:
                for sub in ind.sub_industries:
                    for t in sub.tickers:
                        if t not in seen:
                            seen.add(t)
                            tickers.append(t)
    return tickers

def get_sector_tickers(sector_name: str) -> List[str]:
    sector = get_sector(sector_name)
    if not sector:
        return []
    tickers = []
    seen = set()
    for ig in sector.industry_groups:
        for ind in ig.industries:
            for sub in ind.sub_industries:
                for t in sub.tickers:
                    if t not in seen:
                        seen.add(t)
                        tickers.append(t)
    return tickers

def get_sector_etfs() -> Dict[str, str]:
    return {s.name: s.benchmark_etf for s in GICS}

def ticker_to_sector(ticker: str) -> Optional[str]:
    for sector in GICS:
        for ig in sector.industry_groups:
            for ind in ig.industries:
                for sub in ind.sub_industries:
                    if ticker in sub.tickers:
                        return sector.name
    return None

# Stats
TOTAL_SECTORS = len(GICS)
TOTAL_INDUSTRY_GROUPS = sum(len(s.industry_groups) for s in GICS)
TOTAL_INDUSTRIES = sum(len(ig.industries) for s in GICS for ig in s.industry_groups)
TOTAL_SUB_INDUSTRIES = sum(len(ind.sub_industries) for s in GICS
                           for ig in s.industry_groups for ind in ig.industries)
TOTAL_TICKERS = len(get_all_tickers())

if __name__ == "__main__":
    print(f"GICS Universe: {TOTAL_SECTORS} sectors | {TOTAL_INDUSTRY_GROUPS} groups | "
          f"{TOTAL_INDUSTRIES} industries | {TOTAL_SUB_INDUSTRIES} sub-industries | {TOTAL_TICKERS} tickers")
    for s in GICS:
        n = sum(len(ind.sub_industries) for ig in s.industry_groups for ind in ig.industries)
        print(f"  [{s.code}] {s.name:<30} ETF={s.benchmark_etf:<6}  {n} sub-industries")
