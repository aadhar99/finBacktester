"""
Top 100 NSE stocks by market capitalization and liquidity.

This list includes the most liquid and actively traded stocks on NSE,
covering multiple sectors for diversified backtesting.

Updated: 2024
Source: NSE India, market cap and liquidity data
"""

# Nifty 50 stocks (core large caps)
NIFTY_50 = [
    # Top 10 by market cap
    "RELIANCE",      # Reliance Industries
    "TCS",           # Tata Consultancy Services
    "HDFCBANK",      # HDFC Bank
    "INFY",          # Infosys
    "ICICIBANK",     # ICICI Bank
    "HINDUNILVR",    # Hindustan Unilever
    "ITC",           # ITC Limited
    "SBIN",          # State Bank of India
    "BHARTIARTL",    # Bharti Airtel
    "KOTAKBANK",     # Kotak Mahindra Bank

    # Next 40 Nifty 50 stocks
    "BAJFINANCE",    # Bajaj Finance
    "LT",            # Larsen & Toubro
    "HCLTECH",       # HCL Technologies
    "ASIANPAINT",    # Asian Paints
    "AXISBANK",      # Axis Bank
    "MARUTI",        # Maruti Suzuki
    "SUNPHARMA",     # Sun Pharmaceutical
    "TITAN",         # Titan Company
    "ULTRACEMCO",    # UltraTech Cement
    "NESTLEIND",     # Nestle India
    "WIPRO",         # Wipro
    "BAJAJFINSV",    # Bajaj Finserv
    "ONGC",          # Oil & Natural Gas Corp
    "NTPC",          # NTPC
    "TECHM",         # Tech Mahindra
    "M&M",           # Mahindra & Mahindra
    "POWERGRID",     # Power Grid Corporation
    "TATAMOTORS",    # Tata Motors
    "ADANIENT",      # Adani Enterprises
    "TATASTEEL",     # Tata Steel
    "HINDALCO",      # Hindalco Industries
    "COALINDIA",     # Coal India
    "INDUSINDBK",    # IndusInd Bank
    "JSWSTEEL",      # JSW Steel
    "DRREDDY",       # Dr. Reddy's Laboratories
    "DIVISLAB",      # Divi's Laboratories
    "CIPLA",         # Cipla
    "EICHERMOT",     # Eicher Motors
    "HEROMOTOCO",    # Hero MotoCorp
    "GRASIM",        # Grasim Industries
    "ADANIPORTS",    # Adani Ports
    "APOLLOHOSP",    # Apollo Hospitals
    "BPCL",          # Bharat Petroleum
    "BRITANNIA",     # Britannia Industries
    "TATACONSUM",    # Tata Consumer Products
    "SBILIFE",       # SBI Life Insurance
    "HDFCLIFE",      # HDFC Life Insurance
    "SHRIRAMFIN",    # Shriram Finance
    "LTIM",          # LTIMindtree
    "BEL",           # Bharat Electronics
]

# Nifty Next 50 (mid-large caps)
NIFTY_NEXT_50 = [
    "VEDL",          # Vedanta
    "ADANIPOWER",    # Adani Power
    "JINDALSTEL",    # Jindal Steel & Power
    "DLF",           # DLF Limited
    "GODREJCP",      # Godrej Consumer Products
    "SIEMENS",       # Siemens
    "PIDILITIND",    # Pidilite Industries
    "BANKBARODA",    # Bank of Baroda
    "TATAPOWER",     # Tata Power
    "PNB",           # Punjab National Bank
    "AMBUJACEM",     # Ambuja Cements
    "ACC",           # ACC Limited
    "MCDOWELL-N",    # United Spirits
    "DABUR",         # Dabur India
    "HAVELLS",       # Havells India
    "BERGEPAINT",    # Berger Paints
    "INDIGO",        # InterGlobe Aviation (IndiGo)
    "SAIL",          # Steel Authority of India
    "NMDC",          # NMDC Limited
    "IOC",           # Indian Oil Corporation
    "ICICIGI",       # ICICI Lombard General Insurance
    "BAJAJ-AUTO",    # Bajaj Auto
    "CHOLAFIN",      # Cholamandalam Investment
    "ABB",           # ABB India
    "LICHSGFIN",     # LIC Housing Finance
    "LUPIN",         # Lupin
    "MARICO",        # Marico
    "MOTHERSON",     # Samvardhana Motherson
    "TORNTPHARM",    # Torrent Pharmaceuticals
    "TVSMOTOR",      # TVS Motor Company
    "PAGEIND",       # Page Industries
    "BOSCHLTD",      # Bosch Limited
    "COLPAL",        # Colgate-Palmolive India
    "SRF",           # SRF Limited
    "ZYDUSLIFE",     # Zydus Lifesciences
    "GAIL",          # GAIL (India)
    "ESCORTS",       # Escorts Kubota
    "CUMMINSIND",    # Cummins India
    "BALKRISIND",    # Balkrishan Industries
    "MUTHOOTFIN",    # Muthoot Finance
    "HINDPETRO",     # Hindustan Petroleum
    "OFSS",          # Oracle Financial Services
    "BIOCON",        # Biocon
    "CONCOR",        # Container Corporation of India
    "NAUKRI",        # Info Edge (Naukri)
    "IRCTC",         # Indian Railway Catering & Tourism
    "ZOMATO",        # Zomato
    "PAYTM",         # Paytm (One97 Communications)
    "POLICYBZR",     # PB Fintech (PolicyBazaar)
    "DMART",         # Avenue Supermarts (DMart)
]

# Combine to get top 100
NSE_TOP_100 = NIFTY_50 + NIFTY_NEXT_50

# Sector classification for diversification
SECTORS = {
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "TORNTPHARM"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
    "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "JINDALSTEL", "SAIL"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "COALINDIA", "NTPC"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "CHOLAFIN"],
    "Telecom": ["BHARTIARTL"],
    "Cement": ["ULTRACEMCO", "AMBUJACEM", "ACC"],
}

def get_nse_100():
    """Get list of top 100 NSE stocks."""
    return NSE_TOP_100.copy()

def get_nifty_50():
    """Get list of Nifty 50 stocks."""
    return NIFTY_50.copy()

def get_stocks_by_sector(sector: str):
    """Get stocks from a specific sector."""
    return SECTORS.get(sector, []).copy()

def get_diversified_portfolio(n_stocks: int = 20, n_per_sector: int = 3):
    """
    Get a diversified portfolio by selecting stocks from different sectors.

    Args:
        n_stocks: Total number of stocks to select
        n_per_sector: Max stocks per sector

    Returns:
        List of stock symbols
    """
    portfolio = []
    for sector, stocks in SECTORS.items():
        # Take up to n_per_sector from each sector
        sector_picks = stocks[:min(n_per_sector, len(stocks))]
        portfolio.extend(sector_picks)

        if len(portfolio) >= n_stocks:
            break

    return portfolio[:n_stocks]
