# config/loader_config.py

# Stock tickers configuration
TICKERS_IND = {
    "Information Technology": {
        "AAPL", "ACN", "ADBE", "ADI", "ADSK", "ADP", "AKAM", "AMAT", "AMD", "ANET", "ANSS", 
        "APH", "AVGO", "BR", "CDNS", "CDW", "CRM", "CRWD", "CSCO", "CTSH", "DELL", "ENPH", 
        "EPAM", "FFIV", "FICO", "FSLR", "FTNT", "GDDY", "GEN", "GRMN", "HPE", "HPQ", "IBM", 
        "INTC", "INTU", "IT", "JNPR", "KEYS", "KLAC", "LRCX", "MCHP", "MPWR", "MSI", "MSFT", 
        "MU", "NOW", "NTAP", "NVDA", "NXPI", "ON", "ORCL", "PANW", "PAYC", "PAYX", "PLTR", 
        "PTC", "QCOM", "SMCI", "SNPS", "STX", "SWKS", "TEL", "TER", "TXN", "TYL", "VRSN", 
        "VRSK", "WDC", "WDAY", "ZBRA"
    },
    
    "Health Care": {
        "ABBV", "ABT", "ALGN", "AMGN", "BAX", "BDX", "BIIB", "BMY", "BSX", "CAH", "CI", 
        "CNC", "COO", "CRL", "CVS", "DGX", "DHR", "DVA", "DXCM", "ELV", "EW", "GEHC", "GILD", 
        "HCA", "HOLX", "HSIC", "HUM", "IDXX", "INCY", "IQV", "ISRG", "JNJ", "LH", "LLY", 
        "MCK", "MDT", "MOH", "MRK", "MRNA", "MTD", "PFE", "PODD", "REGN", "RMD", "STE", 
        "SYK", "TECH", "TMO", "UHS", "UNH", "VTRS", "VRTX", "WST", "ZBH", "ZTS", "DAY", "SW"
    },
    
    "Financials": {
        "BAC", "GL", "RVTY", "TRR", "VICI", "WT"
    },
    
    "Communication Services": {
        "ATVI", "CHTR", "CMCSA", "DISH", "EA", "FOX", "FOXA", "GOOG", "GOOGL", "IPG", "LBTYA", 
        "META", "MTCH", "NFLX", "NWS", "NWSA", "OMC", "PARA", "ROKU", "T", "TTWO", "TMUS", 
        "VZ", "WBD"
    },
    
    "Consumer Discretionary": {
        "ABNB", "AMZN", "APTV", "AMC", "AZO", "BBY", "BKNG", "BLDR", "BWA", "CCL", "CMG", 
        "CZR", "DECK", "DG", "DHI", "DIS", "DLTR", "DPZ", "DRI", "EBAY", "EXPE", "F", "FXLV", 
        "GM", "GPC", "HAS", "HD", "HLT", "KMX", "KVUE", "LEN", "LKQ", "LOW", "LULU", "LVS", 
        "LYV", "MAR", "MAS", "MCD", "MGM", "MHK", "NCLH", "NKE", "NVR", "ORLY", "PHM", "POOL", 
        "RCL", "RL", "ROST", "SBUX", "TGT", "TJX", "TPR", "TSCO", "TSLA", "UBER", "ULTA", 
        "WHR", "WYNN", "YUM"
    },
    
    "Industrials": {
        "A", "ALLE", "AME", "AXON", "BA", "BALL", "BKR", "CARR", "CAT", "CBRE", "CHRW", "CMI", 
        "CPRT", "CSX", "CTAS", "DAL", "DE", "DOV", "EFX", "EMR", "ETN", "EXPD", "FAST", "FDX", 
        "GD", "GE", "GEV", "GNRC", "GWW", "HII", "HON", "HUBB", "HWM", "IEX", "IR", "IRM", 
        "ITW", "J", "JBL", "JBHT", "JCI", "LDOS", "LHX", "LII", "LMT", "LUV", "MMM", "NDSN", 
        "NOC", "NSC", "ODFL", "OTIS", "PCAR", "PH", "PNR", "PWR", "ROK", "ROL", "ROP", "RSG", 
        "RTX", "SNA", "SWK", "TDG", "TDY", "TFX", "TRGP", "TRMB", "TT", "TXT", "UAL", "UNP", 
        "UPS", "URI", "WAB", "WAT", "WM", "XYL", "AOS"
    },
    
    "Consumer Staples": {
        "ADM", "AMCR", "BF.B", "BG", "CAG", "CHD", "CL", "CLX", "COST", "CPB", "CTVA", "EL", 
        "GIS", "HRL", "HSY", "K", "KDP", "KHC", "KMB", "KO", "KR", "LW", "MDLZ", "MKC", "MNST", 
        "MO", "PEP", "PG", "PM", "POST", "SJM", "STZ", "SYY", "TAP", "TSN", "WBA", "WMT"
    },
    
    "Energy": {
        "APA", "COP", "CTRA", "CVX", "DVN", "EOG", "EQT", "FANG", "HAL", "HES", "KMI", "MPC", 
        "OKE", "OXY", "PSX", "PXD", "SLB", "TPL", "VLO", "WMB", "XOM"
    },
    
    "Utilities": {
        "AEE", "AEP", "AES", "ATO", "AWK", "CEG", "CMS", "CNP", "D", "DTE", "DUK", "ED", "EIX", 
        "ES", "ETR", "EVRG", "EXC", "FE", "LNT", "NEE", "NI", "NRG", "PCG", "PEG", "PNW", "PPL", 
        "SO", "SRE", "VST", "WEC", "XEL"
    },
    
    "Real Estate": {
        "AMT", "ARE", "AVB", "BXP", "CCI", "CPT", "COR", "CSGP", "DLR", "DOC", "EQIX", "EQR", 
        "ESS", "EXR", "FRT", "HST", "INVH", "KIM", "MAA", "O", "PLD", "PSA", "REG", "SBAC", 
        "SPG", "UDR", "VTR", "WELL"
    },
    
    "Materials": {
        "ALB", "APD", "AVY", "CE", "CF", "DD", "DOW", "ECL", "EMN", "FCX", "FMC", "IFF", "IP", 
        "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SHW", "SOLV", "STLD", "VLTO", 
        "VMC"
    }}

    # S&P 500 companies excluding financials
TICKERS = {
    'sp500_ex_financials': {
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADSK', 'ADP', 'AEE', 
        'AEP', 'AES', 'AKAM', 'ALB', 'ALGN', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMC', 
        'AMZN', 'AMT', 'ANET', 'ANSS', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 
        'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AZO', 'BA', 'BALL', 'BAX', 'BBY', 'BDX', 'BF.B', 
        'BG', 'BIIB', 'BKR', 'BKNG', 'BLDR', 'BMY', 'BR', 'BSX', 'BWA', 'BXP', 'CAG', 'CAH', 
        'CAT', 'CARR', 'CCI', 'CBRE', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CHD', 'CHRW', 
        'CHTR', 'CI', 'CL', 'CLX', 'CMG', 'CMI', 'CMS', 'CMCSA', 'CNC', 'CNP', 'COO', 'COP', 
        'COR', 'COST', 'CPRT', 'CPB', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 
        'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DECK', 'DE', 
        'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 
        'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 
        'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 
        'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPE', 'EXPD', 'EXR', 'F', 'FANG', 'FAST', 
        'FCX', 'FDX', 'FE', 'FFIV', 'FICO', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FXLV', 
        'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GM', 'GNRC', 'GOOG', 'GOOGL', 
        'GPC', 'GRMN', 'GWW', 'HAL', 'HAS', 'HCA', 'HD', 'HES', 'HII', 'HLT', 'HOLX', 'HON', 
        'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'IDXX', 'IEX', 
        'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 
        'ITW', 'J', 'JBL', 'JBHT', 'JCI', 'JNJ', 'JNPR', 'K', 'KDP', 'KEYS', 'KHC', 'KIM', 
        'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'LBTYA', 'LDOS', 'LEN', 'LH', 'LHX', 
        'LII', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 
        'LYB', 'LYV', 'MAA', 'MAR', 'MAS', 'MCD', 'MCK', 'MCHP', 'MDLZ', 'MDT', 'META', 'MGM', 
        'MHK', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 
        'MSI', 'MSFT', 'MTD', 'MTCH', 'MU', 'NCLH', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 
        'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 
        'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PCAR', 
        'PCG', 'PAYC', 'PAYX', 'PEG', 'PEP', 'PFE', 'PG', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 
        'PM', 'PNR', 'PNW', 'PODD', 'POOL', 'POST', 'PPG', 'PPL', 'PSA', 'PSX', 'PTC', 'PXD', 
        'PWR', 'QCOM', 'RCL', 'REG', 'REGN', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROKU', 'ROST', 
        'RSG', 'RTX', 'SBAC', 'SBUX', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 
        'SPG', 'SRE', 'STE', 'STLD', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYK', 'SYY', 'T', 
        'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPL', 
        'TPR', 'TRGP', 'TRMB', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UBER', 
        'UAL', 'UDR', 'UHS', 'ULT', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'VLO', 'VLTO', 'VMC', 
        'VRSN', 'VRSK', 'VRTX', 'VST', 'VTRS', 'VTR', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 
        'WDAY', 'WEC', 'WELL', 'WHR', 'WM', 'WMB', 'WMT', 'WST', 'WYNN', 'XEL', 'XOM', 'XYL', 
        'YUM', 'ZBH', 'ZBRA', 'ZTS'
    },
}

# Date ranges
DATE_RANGES = {
    'recent': {
        'start': '2020-01-01',
        'end': '2024-12-31'
    },
    'full_period': {
        'start': '2005-01-01',
        'end': '2024-12-31'
    }
}


class LoaderConfig:
    @staticmethod
    def get_ticker_group(group_name):
        if group_name not in TICKERS:
            raise ValueError(f"Ticker group '{group_name}' not found")
        return TICKERS[group_name]
    
    @staticmethod
    def get_date_range(range_name):
        if range_name not in DATE_RANGES:
            raise ValueError(f"Date range '{range_name}' not found")
        return DATE_RANGES[range_name]
    


    
