import wrds
import pandas as pd
from datetime import datetime, timedelta

class WRDSLoader:
    def __init__(self, username):
        """Initialize WRDS connection with username."""
        self.db = wrds.Connection(wrds_username=username)
        self.db.create_pgpass_file()
    
    def get_fundamental_data(self, tickers, start_date=None, end_date=None, fields=None):
        """
        Get fundamental data for specified tickers from Compustat.
        
        Parameters:
        -----------
        tickers : str or list
            Single ticker or list of tickers
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        fields : list, optional
            List of fundamental fields to retrieve. If None, fetches a default set.
        
        Returns:
        --------
        pandas.DataFrame
            Dataframe containing fundamental data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        if fields is None:
            fields = ['atq', 'ltq', 'seqq', 'ibq', 'cshoq', 'epspxq', 'revtq', 'niq', 'cogsq', 
                      'capsq', 'cheq', 'dlttq', 'dlcq', 'oiadpq', 'prccq', 'dvpsxq', 
                      'actq', 'lctq', 'txpq', 'txtq', 'dpq', 'invtq', 'pstkq', 'ajexq']
        
        # Build field selection string  
        field_selection = ', '.join([f'f.{field}' for field in fields])
        
        query = f"""
            WITH ticker_permnos AS (
            SELECT DISTINCT permno, ticker
            FROM crsp.stocknames
            WHERE ticker IN ({','.join([f"'{t.upper()}'" for t in tickers])})
            ),
            gvkey_links AS (
            SELECT l.gvkey, p.ticker, p.permno, l.linkdt, l.linkenddt
            FROM crsp_a_ccm.ccmxpf_linktable l
            JOIN ticker_permnos p ON l.lpermno = p.permno
            WHERE l.linktype IN ('LU', 'LC')
            AND l.linkprim IN ('P', 'C')
            )
            SELECT
            f.gvkey,
            f.datadate,
            g.ticker,
            g.permno,
            f.fyearq,
            {field_selection}
            FROM comp.fundq f
            JOIN gvkey_links g ON f.gvkey = g.gvkey
            WHERE f.indfmt = 'INDL'
            AND f.datafmt = 'STD'
            AND f.consol = 'C'
            AND f.datadate BETWEEN g.linkdt AND COALESCE(g.linkenddt, '2099-12-31')
            """
        
        if start_date:
            query += f" AND f.datadate >= '{start_date}'"
        if end_date:
            query += f" AND f.datadate <= '{end_date}'"
        
        query += " ORDER BY g.ticker, f.datadate"
        
        return self.db.raw_sql(query)

    def get_fama_french_factors(self, start_date=None, end_date=None, frequency='daily', model='five_factor'):
        """
        Get Fama-French factor data from WRDS.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        frequency : str, optional (default='daily')
            Data frequency: 'daily' or 'monthly'
        model : str, optional (default='five_factor')
            Factor model: 'three_factor' or 'five_factor'
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe containing Fama-French factors
        """
        # For five-factor model (with all 7 factors)
        if model in ['five_factor', '5_factor']:
            if frequency == 'daily':
                table = 'ff.fivefactors_daily'
            elif frequency == 'monthly':
                table = 'ff.fivefactors_monthly'
            else:
                raise ValueError("Frequency must be 'daily' or 'monthly'")
        # For three-factor model
        elif model in ['three_factor', '3_factor']:
            if frequency == 'daily':
                table = 'ff.factors_daily'
            elif frequency == 'monthly':
                table = 'ff.factors_monthly'
            else:
                raise ValueError("Frequency must be 'daily' or 'monthly'")
        else:
            raise ValueError("Model must be 'three_factor' or 'five_factor'")
        
        # Build query - explicitly select columns for clarity
        query = f"""
            SELECT date, mktrf, smb, hml"""
        
        # Add additional factors for five-factor model
        if model in ['five_factor', '5_factor']:
            query += ", rmw, cma"
        
        # Add rf and umd which are in both tables
        query += ", rf, umd"
        
        # Complete the query
        query += f"""
            FROM {table}"""
        
        if start_date:
            query += f" WHERE date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
        elif end_date:
            query += f" WHERE date <= '{end_date}'"
            
        query += f" ORDER BY date ASC"
        
        # Get the data
        df = self.db.raw_sql(query)
        
        return df

    def get_monthly_stock_data(self, tickers, start_date=None, end_date=None):
        """
        Get monthly stock price data for specified tickers from CRSP MSF.
        
        Parameters:
        -----------
        tickers : str or list
            Single ticker or list of tickers
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        query = f"""
            WITH ticker_permnos AS (
                SELECT DISTINCT permno, comnam, ticker, namedt, nameenddt
                FROM crsp.stocknames
                WHERE ticker IN ({','.join([f"'{t.upper()}'" for t in tickers])})
            )
            SELECT 
                d.permno,
                d.date,
                s.ticker,
                d.prc,
                CASE WHEN d.prc < 0 THEN ABS(d.prc) ELSE d.prc END as price_unadj,
                d.ret,     -- Monthly return including distributions
                d.retx,    -- Monthly return excluding distributions
                d.vol,     -- Monthly volume
                d.shrout,  -- Shares outstanding
                d.cfacpr as price_adjustment_factor,
                d.cfacshr as shares_adjustment_factor
            FROM crsp.msf d  -- Using monthly stock file instead of daily
            JOIN ticker_permnos s ON d.permno = s.permno
            WHERE d.date BETWEEN s.namedt AND s.nameenddt
        """
        
        if start_date:
            query += f" AND d.date >= '{start_date}'"
        if end_date:
            query += f" AND d.date <= '{end_date}'"
            
        query += " ORDER BY d.date ASC"
        
        # Get the data
        df = self.db.raw_sql(query)
            
        return df
    
    def get_macro_factors(self, start_date=None, end_date=None, country='USA'):
        """
        Get macroeconomic factors from WRDS databases.
        
        Retrieves:
        - Interest rates (Treasury rates, Fed Funds Rate)
        - Economic growth (GDP)
        - Inflation measures (CPI)
        - Unemployment data
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        country : str, optional (default='USA')
            Country ISO code for economic indicators (e.g., 'USA', 'CAN')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing macroeconomic factors
        """
        # Initialize empty DataFrame for macro data
        macro_data = pd.DataFrame()
        data_sources = []
        
        # Step 1: Get interest rates from rates_monthly table
        try:
            query = """
            SELECT 
                date, 
                tb3ms AS tb3m_rate, 
                gs10 AS tb10y_rate, 
                fedfunds AS fed_funds_rate,
                tb1yr AS tb1y_rate,
                gs2 AS gs2_rate,
                aaa AS aaa_rate,
                baa AS baa_rate
            FROM frb.rates_monthly
            WHERE 1=1
            """
            
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
                
            query += " ORDER BY date"
            
            interest_data = self.db.raw_sql(query)
            
            if not interest_data.empty:
                print(f"Successfully retrieved interest rates: {len(interest_data)} rows")
                
                # Calculate term spread (10-year minus 3-month)
                interest_data['term_spread'] = interest_data['tb10y_rate'] - interest_data['tb3m_rate']
                
                # Calculate credit spread (BAA minus AAA)
                if 'aaa_rate' in interest_data.columns and 'baa_rate' in interest_data.columns:
                    interest_data['credit_spread'] = interest_data['baa_rate'] - interest_data['aaa_rate']
                    
                data_sources.append(interest_data)
        except Exception as e:
            print(f"Error retrieving interest rates: {e}")
        
        # Step 2: Get economic indicators with correct columns
        try:
            # Query for economic indicators using the actual column structure
            econ_query = f"""
            SELECT
                Datadate as date,
                gdpr1,
                gdpr2,
                cpir,
                unemp
            FROM comp_na_daily_all.ecind_mth
            WHERE econiso = '{country}'
            """
            
            if start_date:
                econ_query += f" AND Datadate >= '{start_date}'"
            if end_date:
                econ_query += f" AND Datadate <= '{end_date}'"
                
            econ_query += " ORDER BY Datadate"
            
            econ_data = self.db.raw_sql(econ_query)
            
            if not econ_data.empty:
                print(f"Successfully retrieved economic indicators: {len(econ_data)} rows")
                
                # Rename columns for clarity
                econ_data.rename(columns={
                    'gdpr1': 'gdp_monthly',
                    'gdpr2': 'gdp_quarterly',
                    'cpir': 'cpi',
                    'unemp': 'unemployment_rate'
                }, inplace=True)
                
                # Calculate growth rates
                if 'gdp_monthly' in econ_data.columns:
                    econ_data['gdp_growth_yoy'] = econ_data['gdp_monthly'].pct_change(12) * 100
                
                if 'gdp_quarterly' in econ_data.columns:
                    econ_data['gdp_growth_qoq'] = econ_data['gdp_quarterly'].pct_change(1) * 100
                
                if 'cpi' in econ_data.columns:
                    econ_data['inflation_yoy'] = econ_data['cpi'].pct_change(12) * 100
                
                if 'unemployment_rate' in econ_data.columns:
                    econ_data['unemployment_change'] = econ_data['unemployment_rate'].diff()
                
                data_sources.append(econ_data)
        except Exception as e:
            print(f"Error retrieving economic indicators: {e}")
        
        # Merge all data sources
        if len(data_sources) > 1:
            macro_data = data_sources[0]
            
            for df in data_sources[1:]:
                macro_data = pd.merge(macro_data, df, on='date', how='outer')
        elif len(data_sources) == 1:
            macro_data = data_sources[0]
        else:
            print("No macroeconomic data could be retrieved")
            return pd.DataFrame()
        
        # Sort by date and forward fill to handle different frequencies
        macro_data = macro_data.sort_values('date')
        
        # Create year_month for easier merging with monthly stock data
        macro_data['year_month'] = pd.to_datetime(macro_data['date']).dt.to_period('M').dt.to_timestamp('M')
        
        # Forward fill missing values
        macro_data = macro_data.ffill()
        
        print(f"Final macro data has {len(macro_data)} rows and columns: {macro_data.columns.tolist()}")
        
        return macro_data
    
    def explore_tables(self):
        """Helper function to explore the Compustat schema"""
        # Get table structure
        query = """
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = 'comp'
        ORDER BY table_name
        """
        tables = self.db.raw_sql(query)
        print("Available tables in comp schema:", tables['table_name'].tolist())
        
        # Get columns for company table
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'comp' AND table_name = 'company'
        ORDER BY column_name
        """
        columns = self.db.raw_sql(query)
        print("Columns in comp.company:", columns['column_name'].tolist())
        
        return tables, columns
    
    def close_connection(self):
        """Close the WRDS connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()


