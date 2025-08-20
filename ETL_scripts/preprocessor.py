# ETL_scripts/preprocessor.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:

    @staticmethod
    def check_wrds_coverage(df, start_date, end_date):
        """
        Check which tickers have incomplete monthly data from WRDS,
        including checks for complete date range coverage.
        
        Parameters:
        df: DataFrame from WRDS with columns including 'date' and 'ticker'/'permno'
        start_date: str, start date in 'YYYY-MM-DD' format
        end_date: str, end date in 'YYYY-MM-DD' format
        
        Returns:
        dict: Dictionary with incomplete tickers and their coverage details
        """
        # Convert dates to datetime if they aren't already
        df['date'] = pd.to_datetime(df['date'])
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Create expected monthly dates
        expected_dates = pd.date_range(start=start_date,
                                    end=end_date,
                                    freq='M')
        
        # Identify ticker column (could be 'ticker' or 'permno')
        ticker_col = 'ticker' if 'ticker' in df.columns else 'permno'
        
        incomplete_tickers = {}
        
        # Check each ticker
        for ticker in df[ticker_col].unique():
            ticker_data = df[df[ticker_col] == ticker]
            ticker_dates = pd.DatetimeIndex(ticker_data['date'].unique())
            
            # Find missing months
            missing_dates = expected_dates.difference(ticker_dates)
            
            # Check first and last appearance dates
            first_appearance = ticker_dates.min()
            last_appearance = ticker_dates.max()
            
            # Calculate time differences in months
            start_diff_months = (first_appearance.year - start_date_dt.year) * 12 + (first_appearance.month - start_date_dt.month)
            end_diff_months = (end_date_dt.year - last_appearance.year) * 12 + (end_date_dt.month - last_appearance.month)
            
            # Mark as incomplete only if start or end dates don't match the expected range
            if start_diff_months > 0 or end_diff_months > 0:
                incomplete_tickers[ticker] = {
                    'first_appearance': first_appearance,
                    'last_appearance': last_appearance,
                    'starts_on_time': start_diff_months == 0,
                    'ends_on_time': end_diff_months == 0,
                    'months_missing_at_start': start_diff_months,
                    'months_missing_at_end': end_diff_months,
                    'missing_internal_months': len(missing_dates),
                    'missing_dates': missing_dates,
                    'coverage_percentage': (len(ticker_dates) / len(expected_dates)) * 100
                }
        
        return incomplete_tickers
    
    @staticmethod
    def print_coverage_summary(incomplete_tickers, start_date, end_date):
        """
        Print a summary of tickers with incomplete date range coverage.
        
        Parameters:
        incomplete_tickers: dict, output from check_wrds_coverage function
        start_date: str, expected start date in 'YYYY-MM-DD' format
        end_date: str, expected end date in 'YYYY-MM-DD' format
        """
        print(f"\nFound {len(incomplete_tickers)} tickers with incomplete date range coverage:")
        print(f"Expected date range: {pd.to_datetime(start_date).strftime('%Y-%m')} to {pd.to_datetime(end_date).strftime('%Y-%m')}")
        
        if len(incomplete_tickers) == 0:
            print("All tickers have complete coverage for the specified date range.")
            return
            
        # Sort tickers by their coverage percentage
        sorted_tickers = sorted(incomplete_tickers.items(), 
                                key=lambda x: x[1]['coverage_percentage'])
        
        for ticker, details in sorted_tickers:
            print(f"\nTicker: {ticker}")
            print(f"Actual period: {details['first_appearance'].strftime('%Y-%m')} to {details['last_appearance'].strftime('%Y-%m')}")
            
            if details['months_missing_at_start'] > 0:
                print(f"Missing {details['months_missing_at_start']} months at the beginning")
            
            if details['months_missing_at_end'] > 0:
                print(f"Missing {details['months_missing_at_end']} months at the end")
                
            print(f"Coverage: {details['coverage_percentage']:.2f}%")
    
    @staticmethod
    def filter_complete_data(df, incomplete_tickers, start_date, end_date):
        """
        Filter out tickers with incomplete date range coverage and return a new DataFrame 
        containing only tickers with complete coverage.
        
        Parameters:
        df: DataFrame from WRDS with columns including 'date' and 'ticker'/'permno'
        incomplete_tickers: dict, output from check_wrds_coverage function
        start_date: str, expected start date in 'YYYY-MM-DD' format
        end_date: str, expected end date in 'YYYY-MM-DD' format
        
        Returns:
        pandas.DataFrame: Filtered DataFrame with only complete tickers
        """
        # Identify ticker column (could be 'ticker' or 'permno')
        ticker_col = 'ticker' if 'ticker' in df.columns else 'permno'
        
        # Get list of incomplete tickers
        incomplete_ticker_list = list(incomplete_tickers.keys())
        
        # Get all unique tickers
        all_tickers = df[ticker_col].unique()
        
        # Calculate number of tickers being removed
        num_incomplete = len(incomplete_ticker_list)
        num_total = len(all_tickers)
        
        print(f"Filtering out {num_incomplete} tickers with incomplete coverage " 
              f"({num_incomplete/num_total*100:.2f}% of {num_total} total tickers)")
        
        # Create filtered DataFrame with only tickers that have complete coverage
        filtered_df = df[~df[ticker_col].isin(incomplete_ticker_list)].copy()
        
        # Print summary of filtered data
        num_remaining = len(filtered_df[ticker_col].unique())
        print(f"Remaining tickers: {num_remaining}")
        print(f"Rows in original DataFrame: {len(df):,}")
        print(f"Rows in filtered DataFrame: {len(filtered_df):,}")
        print(f"Removed {len(df) - len(filtered_df):,} rows ({(len(df) - len(filtered_df))/len(df)*100:.2f}% of data)")
        
        return filtered_df

    def filter_by_observation_count(df, start_date, end_date, expected_count=180):
        """
        Filter data to include only tickers with the expected number of observations.
        
        Parameters:
        df: DataFrame with ticker and date columns
        start_date: str, start date in 'YYYY-MM-DD' format
        end_date: str, end date in 'YYYY-MM-DD' format
        expected_count: int, expected number of observations per ticker
        
        Returns:
        tuple: (filtered_df, excluded_tickers_dict)
            - filtered_df: DataFrame containing only tickers with expected number of observations
            - excluded_tickers_dict: Dictionary with excluded tickers and their observation counts
        """
        # Identify ticker column
        ticker_col = 'ticker' if 'ticker' in df.columns else 'permno'
        
        # Convert dates to datetime if not already
        df['date'] = pd.to_datetime(df['date'])
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Filter data for the date range
        date_filtered_df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
        
        # Count observations per ticker
        ticker_counts = date_filtered_df[ticker_col].value_counts()
        
        # Identify tickers with expected count
        compliant_tickers = ticker_counts[ticker_counts == expected_count].index
        
        # Create a dictionary of excluded tickers with their counts
        excluded_tickers = {}
        for ticker, count in ticker_counts[ticker_counts != expected_count].items():
            excluded_tickers[ticker] = count
        
        # Filter the DataFrame
        filtered_df = date_filtered_df[date_filtered_df[ticker_col].isin(compliant_tickers)].copy()
        
        # Print summary
        total_tickers = len(ticker_counts)
        kept_tickers = len(compliant_tickers)
        removed_tickers = total_tickers - kept_tickers
        
        print(f"Date range: {start_date} to {end_date}")
        print(f"Expected observations per ticker: {expected_count}")
        print(f"Total tickers: {total_tickers}")
        print(f"Tickers with exact {expected_count} observations: {kept_tickers} ({kept_tickers/total_tickers*100:.2f}%)")
        print(f"Excluded tickers: {removed_tickers} ({removed_tickers/total_tickers*100:.2f}%)")
        print(f"Original row count: {len(date_filtered_df)}")
        print(f"Filtered row count: {len(filtered_df)}")
        
        # Show distribution of observation counts
        count_distribution = ticker_counts.value_counts().sort_index()
        print("\nDistribution of observation counts:")
        for count, freq in count_distribution.items():
            print(f"{count} observations: {freq} tickers")
        
        return filtered_df, excluded_tickers

    # This is a good function that is very useful
    @staticmethod
    def adjust_prices(df):
        """Adjust prices for splits and dividends"""
        if not df.empty:
            # Convert negative prices to positive
            df['price_unadj'] = df['price_unadj'].abs()
            
            # Calculate adjusted prices
            df['price_adjusted'] = df['price_unadj'] / df['price_adjustment_factor']
            df['volume_adjusted'] = df['vol'] * df['shares_adjustment_factor']
            df['shares_outstanding_adjusted'] = df['shrout'] * df['shares_adjustment_factor']
            
            # Calculate market cap
            df['market_cap'] = df['price_adjusted'] * df['shares_outstanding_adjusted'] * 1000
            
            # Sort values
            df = df.sort_values(['ticker', 'date'])
            
            # Calculate cumulative returns
            df['cum_ret'] = (1 + df.groupby('ticker')['ret'].transform(
                lambda x: (1 + x).cumprod() - 1
            ))
        
        return df

    @staticmethod
    # also a function that can help, and something that should be discussed in a feature engineering part
    def calculate_financial_ratios(merged_data):
        """
        Calculate financial ratios from merged stock and fundamental data.
        
        Parameters:
        -----------
        merged_data : pandas.DataFrame
            DataFrame containing merged stock and fundamental data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added financial ratios
        """
        data = merged_data.copy()
        
        # Leverage ratios
        if all(col in data.columns for col in ['seqq', 'ltq']):
            data['debt_equity'] = data['ltq'] / data['seqq']
        
        # Debt to assets
        if all(col in data.columns for col in ['ltq', 'atq']):
            data['leverage'] = data['ltq'] / data['atq']
        
        # Valuation ratios
        if all(col in data.columns for col in ['ibq', 'cshoq', 'price_adjusted']):
            # Earnings yield (inverse of PE-ratio)
            data['earnings_yield'] =  data['epspxq'] / data['price_adjusted']
        
            # Book to price ratio (Inverse of PB-ratio)
        if all(col in data.columns for col in ['seqq', 'cshoq', 'price_adjusted']):
            data['bp_ratio'] =  (data['seqq'] / data['cshoq']) / data['price_adjusted'] 
        
        # Profitability ratios
        if all(col in data.columns for col in ['niq', 'atq']):
            data['roa'] = data['niq'] / data['atq']
        
        if all(col in data.columns for col in ['niq', 'seqq']):
            data['roe'] = data['niq'] / data['seqq']
        
        if all(col in data.columns for col in ['oiadpq', 'atq']):
            data['operating_margin'] = data['oiadpq'] / data['revtq'] if 'revtq' in data.columns else np.nan
        
        # Liquidity ratios
        if all(col in data.columns for col in ['actq', 'lctq']):
            data['current_ratio'] = data['actq'] / data['lctq']
        
        if all(col in data.columns for col in ['actq', 'invtq', 'lctq']):
            data['quick_ratio'] = (data['actq'] - data['invtq']) / data['lctq']
        
        # Efficiency ratios
        if all(col in data.columns for col in ['revtq', 'atq']):
            data['asset_turnover'] = data['revtq'] / data['atq']
        
        # Return on Invested Capital (ROIC)
        if all(col in data.columns for col in ['oiadpq', 'txtq', 'atq', 'lctq', 'cheq']):
            # Calculate effective tax rate
            data['tax_rate'] = data['txtq'] / (data['ibq'] + 0.000001)  # Adding small value to avoid division by zero
            # Cap tax rate between 0 and 1 to handle unusual values
            data['tax_rate'] = data['tax_rate'].clip(0, 1)
                
            # Calculate NOPAT (Net Operating Profit After Tax)
            data['nopat'] = data['oiadpq'] * (1 - data['tax_rate'])
                
            # Calculate Invested Capital
            data['invested_capital'] = data['atq'] - data['lctq'] - data['cheq']
                
            # Calculate ROIC
            data['roic'] = data['nopat'] / data['invested_capital']

        # 2. NOPLAT Growth Rate
        if all(col in data.columns for col in ['nopat', 'invested_capital', 'revtq']):
            # First ensure data is sorted properly
            data = data.sort_values(['ticker', 'date_stock'])

            # Group by ticker and quarter_date to calculate quarterly changes
            quarterly_grouped = data.groupby(['ticker', 'quarter_date']).first().reset_index()
            quarterly_grouped = quarterly_grouped.sort_values(['ticker', 'quarter_date'])

            # Calculate growth metrics on this quarterly data
            quarterly_growth = quarterly_grouped.groupby('ticker')

            # Now calculate your growth metrics
            quarterly_growth_data = quarterly_grouped.copy()
            quarterly_growth_data['nopat_growth_qoq'] = quarterly_growth['nopat'].pct_change(1)
            quarterly_growth_data['nopat_growth_yoy'] = quarterly_growth['nopat'].pct_change(4)
            quarterly_growth_data['invested_capital_growth_qoq'] = quarterly_growth['invested_capital'].pct_change(1)
            quarterly_growth_data['invested_capital_growth_yoy'] = quarterly_growth['invested_capital'].pct_change(4)
            quarterly_growth_data['revenue_growth_qoq'] = quarterly_growth['revtq'].pct_change(1)
            quarterly_growth_data['revenue_growth_yoy'] = quarterly_growth['revtq'].pct_change(4)

            # Then merge these growth metrics back to your monthly data
            data = data.merge(
                quarterly_growth_data[['ticker', 'quarter_date', 'nopat_growth_qoq', 'nopat_growth_yoy', 
                                    'invested_capital_growth_qoq', 'invested_capital_growth_yoy',
                                    'revenue_growth_qoq', 'revenue_growth_yoy']],
                on=['ticker', 'quarter_date'],
                how='left'
            )
                
        # Handle inf and NaN values
        for col in data.columns:
            if data[col].dtype.kind in 'fc':  # Float or complex
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        
        return data
            

    @staticmethod
    def expand_quarterly_to_monthly(fundamentals_df, start_date='2005-01-01', end_date='2024-12-31'):
        """
        Expands quarterly fundamental data to monthly with a 1-month lag to avoid look-ahead bias.
        Ensures complete date range from start_date to end_date and properly fills missing values.
        
        Parameters:
        -----------
        fundamentals_df : pandas.DataFrame
            DataFrame containing quarterly fundamental data from Compustat
        start_date : str
            Desired start date in 'YYYY-MM-DD' format
        end_date : str
            Desired end date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pandas.DataFrame
            Expanded fundamental data with monthly frequency and complete date range
        """
        # Make a copy to avoid modifying the original
        fundamentals = fundamentals_df.copy()
        
        # Ensure date column is datetime
        fundamentals['datadate'] = pd.to_datetime(fundamentals['datadate'])
        
        # Add a 1-month lag to implement point-in-time data with realistic availability
        fundamentals['available_date'] = fundamentals['datadate'] + pd.DateOffset(months=1)
        
        # Sort by ticker and date
        fundamentals = fundamentals.sort_values(['ticker', 'available_date'])
        
        # Convert desired date range to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Create full date range with month starts (MS)
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Get all unique tickers
        all_tickers = fundamentals['ticker'].unique()
        print(f"Processing {len(all_tickers)} unique tickers")
        
        # Initialize list to store monthly expanded data
        monthly_fundamentals = []
        
        # Process each ticker separately
        for ticker in all_tickers:
            # Get data for this ticker
            group = fundamentals[fundamentals['ticker'] == ticker]
            
            # Check if we have data for this ticker
            if len(group) == 0:
                continue
                
            # Create dataframe with all months for this ticker
            ticker_months = pd.DataFrame({
                'date': full_date_range,
                'ticker': ticker
            })
            
            # For each month, find the most recent quarter's data that would be available
            for idx, row in ticker_months.iterrows():
                month_date = row['date']
                
                # Find the most recent quarter's data available before this month
                available_data = group[group['available_date'] <= month_date]
                
                if not available_data.empty:
                    recent_quarter = available_data.iloc[-1:].copy()
                    
                    # Add all fundamental fields to the month row
                    for col in recent_quarter.columns:
                        if col not in ['ticker', 'datadate', 'available_date', 'fyear']:
                            ticker_months.loc[idx, col] = recent_quarter[col].values[0]
                    
                    # Add original quarter date and fiscal year for reference
                    ticker_months.loc[idx, 'quarter_date'] = recent_quarter['datadate'].values[0]
                    if 'fyear' in recent_quarter.columns:
                        ticker_months.loc[idx, 'fiscal_year'] = recent_quarter['fyear'].values[0]
            
            # For any NaN values, apply forward fill first
            ticker_months = ticker_months.sort_values('date').fillna(method='ffill')
            
            # Then backward fill any remaining NaNs
            ticker_months = ticker_months.fillna(method='bfill')
            
            # Add to our collection
            monthly_fundamentals.append(ticker_months)
        
        # Combine all tickers
        if not monthly_fundamentals:
            return pd.DataFrame()  # Return empty dataframe if no data
            
        expanded_fundamentals = pd.concat(monthly_fundamentals, ignore_index=True)
        
        # Create year-month column for expanded fundamentals
        expanded_fundamentals['year_month'] = expanded_fundamentals['date'].dt.to_period('M')
        
        # Check if we still have missing values (excluding columns that should be allowed to be NaN)
        # Adjust these columns as needed for your specific dataset
        columns_to_check = [col for col in expanded_fundamentals.columns 
                            if col not in ['date', 'ticker', 'year_month', 'quarter_date', 'fiscal_year']]
        
        missing_count = expanded_fundamentals[columns_to_check].isna().sum().sum()
        
        if missing_count > 0:
            print(f"Warning: {missing_count} values still missing after forward/backward fill")
            # Identify columns with missing values
            cols_with_missing = expanded_fundamentals[columns_to_check].isna().sum()
            cols_with_missing = cols_with_missing[cols_with_missing > 0].sort_values(ascending=False)
            print("Top 5 columns with most missing values:")
            print(cols_with_missing.head())
        
        # Verify date range
        actual_min_date = expanded_fundamentals['date'].min()
        actual_max_date = expanded_fundamentals['date'].max()
        print(f"Date range: {actual_min_date.strftime('%Y-%m-%d')} to {actual_max_date.strftime('%Y-%m-%d')}")
        
        # Make sure each ticker has the full date range
        ticker_date_counts = expanded_fundamentals.groupby('ticker')['date'].count()
        expected_dates = len(full_date_range)
        tickers_with_wrong_dates = ticker_date_counts[ticker_date_counts != expected_dates]
        
        if len(tickers_with_wrong_dates) > 0:
            print(f"Warning: {len(tickers_with_wrong_dates)} tickers don't have exactly {expected_dates} dates")
            print("Sample of tickers with incorrect date counts:")
            print(tickers_with_wrong_dates.head())
        
        return expanded_fundamentals

    @staticmethod
    def merge_fundamentals_with_stock_data(expanded_fundamentals, stock_df, merge_on='ticker'):
        """
        Merges expanded monthly fundamental data with monthly stock data.
        Only keeps tickers that exist in both datasets and ensures no duplicates.
        
        Parameters:
        -----------
        expanded_fundamentals : pandas.DataFrame
            DataFrame containing expanded monthly fundamental data
        stock_df : pandas.DataFrame
            DataFrame containing monthly stock data
        merge_on : str, optional
            Column to use for merging ('ticker' or 'permno'), default is 'ticker'
            
        Returns:
        --------
        pandas.DataFrame
            Merged dataframe with monthly stock data and expanded fundamental data
        """
        # Make copies to avoid modifying the originals
        fundamentals = expanded_fundamentals.copy()
        stock = stock_df.copy()
        
        # Ensure date columns are datetime
        fundamentals['date'] = pd.to_datetime(fundamentals['date'])
        stock['date'] = pd.to_datetime(stock['date'])
        
        # Create year_month column for both datasets using the same method
        fundamentals['year_month'] = fundamentals['date'].dt.strftime('%Y-%m')
        stock['year_month'] = stock['date'].dt.strftime('%Y-%m')
        
        # Check for duplicate dates within each ticker before merging
        print("\nChecking fundamentals data for duplicates before merge:")
        for ticker in fundamentals[merge_on].unique():
            ticker_data = fundamentals[fundamentals[merge_on] == ticker]
            duplicates = ticker_data[ticker_data.duplicated(subset=['year_month'], keep=False)]
            if not duplicates.empty:
                print(f"  Ticker {ticker} has {len(duplicates)//2} duplicated months in fundamentals")
                
        print("\nChecking stock data for duplicates before merge:")
        for ticker in stock[merge_on].unique():
            ticker_data = stock[stock[merge_on] == ticker]
            duplicates = ticker_data[ticker_data.duplicated(subset=['year_month'], keep=False)]
            if not duplicates.empty:
                print(f"  Ticker {ticker} has {len(duplicates)//2} duplicated months in stock data")
        
        # Remove any duplicates (keeping the first occurrence)
        fundamentals = fundamentals.drop_duplicates(subset=[merge_on, 'year_month'], keep='first')
        stock = stock.drop_duplicates(subset=[merge_on, 'year_month'], keep='first')
        
        # Get common tickers
        fundamentals_tickers = set(fundamentals[merge_on].unique())
        stock_tickers = set(stock[merge_on].unique())
        common_tickers = fundamentals_tickers.intersection(stock_tickers)
        
        print(f"\nFundamentals tickers: {len(fundamentals_tickers)}")
        print(f"Stock data tickers: {len(stock_tickers)}")
        print(f"Common tickers: {len(common_tickers)}")
        
        # Filter both datasets to only include common tickers
        filtered_fundamentals = fundamentals[fundamentals[merge_on].isin(common_tickers)]
        filtered_stock = stock[stock[merge_on].isin(common_tickers)]
        
        # Check date ranges before merging
        fundamentals_dates = filtered_fundamentals['year_month'].unique()
        stock_dates = filtered_stock['year_month'].unique()
        
        print(f"\nFundamentals has data for {len(fundamentals_dates)} months")
        print(f"First month in fundamentals: {min(fundamentals_dates)}")
        print(f"Last month in fundamentals: {max(fundamentals_dates)}")
        
        print(f"Stock data has data for {len(stock_dates)} months")
        print(f"First month in stock data: {min(stock_dates)}")
        print(f"Last month in stock data: {max(stock_dates)}")
        
        # Find common months
        common_months = set(fundamentals_dates).intersection(set(stock_dates))
        print(f"Common months between datasets: {len(common_months)}")
        print(f"First common month: {min(common_months)}")
        print(f"Last common month: {max(common_months)}")
        
        # Merge the datasets
        merged_data = pd.merge(
            filtered_stock,
            filtered_fundamentals,
            on=[merge_on, 'year_month'],
            how='inner',
            suffixes=('_stock', '_fundamentals')
        )
        
        # Verify no duplicates in the merged data
        merged_data_check = merged_data.groupby([merge_on, 'year_month']).size().reset_index(name='count')
        if (merged_data_check['count'] > 1).any():
            print("\nWARNING: There are still duplicates in the merged data!")
            duplicates_found = merged_data_check[merged_data_check['count'] > 1]
            print(f"Number of duplicate entries: {len(duplicates_found)}")
            print("Sample duplicates:")
            print(duplicates_found.head())
        else:
            print("\nNo duplicates found in the merged data. Good!")
        
        # Final dataset stats
        final_ticker_count = merged_data[merge_on].nunique()
        final_month_count = merged_data['year_month'].nunique()
        
        print(f"\nFinal dataset contains {final_ticker_count} unique tickers across {final_month_count} months")
        print(f"First month in merged data: {merged_data['year_month'].min()}")
        print(f"Last month in merged data: {merged_data['year_month'].max()}")
        
        return merged_data
    
    @staticmethod
    def append_factors_to_stocks(stocks_df, factors_df, date_column='date'):
        """
        Appends factor data to stock data based on matching year and month.
        
        Parameters:
        -----------
        stocks_df : pandas.DataFrame
            DataFrame containing stock data. Can be a single stock or multiple stocks.
            Must have a datetime column (specified by date_column parameter).
        
        factors_df : pandas.DataFrame
            DataFrame containing factor data.
            Must have the same datetime column as stocks_df.
        
        date_column : str, default='date'
            The name of the column containing the date information in both DataFrames.
            
        Returns:
        --------
        pandas.DataFrame
            A copy of the stocks_df with factor columns appended.
        """
        # Convert date columns to datetime if they aren't already
        # if not pd.api.types.is_datetime64_dtype(stocks_df[date_column]):
        #     stocks_df = stocks_df.copy()
        #     stocks_df[date_column] = pd.to_datetime(stocks_df[date_column])
        
        # if not pd.api.types.is_datetime64_dtype(factors_df[date_column]):
        #     factors_df = factors_df.copy()
        #     factors_df[date_column] = pd.to_datetime(factors_df[date_column])
        
        # Create year-month columns for matching
        stocks_df['year_month'] = stocks_df[date_column].dt.strftime('%Y-%m')
        factors_df['year_month'] = factors_df[date_column].dt.strftime('%Y-%m')
        
        # Get list of factor columns to append (all columns except date and year_month)
        factor_columns = [col for col in factors_df.columns 
                        if col not in [date_column, 'year_month']]
        
        # Create a mapping dictionary from year_month to factor values
        factor_dict = {}
        for col in factor_columns:
            # Create a dict mapping year_month to the factor value
            temp_dict = dict(zip(factors_df['year_month'], factors_df[col]))
            factor_dict[col] = temp_dict
        
        # Create result dataframe
        result_df = stocks_df.copy()
        
        # Add each factor column
        for col in factor_columns:
            result_df[col] = result_df['year_month'].map(factor_dict[col])
        
        # Drop the temporary year_month column
        result_df = result_df.drop(columns=['year_month'])
        
        return result_df

    
    @staticmethod
    # Also very good, and something to be written about for feature engineering
    def calculate_technical_indicators(df, window_short=6, window_medium=12, window_long=24):
        """Calculate technical indicators like moving averages, RSI, etc."""
        result = df.copy()
        
        # Initialize columns to avoid SettingWithCopyWarning
        tech_columns = [
            f'sma_{window_short}m', f'sma_{window_medium}m', f'sma_{window_long}m',
            'volatility_12m', 'volatility_24m', 'rsi_6m', 'ema_3m', 'ema_9m',
            'macd', 'macd_signal', 'mom_3m', 'mom_6m', 'mom_12m'
        ]
        
        for col in tech_columns:
            result[col] = float('nan')
        
        # Group by ticker to calculate indicators for each stock
        for ticker, group in result.groupby('ticker'):
            # Sort by date
            group = group.sort_values('date_stock')
            
            # Moving averages
            group[f'sma_{window_short}m'] = group['price_adjusted'].rolling(window=window_short).mean()
            group[f'sma_{window_medium}m'] = group['price_adjusted'].rolling(window=window_medium).mean()
            group[f'sma_{window_long}m'] = group['price_adjusted'].rolling(window=window_long).mean()
            
            # Volatility (standard deviation of returns)
            group['volatility_12m'] = group['ret'].rolling(window=12).std()
            group['volatility_24m'] = group['ret'].rolling(window=24).std()
            
            # RSI (Relative Strength Index)
            delta = group['price_adjusted'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
            rs = gain / loss
            group['rsi_6m'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD (with monthly parameters)
            group['ema_3m'] = group['price_adjusted'].ewm(span=3, adjust=False).mean()
            group['ema_9m'] = group['price_adjusted'].ewm(span=9, adjust=False).mean()
            group['macd'] = group['ema_3m'] - group['ema_9m']
            group['macd_signal'] = group['macd'].ewm(span=3, adjust=False).mean()
            
            # Price momentum indicators
            group['mom_3m'] = group['price_adjusted'] / group['price_adjusted'].shift(3) - 1
            group['mom_6m'] = group['price_adjusted'] / group['price_adjusted'].shift(6) - 1
            group['mom_12m'] = group['price_adjusted'] / group['price_adjusted'].shift(12) - 1
            
            # Update the result dataframe
            # Use ticker and date to ensure correct matching
            idx = result['ticker'] == ticker
            result.loc[idx, tech_columns] = group[tech_columns].values
        
        return result

    @staticmethod
    def analyze_missing_fundamentals(df, unique_tickers):
        """
        Analyze missing fundamental data by ticker and by metric.
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with fundamental data
        unique_tickers : list
            List of tickers to analyze
        Returns:
        --------
        tuple
            (missing_by_ticker, missing_by_field, tickers_to_exclude, fields_to_exclude)
        """
        print(f"Total unique tickers in analysis: {len(unique_tickers)}")
        
        # Filter df to only include tickers in unique_tickers
        tickers_set = set(unique_tickers)
        filtered_df = df[df['ticker'].isin(tickers_set)].copy()
        
        print(f"Tickers found in fundamental data: {filtered_df['ticker'].nunique()}")
        
        # Find tickers with no data at all
        missing_tickers = tickers_set - set(filtered_df['ticker'].unique())
        if missing_tickers:
            print(f"\nWARNING: {len(missing_tickers)} tickers have no fundamental data at all:")
            print(sorted(list(missing_tickers)))
        
        # Calculate missing values by ticker - NOW USING ONLY TICKERS IN unique_tickers
        missing_by_ticker = pd.DataFrame(index=filtered_df['ticker'].unique())
        
        # Get all fields except metadata columns
        metadata_cols = ['gvkey', 'datadate', 'ticker', 'fyearq', 'permno']
        field_cols = [col for col in filtered_df.columns if col not in metadata_cols]
        
        # For each ticker, calculate % of missing values for each field
        for ticker in missing_by_ticker.index:
            ticker_data = filtered_df[filtered_df['ticker'] == ticker]
            for field in field_cols:
                missing_pct = ticker_data[field].isna().mean() * 100
                missing_by_ticker.loc[ticker, field] = missing_pct
        
        # Calculate overall missing values by ticker
        missing_by_ticker['overall_missing_pct'] = missing_by_ticker.mean(axis=1)
        missing_by_ticker = missing_by_ticker.sort_values('overall_missing_pct', ascending=False)
        
        # Calculate missing values by field - using only the filtered data
        missing_by_field = pd.DataFrame(index=field_cols)
        missing_by_field['missing_pct'] = filtered_df[field_cols].isna().mean() * 100
        missing_by_field = missing_by_field.sort_values('missing_pct', ascending=False)
        
        # Identify potentially problematic tickers and fields
        # Tickers with > 10% missing data overall
        tickers_to_exclude = missing_by_ticker[missing_by_ticker['overall_missing_pct'] > 10].index.tolist()
        # Fields with > 10% missing data overall
        fields_to_exclude = missing_by_field[missing_by_field['missing_pct'] > 10].index.tolist()
        
        return missing_by_ticker, missing_by_field, tickers_to_exclude, fields_to_exclude

    # Step 3: Function to visualize the missing data
    @staticmethod
    def visualize_missing_data(missing_by_ticker, missing_by_field):
        """
        Create visualizations for missing data analysis.
        Parameters:
        -----------
        missing_by_ticker : pandas.DataFrame
            DataFrame with missing data percentages by ticker
        missing_by_field : pandas.DataFrame
            DataFrame with missing data percentages by field
        """
        # Plot missing data by ticker (top 50 instead of 30)
        plt.figure(figsize=(20, 10))
        top_missing_tickers = missing_by_ticker['overall_missing_pct'].sort_values(ascending=False).head(50)
        # Just use ticker names without industry information
        sns.barplot(x=top_missing_tickers.values, y=top_missing_tickers.index)
        plt.title('Top 50 Tickers with Most Missing Fundamental Data', fontsize=16)
        plt.xlabel('Percent Missing (%)', fontsize=12)
        plt.ylabel('Ticker', fontsize=12)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig('missing_data_by_ticker.png')
        plt.close()
        
        # Plot missing data by field (separate figure)
        plt.figure(figsize=(20, 10))
        sns.barplot(x=missing_by_field['missing_pct'].values, y=missing_by_field.index)
        plt.title('Missing Data by Fundamental Field', fontsize=16)
        plt.xlabel('Percent Missing (%)', fontsize=12)
        plt.ylabel('Field', fontsize=12)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig('missing_data_by_field.png')
        plt.close()

    # Step 4: Function to get a summary of time coverage by ticker
    @staticmethod
    def analyze_time_coverage(df):
        """
        Analyze time coverage of fundamental data by ticker.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with fundamental data
            
        Returns:
        --------
        pandas.DataFrame
            Summary of time coverage by ticker
        """
        # Convert datadate to datetime if it's not already
        df = df.copy()
        df['datadate'] = pd.to_datetime(df['datadate'])
        
        # Create a coverage DataFrame
        coverage = pd.DataFrame(index=df['ticker'].unique())
        
        # Set the standard expectation of 80 quarters (20 years)
        expected_quarters = 80
        
        for ticker in coverage.index:
            ticker_data = df[df['ticker'] == ticker]
            
            if not ticker_data.empty:
                # Get basic date range info
                coverage.loc[ticker, 'start_date'] = ticker_data['datadate'].min()
                coverage.loc[ticker, 'end_date'] = ticker_data['datadate'].max()
                
                # Calculate years covered
                start = pd.to_datetime(coverage.loc[ticker, 'start_date'])
                end = pd.to_datetime(coverage.loc[ticker, 'end_date'])
                coverage.loc[ticker, 'years_covered'] = (end - start).days / 365.25
                
                # Extract year and quarter to count unique year-quarters
                ticker_data['year'] = ticker_data['datadate'].dt.year
                ticker_data['quarter'] = ticker_data['datadate'].dt.quarter
                
                # Count unique year-quarters (this ensures we only count one observation per quarter)
                unique_year_quarters = ticker_data.groupby(['year', 'quarter']).size().reset_index().shape[0]
                coverage.loc[ticker, 'quarters_count'] = unique_year_quarters
                
                # Expected quarters is fixed at 80
                coverage.loc[ticker, 'expected_quarters'] = expected_quarters
                
                # Calculate completeness as simple percentage of quarters_count / expected_quarters
                # Cap at 100% for cases where we might have more than 80 quarters
                completeness = (coverage.loc[ticker, 'quarters_count'] / expected_quarters) * 100
                coverage.loc[ticker, 'completeness'] = round(min(100, completeness), 2)
        
        # Sort by completeness
        coverage = coverage.sort_values('completeness', ascending=False)
        return coverage

    @staticmethod
    def clean_with_gvkey_strategy(df, strategy='most_recent'):
        """
        Clean data by keeping only one observation per ticker-quarter with different strategies.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with fundamental data
        strategy : str
            Strategy to choose which gvkey to keep:
            - 'most_recent': Keep the most recent (highest) gvkey
            - 'most_data': Keep the gvkey with the least NAs
            - 'longest_history': Keep the gvkey with the longest history
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame with one row per ticker-quarter
        """
        # Create a copy of the dataframe
        cleaned_df = df.copy()
        
        # Convert datadate to datetime if not already
        cleaned_df['datadate'] = pd.to_datetime(cleaned_df['datadate'])
        
        # Extract year and quarter
        cleaned_df['year'] = cleaned_df['datadate'].dt.year
        cleaned_df['quarter'] = cleaned_df['datadate'].dt.quarter
        
        # Count rows before cleaning
        rows_before = cleaned_df.shape[0]
        
        if strategy == 'most_recent':
            # Sort by ticker, year, quarter, and gvkey (descending to keep most recent gvkey)
            cleaned_df = cleaned_df.sort_values(['ticker', 'year', 'quarter', 'gvkey'], 
                                            ascending=[True, True, True, False])
            
        elif strategy == 'most_data':
            # Calculate NA count for each row
            metadata_cols = ['gvkey', 'datadate', 'ticker', 'fyearq', 'year', 'quarter']
            data_cols = [col for col in cleaned_df.columns if col not in metadata_cols]
            cleaned_df['na_count'] = cleaned_df[data_cols].isna().sum(axis=1)
            
            # Sort by ticker, year, quarter, and na_count (ascending to keep row with least NAs)
            cleaned_df = cleaned_df.sort_values(['ticker', 'year', 'quarter', 'na_count'], 
                                            ascending=[True, True, True, True])
            
        elif strategy == 'longest_history':
            # First, calculate history length for each gvkey
            gvkey_history = cleaned_df.groupby(['ticker', 'gvkey']).size().reset_index(name='history_length')
            
            # Merge this back to the main dataframe
            cleaned_df = cleaned_df.merge(gvkey_history, on=['ticker', 'gvkey'])
            
            # Sort by ticker, year, quarter, and history_length (descending to keep gvkey with longest history)
            cleaned_df = cleaned_df.sort_values(['ticker', 'year', 'quarter', 'history_length'], 
                                            ascending=[True, True, True, False])
        
        # Keep only the first occurrence of each ticker-year-quarter combination
        cleaned_df = cleaned_df.drop_duplicates(subset=['ticker', 'year', 'quarter'], keep='first')
        
        # Remove the temporary columns
        if strategy == 'most_data':
            cleaned_df = cleaned_df.drop(columns=['year', 'quarter', 'na_count'])
        elif strategy == 'longest_history':
            cleaned_df = cleaned_df.drop(columns=['year', 'quarter', 'history_length'])
        else:
            cleaned_df = cleaned_df.drop(columns=['year', 'quarter'])
        
        # Count rows after cleaning
        rows_after = cleaned_df.shape[0]
        
        print(f"Rows before cleaning: {rows_before}")
        print(f"Rows after cleaning: {rows_after}")
        print(f"Removed {rows_before - rows_after} duplicate ticker-quarter observations")
        print(f"Strategy used: {strategy}")
        
        return cleaned_df


    @staticmethod
    def filter_dataframe_by_tickers(df, tickers_list):
        """
        Filter a DataFrame to keep only rows where ticker is in the provided list.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing ticker data
        tickers_list : list
            List of tickers to keep
        
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame containing only specified tickers
        """
        # Convert tickers_list to a set for faster lookup
        tickers_set = set(tickers_list)
        
        # Filter the DataFrame
        filtered_df = df[df['ticker'].isin(tickers_set)].copy()
        
        # Print stats about the filtering
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Filtered DataFrame shape: {filtered_df.shape}")
        print(f"Kept {filtered_df['ticker'].nunique()} unique tickers out of {len(tickers_set)} requested")
        
        # Check if any requested tickers are missing from the data
        found_tickers = set(filtered_df['ticker'].unique())
        missing_tickers = tickers_set - found_tickers
        if missing_tickers:
            print(f"Warning: {len(missing_tickers)} requested tickers were not found in the data:")
            print(sorted(list(missing_tickers)))
        
        return filtered_df

    @staticmethod
    def fill_returns_and_calculate_excess(df):
        """
        Fill missing values for ret, retx, and cum_ret columns and calculate excess returns.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing ticker data with columns:
            - ticker: stock ticker
            - date: date of observation
            - price_adjusted: adjusted price
            - price_unadj: unadjusted price
            - ret: returns (may contain NaN)
            - retx: returns excluding dividends (may contain NaN)
            - cum_ret: cumulative returns (may contain NaN)
            - rf: risk-free rate
        
        Returns:
        --------
        df : pandas DataFrame
            DataFrame with filled return values and excess return calculations
        """
        import pandas as pd
        import numpy as np
        
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Ensure the data is sorted by ticker and date
        df = df.sort_values(['ticker', 'date_stock'])
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()
            
            # 1. Calculate and fill simple return (ret)
            if 'ret' in df.columns:
                # Calculate returns using adjusted price
                ticker_data['calc_ret'] = ticker_data['price_adjusted'].pct_change()
                
                # Set first row return to 0
                ticker_data.loc[ticker_data.index[0], 'calc_ret'] = 0
                
                # Fill missing returns
                missing_return_mask = ticker_data['ret'].isna()
                df.loc[mask & missing_return_mask, 'ret'] = ticker_data.loc[missing_return_mask, 'calc_ret']
            
            # 2. Calculate and fill returns excluding dividends (retx)
            if 'retx' in df.columns:
                # Calculate retx using unadjusted price
                ticker_data['calc_retx'] = ticker_data['price_unadj'].pct_change()
                
                # Set first row retx to 0
                ticker_data.loc[ticker_data.index[0], 'calc_retx'] = 0
                
                # Fill missing retx
                missing_retx_mask = ticker_data['retx'].isna()
                df.loc[mask & missing_retx_mask, 'retx'] = ticker_data.loc[missing_retx_mask, 'calc_retx']
            
            # 3. Calculate and fill cumulative returns (cum_ret)
            if 'cum_ret' in df.columns:
                # Get filled returns for this calculation
                filled_returns = df.loc[mask, 'ret'].copy()
                
                # Check if cum_ret needs to be calculated entirely or partially
                if ticker_data['cum_ret'].isna().all():
                    # Calculate cum_ret from scratch using filled returns
                    df.loc[mask, 'cum_ret'] = (1 + filled_returns).cumprod() - 1
                else:
                    # Mixed case: some cum_ret values exist, others need filling
                    cum_ret_values = ticker_data['cum_ret'].copy()
                    
                    # Iterate through the series
                    for i in range(len(ticker_data)):
                        if i == 0:
                            # First row: if NaN, set to 0 or use existing value
                            if pd.isna(cum_ret_values.iloc[0]):
                                cum_ret_values.iloc[0] = 0
                        else:
                            # Other rows: if NaN, calculate based on previous cum_ret and current return
                            if pd.isna(cum_ret_values.iloc[i]):
                                prev_cum_ret = cum_ret_values.iloc[i-1]
                                curr_ret = filled_returns.iloc[i]
                                
                                if not pd.isna(prev_cum_ret) and not pd.isna(curr_ret):
                                    cum_ret_values.iloc[i] = (1 + prev_cum_ret) * (1 + curr_ret) - 1
                    
                    # Update the DataFrame with calculated values
                    df.loc[mask, 'cum_ret'] = cum_ret_values
        
        # 4. Calculate excess return
        if 'ret' in df.columns and 'rf' in df.columns:
            df['excess_ret'] = df['ret'] - df['rf']
        
        return df

