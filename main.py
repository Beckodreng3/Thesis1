import ETL_scripts 
from ETL_scripts import WRDSLoader

if __name__ == "__main__":
    # Use with context manager
    with WRDSLoader(username='your_username') as wrds_loader:
        # Get stock data for Apple and Microsoft for the last year
        stock_data = wrds_loader.get_stock_data(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        print(stock_data.head())
        
        # Get fundamental data
        fundamentals = wrds_loader.get_fundamentals(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        print(fundamentals.head())