"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handle loading and basic preprocessing of financial datasets"""
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader with path to data files
        
        Parameters:
        -----------
        data_path : str or None
            Path to the directory containing raw data files.
            If None, it will search for common locations.
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Try to find data automatically
            self.data_path = self._find_data_path()
            
        print(f"Data path set to: {self.data_path.absolute()}")
        
    def _find_data_path(self):
        """
        Automatically search for data files in common locations
        """
        # Start from current directory and go up
        current_dir = Path.cwd()
        
        # Possible locations to check
        search_locations = [
            current_dir / 'data' / 'raw',
            current_dir / 'Data' / 'raw',
            current_dir.parent / 'data' / 'raw',
            current_dir.parent / 'Data' / 'raw',
            current_dir.parent.parent / 'data' / 'raw',
            current_dir / 'data',
            current_dir / 'Data',
            current_dir.parent / 'data',
            current_dir.parent / 'Data',
            current_dir,  # Files might be directly in current directory
            current_dir.parent,  # Files might be one level up
        ]
        
        for location in search_locations:
            if location.exists():
                # Check if it has any data files
                files = list(location.glob('*.xls')) + list(location.glob('*.xlsx')) + list(location.glob('*.csv'))
                if files:
                    print(f"Found data at: {location}")
                    return location
                    
        # If not found, create the default path
        default_path = current_dir / 'data' / 'raw'
        default_path.mkdir(parents=True, exist_ok=True)
        print(f"No data found. Created default path at: {default_path}")
        print("Please place your data files in this folder.")
        return default_path
        
    def load_news_data(self, filename=None):
        """
        Load and preprocess the financial news dataset
        
        Parameters:
        -----------
        filename : str or None
            Name of the news data file. If None, will auto-discover.
        
        Returns:
        --------
        pd.DataFrame: Processed news dataframe
        """
        # Try to find the news file automatically if not specified
        if filename is None:
            # Look for files with common names
            possible_names = [
                'raw_analyst_ratings.xls',
                'raw_analyst_ratings.xlsx', 
                'raw_analyst_ratings.csv',
                'analyst_ratings.xls',
                'news_data.xls',
                'financial_news.xls'
            ]
            
            found_file = None
            for name in possible_names:
                if (self.data_path / name).exists():
                    found_file = name
                    break
                    
            if found_file is None:
                # List all files in directory
                all_files = list(self.data_path.glob('*'))
                print(f"Files in {self.data_path}: {[f.name for f in all_files]}")
                raise FileNotFoundError(f"No news data file found in {self.data_path}")
            
            filename = found_file
            
        print(f"Loading news data from {self.data_path / filename}...")
        
        # Load the data based on file extension
        file_path = self.data_path / filename
        if file_path.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        print(f"Loaded {len(df)} news articles")
        
        # Basic preprocessing
        df = self._preprocess_news_data(df)
        
        return df
    
    def _preprocess_news_data(self, df):
        """
        Preprocess news dataframe:
        - Convert date column to datetime
        - Handle missing values
        - Extract date components
        """
        # Try to find the date column (case insensitive)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
                
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['date_only'] = df['date'].dt.date
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour
            df['day_name'] = df['date'].dt.day_name()
            
        # Drop rows with missing headlines
        if 'headline' in df.columns:
            df = df.dropna(subset=['headline'])
        else:
            # Try to find headline column
            for col in df.columns:
                if 'headline' in col.lower() or 'title' in col.lower():
                    df = df.rename(columns={col: 'headline'})
                    break
                    
        # Fill missing stock symbols with 'UNKNOWN'
        if 'stock' in df.columns:
            df['stock'] = df['stock'].fillna('UNKNOWN')
        else:
            # Try to find stock column
            for col in df.columns:
                if 'stock' in col.lower() or 'ticker' in col.lower() or 'symbol' in col.lower():
                    df = df.rename(columns={col: 'stock'})
                    break
                    
        print(f"After preprocessing: {len(df)} articles")
        
        return df
    
    def load_stock_data(self, ticker_symbols=None):
        """
        Load historical stock price data for given tickers
        
        Parameters:
        -----------
        ticker_symbols : list or None
            List of stock ticker symbols to load. If None, will auto-discover.
            
        Returns:
        --------
        dict: Dictionary with ticker as key and dataframe as value
        """
        if ticker_symbols is None:
            # Auto-discover stock files
            ticker_symbols = []
            for ext in ['*.csv', '*.xls', '*.xlsx']:
                for file in self.data_path.glob(ext):
                    # Get filename without extension
                    name = file.stem.upper()
                    # Skip if it looks like the news file
                    if 'analyst' not in name.lower() and 'news' not in name.lower():
                        ticker_symbols.append(name)
            ticker_symbols = list(set(ticker_symbols))  # Remove duplicates
            print(f"Auto-discovered tickers: {ticker_symbols}")
            
        stock_data = {}
        
        for ticker in ticker_symbols:
            print(f"Loading {ticker} data...")
            
            # Try different file formats and cases
            file_options = [
                self.data_path / f"{ticker}.csv",
                self.data_path / f"{ticker}.xls",
                self.data_path / f"{ticker}.xlsx",
                self.data_path / f"{ticker.lower()}.csv",
                self.data_path / f"{ticker.lower()}.xls",
                self.data_path / f"{ticker.capitalize()}.csv",
            ]
            
            file_found = None
            for file_path in file_options:
                if file_path.exists():
                    file_found = file_path
                    break
                    
            if file_found is None:
                print(f"Warning: No data file found for {ticker}")
                continue
                
            print(f"  Found file: {file_found.name}")
                
            # Load the data
            if file_found.suffix == '.csv':
                df = pd.read_csv(file_found)
            else:
                df = pd.read_excel(file_found)
                
            # Preprocess stock data
            df = self._preprocess_stock_data(df, ticker)
            stock_data[ticker] = df
            
            print(f"  Loaded {len(df)} trading days for {ticker}")
            
        return stock_data
    
    def _preprocess_stock_data(self, df, ticker):
        """
        Preprocess stock dataframe:
        - Convert Date to datetime
        - Sort by date
        - Calculate daily returns
        """
        # Find date column (case insensitive)
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
                
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            df.set_index(date_col, inplace=True)
            
        # Find price columns (case insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() == 'close' or col.lower() == 'adj close':
                close_col = col
                break
                
        if close_col:
            # Ensure we have Close column
            df['Close'] = df[close_col]
            
            # Calculate daily returns
            df['daily_return'] = df['Close'].pct_change() * 100
            
        # Add ticker column
        df['ticker'] = ticker
        
        return df


class TextPreprocessor:
    """Handle text preprocessing for NLP tasks"""
    
    @staticmethod
    def get_headline_lengths(df):
        """
        Calculate statistics for headline lengths
        
        Returns:
        --------
        dict: Statistics including mean, median, min, max
        """
        if 'headline' not in df.columns:
            return {}
            
        lengths = df['headline'].str.len()
        
        stats = {
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'std_length': lengths.std()
        }
        
        return stats
    
    @staticmethod
    def get_word_count_distribution(df):
        """Calculate word count statistics for headlines"""
        if 'headline' not in df.columns:
            return {}
            
        word_counts = df['headline'].str.split().str.len()
        
        stats = {
            'mean_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'min_words': word_counts.min(),
            'max_words': word_counts.max()
        }
        
        return stats