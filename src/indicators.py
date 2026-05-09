"""
Technical indicators calculation module using TA-Lib
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate various technical indicators for stock price data"""
    
    def __init__(self):
        """Initialize TechnicalIndicators class"""
        self._check_talib()
        
    def _check_talib(self):
        """Check if TA-Lib is available"""
        try:
            import talib
            self.talib = talib
            self.talib_available = True
        except ImportError:
            print("Warning: TA-Lib not installed. Using custom implementations.")
            self.talib_available = False
            
    def add_moving_averages(self, df, windows=[10, 20, 50]):
        """
        Add Simple and Exponential Moving Averages
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Close' column
        windows : list
            Window sizes for moving averages
            
        Returns:
        --------
        pd.DataFrame: Dataframe with MA columns added
        """
        df = df.copy()
        close = df['Close'].values
        
        for window in windows:
            if self.talib_available:
                sma = self.talib.SMA(close, timeperiod=window)
                ema = self.talib.EMA(close, timeperiod=window)
            else:
                sma = df['Close'].rolling(window=window).mean()
                ema = df['Close'].ewm(span=window, adjust=False).mean()
                
            df[f'SMA_{window}'] = sma
            df[f'EMA_{window}'] = ema
            
        return df
    
    def add_rsi(self, df, period=14):
        """
        Add Relative Strength Index (RSI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Close' column
        period : int
            RSI calculation period (typically 14)
            
        Returns:
        --------
        pd.DataFrame: Dataframe with RSI column added
        """
        df = df.copy()
        close = df['Close'].values
        
        if self.talib_available:
            df['RSI'] = self.talib.RSI(close, timeperiod=period)
        else:
            # Custom RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
        return df
    
    def add_macd(self, df, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Close' column
        fast, slow, signal : int
            MACD parameters
            
        Returns:
        --------
        pd.DataFrame: Dataframe with MACD columns added
        """
        df = df.copy()
        close = df['Close'].values
        
        if self.talib_available:
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
        else:
            # Custom MACD calculation
            exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
        return df
    
    def add_bollinger_bands(self, df, period=20, std_dev=2):
        """
        Add Bollinger Bands
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Close' column
        period : int
            Moving average period
        std_dev : int
            Number of standard deviations
            
        Returns:
        --------
        pd.DataFrame: Dataframe with Bollinger Band columns added
        """
        df = df.copy()
        
        if self.talib_available:
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.talib.BBANDS(
                df['Close'].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
        else:
            df['BB_middle'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * std_dev)
            df['BB_lower'] = df['BB_middle'] - (bb_std * std_dev)
            
        return df
    
    def add_all_indicators(self, df):
        """
        Add all technical indicators to the dataframe
        
        Returns:
        --------
        pd.DataFrame: Dataframe with all indicators
        """
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        
        return df
    
    def get_indicator_summary(self, df):
        """
        Get summary statistics of technical indicators
        
        Returns:
        --------
        dict: Summary of indicator values
        """
        summary = {}
        
        indicator_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                            'BB_upper', 'BB_middle', 'BB_lower']
        
        for col in indicator_columns:
            if col in df.columns:
                summary[col] = {
                    'current': df[col].iloc[-1] if not pd.isna(df[col].iloc[-1]) else None,
                    'mean': df[col].mean(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
                
        # Add RSI signals
        if 'RSI' in df.columns:
            current_rsi = df['RSI'].iloc[-1]
            if current_rsi > 70:
                summary['rsi_signal'] = 'Overbought'
            elif current_rsi < 30:
                summary['rsi_signal'] = 'Oversold'
            else:
                summary['rsi_signal'] = 'Neutral'
                
        return summary