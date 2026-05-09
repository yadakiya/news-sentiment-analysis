"""
Sentiment analysis module using multiple approaches
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import numpy as np


class SentimentAnalyzer:
    """
    Perform sentiment analysis on financial headlines
    Supports multiple sentiment analysis tools
    """
    
    def __init__(self, method='vader'):
        """
        Initialize sentiment analyzer
        
        Parameters:
        -----------
        method : str
            'vader' or 'textblob' - which sentiment tool to use
        """
        self.method = method
        self._initialize_analyzer()
        
    def _initialize_analyzer(self):
        """Initialize the chosen sentiment analyzer"""
        if self.method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.method == 'textblob':
            self.analyzer = None  # TextBlob doesn't need initialization
        else:
            raise ValueError("Method must be 'vader' or 'textblob'")
            
    def get_sentiment_score(self, text):
        """
        Get sentiment score for a single headline
        
        Parameters:
        -----------
        text : str
            Headline text to analyze
            
        Returns:
        --------
        float: Sentiment score from -1 (negative) to +1 (positive)
        """
        if pd.isna(text) or text == '':
            return 0.0
            
        if self.method == 'vader':
            scores = self.analyzer.polarity_scores(text)
            return scores['compound']
        else:  # textblob
            blob = TextBlob(text)
            return blob.sentiment.polarity
            
    def batch_sentiment(self, texts):
        """
        Get sentiment scores for multiple headlines
        
        Parameters:
        -----------
        texts : list or pd.Series
            Collection of headlines
            
        Returns:
        --------
        list: Sentiment scores
        """
        return [self.get_sentiment_score(text) for text in texts]
    
    def classify_sentiment(self, score, threshold=0.05):
        """
        Classify sentiment as positive, neutral, or negative
        
        Parameters:
        -----------
        score : float
            Sentiment score between -1 and 1
        threshold : float
            Threshold for classification (default: 0.05)
            
        Returns:
        --------
        str: 'positive', 'neutral', or 'negative'
        """
        if score > threshold:
            return 'positive'
        elif score < -threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def add_sentiment_to_dataframe(self, df, headline_col='headline'):
        """
        Add sentiment scores and classifications to dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe containing headlines
        headline_col : str
            Name of column containing headlines
            
        Returns:
        --------
        pd.DataFrame: Dataframe with added sentiment columns
        """
        df = df.copy()
        
        # Add sentiment scores
        df['sentiment_score'] = self.batch_sentiment(df[headline_col])
        
        # Add sentiment classification
        df['sentiment_class'] = df['sentiment_score'].apply(self.classify_sentiment)
        
        return df
    
    @staticmethod
    def aggregate_daily_sentiment(df):
        """
        Aggregate sentiment scores by date and stock
        
        Returns:
        --------
        pd.DataFrame: Daily aggregated sentiment by stock
        """
        # Group by date and stock
        daily_sentiment = df.groupby(['date_only', 'stock']).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_class': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date', 'stock', 'avg_sentiment', 'sentiment_std', 
                                   'article_count', 'dominant_sentiment']
        
        return daily_sentiment