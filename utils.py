import pandas as pd
import os
from logger import logger

def load_data(data_path='data.csv'):
    """
    Loads stock data from CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        tuple: (df, stock_symbols) where df is the DataFrame and stock_symbols is a list of unique symbols.
               Returns (pd.DataFrame(), []) if file not found.
    """
    try:
        if not os.path.exists(data_path):
            logger.error(f"Data file '{data_path}' not found.")
            return pd.DataFrame(), []

        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            logger.warning("Some dates could not be parsed. Check date format in CSV.")

        stock_symbols = df['Symbol'].unique().tolist()
        logger.info(f"Loaded data with {len(df)} rows and {len(stock_symbols)} unique stock symbols.")
        return df, stock_symbols

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), []

def preprocess_data(df, stock_symbol):
    """
    Preprocesses data for a specific stock symbol.

    Args:
        df (pd.DataFrame): Full dataset.
        stock_symbol (str): Stock symbol to filter.

    Returns:
        pd.DataFrame: Preprocessed data for the stock.
    """
    try:
        df_stock = df[df['Symbol'] == stock_symbol].copy()
        if df_stock.empty:
            logger.warning(f"No data found for stock '{stock_symbol}'.")
            return pd.DataFrame()

        df_stock.sort_values('Date', inplace=True)
        df_stock.reset_index(drop=True, inplace=True)

        # Fill missing values in sentiment columns with 0
        sentiment_cols = [col for col in df.columns if 'News -' in col]
        df_stock[sentiment_cols] = df_stock[sentiment_cols].fillna(0)

        # Create time-based features
        df_stock['Year'] = df_stock['Date'].dt.year
        df_stock['Month'] = df_stock['Date'].dt.month
        df_stock['Day'] = df_stock['Date'].dt.day
        df_stock['DayOfWeek'] = df_stock['Date'].dt.dayofweek

        # Create Lag Features (previous day's close)
        df_stock['Lag_1_Close'] = df_stock['Close'].shift(1)

        # Create Moving Averages
        df_stock['MA_7'] = df_stock['Close'].rolling(window=7).mean()
        df_stock['MA_30'] = df_stock['Close'].rolling(window=30).mean()

        # Create Rolling Volatility
        df_stock['Volatility_7'] = df_stock['Close'].rolling(window=7).std()

        # Sentiment ratios
        epsilon = 1e-6
        df_stock['Positive_Sentiment_Ratio'] = df_stock['News - Positive Sentiment'] / (df_stock['News - Volume'] + epsilon)
        df_stock['Negative_Sentiment_Ratio'] = df_stock['News - Negative Sentiment'] / (df_stock['News - Volume'] + epsilon)

        # Drop rows with NaN values created by shifts and rolling windows
        df_stock.dropna(inplace=True)

        logger.info(f"Preprocessed data for '{stock_symbol}': {len(df_stock)} rows after preprocessing.")
        return df_stock

    except Exception as e:
        logger.error(f"Error preprocessing data for '{stock_symbol}': {str(e)}")
        return pd.DataFrame()

def validate_stock_symbol(stock_symbol, available_symbols):
    """
    Validates if the stock symbol is available.

    Args:
        stock_symbol (str): Symbol to validate.
        available_symbols (list): List of available symbols.

    Returns:
        bool: True if valid, False otherwise.
    """
    if stock_symbol not in available_symbols:
        logger.warning(f"Invalid stock symbol '{stock_symbol}'. Available: {available_symbols}")
        return False
    return True
