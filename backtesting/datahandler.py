import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from typing import List, Dict

class BaseDataHandler:
    def __init__(self, 
             symbol: str, 
             start_time: datetime, 
             end_time: datetime,
            #  use_cryptoquant: bool = True,
            #  use_binance: bool = False,
            #  use_glassnode: bool = False,
             limit: int = 100000,
             flatten: bool = True,
             window: str = "hour"):
        self.symbol = symbol
        self.start_dt = start_time  # keep original datetime
        self.end_dt = end_time
        self.start_time = self.convert_to_unix_ms(start_time)  # for APIs like CryptoQuant
        self.end_time = self.convert_to_unix_ms(end_time)

        # self.use_cryptoquant = use_cryptoquant
        # self.use_binance = use_binance
        # self.use_glassnode = use_glassnode

        self.limit = limit
        self.flatten = flatten
        self.window = window

        # Data containers
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.processed_data: pd.DataFrame = pd.DataFrame()
        self.features: pd.DataFrame = pd.DataFrame()

        handler.fetch_yfinance_data()
        handler.preprocess()

    def convert_to_unix_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    # def add_symbol(self, symbol: str):
    #     if symbol not in self.symbol:
    #         self.symbol.append(symbol)
    #         self.fetch_data(symbol)

    def load_from_disc(self, path: str):
        self.raw_data = pd.read_csv(path)

    def fetch_yfinance_data(self):
        interval = self.window if self.window in ['1m', '2m', '5m', '15m', '30m', '1h', '90m', '1d'] else '1h'
        data = yf.download(
            self.symbol,
            start=self.start_dt.strftime('%Y-%m-%d'),
            end=self.end_dt.strftime('%Y-%m-%d'),
            interval=interval,
            progress=False
        )

        data.rename(columns={
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Adj Close': 'adj_close', 
            'Volume': 'volume'
        }, inplace=True)

        data.dropna(inplace=True)
        self.raw_data = data

    def fetch_data(self, symbol: str):
        # if self.use_cryptoquant:
        base_url = "https://api.datasource.cybotrade.rs/cryptoquant/"
        # elif self.use_glassnode:
        #     base_url = "https://api.datasource.cybotrade.rs/glassnode/"
        # elif self.use_binance:
        #     base_url = "https://api.datasource.cybotrade.rs/coinglass/"
        # else:
        #     raise ValueError("No data source selected")

        endpoint = f"price/{symbol}"
        url = base_url + endpoint

        params = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "limit": self.limit,
            "window": self.window,
            "flatten": self.flatten
        }
        headers = {
            "Authorization": "Bearer Jpef9rVtVwUCNHwnAPB7jrnNuqm4YOVCWgMnps61zt2mRNCs"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = pd.DataFrame(response.json().get("result", []))
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
            self.raw_data = data
        else:
            print(f"Error fetching data for {symbol}: {response.status_code} - {response.text}")

    def get_data(self) -> pd.DataFrame:
        return self.processed_data

    def preprocess(self):
        df = self.raw_data
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.fillna(method='ffill', inplace=True)
        self.processed_data = df

        # Compute log return using close price (added here)
        if 'close' in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df.dropna(inplace=True)  # Drop NaN values caused by the shift operation
        
        # Compute rolling volatility (standard deviation of log returns)
        df['volatility'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)  # Annualized

        self.processed_data = df

    def generate_features(self):
        for sym, df in self.processed_data.items():
            df['returns'] = df['price'].pct_change()
            df['rolling_mean_7'] = df['price'].rolling(window=7).mean()
            df['rolling_mean_30'] = df['price'].rolling(window=30).mean()
            self.features[sym] = df

    def resample_data(self, window: str):
        if self.processed_data.empty:
            raise ValueError("No processed data available. Please load and preprocess data first.")
        
        df_resampled = self.processed_data.resample(window).agg({
            # 'price': 'last',  # Last price in the period
            # 'returns': 'sum',  # Sum of returns (for period-based returns)
            # 'rolling_mean_7': 'last',  # Last rolling mean value in the period
            # 'rolling_mean_30': 'last',  # Last rolling mean value in the period
        })
        df_resampled.fillna(method='ffill', inplace=True)
        self.processed_data = df_resampled
        return df_resampled

    def export(self, path: str):
        # Save the processed data to the specified path with the symbol in the filename
        self.processed_data.to_csv(f"{path}/{self.symbol}_data.csv")
        print(f"Data exported to {path}/{self.symbol}_data.csv")

# #         self.window = window
#         self.preprocess()  # Re-preprocess after changing window
#         self.generate_features()

handler = BaseDataHandler(symbol='BTC',
                          start_time=datetime(2025, 1, 1),
                          end_time=datetime(2025, 4, 15),
                          window="1h")
handler.export("/Users/pohsharon/Downloads/UMH")
print(handler.processed_data.head())

class DataHandler:
    def __init__(self, data):
        self.data = data

    
    def fetch_data(self):
        # Placeholder for data fetching logic
        pass
    
    def preprocess_data(self):
        # Placeholder for data preprocessing logic
        pass
    
    def get_data(self) -> pd.DataFrame:
        return self.data