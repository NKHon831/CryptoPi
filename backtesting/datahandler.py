import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaseDataHandler:
    def __init__(self, 
             symbol: str, 
             start_time: datetime, 
             end_time: datetime,
             limit: int = 100000,
             flatten: bool = True,
             window: str = "hour"):
        self.symbol = symbol
        self.start_dt = start_time
        self.end_dt = end_time
        self.start_time = self.convert_to_unix_ms(start_time)
        self.end_time = self.convert_to_unix_ms(end_time)

        # Default values
        self.limit = limit
        self.flatten = flatten
        self.window = window

        # Data containers
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.processed_data: pd.DataFrame = pd.DataFrame()

        # Fetch OHLC
        self.fetch_yfinance_data()

        # Fetch funding rate
        funding_params = {
            "exchange": "binance",
            "window": "hour",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "flatten": "true",
            # "limit": 2
        }

        # Derive log return, volatility
        self.preprocess()

    def convert_to_unix_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

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
    
    def get_processed_data(self) -> pd.DataFrame:
        return self.processed_data

    def preprocess(self):
        df = self.raw_data
        df.sort_index(inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        # Compute log return using close price
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df.dropna(inplace=True)  # Drop NaN values caused by the shift operation
        
        # Compute rolling volatility (standard deviation of log returns)
        df["volatility_10"] = df["log_returns"].rolling(10).std()
        df["vol_adj_returns"] = df["log_returns"] / df["volatility_10"]

        # Compute Indicators and Add to DataFrame
        df["EMA_50"] = self.compute_ema(df, period=50)
        df["EMA_200"] = self.compute_ema_200(df, period=200)
        df["RSI_14"] = self.compute_rsi(df, period=14)
        df["MACD"], df["MACD_Signal"] = self.compute_macd(df)
        df["ATR_14"] = self.compute_atr(df, period=14)
        df["SAR"] = self.compute_sar(df)
        df["SLOPE_14"] = self.compute_slope(df, period=14)
        df["ADX_14"] = self.compute_adx(df, period=14)
        df["OBV"] = self.compute_obv_vectorized(df)

        # Fill any NaNs from rolling calculations
        df.fillna(0, inplace=True)
        self.processed_data = df

    # Moving Averages: EMA 50 & EMA 200
    def compute_ema(self, df, column="close", period=50):
        return df[column].ewm(span=period, adjust=False).mean()

    def compute_ema_200(self, df, column="close", period=200):
        return df[column].ewm(span=period, adjust=False).mean()
    
    # Momentum Indicators: RSI & MACD
    def compute_rsi(self, df, column="close", period=14):
        delta = df[column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, df, column="close", short_period=12, long_period=26, signal_period=9):
        short_ema = df[column].ewm(span=short_period, adjust=False).mean()
        long_ema = df[column].ewm(span=long_period, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line


    # Volatility Indicator: ATR
    def compute_atr(self, df, period=14):
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum(abs(df["high"] - df["close"].shift()),
                                    abs(df["low"] - df["close"].shift())))
        return tr.rolling(window=period).mean()

    # Trend Indicators: SAR, Slope & ADX
    def compute_slope(self, df, column="close", period=14):
        return df[column].diff(period) / period

    def compute_sar(self, df, acceleration=0.02, maximum=0.2):
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        sar = np.zeros(len(df))
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = high[0]  # extreme point
        sar[0] = low[0]  # initial SAR value
        
        for i in range(1, len(df)):
            prev_sar = sar[i - 1]
            if trend == 1:
                sar[i] = prev_sar + af * (ep - prev_sar)
                if low[i] < sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:
                sar[i] = prev_sar - af * (prev_sar - ep)
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)

        return sar

    def compute_adx(self, df, period=14):
        plus_dm = np.maximum(df["high"].diff(), 0)
        minus_dm = np.maximum(-df["low"].diff(), 0)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum(abs(df["high"] - df["close"].shift()),
                                    abs(df["low"] - df["close"].shift())))
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    # Volume Indicator: OBV
    def compute_obv_vectorized(self, df):
        direction = np.sign(df["close"].diff()).fillna(0)
        return (direction * df["volume"]).cumsum()
    
    def export(self, path: str):
        # Save the processed data to the specified path with the symbol in the filename
        self.processed_data.to_csv(f"{path}/{self.symbol}_data.csv")
        print(f"Data exported to {path}/{self.symbol}_data.csv")

# Test the BaseDataHandler class
handler = BaseDataHandler(symbol='BTC-USD',
                          start_time=datetime(2025, 1, 1),
                          end_time=datetime(2025, 4, 16),
                          window="hour")
handler.export("/Users/pohsharon/Downloads/UMH")
print(handler.processed_data.tail())