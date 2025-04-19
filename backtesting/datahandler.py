# python3 -m backtesting.datahandler 
# Caching strategy:
# - requests_cache: caches raw API HTTP responses
# - joblib: caches the final merged + processed DataFrame
# - unique filenames ensure different parameter combinations don’t overwrite each other


import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timezone
import os
from functools import reduce
from config import Config
from functools import lru_cache
import requests
import requests_cache
import time
import os
import joblib

class BaseDataHandler:
    def __init__(self, 
             symbol: str, 
             start_time: datetime, 
             end_time: datetime,
             window: str,
             limit: int = 100000,
             flatten: bool = True):

        self.symbol = symbol
        self.start_dt = start_time
        self.end_dt = end_time
        self.start_time = self.convert_to_unix_ms(start_time)
        self.end_time = self.convert_to_unix_ms(end_time)

        # Default values
        self.limit = limit
        self.flatten = flatten
        self.window = window

        # self.l1_cache = L1Cache(ttl_seconds=3600)

        # Data containers
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.processed_data: pd.DataFrame = pd.DataFrame()
        self.fetch_binance_data()

    def convert_to_unix_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    def load_from_disc(self, path: str):
        self.raw_data = pd.read_csv(path)

    def _generate_cache_key(self):
        return (self.window, self.start_time, self.end_time)
    
    def _get_cache_key(self):
        return f"{self.symbol}_{self.window}_{self.start_time}_{self.end_time}"

    def fetch_binance_data(self):
        requests_cache.install_cache('api_cache', expire_after=3600)
        config = Config()
        api_key = config.CYBOTRADE_API_KEY
        url = "https://api.datasource.cybotrade.rs/binance-linear/candle"
        headers = {"X-api-key": api_key}

        window = "1d" if self.window == "24h" else self.window

        params = {
            "symbol": "BTCUSDT",
            "interval": window,
            "start_time": self.start_time,
            "end_time": self.end_time ,
        }

        # Check if the request is already cached
        try:
            # Fetching data from the API
            session = requests_cache.CachedSession('api_cache', expire_after=3600)
            start=time.time()
            response = session.get(url, headers=headers, params=params)
            print(f"⏱️ Binance Data Duration: {time.time() - start:.4f}s")
            
            # Check if the data was fetched from the cache
            # if response.from_cache:
            #     print("✅ Binance Data fetched from cache!")
            # else:
            #     print("📡 Binance Data fetched from API!")

            # Check for errors in the response
            
            response.raise_for_status()

            data = response.json()

            if 'data' not in data or not data['data']:
                # print("⚠️ No 'data' returned in response.")
                return pd.DataFrame()

            df = pd.json_normalize(data['data'])

            # Ensure required columns are present
            expected_cols = ["start_time", "close", "high", "low", "open", "volume", ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            # if missing_cols:
            #     pass
            #     # print(f"⚠️ Missing columns: {missing_cols}")

            # Convert timestamp
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
            df = df.rename(columns={"start_time": "timestamp"})
            df.set_index("timestamp", inplace=True)

            self.raw_data = df
            self.processed_data = df

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return pd.DataFrame()
        
    def get_processed_data(self) -> pd.DataFrame:
        return self.processed_data

    def export(self, path: str, filename: str):
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        # Add .csv extension if not present
        if not filename.endswith(".csv"):
            filename += ".csv"

        # Construct full file path using just the filename
        full_path = os.path.join(path, filename)

        try:
            self.processed_data.to_csv(full_path)
            print(f"✅ Binance Data exported to: {full_path}")
        except Exception as e:
            print(f"❌ Failed to export data: {e}")

class RegimeModelData(BaseDataHandler):
    def __init__(self, 
                 symbol: str, 
                 start_time: datetime, 
                 end_time: datetime,
                 window: str,
                 limit: int = 100000,
                 flatten: bool = True,):
        super().__init__(symbol, start_time, end_time, window, limit, flatten)
        start_str = str(start_time).replace(":", "-").replace(" ", "_")
        end_str = str(end_time).replace(":", "-").replace(" ", "_")
        filename = f"{symbol}_{window}_{start_str}_{end_str}.pkl"
        # Automatically fetch the OHLC data when RegimeModelData is initialized
        cache_file = f"cache/processed/{filename}.pkl"
        start_timer = time.time()
        if os.path.exists(cache_file):
            print("📥 Loaded processed regime data from cache.")
            clean_old_cache(cache_dir="cache/processed", max_age_seconds=60*60*24)
            self.processed_data = joblib.load(cache_file)
        else:
            # self.fetch_binance_data()
            self.preprocess()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            joblib.dump(self.processed_data, cache_file)
            print("ℹ️ Saved processed regime data to cache.")

        end_timer = time.time()
        print(f"⏱️ RegimeModelData duration: {end_timer - start_timer:.4f}s")

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

class FinalAlphaModelData(BaseDataHandler):
    def __init__(
            self, 
            symbol: str, 
            start_time: datetime, 
            end_time: datetime, 
            window: str,
            **kwargs):
        super().__init__(symbol, start_time, end_time, window, **kwargs)
        config = Config()
        self.api_key = config.CYBOTRADE_API_KEY
        self.base_url = "https://api.datasource.cybotrade.rs"
        self.headers = {"X-api-key": self.api_key}

        # Global parameters for most endpoints
        self.common_params = {
            "a": symbol,
            "i": self.window,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

        # Endpoint-specific configuration
        self.endpoint_config = {
            "glassnode/addresses/min_10k_count": {}, # add_10_k
            # "glassnode/addresses/min_100_count": {}, # add_100_btc
            # "glassnode/addresses/new_non_zero_count": {}, # new_adds
            "glassnode/addresses/accumulation_count": {}, # new_adds
            "glassnode/addresses/count": {}, # total_adds
            # "glassnode/supply/active_more_1y_percent": {}, # s_last_act_1y
            # "glassnode/blockchain/block_count": {}, # blocks_mined
            # "glassnode/mining/hash_rate_mean": {}, # hash_rate
            # "glassnode/supply/inflation_rate": {}, # inflat_rate
            # "glassnode/mining/revenue_from_fees": {}, # min_rev_fees
            # "glassnode/distribution/balance_exchanges": {}, # ex_balance
            # "glassnode/distribution/balance_exchanges": {"a": "USDT"}, # ex_balance_usdt
            # "glassnode/transactions/transfers_to_exchanges_count_pit": {}, # ex_deposits
            # "glassnode/transactions/transfers_volume_to_exchanges_sum": {}, # ex_inflow_vol
            # # ex_inflow_total
            # # "glassnode/distribution/exchange_net_position_change_pit": {"i": "24h"}, (Only allow window=24h) # net_pos_chan
            # # "glassnode/distribution/exchange_net_position_change_pit": {"i": "24h", "a": "USDT"}, (Only allow window=24h) # net_pos_usdt 
            # "glassnode/transactions/transfers_volume_exchanges_net_pit": {}, # netflow_vol
            # # "glassnode/transactions/transfers_volume_exchanges_net_pit": {"a": "USDT"}, # netflow_vol_usdt
            # "glassnode/transactions/transfers_volume_from_exchanges_mean_pit": {}, # outflow_mean
            # "glassnode/transactions/transfers_volume_from_exchanges_sum_pit": {}, # outflow_total
            # "glassnode/transactions/transfers_from_exchanges_count_pit": {}, # withdrawals
            # "glassnode/indicators/net_realized_profit_loss": {}, # profit_loss
            # "glassnode/indicators/net_unrealized_profit_loss": {}, # nupl
            # # utx_profit
            # "glassnode/indicators/realized_loss": {}, # real_loss
            # # p_l_ration
            # "glassnode/indicators/realized_profit": {}, # real_profit
            # "glassnode/indicators/sopr": {}, # sopr
            # "glassnode/supply/loss_sum": {}, # supply_loss
            # # "glassnode/indicators/difficulty_ribbon": {"i": "24h"}, (Only allows window=24h) # dif_ribbon
            # # ent_adj_count
            # "glassnode/transactions/transfers_volume_entity_adjusted_sum_pit": {}, # ent_vol_total
            # "glassnode/transactions/transfers_volume_within_exchanges_sum_pit": {}, # in_house_vol
            # "glassnode/transactions/transfers_volume_between_exchanges_sum_pit": {}, # inter_ex
            # "glassnode/indicators/liveliness": {}, # liveliness
            # "glassnode/indicators/nvt": {}, # nvt_ratio
            # "glassnode/indicators/nvts": {}, # nvts_signal
            # "glassnode/indicators/reserve_risk": {}, # reserve_risk
            # "glassnode/indicators/rhodl_ratio": {}, # rhodl_ratio
            # "glassnode/indicators/seller_exhaustion_constant": {}, # seller_exhaustion
            # "glassnode/indicators/stock_to_flow_deflection": {}, # s_flow_def1
            # # "glassnode/transactions/transfers_count": {}, # Not sure why does not support BTC # tra_count
            # "glassnode/blockchain/utxo_created_count": {}, # utx_created
            # "glassnode/indicators/velocity": {}, # velocity
        }

    def fetch_all_endpoints(self):
        clean_old_cache(cache_dir="cache/endpoints", max_age_seconds=60 * 60 * 24)  # e.g., delete after 24 hours
        all_data = {}
        headers = {
            "X-Api-Key": self.api_key
        }

        # Fetch data from each endpoint
        for endpoint, custom_params in self.endpoint_config.items():
            # print(f"Fetching: /{endpoint}")
            url = f"{self.base_url}/{endpoint}"
            params = self.common_params

            # Format filename safely
            endpoint_str = endpoint.replace("/", "_")
            start_str = str(self.start_time).replace(":", "-").replace(" ", "_")
            end_str = str(self.end_time).replace(":", "-").replace(" ", "_")
            cache_file = f"cache/endpoints/{self.symbol}_{self.window}_{start_str}_{end_str}_{endpoint_str}.pkl"

            start_time_fetch = time.time()

            try:
                if os.path.exists(cache_file):
                    print(f"📦 Loaded from cache")
                    data = joblib.load(cache_file)

                    if 'data' not in data or not data['data']:
                        print(f"⚠️ Cached file is invalid or empty. Deleting and refetching...")
                        os.remove(cache_file)
                        raise ValueError("Invalid cache")
                else:
                    raise FileNotFoundError()

            except (ValueError, FileNotFoundError):
                print(f"🌐 Fetching from API: {endpoint}")
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                if 'data' not in data or not data['data']:
                    raise ValueError("Empty or invalid API response")
                
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                joblib.dump(data, cache_file)

            df = pd.json_normalize(data['data'])
            df.columns = [f"{endpoint}_{col}" for col in df.columns]

            timestamp_col = next((col for col in df.columns if col.endswith("start_time")), None)
            if timestamp_col:
                df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

            all_data[endpoint] = df

            duration = time.time() - start_time_fetch  # ⏱️ End timing
            print(f"⏱️ Duration for {endpoint}: {duration:.2f} seconds\n")

        # Filter only DataFrames with 'timestamp'
        dataframes = [df for df in all_data.values() if "timestamp" in df.columns]

        # Merge on 'timestamp'
        if dataframes:
            try:
                combined_df = reduce(lambda left, right: pd.merge(left, right, how="outer", on="timestamp"), dataframes)
            except Exception as e:
                print(f"❌ Error during merging: {e}")
                combined_df = pd.DataFrame()
        else:
            print("⚠️ No dataframes with a 'timestamp' column to merge.")
            combined_df = pd.DataFrame()

        self.processed_data = combined_df

        # Convert timestamp to datetime
        if "timestamp" in self.processed_data.columns:
            self.processed_data["timestamp"] = pd.to_datetime(self.processed_data["timestamp"], unit="ms", utc=True)
            # print("✅ Timestamp column converted to datetime.")
        else:
            print("⚠️ 'timestamp' column not found in processed_data.")

        result = pd.merge(self.raw_data, combined_df, left_on='timestamp', right_on='timestamp', how='outer')
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))
        self.processed_data = result
        return result
    
class BenchmarkData:
    def __init__(self, symbol: str, start_time: datetime, end_time: datetime, interval: str):
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval

    def fetch_yfinance_data(self):
        btc = yf.download(
            self.symbol,
            start=self.start_time,
            end=self.end_time,
            interval=self.interval,
            progress=False
        )
        btc = btc[['Close']].rename(columns={'Close': 'benchmark'})
        btc.reset_index(inplace=True)
        print(btc.head())
        return btc
    
def clean_old_cache(cache_dir="cache/endpoints", max_age_seconds=60 * 60 * 24):
    """
    Deletes cache files older than max_age_seconds from the specified cache directory.
    """
    current_time = time.time()

    # Loop through files in the cache directory
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is older than the specified max age
            file_age = current_time - os.path.getmtime(file_path)

            if file_age > max_age_seconds:
                print(f"Deleting old cache file: {file_path}")
                os.remove(file_path)

# # OHLC only
# ohlc = BaseDataHandler(symbol='BTC-USD',
#                       start_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
#                       end_time=datetime(2025, 4, 1, tzinfo=timezone.utc),
#                       window="1h")
# raw_ohlc = ohlc.raw_data
# # ohlc.export("/Users/pohsharon/Downloads/UMH", "ohlc") # Change path to your desired export path
# print(raw_ohlc.tail)

# Regime Model Data Frame
# regime_model = RegimeModelData(symbol='BTC-USD',
#                         start_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
#                         end_time=datetime(2025, 4, 13, tzinfo=timezone.utc),
#                         window="1h")
# regime_model.preprocess()
# print(regime_model.processed_data.tail())

# # Test the FinalAlphaModel class
model = FinalAlphaModelData(symbol='BTC',
                          start_time=datetime(2022, 12, 1, tzinfo=timezone.utc),
                          end_time=datetime(2023, 1, 4, tzinfo=timezone.utc),
                          window="24h")

df = model.fetch_all_endpoints()
model.export("/Users/pohsharon/Downloads/UMH", "final_alpha") # Change path to your desired export path
print(df.head())

# Benchmark Data
# benchmark = BenchmarkData(
#     symbol='BTC-USD',
#     start_time=datetime(2024, 3, 1),
#     end_time=datetime(2024, 4, 1),
#     interval='60m'
# )
# btc_data = benchmark.fetch_yfinance_data()

'''
Base Data & Regime Model Interval
1m 3m 5m 10m 15m 30m 1h 2h 4h 6h 12h 1d 3d 1w 1M
 
Final Alpha Model Interval
1h 24h

Benchmark Interval
1m - Only available for last 7 days
2m - Only available for last 60 days
5m - Only available for last 60 days
15m	- Only available for last 60 days
30m	- Only available for last 60 days
60m/1h	- Only available for last 730 days
90m	- Only available for last 60 days
1d 5d 1wk 1mo 3mo - Available for many years
'''