INPUT_CSV_PATH = r".\\Input\\BTC-USD_data.csv"
OUTPUT_CSV_PATH = r".\\output\\regime_csv"
# OUTPUT_JPG_PATH = r".\\output\\regime_graph"
N_HMM_STATES = 3
HMM_FEATURES = ["log_returns", "volatility_10", "RSI_14", "MACD", "ATR_14", "OBV"]
MODEL_PATH_PREFIX = r".\\models\\btc_hmm\\btc"
TEST_DATA_PATH = r"C:\\Users\\riche\\OneDrive\\Documents\\UMH25\\Input\\BTC-USD_data.csv"