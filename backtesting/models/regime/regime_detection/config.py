INPUT_CSV_PATH = r".\\Input\\final-alpha-model-data.csv"
OUTPUT_CSV_PATH = r".\\output\\regime_csv"

HMM_FEATURES = ['open', 'high', 'low', 'close', 'min_100_count', 'liveliness']
MODEL_PATH_PREFIX = r".\\models\\btc_hmm\\btc"
TEST_DATA_PATH = r"C:\\Users\\riche\\OneDrive\\Documents\\UMH25\\Input\\BTC-USD_data.csv"
LOG_RETURN_THRESHOLD = 0.001

PERFORM_TUNING = False

# HMM hyperparameters
N_HMM_STATES = 3  # This will be updated by hyperparameter tuning
HMM_COVARIANCE_TYPE = "full"  # This will be updated by hyperparameter tuning
HMM_N_ITER = 300  # This will be updated by hyperparameter tuning
HMM_TOL = 0.01