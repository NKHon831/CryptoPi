import pandas as pd
import joblib
from market_regime_hmm import MarketRegimeHMM
from config import *

# Define the paths for loading the pretrained model and scaler
MODEL_PATH = '.\\models\\btc_hmm\\btc_hmm.pkl'  # Adjust the path as needed
SCALER_PATH = '.\\models\\btc_hmm\\btc_scaler.pkl'  # Adjust the path as needed

def test_model():
    # Step 1: Generate a synthetic data row (one row of data)
    data = {
        'close': [45000],  # Example price (close price)
        'log_returns': [0.01],  # Example log return
        'volatility_10': [0.02],  # Example 10-day volatility
        'RSI_14': [50],  # Example RSI 14
        'MACD': [0.5],  # Example MACD value
        'ATR_14': [0.7],  # Example ATR 14
        'OBV': [1000000],  # Example OBV value
    }
    
    # Create a DataFrame with synthetic data
    df_test = pd.DataFrame(data)
    
    # Step 2: Load the pretrained model and scaler into the MarketRegimeHMM object
    print("Loading pretrained model and scaler...")
    model = MarketRegimeHMM(n_states=3, features=['log_returns', 'volatility_10', 'RSI_14', 'MACD', 'ATR_14', 'OBV'])
    
    # Load the model and scaler into the MarketRegimeHMM instance
    model.model = joblib.load(MODEL_PATH)
    model.scaler = joblib.load(SCALER_PATH)
    
    # Step 3: Preprocess the input data (scale the features)
    X_test = df_test[['log_returns', 'volatility_10', 'RSI_14', 'MACD', 'ATR_14', 'OBV']]
    X_scaled = model.scaler.transform(X_test)  # Apply the same scaling as during training
    
    # Step 4: Predict the regime for the test data
    df_test['regime_smoothed'] = model.model.predict(X_scaled)
    
    # Step 5: Label the regimes using the method from MarketRegimeHMM
    df_test = model.label_regimes(df_test)
    
    # Step 6: Print the results
    print("\nTest Results:")
    print(df_test[['close', 'regime_smoothed', 'regime_label']])

# Run the test
if __name__ == "__main__":
    test_model()
