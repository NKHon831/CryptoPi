import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy.stats import mode
import matplotlib.pyplot as plt
from config import *
import os
import joblib

def load_and_prepare_data(path):
    df_raw = pd.read_csv(path, header=None)
    df_raw.columns = df_raw.iloc[0]
    df = df_raw[3:].copy()
    df.rename(columns={df.columns[0]: 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.set_index('Datetime', inplace=True)
    return df

def train_hmm(df, features, n_states, save_prefix=None):
    df_hmm = df[features].dropna()
    X = df_hmm.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)
    
    df.loc[df_hmm.index, "regime"] = model.predict(X_scaled)
    df["regime"] = df["regime"].ffill().bfill()
    
    if save_prefix:
        joblib.dump(model, f"{save_prefix}_hmm.pkl")
        joblib.dump(scaler, f"{save_prefix}_scaler.pkl")
        print(f"[Saved] Model to {save_prefix}_hmm.pkl and scaler to {save_prefix}_scaler.pkl")
    
    return df, model, scaler

def smooth_regime(df, window=7):
    def rolling_mode(series):
        return series.rolling(window, center=False).apply(lambda x: mode(x, keepdims=True).mode[0], raw=False)
    df["regime_smoothed"] = rolling_mode(df["regime"]).ffill().bfill().astype(int)
    return df

def compute_regime_blocks(df):
    df["regime_change"] = df["regime_smoothed"].diff().ne(0).astype(int)
    df["block_id"] = df["regime_smoothed"].ne(df["regime_smoothed"].shift()).cumsum()
    return df

def regime_sma(df, column, window):
    return df.groupby("block_id")[column].apply(lambda x: x.rolling(window, min_periods=1).mean()).reset_index(level=0, drop=True)

def generate_regime_features(df):
    df["SMA_50_regime"] = regime_sma(df, "close", 50)
    df["SMA_200_regime"] = regime_sma(df, "close", 200)
    df["volatility_10_regime"] = regime_sma(df, "volatility_10", 10)
    df["RSI_14_regime"] = regime_sma(df, "RSI_14", 14)
    return df

def one_hot_encode_regime(df):
    df["regime_smoothed"] = df["regime_smoothed"].map({0: 1, 1: 2, 2: 3})
    df["regime_smoothed_backup"] = df["regime_smoothed"]
    df = pd.get_dummies(df, columns=["regime_smoothed"], prefix="regime")
    for col in ["regime_1", "regime_2", "regime_3"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(int)
    if "regime_smoothed" not in df.columns:
        df["regime_smoothed"] = df["regime_smoothed_backup"]
    df.drop(columns=["regime_smoothed_backup"], inplace=True)
    return df

def add_lag_features(df, base_col="log_returns", lags=7):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[base_col].shift(lag)
    return df.dropna()

def load_hmm_model(path_prefix):
    model = joblib.load(f"{path_prefix}_hmm.pkl")
    scaler = joblib.load(f"{path_prefix}_scaler.pkl")
    print(f"[Loaded] Model and scaler from {path_prefix}_*.pkl")
    return model, scaler

def predict_regimes(df_new, features, model, scaler):
    df_input = df_new[features].dropna()
    X = scaler.transform(df_input.values)
    regimes = model.predict(X)
    
    df_new.loc[df_input.index, "regime"] = regimes
    df_new["regime"] = df_new["regime"].ffill().bfill()
    return df_new

def generate_output_paths(input_csv_path, output_csv_dir, output_jpg_dir):
    # Ensure output directories exist
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_jpg_dir, exist_ok=True)

    # Extract base file name without extension
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]  # e.g., "BTC-USD_data"

    # Build paths
    csv_output_path = os.path.join(output_csv_dir, f"{base_name}_regime.csv")
    regime_plot_path = os.path.join(output_jpg_dir, f"{base_name}_regime-change-plot.jpg")
    dist_plot_path = os.path.join(output_jpg_dir, f"{base_name}_frequency-distribution.jpg")

    return csv_output_path, regime_plot_path, dist_plot_path

def plot_regimes(df, path):
    regime_colors = {1: "red", 2: "green", 3: "blue"}
    regime_labels = {1: "Regime 1", 2: "Regime 2", 3: "Regime 3"}
    plt.figure(figsize=(12, 6))
    for regime, color in regime_colors.items():
        subset = df[df["regime_smoothed"] == regime]
        plt.plot(subset.index, subset["close"], label=regime_labels[regime], color=color)
    for change in df[df["regime_change"] == 1].index:
        plt.axvline(change, color='black', linestyle='--', alpha=0.5)
    plt.title("Market Regimes Detected by HMM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)

def plot_regime_distribution(df, path):
    colors = ["red", "green", "blue"]
    counts = df["regime_smoothed"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=colors)
    plt.title("Regime Frequency Distribution")
    plt.xlabel("Regime")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)

def export_to_csv(df, path):
    df.to_csv(path)
    print(f"Data exported to {path}")

def main():
    csv_out, regime_plot, dist_plot = generate_output_paths(INPUT_CSV_PATH, OUTPUT_CSV_PATH, OUTPUT_JPG_PATH)
    df = load_and_prepare_data(INPUT_CSV_PATH)
    # Train and save
    df, model, scaler = train_hmm(df, HMM_FEATURES, N_HMM_STATES, save_prefix="models/btc_hmm")
    df = smooth_regime(df)
    df = compute_regime_blocks(df)
    df = generate_regime_features(df)
    df = one_hot_encode_regime(df)
    df = add_lag_features(df)
    plot_regimes(df, regime_plot)
    plot_regime_distribution(df, dist_plot)
    export_to_csv(df, csv_out)

if __name__ == "__main__":
    main()