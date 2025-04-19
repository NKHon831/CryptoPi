import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

class MarketRegimeHMM:
    def __init__(self, n_states=3, features=None, model_path_prefix=None):
        self.n_states = n_states
        self.features = features or []
        self.model_path_prefix = model_path_prefix
        self.model = None
        self.scaler = None

    @staticmethod
    def load_and_prepare_data(path):
        df_raw = pd.read_csv(path)
        # df_raw.columns = df_raw.iloc[0]
        # df = df_raw[3:].copy()
        # df.rename(columns={df.columns[0]: 'Datetime'}, inplace=True)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')
        df_raw.set_index('timestamp', inplace=True)
        return df_raw
    
    def fit(self, df):
        df_hmm = df[self.features].dropna()
        X = df_hmm.values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=1000, random_state=42)
        self.model.fit(X_scaled)
        
        df.loc[df_hmm.index, "regime"] = self.model.predict(X_scaled)
        df["regime"] = df["regime"].ffill().bfill()
        
        return df

    def predict(self, df):
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be loaded or trained before prediction.")

        df_hmm = df[self.features].dropna()
        X_scaled = self.scaler.transform(df_hmm.values)
        df.loc[df_hmm.index, "regime"] = self.model.predict(X_scaled)
        df["regime"] = df["regime"].ffill().bfill()
        return df

    def smooth_regime(self, df, window=7):
        def rolling_mode(series):
            return series.rolling(window, center=False).apply(lambda x: mode(x, keepdims=True).mode[0], raw=False)
        df["regime_smoothed"] = rolling_mode(df["regime"]).ffill().bfill().astype(int)
        return df

    def compute_blocks(self, df):
        df["regime_change"] = df["regime_smoothed"].diff().ne(0).astype(int)
        df["block_id"] = df["regime_smoothed"].ne(df["regime_smoothed"].shift()).cumsum()
        return df

    def label_regimes(self, df):
        """
        Label regimes as 'Bear', 'Bull', or 'Neutral' based on their characteristics.
        
        This method analyzes each regime's average returns and volatility to determine
        appropriate market condition labels.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing regime classifications and financial data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added 'regime_label' column
        """
        # Create a temporary DataFrame to store regime statistics
        regime_stats = {}
        
        # For each regime, calculate average return and volatility
        for regime in range(self.n_states):
            regime_data = df[df['regime_smoothed'] == regime]
            
            # Ensure 'close' column is numeric
            regime_data['close'] = pd.to_numeric(regime_data['close'], errors='coerce')
            
            # Calculate average returns (assuming you have a 'returns' column)
            # If not, we'll calculate it from close prices
            if 'returns' not in df.columns and 'close' in df.columns:
                returns = regime_data['close'].pct_change()
            else:
                returns = regime_data.get('returns', pd.Series(0, index=regime_data.index))
            
            avg_return = returns.mean()
            volatility = returns.std()
            
            regime_stats[regime] = {
                'avg_return': avg_return,
                'volatility': volatility
            }
        
        # Determine which regime is which based on returns and volatility
        # Bear: Negative returns
        # Bull: Positive returns
        # Neutral: Returns close to zero or lower volatility
        
        # Sort regimes by average return
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['avg_return'])
        
        # Assign labels based on their position in sorted order
        regime_labels = {}
        regime_labels[sorted_regimes[0][0]] = 'Bear'  # Lowest returns -> Bear
        regime_labels[sorted_regimes[-1][0]] = 'Bull'  # Highest returns -> Bull
        
        # The middle regime (or the one with lowest volatility if only 2 regimes)
        if len(sorted_regimes) > 2:
            regime_labels[sorted_regimes[1][0]] = 'Neutral'
        else:
            # If only 2 regimes, use volatility to determine which is neutral vs bull/bear
            remaining_regime = [r for r in range(self.n_states) if r not in regime_labels.keys()][0]
            regime_labels[remaining_regime] = 'Neutral'
        
        # Add verbose statistics and labels to a class attribute for reference
        self.regime_characteristics = {
            regime: {
                'stats': stats,
                'label': regime_labels[regime]
            }
            for regime, stats in regime_stats.items()
        }
        
        # Add labels to the dataframe
        df['regime_label'] = df['regime_smoothed'].map(regime_labels)
        
        # Print information about the regimes
        print("\nRegime Classification Results:")
        print("-----------------------------")
        for regime, info in self.regime_characteristics.items():
            print(f"Regime {regime} => {info['label']}")
            print(f"  Average Return: {info['stats']['avg_return']:.4%}")
            print(f"  Volatility: {info['stats']['volatility']:.4%}")
        
        return df

    def save(self, model_path_prefix=None):
        # Use the provided model_path_prefix or the instance's model_path_prefix
        if model_path_prefix is None:
            model_path_prefix = self.model_path_prefix
        
        if model_path_prefix is None:
            raise ValueError("Model path prefix is not set.")
        
        joblib.dump(self.model, f"{model_path_prefix}_hmm.pkl")
        joblib.dump(self.scaler, f"{model_path_prefix}_scaler.pkl")
        print(f"[Saved] Model and scaler to {model_path_prefix}_*.pkl")

    def load(self):
        if self.model_path_prefix is None:
            raise ValueError("Model path prefix is not set.")
        self.model = joblib.load(f"{self.model_path_prefix}_hmm.pkl")
        self.scaler = joblib.load(f"{self.model_path_prefix}_scaler.pkl")
        print(f"[Loaded] Model and scaler from {self.model_path_prefix}_*.pkl")
