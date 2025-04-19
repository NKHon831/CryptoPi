from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from config import LOG_RETURN_THRESHOLD

class MarketRegimeHMM:
    def __init__(self, n_states=3, features=None, covariance_type='full', n_iter=100, tol=1e-4, model_path_prefix=None):
        self.n_states = n_states
        self.features = features if features is not None else []
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.model = None
        self.scaler = None
        self.model_path_prefix = model_path_prefix

    @staticmethod
    def load_and_prepare_data(path):
        df_raw = pd.read_csv(path)
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
        Label regimes as '-1', '1', or '0' based on their signal.
        
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
        regime_labels[sorted_regimes[0][0]] = '-1'  # Lowest returns -> Bear
        regime_labels[sorted_regimes[-1][0]] = '1'  # Highest returns -> Bull
        
        # The middle regime (or the one with lowest volatility if only 2 regimes)
        if len(sorted_regimes) > 2:
            regime_labels[sorted_regimes[1][0]] = '0'
        else:
            # If only 2 regimes, use volatility to determine which is neutral vs bull/bear
            remaining_regime = [r for r in range(self.n_states) if r not in regime_labels.keys()][0]
            regime_labels[remaining_regime] = '0'
        
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

    def tune_hyperparameters(self, df, param_grid=None, cv=5, scoring='neg_log_likelihood'):
        """
        Perform hyperparameter tuning using Grid Search CV.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data with features for the HMM model
        param_grid : dict, optional
            Dictionary with parameters names as keys and lists of parameter values
            Default grid searches over n_states and covariance_type
        cv : int, optional
            Number of folds for cross-validation
        scoring : str, optional
            Scoring metric to use ('neg_log_likelihood' by default)
            
        Returns:
        --------
        dict
            Best parameters found during grid search
        float
            Best score achieved
        pandas.DataFrame
            DataFrame with all grid search results
        """
        
        # Default parameter grid if none is provided
        if param_grid is None:
            param_grid = {
                'covariance_type': ['diag', 'full', 'tied', 'spherical'],
                'n_iter': [100, 200],
                'tols':[1e-2, 1e-3, 1e-4],
            }
        
        # Extract features data
        X = df[self.features].values
        
        # We'll store results here
        results = []
        
        # Create time series split for CV
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Track best parameters and score
        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Generate all parameter combinations
        from itertools import product
        param_combinations = list(dict(zip(param_grid.keys(), values)) 
                                for values in product(*param_grid.values()))
        
        print(f"Performing grid search over {len(param_combinations)} parameter combinations...")
        
        for params in param_combinations:
            print(f"Testing parameters: {params}")
            fold_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and fit HMM model with current parameters
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=params['covariance_type'],
                    n_iter=params['n_iter'],
                    tol=params['tols'],
                    random_state=42
                )
                
                try:
                    model.fit(X_train_scaled)
                    # Calculate score (higher is better for log likelihood)
                    score = model.score(X_test_scaled)
                    fold_scores.append(score)
                except Exception as e:
                    print(f"Error with parameters {params}: {str(e)}")
                    fold_scores.append(-np.inf)
            
            # Average score across folds
            avg_score = np.mean(fold_scores) if fold_scores else -np.inf
            
            # Save results
            results.append({**params, 'score': avg_score})
            
            # Update best parameters if we found a better score
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")
        
        return best_params, best_score, results_df

    @staticmethod
    def save_best_params(best_params, config_file='config.py'):
        """
        Save the best parameters to the config.py file.
        
        Parameters:
        -----------
        best_params : dict
            Dictionary of best parameters from hyperparameter tuning
        config_file : str
            Path to the config file
        """
        import re
        
        # Read existing config file
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Update parameters in the config file
        for param_name, param_value in best_params.items():
            if param_name == 'n_states':
                pattern = r'N_HMM_STATES\s*=\s*\d+'
                replacement = f'N_HMM_STATES = {param_value}'
                config_content = re.sub(pattern, replacement, config_content)
            
            # Add other parameters as needed
            # For example, if you want to save covariance_type:
            elif param_name == 'covariance_type':
                # Check if HMM_COVARIANCE_TYPE already exists in the config
                if 'HMM_COVARIANCE_TYPE' in config_content:
                    pattern = r'HMM_COVARIANCE_TYPE\s*=\s*[\'"]\w+[\'"]'
                    replacement = f'HMM_COVARIANCE_TYPE = "{param_value}"'
                    config_content = re.sub(pattern, replacement, config_content)
                else:
                    # Add the parameter if it doesn't exist
                    config_content += f'\nHMM_COVARIANCE_TYPE = "{param_value}"'
            
            elif param_name == 'n_iter':
                if 'HMM_N_ITER' in config_content:
                    pattern = r'HMM_N_ITER\s*=\s*\d+'
                    replacement = f'HMM_N_ITER = {param_value}'
                    config_content = re.sub(pattern, replacement, config_content)
                else:
                    config_content += f'\nHMM_N_ITER = {param_value}'
            
            elif param_name == 'tols':
                if 'HMM_TOL' in config_content:
                    pattern = r'HMM_TOL\s*=\s*[\d\.eE+-]+'
                    replacement = f'HMM_TOL = {param_value}'
                    config_content = re.sub(pattern, replacement, config_content)
                else:
                    config_content += f'\HMM_TOL = {param_value}'
        
        # Write updated config back to file
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"Best parameters saved to {config_file}")

    @staticmethod
    def plot_price_with_regime(df):

        if 'close' not in df.columns or 'regime_label' not in df.columns:
            print("Missing 'close' or 'regime_label' column. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(20, 6))

        # Define colors based on your label encoding
        colors = {
            -1: 'red',    # Bear
            0: 'gray',   # Neutral
            1: 'green',  # Bull
        }

        # Plot price
        ax.plot(df.index, df['close'], label='Price', color='black')

        # Overlay regimes
        prev_regime = None
        start_idx = None
        for i in range(len(df)):
            curr_regime = df['regime_label'].iloc[i]
            if prev_regime is None:
                prev_regime = curr_regime
                start_idx = df.index[i]
            if curr_regime != prev_regime or i == len(df) - 1:
                end_idx = df.index[i] if i == len(df) - 1 else df.index[i - 1]
                ax.axvspan(start_idx, end_idx, color=colors.get(prev_regime, 'blue'), alpha=0.2)
                prev_regime = curr_regime
                start_idx = df.index[i]

        ax.set_title("Price with Regime Backgrounds")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()