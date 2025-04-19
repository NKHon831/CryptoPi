## MarketRegimeHMM

A class to identify market regimes using a Gaussian Hidden Markov Model (HMM). This tool is especially useful for detecting patterns in financial time series, labeling different market conditions like bull, bear, and neutral regimes.

### `MarketRegimeHMM()` Class

#### Parameters

- n_states (int): Number of hidden states (regimes) for the HMM.
- features (list): List of column names used as features for modeling.
- covariance_type (str): Type of covariance ('full', 'diag', etc.) for the HMM.
- n_iter (int): Maximum number of iterations during model training.
- tol (float): Convergence threshold.
- model_path_prefix (str): Optional prefix for saving/loading models.

#### Methods

| Method                                                                        | Description                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| load_and_prepare_data(path)                                                   | Loads and formats a CSV file.                                                                                                                                                                                                                                             |
| fit(df)                                                                       | Fits the HMM on the specified features in the DataFrame.                                                                                                                                                                                                                  |
| predict(df)                                                                   | Predict regimes using the trained model.                                                                                                                                                                                                                                  |
| smooth_regime(df, window=7)                                                   | Applies rolling mode to smooth the regime predictions.                                                                                                                                                                                                                    |
| compute_blocks(df)                                                            | Identifies distinct regime blocks and changes.                                                                                                                                                                                                                            |
| label_regimes(df)                                                             | Labels each smoothed regime based on average return and volatility.                                                                                                                                                                                                       |
| save(model_path_prefix=None)                                                  | Saves the trained HMM model and the associated scaler using joblib.                                                                                                                                                                                                       |
| load()                                                                        | Loads a previously saved HMM model and scaler from disk.                                                                                                                                                                                                                  |
| tune_hyperparameters(df, param_grid=None, cv=5, scoring='neg_log_likelihood') | Performs hyperparameter tuning using Grid Search with time-series cross-validation.                                                                                                                                                                                       |
| save_best_params(best_params, config_file='config.py')                        | This method updates a configuration file (config.py) with the best hyperparameters found during model tuning, such as the number of HMM states or iteration count. This allows reproducibility and consistency across experiments by persisting optimal parameter values. |
| plot_price_with_regime(df)                                                    | This method visualizes the price trend of an asset alongside its detected market regimes (bull, bear, or neutral). It uses color-coded background shading to indicate different regimes over time.                                                                        |
