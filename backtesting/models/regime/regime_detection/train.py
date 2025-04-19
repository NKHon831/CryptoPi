import os
import pandas as pd
from market_regime_hmm import MarketRegimeHMM  # Ensure this imports your class correctly
from config import INPUT_CSV_PATH, OUTPUT_CSV_PATH, N_HMM_STATES, HMM_FEATURES, MODEL_PATH_PREFIX
import matplotlib.pyplot as plt
from pprint import pprint
from config import LOG_RETURN_THRESHOLD, PERFORM_TUNING, HMM_COVARIANCE_TYPE, HMM_N_ITER, HMM_TOL

def main():
    # Step 1: Load and prepare data
    print("Loading data...")
    df = MarketRegimeHMM.load_and_prepare_data(INPUT_CSV_PATH)

    # Step 2: Initialize the MarketRegimeHMM model
    print("Initializing MarketRegimeHMM model...")
    model = MarketRegimeHMM(n_states=N_HMM_STATES, features=HMM_FEATURES)
    # Optional: Perform hyperparameter tuning    
    if PERFORM_TUNING:
        print("Performing hyperparameter tuning...")
        # Define parameter grid for tuning
        param_grid = {
            'covariance_type': ['diag', 'full', 'tied', 'spherical'],
            'n_iter': [100, 200, 300],
            'tols': [1e-2, 1e-3, 1e-4]
        }
        
        # Perform tuning
        best_params, best_score, results_df = model.tune_hyperparameters(df, param_grid)
        
        # Save best parameters to config file
        model.save_best_params(best_params)
        
        # Update model with best parameters
        model = MarketRegimeHMM(
            features=HMM_FEATURES,
            covariance_type=best_params['covariance_type'],
            n_iter=best_params['n_iter'],
            tol=best_params['tols']
        )
        
        # Optionally save tuning results
        results_df.to_csv(os.path.join(OUTPUT_CSV_PATH, 'hyperparameter_tuning_results.csv'))
    else:
        # Using parameters from config
        model = MarketRegimeHMM(
            n_states=N_HMM_STATES, 
            features=HMM_FEATURES,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL
        )
    # Step 3: Fit the model
    print("Fitting the model...")
    df = model.fit(df)

    # Step 4: Save the model and scaler
    print("Saving model and scaler...")
    model.save(MODEL_PATH_PREFIX)

    # Step 5: Perform smoothing and compute blocks
    print("Smoothing regimes...")
    df = model.smooth_regime(df, window=7)
    df = model.compute_blocks(df)

    # Step 6: Label the regimes (Bear, Bull, Neutral)
    print("Labeling regimes...")
    df = model.label_regimes(df)
    counts = df['regime_label'].value_counts().sort_index()
    print("Regime label counts:")
    print(counts)

    # Step 7: Save the output CSV
    print(f"Saving regime CSV to {OUTPUT_CSV_PATH}...")
    os.makedirs(OUTPUT_CSV_PATH, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_CSV_PATH, 'regime_data.csv'))

    # Step 8: Optionally save graphs or plots if needed
    # print("Saving regime plot...")
    # os.makedirs(OUTPUT_JPG_PATH, exist_ok=True)
    # # Assuming model has a plot function, if not, you can implement one in MarketRegimeHMM class
    # model.plot_regimes(df)  # This would need to be implemented if desired
    # plt.savefig(os.path.join(OUTPUT_JPG_PATH, 'regime_plot.jpg'))

    print("Training complete.")

    # df, report, cm = model.evaluate_prediction_using_log_return(df)
    # pprint(report)
    model.plot_price_with_regime(df)

    regime_returns = df.groupby('regime_label')['log_returns'].mean()
    print(regime_returns)

if __name__ == "__main__":
    main()
