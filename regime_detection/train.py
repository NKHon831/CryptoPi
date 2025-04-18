import os
import pandas as pd
from market_regime_hmm import MarketRegimeHMM  # Ensure this imports your class correctly
from config import INPUT_CSV_PATH, OUTPUT_CSV_PATH, OUTPUT_JPG_PATH, N_HMM_STATES, HMM_FEATURES, MODEL_PATH_PREFIX
import matplotlib.pyplot as plt

def main():
    # Step 1: Load and prepare data
    print("Loading data...")
    df = MarketRegimeHMM.load_and_prepare_data(INPUT_CSV_PATH)

    # Step 2: Initialize the MarketRegimeHMM model
    print("Initializing MarketRegimeHMM model...")
    model = MarketRegimeHMM(n_states=N_HMM_STATES, features=HMM_FEATURES)

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

if __name__ == "__main__":
    main()
