# Repository Structure Guide

1. `assets/`:
   Contains static assets such as images, logos, and screenshots used in documentation or UI (e.g., Streamlit dashboards or README.md).

2. `backtesting/`:
   Core engine for simulating trading strategies on historical data.

3. `docs/`:
   Contains external-facing documentation files.

4. `models/`:
   Houses the machine learning models used in the pipeline.

   - `alpha/`: Transformer-based model that acts as the final predictor of future price movements.
   - `regime/`: Detects market regimes using a classification model on market and on-chain indicators.
   - `sentiment/`: Extracts sentiment scores and labels from news and Twitter posts using a Hugging Face model.

5. `scripts/`:
   Command-line scripts for automating common workflows.

_Note_:

1. **Important**: Please create new feature branches before commiting new code.
2. `.gitkeep` is an empty file used to ensure that otherwise-empty directories are tracked by Git (since Git ignores empty folders by default). It has no functional impact on the project.
