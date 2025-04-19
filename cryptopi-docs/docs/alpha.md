## Final Alpha Model

This module provides a unified interface for training and evaluating machine learning models, including a Transformer-based deep learning model for sequential data and an XGBoost model for tabular data.

### BaseModel

Abstract base class to enforce a consistent interface across all models. All model classes should inherit from BaseModel.

### TransformerModel

This is a PyTorch-based Transformer model designed for time series or sequential data. The model processes input sequences using self-attention mechanisms and is trained to perform binary classification.

#### Parameters

| Parameter       | Type  | Description                                     |
| --------------- | ----- | ----------------------------------------------- |
| input_dim       | int   | Number of features in each time step.           |
| seq_len         | int   | Sequence length expected per sample.            |
| d_model         | int   | Dimension of the Transformer model.             |
| nhead           | int   | Number of attention heads.                      |
| num_layers      | int   | Number of Transformer encoder layers.           |
| dropout         | float | Dropout rate.                                   |
| lr              | float | Learning rate for the optimizer.                |
| epochs          | int   | Number of training epochs.                      |
| batch_size      | int   | Batch size for training.                        |
| device          | str   | Device to train on ('cuda' or 'cpu').           |
| checkpoint_path | str   | File path for saving/loading model checkpoints. |

#### Methods

| Method            | Description                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------- |
| train(X, y)       | Trains the Transformer model with early stopping based on validation loss.                  |
| predict(X)        | Returns binary class predictions for the input data.                                        |
| predict_proba(X)  | Returns class probabilities for the input data.                                             |
| \_preprocess_x(X) | Reshapes and normalizes input data to match the sequence format required by the Transformer |
| save_model()      | Saves the model checkpoint to the specified path.                                           |
| load_model()      | Loads the model checkpoint from the specified path.                                         |

### XGBoostModel

This is a wrapper around XGBoost’s XGBClassifier, suitable for tabular classification problems.

#### Parameters

`**kwargs`: Any parameters accepted by `xgb.XGBClassifier`.

#### Methods

| Method                           | Description                                                             |
| -------------------------------- | ----------------------------------------------------------------------- |
| train(X, y) -> None              | Fits the XGBoost model using the input feature set and target variable. |
| predict(X) -> pd.Series          | Returns predicted labels for the input data.                            |
| predict_proba(X) -> pd.DataFrame | Returns prediction probabilities for each class.                        |

### AlphaLogisticRegression

Initializes the logistic regression model and standard scaler.

#### Methods

| Method                                                                             | Description                                                                      |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| load_data() -> pd.DataFrame                                                        | Loads time-series alpha factor data from a CSV file.                             |
| split_data(X, y, train_start_date, train_end_date, test_start_date, test_end_date) | Splits the input data into training and testing sets using time windows.         |
| train_test_split(df, labels)                                                       | Performs a traditional sklearn-style train-test split (not time-aware).          |
| preprocess(df)                                                                     | Preprocesses raw OHLCV data by computing log returns and removing price columns. |
| generate_label(df)                                                                 | Generates binary labels from log_returns.                                        |
| feature_selection(X_train, y_train)                                                | Selects top 20 features using ANOVA F-test.                                      |
