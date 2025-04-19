import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import your models
from alpha.xgboost import XGBoostModel
from alpha.transformer import TransformerModel

def generate_dummy_data(num_samples=1000, sequence_length=10, n_features=16):
    total_features = sequence_length * n_features
    X_np = np.random.rand(num_samples, total_features)
    y_np = np.random.randint(0, 2, size=(num_samples,))
    
    X = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(total_features)])
    y = pd.Series(y_np)
    return X, y

def test_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoostModel...")
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)
    preds = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"XGBoost Accuracy: {acc:.4f}\n")

def test_transformer(X_train, y_train, X_test, y_test, sequence_length, n_features):
    print("Training TransformerModel...")
    transformer_model = TransformerModel(
        sequence_length=sequence_length,
        n_features=n_features,
        epochs=5  # Adjust for quick test
    )
    transformer_model.train(X_train, y_train)
    preds = transformer_model.predict(X_test)

    acc = accuracy_score(y_test, preds.astype(int))
    print(f"Transformer Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    # Set configuration
    sequence_length = 10
    n_features = 16

    # Generate data
    X, y = generate_dummy_data(num_samples=1000, sequence_length=sequence_length, n_features=n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run tests
    test_xgboost(X_train, y_train, X_test, y_test)
    test_transformer(X_train, y_train, X_test, y_test, sequence_length, n_features)
