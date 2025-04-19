import xgboost as xgb
import pandas as pd
from .base import BaseModel

class XGBoostModel(BaseModel):
    
    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model with optional parameters.

        Parameters:
        **kwargs: Additional parameters for XGBoost.
        """
        self.model = xgb.XGBClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the XGBoost model using the provided features and target variable.

        Parameters:
        X (pd.DataFrame): Feature set for training.
        y (pd.Series): Target variable for training.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target variable using the provided features.

        Parameters:
        X (pd.DataFrame): Feature set for prediction.

        Returns:
        pd.Series: Predicted values.
        """
        return self.model.predict(X)
    
