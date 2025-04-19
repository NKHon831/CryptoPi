from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def train(self, X:pd.DataFrame, y:pd.Series) -> None:
        """
        Train the model using the provided features and target variable.

        Parameters:
        X (pd.DataFrame): Feature set for training.
        y (pd.Series): Target variable for training.
        """
        pass

    @abstractmethod
    def predict(self, X:pd.DataFrame) -> pd.Series:
        """
        Predict the target variable using the provided features.

        Parameters:
        X (pd.DataFrame): Feature set for prediction.

        Returns:
        pd.Series: Predicted values.
        """
        pass
