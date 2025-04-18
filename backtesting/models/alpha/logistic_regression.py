import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessor from sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Model Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Feature Selection and Feature Importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier

# Class encoder
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LogisticRegression

class AlphaLogisticRegression():

    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced")
        self.scaler = StandardScaler()  
    
    # Load the data TODO: Change the way of loading data using datahandler
    def load_data(self):
        df = pd.read_csv("data/final-alpha-model-data.csv", index_col=0, parse_dates=True)

        return df
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_start_date: str | pd.Timestamp = "2020-01-01",  # Start of training period
        train_end_date: str | pd.Timestamp = "2021-12-31",  # End of training period (for backtesting)
        test_start_date: str | pd.Timestamp = "2022-01-01",  # Start of testing period (for forward testing)
        test_end_date: str | pd.Timestamp = "2022-12-31"  # End of testing period
    ) -> tuple:
        
        # Ensure the index is a datetime type for X
        X.index = pd.to_datetime(X.index)
        
        # If y does not have a timestamp index, align y to X's index
        y.index = X.index  # Align y with the X index
        
        # Sort the data by index (timestamp)
        X = X.sort_index()
        y = y.sort_index()  # Sort y to align with X
        
        # Apply the date filters for train and test data
        train_mask = (X.index >= pd.to_datetime(train_start_date)) & (X.index <= pd.to_datetime(train_end_date))
        test_mask = (X.index >= pd.to_datetime(test_start_date)) & (X.index <= pd.to_datetime(test_end_date))

        # Split the data based on the masks
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        return X_train, X_test, y_train, y_test

    def train_test_split(self, df: pd.DataFrame, labels: pd.DataFrame) -> tuple:
        return train_test_split(df, labels, test_size=0.2, random_state=42, shuffle=False)

    # Preprocessing logic 
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        # Calculate the log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Drop the OHLC columns
        df = df.drop(columns=["close", "high", "low", "open"])
        df = df.dropna()

        return df
    
    # Labels generation logic
    def generate_label(self, df: pd.DataFrame) -> pd.DataFrame:

        Y = pd.DataFrame(list(np.where(df["log_returns"] >0, 1, -1)))
        
        # Transform Y to a binary classification problem's values (0 and 1)
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)

        Y = pd.DataFrame(encoded_Y, columns=["label"], index=df.index)
        return Y


    # Feature Selection Logic
    def feature_selection(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:

        bestfeatures = SelectKBest(score_func=f_classif, k=len(X_train.keys()))
        fit = bestfeatures.fit(X_train, y_train)

        # Concatenate two dataframes for better visualization
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ["Specs", "Score"]

        # Select the best 20 features
        best_features = list(featureScores.nlargest(20, "Score")["Specs"])
        for i in list(featureScores["Specs"]):
            if (i not in best_features):
                X_train = X_train.drop(i, axis=1)

        return X_train, featureScores
    
    # TODO: Hyperparameter tuning logic


    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train the Logistic Regression model using the provided features and target variable.

        Parameters:
        X (pd.DataFrame): Feature set for training.
        y (pd.Series): Target variable for training.
        """
        
        # Fit the model
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
    
    def evaluate(self, X:pd.DataFrame, y:pd.DataFrame):

        y_pred = self.model.predict(X)
        
        print("Accuracy: ", accuracy_score(y, y_pred))

        return pd.DataFrame({
            "date": X.index,
            "predictions": y_pred
        })

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        """
        Plot the confusion matrix for the model predictions.

        Parameters:
        y_true (pd.Series): True target variable.
        y_pred (pd.Series): Predicted target variable.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


def main():
    LR = AlphaLogisticRegression()
    
    # Load, preprocess and split the data
    df = LR.load_data()
    df = LR.preprocess(df)
    labels= LR.generate_label(df)
    X_train, X_test, y_train, y_test = LR.split_data(df, labels)

    # Feature Selection
    X_train, _ = LR.feature_selection(X_train, y_train)
    X_train_scaled = LR.scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index)

    
    # Transform the testing data using the same preprocessing logic with the training data
    X_test = X_test[X_train.columns]
    X_test_scaled = LR.scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index)

    # Train the model (Backward Testing)
    LR.train(X_train_scaled, y_train)

    df_backtest = LR.evaluate(X_train_scaled, y_train)

    # Forward Testing
    y_pred = LR.predict(X_test_scaled)
    df_forwardtest = pd.DataFrame({
        "date": X_test.index,
        "predictions": y_pred
    })
    LR.plot_confusion_matrix(y_test, y_pred.astype(int))

    df_backtest.to_csv("../../data/dbacktest_predictions.csv", index=False)
    df_forwardtest.to_csv("../../data/forwardtest_predictions.csv", index=False)

    return df_backtest, df_forwardtest


if __name__ == "__main__":
    main()


