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
    
    # def split_data(
    #     self,
    #     X: pd.DataFrame,
    #     y: pd.DataFrame,
    #     date_col: str = "timestamp",
    #     cutoff_date: str | pd.Timestamp = None
    # ) -> tuple:
        
    #     X = X.drop(columns=["timestamp"], errors="ignore")  # Avoid duplication
    #     X[date_col] = X.index  # Assuming the index is the date column

    #     # Ensure timestamp column is datetime
    #     X[date_col] = pd.to_datetime(X[date_col])

    #     # Sort X by date and align y
    #     X = X.sort_values(by=date_col)
    #     y = y.loc[X.index]  # ensure alignment

    #     # Use provided cutoff_date or default to 1 year before the max date
    #     if cutoff_date is None:
    #         cutoff_date = X[date_col].max() - pd.DateOffset(years=1)
    #     else:
    #         cutoff_date = pd.to_datetime(cutoff_date)

    #     # Split based on cutoff
    #     train_idx = X[date_col] < cutoff_date
    #     test_idx = X[date_col] >= cutoff_date

    #     X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    #     y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    #     return X_train, X_test, y_train, y_test

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
    X_train, X_test, y_train, y_test = LR.train_test_split(df, labels)

    # Feature Selection
    X_train, _ = LR.feature_selection(X_train, y_train)
    X_train_scaled = LR.scaler.fit_transform(X_train)

    
    # Transform the testing data using the same preprocessing logic with the training data
    X_test = X_test[X_train.columns]
    X_test_scaled = LR.scaler.transform(X_test)

    # Train the model (Backward Testing)
    LR.train(X_train_scaled, y_train)

    LR.evaluate(X_train_scaled, y_train)

    
    # Forward Testing
    y_pred = LR.predict(X_test_scaled)
    LR.plot_confusion_matrix(y_test, y_pred.astype(int))

if __name__ == "__main__":
    main()


