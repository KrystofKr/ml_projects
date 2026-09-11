from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from titanic_ml.data import FEATURE_COLUMNS

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        missing = FEATURE_COLUMNS - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        df["CabinDeck"] = df["Cabin"].str[0]

        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        return df
        