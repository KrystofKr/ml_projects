from pathlib import Path
import pandas as pd

TARGET = "Survived"

FEATURE_COLUMNS = {
    "PassengerId",
    "Pclass",  
    "Name",    
    "Sex",    
    "Age",
    "SibSp",  
    "Parch",  
    "Ticket",    
    "Fare",
    "Cabin",
    "Embarked", 
}

def schema_validation(train: pd.DataFrame, test: pd.DataFrame | None = None) -> None:
    if TARGET not in train.columns:
        raise ValueError(f"Target feature is not included in train data: {TARGET!r}")

    missing_train = FEATURE_COLUMNS - set(train.columns)
    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")

    if not train["PassengerId"].is_unique:
        raise ValueError("PassengerId must be unique in training data.")

    if test is not None:

        if TARGET in test.columns:
            raise ValueError(f"Target feature cannot be included in inference data: {TARGET!r}")

        missing_test = FEATURE_COLUMNS - set(test.columns)
        if missing_test:
            raise ValueError(f"Inference data is missing columns: {sorted(missing_test)}")

        if not test["PassengerId"].is_unique:
            raise ValueError("PassengerId must be unique in inference data.")

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import kagglehub
    except ImportError as exp:
        raise RuntimeError("kagglehub is needed for load of data.") from exp

    data_path = Path(kagglehub.competition_download("titanic"))
    train = data_path / "train.csv"
    test = data_path / "test.csv"

    missing_files = [ file.name for file in (train, test) if not file.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "KaggleHub download is missing required files: "
            f"{', '.join(sorted(missing_files))}."
        )

    train_csv = pd.read_csv(train)
    test_csv = pd.read_csv(test)

    schema_validation(train_csv, test_csv)
    return train_csv, test_csv

def split_features_target(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in train.columns:
        raise KeyError(f"Missing target column: {TARGET}")
    X = train.drop(columns=[TARGET]).copy()
    y = train[TARGET].copy()
    return X,y