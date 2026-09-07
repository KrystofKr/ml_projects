import pandas as pd
import pytest

from titanic_ml.data import schema_validation, split_features_target


def make_valid_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": ["Passenger One", "Passenger Two"],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "PC 17599"],
            "Fare": [7.25, 71.2833],
            "Cabin": [None, "C85"],
            "Embarked": ["S", "C"],
        }
    )


def test_schema_validation_accepts_train_without_test() -> None:
    train = make_valid_train()

    schema_validation(train)

def test_schema_validation_rejects_missing_inference_column() -> None:
    train = make_valid_train()
    test = train.drop(columns=["Survived", "Fare"]).copy()
    with pytest.raises(ValueError, match=r"Inference data is missing columns.*Fare"):
        schema_validation(train, test)

def test_schema_validation_accepts_valid_train_and_inference() -> None:
    train = make_valid_train()
    test = train.drop(columns=["Survived"]).copy()
    schema_validation(train, test)

def test_schema_validation_rejects_target_in_inference() -> None:
    train = make_valid_train()
    test = train.copy()

    with pytest.raises(ValueError,match=r"Target feature cannot be included in inference data.*Survived"):
        schema_validation(train, test)

def test_schema_validation_rejects_duplicate_train_passenger_id() -> None:
    train = make_valid_train()
    train.loc[1, "PassengerId"] = train.loc[0, "PassengerId"]
    with pytest.raises(ValueError, match=r"PassengerId.*unique.*training"):
        schema_validation(train)

def test_schema_validation_rejects_duplicate_inference_passenger_id() -> None:
    train = make_valid_train()
    test = train.drop(columns=["Survived"]).copy()
    test.loc[1, "PassengerId"] = test.loc[0, "PassengerId"]
    with pytest.raises(ValueError, match=r"PassengerId.*unique.*inference"):
        schema_validation(train, test)        

def test_split_features_target_removes_target() -> None:
    train = make_valid_train()
    X,y = split_features_target(train)

    assert "Survived" not in X.columns
    pd.testing.assert_series_equal(y, train["Survived"])

def test_split_features_target_returns_independent_copies() -> None:
    train = make_valid_train()
    original = train.copy()

    X, y = split_features_target(train)    
    X.loc[0, "Age"] = X.loc[1,"Age"]
    y.loc[0] = y.loc[1]

    pd.testing.assert_frame_equal(train, original)

def test_split_features_target_rejects_missing_target() -> None:
    train = make_valid_train().drop(columns=["Survived"])
    with pytest.raises(KeyError, match=r"Missing target column.*Survived"):
        split_features_target(train)    