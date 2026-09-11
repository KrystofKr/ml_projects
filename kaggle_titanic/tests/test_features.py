import pandas as pd
from titanic_ml.features import TitanicFeatureEngineer
import pytest

df = pd.DataFrame(
    {
        "PassengerId": [1, 2, 3],
        "Pclass": [1, 2, 3],
        "Name": [
            "Smith, Mr. John",
            "Smith, Mrs. Anna",
            "Brown, Mr. Peter",
        ],
        "Sex": ["male", "female", "male"],
        "Age": [30, 28, 40],
        "SibSp": [1, 1, 0],
        "Parch": [0, 0, 0],
        "Ticket": ["A123", "A123", "B456"],
        "Fare": [50.0, 50.0, 10.0],
        "Cabin": ["C85", "C85", None],
        "Embarked": ["S", "S", "C"],
    }
)


def test_fit() -> None:
    engineer = TitanicFeatureEngineer()
    df_trimmed = df.drop(columns=["Ticket"]).copy()

    with pytest.raises(ValueError, match=r"Missing.*Ticket"):
        engineer.fit(df_trimmed)

    original = df.copy()
    result = TitanicFeatureEngineer().fit_transform(df)

    assert result["FamilySize"].tolist() == [2,2,1]
    assert result["IsAlone"].tolist() == [0,0,1]
    assert result["FarePerPerson"].tolist() == [25.0, 25.0, 10.0]
    assert result["CabinDeck"].iloc[:2].tolist() == ["C", "C"]
    assert pd.isna(result["CabinDeck"].iloc[2])
    assert {"PassengerId", "Name", "Ticket", "Cabin"}.isdisjoint(result.columns)


    pd.testing.assert_frame_equal(df, original)

