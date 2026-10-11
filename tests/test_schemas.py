import pytest
from pydantic import ValidationError

from app.schemas import HeartFeatures


def valid_payload():
    return {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
    }


def test_valid_payload_is_accepted():
    features = HeartFeatures(**valid_payload())
    assert list(features.to_feature_dict()) == [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]


def test_invalid_category_is_rejected():
    payload = valid_payload()
    payload["cp"] = 4
    with pytest.raises(ValidationError):
        HeartFeatures(**payload)


def test_unknown_fields_are_rejected():
    payload = valid_payload()
    payload["made_up_feature"] = 9
    with pytest.raises(ValidationError):
        HeartFeatures(**payload)
