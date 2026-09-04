import joblib
from pathlib import Path


def load_model():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "sentiment_model.joblib"

    return joblib.load(model_path)


def test_positive_prediction():
    model = load_model()

    prediction = model.predict(
        ["This movie was absolutely fantastic and enjoyable."]
    )[0]

    assert prediction == 1


def test_negative_prediction():
    model = load_model()

    prediction = model.predict(
        ["This movie was terrible and extremely boring."]
    )[0]

    assert prediction == 0