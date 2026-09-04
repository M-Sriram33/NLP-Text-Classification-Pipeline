import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


# Create FastAPI application
app = FastAPI(
    title="IMDb Sentiment Classifier",
    description="Sentiment classification API using TF-IDF and Linear SVM",
    version="1.0.0",
)


# Load trained model
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "sentiment_model.joblib"

model = joblib.load(model_path)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "IMDb Sentiment Classification API",
        "status": "running",
    }


@app.post("/predict")
def predict(request: TextRequest):
    prediction = model.predict([request.text])[0]

    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "sentiment": sentiment
    }