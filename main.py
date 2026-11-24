from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model = joblib.load("logistic_regression_model.pkl")
preprocessor = joblib.load("heart_preprocessor.pkl")
threshold = joblib.load("best_threshold.pkl")

# Create FastAPI application
app = FastAPI(
    title="Heart Disease Classification API",
    description="Logistic Regression based survey status classification",
    version="1.0"
)

# Input schema
class HeartData(BaseModel):

    BMI: float
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Heart Disease Classification API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: HeartData):

    # Convert input into DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Get probability for Yes class
    yes_index = list(model.classes_).index("Yes")

    probability = model.predict_proba(processed_data)[0][yes_index]

    # Apply selected threshold
    prediction = "Yes" if probability >= threshold else "No"

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": float(threshold),
        "message": "This is a classification of self-reported survey status, not a medical diagnosis."
    }