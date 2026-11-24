# Heart Disease Classification

## Project Overview
This project uses Machine Learning to classify self-reported heart disease status using the CDC BRFSS 2020 dataset.

## Dataset
- Source: CDC BRFSS 2020
- Target: HeartDisease
- Features: 17 input features

## Machine Learning
- Algorithm: Logistic Regression
- Data preprocessing: StandardScaler and OneHotEncoder
- Class imbalance handling: Class Weight Balancing
- Threshold tuning using validation data

## Model Evaluation
- Accuracy: 81%
- Recall (HeartDisease Yes): 66%
- Precision (HeartDisease Yes): 27%
- F1-score (HeartDisease Yes): 38%
- ROC-AUC: 0.832
- PR-AUC: 0.344

## API
Built using FastAPI.

### Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

`http://127.0.0.1:8000/docs`

## Project Structure

- `heart_disease_classification.ipynb` – ML notebook
- `main.py` – FastAPI application
- `logistic_regression_model.pkl` – Trained model
- `heart_preprocessor.pkl` – Preprocessing pipeline
- `best_threshold.pkl` – Selected classification threshold
- `requirements.txt` – Dependencies

## Disclaimer
This project classifies self-reported heart disease status from survey data. It is an educational machine learning project and is not intended for medical diagnosis or clinical decision-making.