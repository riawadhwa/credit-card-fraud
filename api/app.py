from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
rf = joblib.load("random_forest.pkl")
dt = joblib.load("decision_tree.pkl")
stacking_model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")
cnn = load_model("cnn_model.keras")

app = FastAPI(title="Credit Card Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.post("/predict")
def predict(transaction: Transaction):

    # ✅ CORRECT FEATURE ORDER (matches training)
    X = np.array([[
        transaction.Time,
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28,
        transaction.Amount   # ✅ Amount LAST
    ]])

    # ✅ CORRECT SCALING (Time + Amount only)
    X[:, [0, -1]] = scaler.transform(X[:, [0, -1]])

    # Base model predictions
    dt_prob = dt.predict_proba(X)[:, 1]
    rf_prob = rf.predict_proba(X)[:, 1]

    # CNN prediction
    X_cnn = X.reshape(1, X.shape[1], 1)
    cnn_prob = cnn.predict(X_cnn, verbose=0).ravel()

    # Stacking
    stack_input = np.hstack([
        dt_prob.reshape(-1, 1),
        rf_prob.reshape(-1, 1),
        cnn_prob.reshape(-1, 1)
    ])

    final_prob = stacking_model.predict_proba(stack_input)[:, 1]

    THRESHOLD = 0.2
    prediction = "Fraud" if final_prob[0] > THRESHOLD else "Legit"

    return {
        "decision_tree_score": float(dt_prob[0]),
        "random_forest_score": float(rf_prob[0]),
        "cnn_score": float(cnn_prob[0]),
        "final_risk_score": float(final_prob[0]),
        "prediction": prediction
    }
