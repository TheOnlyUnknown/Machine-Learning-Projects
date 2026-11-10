import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
# The .pkl files are read from next to this file unless MODEL_DIR says otherwise.
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR))
MAX_BATCH = 200

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(MODEL_DIR / "credit_risk.pkl")
    # Cast to a plain float so the value is always JSON serialisable.
    ml_model["threshold"] = float(joblib.load(MODEL_DIR / "best_thresholds.pkl"))

    yield

    ml_model.clear()


app = FastAPI(title="Credit Risk Studio", lifespan=lifespan)


# The only columns the user will see and provide inputs for
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


def score(applications: list[LoanApplication]) -> list[float]:
    input_df = pd.DataFrame([a.model_dump() for a in applications])
    try:
        probabilities = ml_model["model"].predict_proba(input_df)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model failed to score: {exc}")
    return [float(p) for p in probabilities]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_model}


@app.post("/predict")
def predict(data: LoanApplication):
    probability = score([data])[0]
    threshold = ml_model["threshold"]
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


@app.post("/predict/batch")
def predict_batch(data: list[LoanApplication]):
    """Scores many applications in one call. The UI uses this for the what-if curve."""
    if not data or len(data) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_BATCH} applications.")
    return {"probabilities": score(data), "threshold": ml_model["threshold"]}
